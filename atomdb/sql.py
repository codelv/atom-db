"""
Copyright (c) 2018-2019, Jairus Martin.

Distributed under the terms of the MIT License.

The full license is in the file LICENSE.txt, distributed with this software.

Created on Aug 2, 2018

@author: jrm
"""
import os
import datetime
import sqlalchemy as sa
from functools import wraps
from atom import api
from atom.atom import AtomMeta, with_metaclass
from atom.api import (
    Atom, Subclass, ContainerList, Int, Dict, Instance, Typed, Property, Str,
    ForwardInstance
)
from sqlalchemy.engine import ddl, strategies
from sqlalchemy.sql import schema
from sqlalchemy.orm.query import Query

from .base import (
    ModelManager, ModelSerializer, Model, ModelMeta, find_subclasses
)



DEFAULT_DATABASE = None

# kwargs reserved for sqlalchemy table columns
COLUMN_KWARGS = (
    'autoincrement', 'default', 'doc', 'key', 'index', 'info', 'nullable',
    'onupdate', 'primary_key', 'server_default', 'server_onupdate',
    'quote', 'unique', 'system', 'comment'
)
FK_TYPES = (api.Instance, api.Typed, api.ForwardInstance, api.ForwardTyped)


class Relation(ContainerList):
    """ A member which serves as a fk relation backref

    """
    __slots__ = ('_to',)

    def __init__(self, item, default=None):
        super(Relation, self).__init__(ForwardInstance(item), default=None)
        self._to = None

    @property
    def to(self):
        to = self._to
        if not to:
            to = self._to = resolve_member_type(self.validate_mode[-1])
        return to


def py_type_to_sql_column(model, member, cls, **kwargs):
    """ Convert the python type to an alchemy table column type

    """
    if issubclass(cls, Model):
        name = f'{cls.__model__}.{cls.__pk__}'
        cls.__backrefs__.add((model, member))
        return (sa.Integer, sa.ForeignKey(name))
    elif issubclass(cls, str):
        return sa.String(**kwargs)
    elif issubclass(cls, int):
        return sa.Integer(**kwargs)
    elif issubclass(cls, float):
        return sa.Float(**kwargs)
    elif issubclass(cls, datetime.datetime):
        return sa.DateTime(**kwargs)
    elif issubclass(cls, datetime.date):
        return sa.Date(**kwargs)
    elif issubclass(cls, datetime.time):
        return sa.Time(**kwargs)
    raise NotImplementedError(
        f"A column for {cls} could not be detected, please specify it "
        f"manually by tagging it with .tag(column=<sqlalchemy column>)")


def resolve_member_type(member):
    if hasattr(member, 'resolve'):
        return member.resolve()
    else:
        return member.validate_mode[-1]


def atom_member_to_sql_column(model, member, **kwargs):
    """ Convert the atom member type to an sqlalchemy table column type
    See https://docs.sqlalchemy.org/en/latest/core/type_basics.html

    """
    if isinstance(member, api.Str):
        return sa.String(**kwargs)
    elif isinstance(member, api.Unicode):
        return sa.Unicode(**kwargs)
    elif isinstance(member, api.Bool):
        return sa.Boolean()
    elif isinstance(member, api.Int):
        return sa.Integer()
    elif isinstance(member, api.Long):
        return sa.BigInteger()
    elif isinstance(member, api.Float):
        return sa.Float()
    elif isinstance(member, api.Range):
        # TODO: Add min / max
        return sa.Integer()
    elif isinstance(member, api.FloatRange):
        # TODO: Add min / max
        return sa.Float()
    elif isinstance(member, api.Enum):
        return sa.Enum(*member.items)
    elif isinstance(member, api.IntEnum):
        return sa.SmallInteger()
    elif isinstance(member, FK_TYPES):
        value_type = resolve_member_type(member)
        if value_type is None:
            raise TypeError("Instance and Typed members must specify types")
        return py_type_to_sql_column(model, member, value_type, **kwargs)
    elif isinstance(member, Relation):
        # Relations are for backrefs
        item_type = member.validate_mode[-1]
        if item_type is None:
            raise TypeError("Relation members must specify types")

        # Resolve the item type
        value_type = resolve_member_type(item_type)
        if value_type is None:
            raise TypeError("Relation members must specify types")
        return None  # Relations are just syntactic sugar
    elif isinstance(member, (api.List, api.ContainerList, api.Tuple)):
        item_type = member.validate_mode[-1]
        if item_type is None:
            raise TypeError("List and Tuple members must specify types")

        # Resolve the item type
        value_type = resolve_member_type(item_type)
        if value_type is None:
            raise TypeError("List and Tuple members must specify types")
        return sa.ARRAY(py_type_to_sql_column(
            model, member, value_type, **kwargs))
    elif isinstance(member, api.Bytes):
        return sa.LargeBinary(**kwargs)
    elif isinstance(member, api.Dict):
        return sa.JSON(**kwargs)
    raise NotImplementedError(
        f"A column for {member} could not be detected, please specify it"
        f"manually by tagging it with .tag(column=<sqlalchemy column>)")


def create_table_column(model, member):
    """ Converts an Atom member into a sqlalchemy data type.

    Parameters
    ----------
    model: Model
        The model which owns this member
    member: Member
        The atom member

    Returns
    -------
    column: Column
        An sqlalchemy column

    References
    ----------
    1. https://docs.sqlalchemy.org/en/latest/core/types.html

    """
    # Copy the metadata as we modify it
    metadata = member.metadata.copy() if member.metadata else {}

    # If a column is specified use that
    if 'column' in metadata:
        return metadata['column']

    metadata.pop('store', None)
    column_name = metadata.pop('name', member.name)
    column_type = metadata.pop('type', None)

    # Extract column kwargs from member metadata
    kwargs = {}
    for k in COLUMN_KWARGS:
        if k in metadata:
            kwargs[k] = metadata.pop(k)

    if column_type is None:
        args = atom_member_to_sql_column(model, member, **metadata)
        if args is None:
            return None
        if not isinstance(args, (tuple, list)):
            args = (args,)
    else:
        args = (column_type,)
    return sa.Column(column_name, *args, **kwargs)


def create_table(model, metadata):
    """ Create an sqlalchemy table by inspecting the Model and generating
    a column for each member.

    Parameters
    ----------
    model: SQLModel
        The atom model

    References
    ----------
    1. https://docs.sqlalchemy.org/en/latest/core/metadata.html

    """
    name = model.__model__
    members = model.members()
    columns = []
    for f in model.__fields__:
        column = create_table_column(model, members[f])
        if column is not None:
            columns.append(column)
    return sa.Table(name, metadata, *columns)


class SQLModelSerializer(ModelSerializer):
    """ Uses sqlalchemy to lookup the model.

    """
    def flatten_object(self, obj, scope):
        """ Serialize a model for entering into the database

        Parameters
        ----------
        obj: Model
            The object to unflatten
        scope: Dict
            The scope of references available for circular lookups

        Returns
        -------
        result: Object
            The flattened object

        """
        return obj._id

    async def get_object_state(self, obj, state, scope):
        """ Load the object state if needed. Since the __model__ is not saved
        to the db tables with SQL we know that if it's "probably" there
        because a query was used.
        """
        ModelType = obj.__class__
        if '__model__' in state:
            return state  # Joined already
        return await ModelType.objects.get(_id=state['_id'])

    def _default_registry(self):
        return {m.__model__: m for m in find_subclasses(SQLModel)}


class SQLModelManager(ModelManager):
    """ Manages models via aiopg, aiomysql, or similar libraries supporting
    SQLAlchemy tables. It stores a table for each class

    """
    #: DB URL
    url = Typed(sa.engine.url.URL)

    #: Metadata
    metadata = Instance(sa.MetaData)

    def _default_metadata(self):
        return sa.MetaData(SQLBinding(manager=self))

    def create_tables(self):
        tables = {}
        for cls in find_subclasses(SQLModel):
            table = cls.table
            if table is None:
                table = cls.table = create_table(cls, self.metadata)
            if not table.metadata.bind:
                table.metadata.bind = SQLBinding(manager=self, table=table)
            tables[cls] = table
        return tables

    def _default_url(self):
        env = os.environ
        url = env.get('DATABASE_URL', env.get('MYSQL_URL'))
        if url is None:
            raise EnvironmentError("No DATABASE_URL has been set")
        return sa.engine.url.make_url(url)

    def __get__(self, obj, cls=None):
        """ Retrieve the table for the requested object or class.

        """
        cls = cls or obj.__class__
        if not issubclass(cls, Model):
            return self  # Only return the client when used from a Model
        table = cls.table
        if table is None:
            table = cls.table = create_table(cls, self.metadata)
        return SQLTableProxy(table=table, model=cls)

    def get_database(self):
        db = DEFAULT_DATABASE
        if db is None:
            raise EnvironmentError("No database engine has been set")
        return db

    def get_dialect(self, **kwargs):
        dialect_cls = self.url.get_dialect()
        dialect_args = {}
        for k in sa.util.get_cls_kwargs(dialect_cls):
            if k in kwargs:
                dialect_args[k] = kwargs.pop(k)
        return dialect_cls(**dialect_args)


class SQLTableProxy(Atom):
    table = Instance(sa.Table)
    model = Subclass(Model)

    @property
    def engine(self):
        return self.table.bind.manager.database

    @property
    def connection(self):
        return self.engine.acquire

    def create(self):
        """ A wrapper for create which catches the create queries then executes
        them
        """
        table = self.table
        table.create()
        return table.bind.wait()

    def drop(self):
        table = self.table
        table.drop()
        return table.bind.wait()

    async def execute(self, *args, **kwargs):
        async with self.engine.acquire() as conn:
            return await conn.execute(*args, **kwargs)

    def all(self):
        return self.filter()

    async def fetchall(self, query):
        async with self.engine.acquire() as conn:
            r = await self.execute(query)
            return await r.fetchall()

    async def fetchone(self, query):
        async with self.engine.acquire() as conn:
            r = await self.execute(query)
            return await r.fetchone()

    def filter(self, **filters):
        table = self.table
        q = table.select()
        if filters:
            q = q.where(
                *(table.c[k] == v for k, v in filters.items()))
        return self.fetchall(q)

    def get(self, **filters):
        table = self.table
        q = table.select().where(
            *(table.c[k] == v for k, v in filters.items()))
        return self.fetchone(q)


class SQLBinding(Atom):
    #: Model Manager
    manager = Instance(SQLModelManager)

    #: Dialect
    dialect = Instance(object)

    #: The queue
    queue = ContainerList()

    #: Dialect name
    name = Str()

    engine = property(lambda s: s)
    schema_for_object = schema._schema_getter(None)

    def _default_name(self):
        return self.dialect.name

    def _default_dialect(self):
        return self.manager.get_dialect()

    def contextual_connect(self, **kwargs):
        return self

    def connect(self, **kwargs):
        return self

    def execution_options(self, **kw):
        return self

    def compiler(self, statement, parameters, **kwargs):
        return self.dialect.compiler(
            statement, parameters, engine=self, **kwargs)

    def create(self, entity, **kwargs):
        kwargs["checkfirst"] = False
        node = ddl.SchemaGenerator(self.dialect, self, **kwargs)
        node.traverse_single(entity)

    def drop(self, entity, **kwargs):
        kwargs["checkfirst"] = False
        node = ddl.SchemaDropper(self.dialect, self, **kwargs)
        node.traverse_single(entity)

    def _run_visitor(self, visitorcallable, element, connection=None, **kwargs):
        kwargs["checkfirst"] = False
        node = visitorcallable(self.dialect, self, **kwargs)
        node.traverse_single(element)

    def execute(self, object_, *multiparams, **params):
        self.queue.append((object_, multiparams, params))

    async def wait(self):
        engine = self.manager.database
        result = None
        async with engine.acquire() as conn:
            while self.queue:
                op, args, kwargs = self.queue.pop(0)
                result = await conn.execute(op, args)#, kwargs)
        return result


class SQLMeta(ModelMeta):
    """ Both the pk and _id are aliases to the primary key

    """
    def __new__(meta, name, bases, dct):
        cls = ModelMeta.__new__(meta, name, bases, dct)

        members = cls.members()

        # If a pk field is defined use that insetad of _id
        pk = None
        for m in members.values():
            if m.name == '_id':
                continue
            if m.metadata and m.metadata.get('primary_key'):
                pk = m
                break
        if pk:
            cls._id = pk
            members['_id'] = pk
            cls.__fields__ = tuple((f for f in cls.__fields__ if f != '_id'))

        # Set the pk name
        cls.__pk__ = cls._id.name

        # Will be set to the table model by manager, not done here to avoid
        # import errors that may occur
        cls.table = None
        cls.__backrefs__ = set()
        return cls


class SQLModel(with_metaclass(SQLMeta, Model)):
    """ A model that can be saved and restored to and from a database supported
    by sqlalchemy.

    """

    #: If no other member is tagged with primary_key=True this is used
    _id = Typed(int).tag(store=True, primary_key=True)

    #: Use SQL serializer
    serializer = SQLModelSerializer.instance()

    #: Use SQL object manager
    objects = SQLModelManager.instance()

    async def __setstate__(self, state, scope=None):
        # Check if the state is using labels by looking for the pk field
        pk_label = f'{self.__model__}_{self.__pk__}'
        if pk_label in state:
            # Convert the joined tables into nested states
            table = self.table
            table_name = table.name
            pk = state[pk_label]
            ref = os.urandom(16)
            grouped_state = {'__ref__': ref}

            # Pull known
            for m in self.members().values():
                key = f'{table_name}_{m.name}'
                if isinstance(m, Relation):
                    rel = m.to
                    rel_table_name = rel.__model__
                    nested_state = {}
                    for sm in rel.members().values():
                        key = f'{rel_table_name}_{sm.name}'
                        if key in state:
                            v = state[key]
                            # Use a reference if this is a foreign key
                            if (rel, sm) in self.__backrefs__ and v == pk:
                                v = {'__ref__': ref}
                            nested_state[sm.name] = v
                    if nested_state:
                        # Update any backrefs
                        nested_state['__model__'] = rel_table_name
                        grouped_state[m.name] = [nested_state]
                elif key in state:
                    grouped_state[m.name] = state[key]
            state = grouped_state

        await super().__setstate__(state, scope)

    async def save(self):
        """ Alias to delete this object to the database """
        db = self.objects
        state = self.__getstate__()
        state.pop('__model__', None)
        state.pop('__ref__', None)
        state.pop('_id', None)

        for name, m in self.members().items():
            if isinstance(m, Relation):
                state.pop(name, None)

        table = self.table
        async with db.engine.acquire() as conn:
            if self._id:
                q = table.update().where(
                        table.c[self.__pk__] == self._id).values(**state)
                r = await conn.execute(q)
            else:
                q = table.insert().values(**state)
                r = await conn.execute(q)
                self._id = r.lastrowid
            await conn.execute('commit')
            return r

    async def delete(self):
        """ Alias to delete this object in the database """
        db = self.objects
        table = self.table
        if self._id:
            async with db.engine.acquire() as conn:
                q = table.delete().where(table.c[self.__pk__] == self._id)
                await conn.execute(q)
                await conn.execute('commit')

