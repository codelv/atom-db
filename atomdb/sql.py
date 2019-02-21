"""
Copyright (c) 2018, Jairus Martin.

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
from atom.api import Atom, ContainerList, Int, Dict, Instance, Typed
from .base import ModelManager, ModelSerializer, Model, find_subclasses
from sqlalchemy.engine import ddl, strategies
from sqlalchemy.sql import schema

DEFAULT_DATABASE = None

# kwargs reserved for sqlalchemy table columns
COLUMN_KWARGS = (
    'autoincrement', 'default', 'doc', 'key', 'index', 'info', 'nullable',
    'onupdate', 'primary_key', 'server_default', 'server_onupdate',
    'quote', 'unique', 'system', 'comment'
)


def py_type_to_sql_column(cls, **kwargs):
    """ Convert the python type to an alchemy table column type

    """
    if issubclass(cls, Model):
        name =  f'{cls.__model__}._id'
        return (sa.Integer, sa.ForeignKey(name))
    elif issubclass(cls, str):
        return sa.String(**kwargs)
    elif issubclass(cls, int):
        return sa.Integer(**kwargs)
    elif issubclass(cls, float):
        return sa.Integer(**kwargs)
    elif issubclass(cls, datetime.datetime):
        return sa.DateTime(**kwargs)
    elif issubclass(cls, datetime.date):
        return sa.Date(**kwargs)
    elif issubclass(cls, datetime.time):
        return sa.Time(**kwargs)
    raise NotImplementedError(f"A column for {cls} could not be detected, "
                              "please specifiy it manually by tagging it with "
                              ".tag(column=<sqlalchemy column>)")


def atom_member_to_sql_column(member, **kwargs):
    """ Convert the atom member type to an sqlalchemy table column type

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
    elif isinstance(member, (api.Instance, api.Typed,
                             api.ForwardInstance, api.ForwardTyped)):
        if hasattr(member, 'resolve'):
            value_type = member.resolve()
        else:
            value_type = member.validate_mode[-1]
        if value_type is None:
            raise TypeError("Instance and Typed members must specifiy types")
        return py_type_to_sql_column(value_type, **kwargs)
    elif isinstance(member, (api.List, api.ContainerList, api.Tuple)):
        item_type = member.validate_mode[-1]
        if item_type is None:
            raise TypeError("List and Tuple members must specifiy types")

        # Resolve the item type
        if hasattr(item_type, 'resolve'):
            value_type = item_type.resolve()
        else:
            value_type = item_type.validate_mode[-1]

        if value_type is None:
            raise TypeError("List and Tuple members must specifiy types")
        elif isinstance(value_type, Model):
            name =  f'{value_type.__model__}._id'
            return (sa.Integer, sa.ForeignKey(name))
        return sa.ARRAY(py_type_to_sql_column(value_type, **kwargs))
    elif isinstance(member, api.Bytes):
        return sa.LargeBinary(**kwargs)
    elif isinstance(member, api.Dict):
        return sa.JSON(**kwargs)
    raise NotImplementedError(f"A column for {member} could not be detected, "
                              "please specifiy it manually by tagging it with "
                              ".tag(column=<sqlalchemy column>)")


def create_table_column(member):
    """ Converts an Atom member into a sqlalchemy data type.

    Parameters
    ----------
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
    metadata = member.metadata or {}

    # If a column is specified use that
    if 'column' in metadata:
        return metadata['column']

    metadata.pop('store', None)

    # Extract column kwargs from member metadata
    kwargs = {}
    for k in COLUMN_KWARGS:
        if k in metadata:
            kwargs[k] = metadata.pop(k)

    args = atom_member_to_sql_column(member, **metadata)
    if not isinstance(args, (tuple, list)):
        args = (args,)
    return sa.Column(member.name, *args, **kwargs)


def create_table(model, **metadata):
    """ Create an sqlalchemy table by inspecting the Model and generating
    a column for each member.

    """
    if not issubclass(model, Model):
        raise TypeError("Only Models are supported")
    name = model.__model__
    metadata = sa.MetaData(**metadata)
    members = model.members()
    columns = (create_table_column(members[f]) for f in model.__fields__)
    return sa.Table(name, metadata, *columns)


class SQLModelSerializer(ModelSerializer):
    """ Uses sqlalchemy to lookup the model.

    """
    async def get_object_state(self, obj, _id):
        ModelType = obj.__class__
        result = await ModelType.objects.select('*').where(_id=_id)
        # Convert the result to a dict
        return result

    def _default_registry(self):
        return {m.__model__: m for m in find_subclasses(SQLModel)}


class SQLModelManager(ModelManager):
    """ Manages models via aiopg, aiomysql, or similar libraries supporting
    SQLAlchemy tables. It stores a table for each class

    """
    #: Mapping of model to table used to store the model
    tables = Dict()

    #: DB URL
    url = Typed(sa.engine.url.URL)

    def _default_tables(self):
        """ Create all the tables """
        return {m: create_table(m) for m in find_subclasses(SQLModel)}

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
        if cls not in self.tables:
            table = self.tables[cls] = create_table(cls)
        table = self.tables[cls]
        table.metadata.bind = SQLBinding(manager=self, table=table)
        return SQLTableProxy(table=table)

    def get_database(self):
        db = DEFAULT_DATABASE
        if db is None:
            raise EnvironmentError("No database engine has been set")
        return db

    def get_dialect(self, **kwargs):
        # create url.URL object
        url = self.url

        dialect_cls = url.get_dialect()

        dialect_args = {}
        # consume dialect arguments from kwargs
        for k in sa.util.get_cls_kwargs(dialect_cls):
            if k in kwargs:
                dialect_args[k] = kwargs.pop(k)

        # create dialect
        return dialect_cls(**dialect_args)


class SQLTableProxy(Atom):
    table = Instance(sa.Table)

    def __getattr__(self, name):
        attr = getattr(self.table, name)
        if not callable(attr):
            return attr
        async def wrapped(*args, **kwargs):
            r = attr(*args, **kwargs)
            binding = self.table.bind
            return await binding.run_until_current()
        return wrapped


class SQLBinding(Atom):
    #: Model Manager
    manager = Instance(SQLModelManager)

    #: Dialect
    dialect = Instance(object)

    #: The queue
    _queue = ContainerList()

    #: Dialect name
    name = api.Unicode()

    table = Instance(sa.Table)

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
        return self

    def drop(self, entity, **kwargs):
        kwargs["checkfirst"] = False
        node = ddl.SchemaDropper(self.dialect, self, **kwargs)
        node.traverse_single(entity)
        print((entity, self.dialect, node))
        return self

    def _run_visitor(self, visitorcallable, element, connection=None, **kwargs):
        kwargs["checkfirst"] = False
        node = visitorcallable(self.dialect, self, **kwargs)
        node.traverse_single(element)
        return self

    def execute(self, object_, *multiparams, **params):
        self._queue.append((object_, multiparams, params))
        return self

    def _observe__queue(self, change):
        print(change)

    @property
    def has_pending(self):
        return bool(self._queue)

    async def run_until_current(self):
        engine = self.manager.database
        result = None
        async with engine.acquire() as conn:
            while self._queue:
                op, args, kwargs = self._queue.pop(0)
                result = await conn.execute(op, args)#, kwargs)
        return result


#class SQLBinding(Atom):


    #def _run_visitor(self, *args, **kwargs):
        #""" Binding that defers all calls into a queue then invokes
        #them all async

        #"""

    #async def create_table(self):
        #async with self.manager.database.acquire() as conn:
            #return await conn.execute(self.table.metadata.create_all(conn))

    #async def filter(self, **kwargs):
        #async with self.manager.database.acquire() as conn:
            #return await conn.execute(self.table.filter(**kwargs))

    #async def drop_table(self):
        #async with self.manager.database.acquire() as conn:
            #This can't be paramertized? WTF SQL
            #name = self.table.name
            #return await conn.execute('DROP TABLE IF EXISTS %s;' % name)



class SQLModel(Model):
    """ A model that can be saved and restored to and from a database supported
    by sqlalchemy.

    """
    #: ID of this object in the database
    _id = Int().tag(primary_key=True)

    #: Use SQL serializer
    serializer = SQLModelSerializer.instance()

    #: Use SQL object manager
    objects = SQLModelManager.instance()

    async def save(self):
        """ Alias to delete this object to the database """
        db  = self.objects
        state = self.__getstate__()
        if self._id is not None:
            return await db.insert({'_id': self._id}, state, upsert=True)
        else:
            r = await db.execute(self.table.insert().values(state))
            self._id = r.inserted_id
            return r

    async def delete(self):
        """ Alias to delete this object in the database """
        db = self.objects
        if self._id:
            return await db.excute(db.table.delete().where(_id=self._id))
