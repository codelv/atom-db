"""
Copyright (c) 2018-2019, Jairus Martin.

Distributed under the terms of the MIT License.

The full license is in the file LICENSE.txt, distributed with this software.

Created on Aug 2, 2018

@author: jrm
"""
import os
import logging
import datetime
import weakref
import sqlalchemy as sa
from atom import api
from atom.atom import AtomMeta, with_metaclass
from atom.api import (
    Atom, Subclass, ContainerList, Int, Dict, Instance, Typed, Property, Str,
    ForwardInstance, Value, Bool
)
from functools import wraps
from sqlalchemy.engine import ddl, strategies
from sqlalchemy.sql import schema
from sqlalchemy.orm.query import Query
from sqlalchemy import func


from .base import (
    ModelManager, ModelSerializer, Model, ModelMeta, find_subclasses,
    JSONModel, JSONSerializer
)


# kwargs reserved for sqlalchemy table columns
COLUMN_KWARGS = (
    'autoincrement', 'default', 'doc', 'key', 'index', 'info', 'nullable',
    'onupdate', 'primary_key', 'server_default', 'server_onupdate',
    'quote', 'unique', 'system', 'comment'
)
FK_TYPES = (api.Instance, api.Typed, api.ForwardInstance, api.ForwardTyped)

# ops that can be used with django-style queries
QUERY_OPS = {
    'eq': '__eq__',
    'gt': '__gt__',
    'gte': '__ge__',
    'ge': '__ge__',
    'lt': '__lt__',
    'le': '__le__',
    'lte': '__le__',
    'all': 'all_',
    'any': 'any_',
    #'and': 'and_',
    #'or': 'or_',
    'ne': '__ne__',
    'not': '__ne__',
    'contains': 'contains',
    'endswith': 'endswith',
    'ilike': 'ilike',
    'in': 'in_',
    'is': 'is_',
    'is_distinct_from': 'is_distinct_from',
    'isnot': 'isnot',
    'isnot_distinct_from': 'isnot_distinct_from',
    'like': 'like',
    'match': 'match',
    'notilike': 'notilike',
    'notlike': 'notlike',
    'notin': 'notin_',
    'startswith': 'startswith',
}

# Fields supported on the django style Meta class of a model
VALID_META_FIELDS = ('db_table', 'unique_together', 'abstract')

MULTI_JOIN_ERROR = "Multi-join lookups are not supported, build queries " \
                   "manually using Model.objects.table.select()..."


log = logging.getLogger('atomdb.sql')


def find_sql_models():
    """ Finds all non-abstract imported SQLModels by looking up subclasses
    of the SQLModel.

    Yields
    ------
    cls: SQLModel

    """
    for model in find_subclasses(SQLModel):
        # Get model Meta class
        meta = getattr(model, 'Meta', None)
        if meta:
            # If this is marked as abstract ignore it
            if getattr(meta, 'abstract', False):
                continue
        yield model


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
    if issubclass(cls, JSONModel):
        return sa.JSON(**kwargs)
    elif issubclass(cls, Model):
        name = f'{cls.__model__}.{cls.__pk__}'
        cls.__backrefs__.add((model, member))

        # Determine the type of the foreign key
        column = create_table_column(cls, cls._id)
        return (column.type, sa.ForeignKey(name, **kwargs))
    elif issubclass(cls, str):
        return sa.String(**kwargs)
    elif issubclass(cls, int):
        return sa.Integer(**kwargs)
    elif issubclass(cls, float):
        return sa.Float(**kwargs)
    elif issubclass(cls, dict):
        return sa.JSON(**kwargs)
    elif issubclass(cls, (tuple, list)):
        return sa.ARRAY(**kwargs)
    elif issubclass(cls, datetime.datetime):
        return sa.DateTime(**kwargs)
    elif issubclass(cls, datetime.date):
        return sa.Date(**kwargs)
    elif issubclass(cls, datetime.time):
        return sa.Time(**kwargs)
    elif issubclass(cls, (bytes, bytearray)):
        return sa.LargeBinary(**kwargs)
    raise NotImplementedError(
        f"A type for {member.name} of {model} ({cls}) could not be "
        f"determined automatically, please specify it manually by tagging it "
        f"with .tag(column=<sqlalchemy column>)")


def resolve_member_type(member):
    """ Determine the type specified on a member to determine ForeignKey
    relations.

    Parameters
    ----------
    member: atom.catom.Member
        The member to retrieve the type from
    Returns
    -------
    object: Model or object
        The type specified.

    """
    if hasattr(member, 'resolve'):
        return member.resolve()
    else:
        return member.validate_mode[-1]


def atom_member_to_sql_column(model, member, **kwargs):
    """ Convert the atom member type to an sqlalchemy table column type
    See https://docs.sqlalchemy.org/en/latest/core/type_basics.html

    """
    if hasattr(member, 'get_column_type'):
        # Allow custom members to define the column type programatically
        return member.get_column_type(model)
    elif isinstance(member, api.Str):
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
        if issubclass(value_type, JSONModel):
            return sa.JSON(**kwargs)
        return sa.ARRAY(py_type_to_sql_column(
            model, member, value_type, **kwargs))
    elif isinstance(member, api.Bytes):
        return sa.LargeBinary(**kwargs)
    elif isinstance(member, api.Dict):
        return sa.JSON(**kwargs)
    raise NotImplementedError(
        f"A column for {member.name} of {model} could not be determined "
        f"automatically, please specify it manually by tagging it "
        f"with .tag(column=<sqlalchemy column>)")


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
    if hasattr(member, 'get_column'):
        # Allow custom members to define the column programatically
        return member.get_column(model)

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
    elif isinstance(column_type, (tuple, list)):
        args = column_type
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
    args = []

    # Add columns
    for f in model.__fields__:
        column = create_table_column(model, members[f])
        if column is not None:
            args.append(column)

    # Add table metadata
    meta = getattr(model, 'Meta', None)
    if meta:
        abstract = getattr(meta, 'abstract', False)
        if abstract:
            raise NotImplementedError(
                f"Tables cannot be created for abstract models: {model}")
        unique_together = getattr(meta, 'unique_together', [])
        if unique_together:
            if isinstance(unique_together[0], str):
                unique_together = [unique_together]
            for constraint in unique_together:
                args.append(sa.UniqueConstraint(*constraint))

    return sa.Table(name, metadata, *args)


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
        """ Add all sql and json models to the registry
        """
        registry = JSONSerializer.instance().registry.copy()
        registry.update({m.__model__: m for m in find_sql_models()})
        return registry


class SQLModelManager(ModelManager):
    """ Manages models via aiopg, aiomysql, or similar libraries supporting
    SQLAlchemy tables. It stores a table for each class and when accessed
    on a Model subclass it returns a table proxy binding.

    """
    #: Metadata
    metadata = Instance(sa.MetaData)

    #: Table proxy cache
    proxies = Dict()

    #: Cache results
    cache = Bool(True)

    def _default_metadata(self):
        return sa.MetaData(SQLBinding(manager=self))

    def create_tables(self):
        """ Create sqlalchemy tables for all registered SQLModels

        """
        tables = {}
        for cls in find_sql_models():
            table = cls.__table__
            if table is None:
                table = cls.__table__ = create_table(cls, self.metadata)
            if not table.metadata.bind:
                table.metadata.bind = SQLBinding(manager=self, table=table)
            tables[cls] = table
        return tables

    def __get__(self, obj, cls=None):
        """ Retrieve the table for the requested object or class.

        """
        cls = cls or obj.__class__
        if not issubclass(cls, Model):
            return self  # Only return the client when used from a Model
        try:
            return self.proxies[cls]
        except KeyError:
            table = cls.__table__
            if table is None:
                table = cls.__table__ = create_table(cls, self.metadata)
            proxy = self.proxies[cls] = SQLTableProxy(table=table, model=cls)
            return proxy

    def _default_database(self):
        raise EnvironmentError("No database engine has been set. Use "
                               "SQLModelManager.instance().database = <db>")


class ConnectionProxy(Atom):
    """ An wapper for a connection to be used with async with syntax that
    does nothing but passes the existing connection when entered.

    """
    connection = Value()

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, tb):
        pass


class SQLTableProxy(Atom):
    #: Table this is a proxy to
    table = Instance(sa.Table)

    #: Model which owns the table
    model = Subclass(Model)

    #: Cache of pk: obj using weakrefs
    cache = Typed(weakref.WeakValueDictionary, ())

    #: Key used to pull the connection out of filter kwargs
    connection_kwarg = Str('connection')

    @property
    def engine(self):
        return self.table.bind.manager.database

    def connection(self, connection=None):
        """ Create a new connection or the return given connection as an async
        contextual object.

        Parameters
        ----------
        connection: Database connection or None
            The connection to return

        Returns
        -------
        connection: Database connection
            The database connection or one that may be used with async with

        """
        if connection is None:
            return self.engine.acquire()
        return ConnectionProxy(connection=connection)

    def create_table(self):
        """ A wrapper for create which catches the create queries then executes
        them
        """
        table = self.table
        table.create()
        return table.bind.wait()

    def drop_table(self):
        table = self.table
        table.drop()
        return table.bind.wait()

    async def execute(self, *args, **kwargs):
        async with self.connection() as conn:
            return await conn.execute(*args, **kwargs)

    async def fetchall(self, query, connection=None):
        """ Fetch all results for the query.

        Parameters
        ----------
        query: String or Query
            The query to execute
        connection: Database connection
            The connection to use or a new one will be created

        Returns
        -------
        rows; List
            List of rows returned, NOT objects

        """
        async with self.connection(connection) as conn:
            r = await conn.execute(query)
            return await r.fetchall()

    async def fetchmany(self, query, size=None, connection=None):
        """ Fetch size results for the query.

        Parameters
        ----------
        query: String or Query
            The query to execute
        size: Int or None
            The number of results to fetch
        connection: Database connection
            The connection to use or a new one will be created

        Returns
        -------
        rows: List
            List of rows returned, NOT objects

        """
        async with self.connection(connection) as conn:
            r = await conn.execute(query)
            return await r.fetchmany(size)

    async def fetchone(self, query, connection=None):
        """ Fetch a single result for the query.

        Parameters
        ----------
        query: String or Query
            The query to execute
        connection: Database connection
            The connection to use or a new one will be created

        Returns
        -------
        rows: Object or None
            The row returned or None
        """
        async with self.connection(connection) as conn:
            r = await conn.execute(query)
            return await r.fetchone()

    async def scalar(self, query, connection=None):
        """ Fetch the scalar result for the query.

        Parameters
        ----------
        query: String or Query
            The query to execute
        connection: Database connection
            The connection to use or a new one will be created

        Returns
        -------
        result: Object or None
            The the first column of the first row or None
        """
        async with self.connection(connection) as conn:
            r = await conn.execute(query)
            return await r.scalar()

    async def filter(self, **filters):
        """ Selects the objects matching the given filters

        Parameters
        ----------
        filters: Dict
            The filters to select which objects to delete

        Returns
        -------
        result: List[Model]
            A list of objects matching the response

        """
        restore = self.model.restore
        q = self.query(None, **filters)
        connection = filters.get(self.connection_kwarg)
        return [await restore(row)
                for row in await self.fetchall(q, connection=connection)]

    async def delete(self, **filters):
        """ Delete the objects matching the given filters

        Parameters
        ----------
        filters: Dict
            The filters to select which objects to delete

        Returns
        -------
        result: Response
            The execute response

        """
        q = self.query('delete', **filters)
        connection = filters.get(self.connection_kwarg)
        async with self.connection(connection) as conn:
            return await conn.execute(q)

    async def all(self, **kwargs):
        return await self.filter(**kwargs)

    async def get(self, **filters):
        """ Get a model matching the given criteria

        Parameters
        ----------
        filters: Dict
            The filters to use to retrieve the object

        Returns
        -------
        result: Model or None
            The model matching the query or None.

        """
        q = self.query(None, **filters)
        connection = filters.get(self.connection_kwarg)
        row = await self.fetchone(q, connection=connection)
        if row is not None:
            return await self.model.restore(row)

    async def get_or_create(self, **filters):
        """ Get or create a model matching the given criteria

        Parameters
        ----------
        filters: Dict
            The filters to use to retrieve the object

        Returns
        -------
        result: Tuple[Model, Bool]
            A tuple of the object and a bool indicating if it was just created

        """
        obj = await self.get(**filters)
        if obj is not None:
            return (obj, False)
        connection_kwarg = self.connection_kwarg
        connection = filters.get(connection_kwarg)
        state = {k: v for k, v in filters.items()
                 if '__' not in k and k != connection_kwarg}
        obj = self.model(**state)
        await obj.save(force_insert=True, connection=connection)
        return (obj, True)

    async def create(self, **state):
        """ Create a and save model with the given state.

        The connection parameter is popped from this state.

        Parameters
        ----------
        state: Dict
            The state to use to initialize the object.

        Returns
        -------
        result: Tuple[Model, Bool]
            A tuple of the object and a bool indicating if it was just created

        """
        connection = state.pop(self.connection_kwarg, None)
        obj = self.model(**state)
        await obj.save(force_insert=True, connection=connection)
        return obj

    async def count(self, **filters):
        """ Return a count of objects in the table matching the given filters

        """
        q = self.query('count', **filters)
        connection = filters.get(self.connection_kwarg)
        return await self.scalar(q, connection=connection)

    def query(self, __q__=None, **filters):
        """ Build a django-style query by adding a where clause to the given
        query.

        Parameters
        ----------
        __q__: Query or String
            The query to add a where clause to. Will default to select if not
            given.
        filters: Dict
            A dict where keys are mapped to columns and values. Use __ to
            specify an operator. For example (date__gt=datetime.now())

        Returns
        -------
        query: Query
            An sqlalchemy query which an be used with execute, fetchall, etc..

        References
        ----------
        1. https://docs.sqlalchemy.org/en/latest/core/sqlelement.html
            ?highlight=column#sqlalchemy.sql.operators.ColumnOperators

        """
        connection_kwarg = self.connection_kwarg
        ops = []
        joins = {}

        # Build the filter operations
        for k, v in filters.items():
            if k == connection_kwarg:
                continue # Skip the connection parameter if given

            columns = self.table.c
            cls = self.model

            # Support lookups by model
            if isinstance(v, Model):
                v = v.serializer.flatten_object(v, scope=None)

            # Support related lookups or operations like __in
            if '__' in k:
                args = k.split('__')
                n = len(args)
                if n == 1:
                    raise ValueError("Invalid filter %s on %s" % (k, cls))
                if n > 3:
                    raise NotImplementedError(MULTI_JOIN_ERROR)

                field = args[0]
                op = QUERY_OPS.get(args[-1], None)

                # This is a lookup on a related field or a operation on a
                # related field so figure out which case
                if op is None or n == 3:
                    if n == 2:
                        op = '__eq__'
                    elif op is None:
                        raise NotImplementedError(MULTI_JOIN_ERROR)

                    member = getattr(cls, field, None)
                    if not isinstance(member, FK_TYPES):
                        raise ValueError("Invalid filter %s on %s" % (k, cls))

                    # Set the Model to the related field model
                    cls = resolve_member_type(member)
                    table = cls.objects.table

                    # Specify the join
                    on = columns[self.model.__pk__] == table.c[cls.__pk__]
                    joins[table] = on
                    columns = table.c

                    # Since this is a joined lookup change the field
                    field = args[1]

            else:
                # Simple case
                op = '__eq__'
                field = k

            # Get the column from the sqlalchemy table
            try:
                col = columns[field]
            except KeyError:
                # If the field has a different name assigned use that
                member = getattr(cls, field, None)
                if member is None:
                    raise ValueError(
                        "Invalid filter %s on %s" % (k, self.model))
                name = member.metadata['name']
                col = columns[name]

            ops.append((col, op, v))

        # Build the base query, joining as needed
        q = self.table
        for table, on in joins.items():
            q = q.join(table, on)

        if __q__ is None:
            q = q.select(use_labels=bool(joins))
        elif isinstance(__q__, str):
            # This is needed to support joins
            # select, count, delete, etc...
            q = getattr(q, __q__)()
        else:
            q = __q__

        # Filter the query
        for (col, op, v) in ops:
            q = q.where(getattr(col, op)(v))

        return q


class SQLBinding(Atom):
    #: Model Manager
    manager = Instance(SQLModelManager)

    #: The queue
    queue = ContainerList()

    engine = property(lambda s: s)
    schema_for_object = schema._schema_getter(None)

    @property
    def name(self):
        return self.dialect.name

    @property
    def dialect(self):
        return self.manager.database.dialect

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
                if pk is not None:
                    raise NotImplementedError(
                        "Using multiple primary keys is not yet supported.")
                pk = m

        if pk:
            cls._id = pk
            members['_id'] = pk
            cls.__fields__ = tuple((f for f in cls.__fields__ if f != '_id'))

        # Set the pk name
        cls.__pk__ = (cls._id.metadata or {}).get('name', cls._id.name)

        # Set to the sqlalchemy Table
        cls.__table__ = None

        # Will be set to the table model by manager, not done here to avoid
        # import errors that may occur
        cls.__backrefs__ = set()

        # If a Meta class is defined check it's validity and if extending
        # do not inherit the abstract attribute
        Meta = dct.get('Meta', None)
        if Meta is not None:
            for f in dir(Meta):
                if f.startswith('_'):
                    continue
                if f not in VALID_META_FIELDS:
                    raise TypeError(
                        f'{f} is not a valid Meta field on {cls}.')

            db_table = getattr(Meta, 'db_table', None)
            if db_table:
                cls.__model__ = db_table

        # If this inherited from an abstract model but didn't specify
        # Meta info make the subclass not abstract unless it was redefined
        base_meta = getattr(cls, 'Meta', None)
        if base_meta and getattr(base_meta, 'abstract', None):
            if not Meta:
                class Meta(base_meta):
                    abstract = False
                cls.Meta = Meta
            elif getattr(Meta, 'abstract', None) is None:
                Meta.abstract = False

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

    @classmethod
    async def restore(cls, state, force=None):
        """ Restore an object from the database using the primary key. Save
        a ref in the table's object cache.  If force is True, update
        the cache if it exists.
        """
        try:
            pk = state[f'{cls.__model__}_{cls.__pk__}']
        except KeyError:
            pk = state[cls.__pk__]

        # Check the default for force reloading
        if force is None:
            force = not cls.objects.table.bind.manager.cache

        # Check if this is in the cache
        cache = cls.objects.cache
        obj = cache.get(pk)
        if obj is None:
            # Create and cache it
            obj = cls.__new__(cls)
            cache[pk] = obj

            # This ideally should only be done if created
            await obj.__restorestate__(state)
        elif force:
            await obj.__restorestate__(state)

        return obj

    async def __restorestate__(self, state, scope=None):
        # Holds cleaned state extracted for this model which may come from
        # a DB row using labels or renamed columns
        cleaned_state = {}

        # Check if the state is using labels by looking for the pk field
        pk_label = f'{self.__model__}_{self.__pk__}'

        if pk_label in state:
            # Convert row to dict because it speeds up lookups
            state = dict(state)
            # Convert the joined tables into nested states
            table_name = self.__table__.name
            pk = state[pk_label]

            # Pull known
            for name, m in self.members().items():
                metadata = (m.metadata or {})
                field_name = metadata.get('name', name)
                field_label = f'{table_name}_{field_name}'

                if isinstance(m, FK_TYPES):
                    rel = resolve_member_type(m)
                    if issubclass(rel, SQLModel):
                        # If the related model was joined, the pk field should
                        # exist so automatically restore that as well
                        rel_pk_field = f'{rel.__model__}_{rel.__pk__}'
                        try:
                            rel_id = state[field_label]
                        except KeyError:
                            rel_id = state.get(rel_pk_field)
                        if rel_id:
                            # Lookup in cache first to avoid recursion errors
                            obj = rel.objects.cache.get(rel_id)
                            if obj is None:
                                if rel_pk_field not in state:
                                    continue
                                obj = await rel.restore(state)
                            cleaned_state[name] = obj
                            continue

                elif isinstance(m, Relation):
                    # Through must be a callable which returns a tuple of
                    # the through table model
                    through_factory = metadata.get('through')
                    if through_factory:
                        M2M, this_attr, rel_attr = through_factory()
                        cleaned_state[name] = [
                            getattr(r, rel_attr)
                                for r in await M2M.objects.filter(
                                    **{this_attr: pk})]
                    else:
                        # Skip relations
                        continue

                # Regular fields
                try:
                    cleaned_state[name] = state[field_label]
                except KeyError:
                    continue

        else:
            # If any column names were redefined use those instead
            for name, m in self.members().items():
                field_name = (m.metadata or {}).get('name', name)

                try:
                    v = state[field_name]
                except KeyError:
                    continue

                # Attempt to lookup related fields from the cache
                if isinstance(m, FK_TYPES):
                    rel = resolve_member_type(m)
                    if issubclass(rel, SQLModel):
                        v = rel.objects.cache.get(v)
                        if v is None:
                            # Skip because this will throw a TypeError
                            continue

                cleaned_state[name] = v
        await super().__restorestate__(cleaned_state, scope)

    async def save(self, force_insert=False, force_update=False,
                   connection=None):
        """ Alias to save this object to the database """
        if force_insert and force_update:
            raise ValueError(
                'Cannot use force_insert and force_update together')

        db = self.objects
        state = self.__getstate__()
        state.pop('__model__', None)
        state.pop('__ref__', None)
        state.pop('_id', None)

        # If any column names were redefined use those instead
        for name, m in self.members().items():
            if isinstance(m, Relation):
                state.pop(name, None)
            elif m.metadata and 'name' in m.metadata and name in state:
                state[m.metadata['name']] = state.pop(name)

        table = db.table
        async with db.connection(connection) as conn:
            if force_update or (self._id and not force_insert):
                q = table.update().where(
                        table.c[self.__pk__] == self._id).values(**state)
                r = await conn.execute(q)
                if not r.rowcount:
                    log.warning(
                        f'Did not update "{self}", either no rows with '
                        f'pk={self._id} exist or it has not changed.')
            else:
                if not self._id:
                    # Postgres errors if using None for the pk
                    state.pop(self.__pk__, None)
                q = table.insert().values(**state)
                r = await conn.execute(q)

                # Don't overwrite if force inserting
                if not self._id:
                    if hasattr(r, 'lastrowid'):
                        self._id = r.lastrowid # MySQL
                    else:
                        self._id = await r.scalar() # Postgres

                # Save a ref to the object in the model cache
                db.cache[self._id] = self
            return r

    async def delete(self, connection=None):
        """ Alias to delete this object in the database """
        pk = self._id
        if not pk:
            return
        db = self.objects
        table = db.table
        q = table.delete().where(table.c[self.__pk__] == pk)
        async with db.connection(connection) as conn:
            r = await conn.execute(q)
            if r.rowcount:
                del db.cache[pk]
                del self._id
            return r

