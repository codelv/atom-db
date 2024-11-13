"""
Copyright (c) 2018-2022, Jairus Martin.

Distributed under the terms of the MIT License.

The full license is in the file LICENSE.txt, distributed with this software.

Created on Aug 2, 2018
"""

import asyncio
import datetime
import enum
import functools
import logging
import weakref
from decimal import Decimal
from typing import Any
from typing import Callable as CallableType
from typing import ClassVar
from typing import Dict as DictType
from typing import Generic, Iterator
from typing import List as ListType
from typing import Optional, Sequence
from typing import Set as SetType
from typing import Tuple as TupleType
from typing import Type, TypeVar, Union, cast

import sqlalchemy as sa
from atom import api
from atom.api import (
    Atom,
    Bool,
    ContainerList,
    Dict,
    ForwardInstance,
    ForwardSubclass,
    ForwardTyped,
    Instance,
    Int,
    List,
    Member,
    Set,
    Str,
    Typed,
    Validate,
    Value,
)
from atom.catom import atomclist
from sqlalchemy.engine import ddl
from sqlalchemy.sql import schema
from sqlalchemy.sql.elements import UnaryExpression
from sqlalchemy.sql.operators import asc_op, desc_op
from sqlalchemy.sql.type_api import TypeEngine

from .base import (
    JSONModel,
    JSONSerializer,
    Model,
    ModelManager,
    ModelMeta,
    ModelSerializer,
    RestoreStateFn,
    ScopeType,
    StateType,
    UnresolvableError,
    find_subclasses,
    generate_function,
    is_db_field,
    is_primitive_member,
    resolve_member_types,
)

# kwargs reserved for sqlalchemy table columns
COLUMN_KWARGS = (
    "autoincrement",
    "default",
    "doc",
    "key",
    "index",
    "info",
    "nullable",
    "onupdate",
    "primary_key",
    "server_default",
    "server_onupdate",
    "quote",
    "unique",
    "system",
    "comment",
)
FK_TYPES = (Instance, Typed, ForwardInstance, ForwardTyped)

# Member types that will default to nullable=False unless tagged otherwise
NON_NULL_MEMBERS = (
    api.Bool,
    api.Dict,
    api.Str,
    api.Int,
    api.Float,
    api.Range,
    api.Enum,
    api.FloatRange,
    api.List,
    api.ContainerList,
    api.Tuple,
    api.Set,
    api.Bytes,
)

# kwargs reserved for the serializer
SERIALIZE_KWARGS = ("flatten", "unflatten")

# ops that can be used with django-style queries
QUERY_OPS = {
    "eq": "__eq__",
    "gt": "__gt__",
    "gte": "__ge__",
    "ge": "__ge__",
    "lt": "__lt__",
    "le": "__le__",
    "lte": "__le__",
    "all": "all_",
    "any": "any_",
    "ne": "__ne__",
    "not": "__ne__",
    "contains": "contains",
    "endswith": "endswith",
    "ilike": "ilike",
    "in": "in_",
    "is": "is_",
    "is_distinct_from": "is_distinct_from",
    "isnot": "isnot",
    "isnot_distinct_from": "isnot_distinct_from",
    "like": "like",
    "match": "match",
    "notilike": "notilike",
    "notlike": "notlike",
    "notin": "notin_",
    "startswith": "startswith",
}

# Fields supported on the django style Meta class of a model
VALID_META_FIELDS = (
    "db_name",
    "db_table",
    "unique_together",
    "abstract",
    "constraints",
    "triggers",
    "composite_indexes",
    "get_latest_by",
)

# Constraint naming conventions
CONSTRAINT_NAMING_CONVENTIONS = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    # Using "ck_%(table_name)s_%(constraint_name)s" is preferred but it causes
    # issues using Bool on mysql
    "ck": "ck_%(table_name)s_%(column_0_N_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

log = logging.getLogger("atomdb.sql")

QueryType = Union[str, sa.sql.expression.Executable]
T = TypeVar("T", bound="SQLModel")


def find_sql_models() -> Iterator[Type["SQLModel"]]:
    """Finds all non-abstract imported SQLModels by looking up subclasses
    of the SQLModel.

    Yields
    ------
    cls: SQLModel

    """
    for model in find_subclasses(SQLModel):
        # Get model Meta class
        meta = getattr(model, "Meta", None)
        if meta:
            # If this is marked as abstract ignore it
            if getattr(meta, "abstract", False):
                continue
        yield model


@functools.lru_cache(1024)
def create_related_list(owner: Model, relation: "Relation"):
    class RelatedList(atomclist):
        """A custom list which has methods to query a foreign key
        one to many or many to many relation
        """

        __slots__ = ()

        def sort(self, *, key=None, reverse=False):
            # AtomCListHandler calls super(type(self), self).sort which causes a loop
            # in a loop because we fiddle with the __class__
            super(atomclist, self).sort(key=key, reverse=reverse)

        async def load(self, inplace: bool = True) -> list:
            """Returns a list of the related values."""
            Model = cast(Type[SQLModel], type(owner))
            ThroughModel = relation.through
            RelModel = relation.to
            if ThroughModel is not None:
                # A many to many relation case. For example:
                #
                #   class Pizza(SQLModel):
                #       toppings = Relation(lambda: Topping, through=lambda: PizzaTopping)
                #   class Topping(SQLModel):
                #       name = Str()
                #   class PizzaTopping(SQLModel):
                #       pizza = Instance(Pizza)
                #       topping = Instance(Topping)
                #
                # When we have:
                #   toppings = await pizza.toppings.load()
                # The pizza is the owner, and the toppings member is the relation.
                # So inlining it will be the same as the following:
                #   toppings = [
                #      row.topping await PizzaTopping.objects.select_related(
                #          "topping").filter(pizza=pizza)
                #   ]
                #
                relation_backref = resolve_backref(RelModel, ThroughModel)
                if relation_backref is None:
                    raise UnresolvableError(
                        f"relation between {RelModel} and through model {ThroughModel}"
                        f": Tried {RelModel.__backrefs__}"
                    )
                owner_backref = resolve_backref(Model, ThroughModel)
                if owner_backref is None:
                    raise UnresolvableError(
                        f"relation between {Model} and through model {ThroughModel}"
                        f": Tried {Model.__backrefs__}"
                    )
                items = [
                    getattr(row, relation_backref.name)
                    for row in await ThroughModel.objects.select_related(
                        relation_backref.name
                    ).filter(**{owner_backref.name: owner})
                ]
            else:
                # A many to one relation case. For example:
                #
                #   class Page(SQLModel):
                #       comments = Relation(lambda: Comment)
                #   class Comment(SQLModel):
                #       page = Instance(Page)
                #
                # When we have:
                #   comments = await page.comments.load()
                # The page is the owner, and the comments member is the relation.
                # So inlining it will be the same as the following:
                #   comments = await Comments.objects.filter(page=page)
                owner_backref = resolve_backref(Model, RelModel)
                if owner_backref is None:
                    raise UnresolvableError(
                        f"relation between {Model} and {RelModel}"
                        f": Tried {Model.__backrefs__}"
                    )
                items = await RelModel.objects.filter(**{owner_backref.name: owner})

            if inplace:
                for item in items:
                    if item not in self:
                        self.append(item)
            return items

        async def save(self, connection=None):
            """Save the current list as the complete set of related items. This
            should only be used for small sets of items.
            """
            current = set(self)
            saved = set(await self.load(inplace=False))
            ThroughModel = relation.through
            RelModel = relation.to
            if ThroughModel is not None:
                Model = cast(Type[SQLModel], type(owner))
                owner_backref = resolve_backref(Model, ThroughModel)
                relation_backref = resolve_backref(RelModel, ThroughModel)
                removed_ids = [obj._id for obj in saved.difference(current)]

                if removed_ids:
                    # Remove old
                    await ThroughModel.objects.filter(
                        **{
                            owner_backref.name: owner,
                            f"{relation_backref.name}__in": removed_ids,
                        }
                    ).delete(connection=connection)

                # Add new
                for added_item in current.difference(saved):
                    await ThroughModel.objects.create(
                        **{
                            owner_backref.name: owner,
                            relation_backref.name: added_item,
                        }
                    )
            else:
                for removed_item in saved.difference(current):
                    await removed_item.delete(connection=connection)
                for added_item in current.difference(saved):
                    await added_item.save(connection=connection)

    return RelatedList


class Relation(ContainerList):
    """A member which serves as a fk relation backref"""

    __slots__ = ("_to", "_through")

    def __init__(
        self,
        item: CallableType[[], Type[Model]],
        default: Any = None,
        *,
        through: CallableType[[], Optional[Type[Model]]] = lambda: None,
    ):
        super().__init__(ForwardInstance(item))  # type: ignore
        self._to: Optional[Type[Model]] = None
        self._through = through
        self.tag(store=False)
        self.set_post_getattr_mode(
            api.PostGetAttr.MemberMethod_ObjectValue, "post_getattr"
        )

    def post_getattr(self, obj: Model, value: ListType[Model]):
        """Rewrite class to RelatedList"""
        value.__class__ = create_related_list(obj, self)
        return value

    def resolve(self) -> Type[Model]:
        return self.to

    @property
    def to(self) -> Type[Model]:
        to = self._to
        if to is None:
            types = resolve_member_types(self.validate_mode[-1])
            assert types is not None
            to = self._to = types[0]
        return to

    @property
    def through(self) -> Optional[Type[Model]]:
        """Return the through model"""
        return self._through()


class RelatedInstance(ForwardInstance):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # type: ignore
        self.tag(store=False)


class RelatedTyped(ForwardTyped):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # type: ignore
        self.tag(store=False)


def py_type_to_sql_column(
    model: Type[Model],
    member: Member,
    types: Union[Type, TupleType[Type, ...]],
    **kwargs,
) -> TypeEngine:
    """Convert the python type to an alchemy table column type"""
    if isinstance(types, tuple):
        cls, *subtypes = types
    else:
        cls = types

    if issubclass(cls, JSONModel):
        return sa.JSON(**kwargs)
    elif issubclass(cls, SQLModel):
        name = f"{cls.__model__}.{cls.__pk__}"
        cls.__backrefs__.add((model, member))

        # Determine the type of the foreign key
        column = create_table_column(cls, cls._id)
        return (column.type, sa.ForeignKey(name, **kwargs))
    elif issubclass(cls, str) or (
        hasattr(enum, "StrEnum") and isinstance(cls, enum.StrEnum)
    ):
        return sa.String(**kwargs)
    elif issubclass(cls, int) or (
        hasattr(enum, "IntEnum") and isinstance(cls, (enum.IntEnum, enum.IntFlag))
    ):
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
    elif issubclass(cls, datetime.timedelta):
        return sa.Interval(**kwargs)
    elif issubclass(cls, (bytes, bytearray)):
        return sa.LargeBinary(**kwargs)
    elif issubclass(cls, Decimal):
        return sa.Numeric(**kwargs)
    raise NotImplementedError(
        f"A type for {member.name} of {model} ({cls}) could not be "
        f"determined automatically, please specify it manually by tagging it "
        f"with .tag(column=<sqlalchemy column>) or set `store=False`"
    )


@functools.lru_cache(1024)
def resolve_member_column(
    model: Type["SQLModel"], field: str
) -> TupleType[sa.Column, SetType[str]]:
    """Get the sqlalchemy column for the given model and field.

    Parameters
    ----------
    model: atomdb.sql.Model
        The model to lookup
    field: String
        The field name

    Returns
    -------
    result: tuple[sa.Column, set[str]]
        A tuple the sqlalchemy column for the model field and the set of
        related clauses needed to join across foreign keys.

    """
    if model is None or not field:
        raise ValueError("Invalid field %s on %s" % (field, model))

    # Walk the relations
    related_clauses = set()
    if "__" in field:
        path = field
        *related_parts, field = field.rsplit("__")
        clause = "__".join(related_parts)
        related_clauses.add(clause)

        # Follow the FK lookups
        # Rename so the original lookup path is retained if an error occurs
        rel_model = model
        for part in related_parts:
            m = rel_model.members().get(part)
            if m is None:
                raise ValueError("Invalid field %s on %s" % (path, model))
            rel_model_types = resolve_member_types(m)
            if rel_model_types is None:
                raise ValueError("Invalid field %s on %s" % (path, model))
            rel_model = rel_model_types[0]
        model = rel_model

    # Lookup the member
    members = model.members()
    if field not in members and field == "_id":
        field = model.__pk__  # Support lookup using _id for pk
    m = members.get(field)
    if m is not None:
        if m.metadata:
            # If the field has a different name assigned use that
            field = m.metadata.get("name", field)
        if isinstance(m, Relation):
            # Support looking up columns through a relation by the pk
            model = m.to  # type: ignore

            # Add the through table to the related clauses if needed
            related_clauses.add(field)

            field = model.__pk__

    # Finally get the column from the table
    col = model.objects.table.columns.get(field)
    if col is None:
        raise ValueError("Invalid field %s on %s" % (field, model))
    return col, related_clauses


@functools.lru_cache(1024)
def resolve_backref(model: Type["SQLModel"], through: Type["SQLModel"]) -> Member:
    """Find the member on the through model that refers to the given model."""
    assert model.objects and through.objects  # Force creation
    for other_model, referring_member in model.__backrefs__:
        if other_model is through:
            return referring_member
    raise ValueError(f"No backref relation found between {model} and {through}")


@functools.lru_cache(1024)
def resolve_relation(
    model: Type["SQLModel"], field: str
) -> TupleType[Member, Type[Model], Member, sa.Column]:
    """Lookup a Relation.

    Parameters
    ----------
    model: SQLModel
        The model to lookup.
    field: str
        Path to a Relation, Typed, or Instance marked with store=False

    Returns
    -------
    result: tuple[Member, SQLModel, Member, sa.Column]
        A tuple of the related field on the given model, the other model
        it points to, and the field on that model that points back to this
        model, and the Column.

    """
    relation = model.members().get(field)
    RelModel: Optional[Type[Model]] = None
    if isinstance(relation, Relation):
        # RelModel has a many to one relation back to model
        RelModel = cast(Relation, relation).to
    elif isinstance(relation, FK_TYPES) and not is_db_field(relation):
        # Note: If is_db_field passes the user should use select_related
        # instead of prefetch related.
        types = resolve_member_types(relation)
        if types and len(types) == 1 and issubclass(types[0], Model):
            # RelModel has a one to one relation back to model
            RelModel = types[0]

    if RelModel is not None:
        assert model.objects and RelModel.objects  # Force creation

        m = cast(Member, relation)
        # Find the referring member
        # TODO: This does not support multiple backrefs
        for other_model, referring_member in model.__backrefs__:
            if RelModel is other_model:
                meta = referring_member.metadata or {}
                name = meta.get("name", referring_member.name)
                rel_col = RelModel.objects.table.c[name]
                return (m, other_model, referring_member, rel_col)
    raise ValueError("Invalid prefetch relation '%s' from %s" % (field, model))


def atom_member_to_sql_column(
    model: Type["SQLModel"], member: Member, **kwargs
) -> TypeEngine:
    """Convert the atom member type to an sqlalchemy table column type
    See https://docs.sqlalchemy.org/en/latest/core/type_basics.html

    """
    if hasattr(member, "get_column_type"):
        # Allow custom members to define the column type programatically
        return member.get_column_type(model)  # type: ignore
    elif isinstance(member, api.Str):
        return sa.String(**kwargs)
    elif hasattr(api, "Unicode") and isinstance(member, api.Unicode):  # type: ignore
        return sa.Unicode(**kwargs)  # type: ignore
    elif isinstance(member, api.Bool):
        return sa.Boolean()
    elif isinstance(member, api.Int):
        return sa.Integer()
    elif hasattr(api, "Long") and isinstance(member, api.Long):  # type: ignore
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
        model_name = model.__model__.replace(".", "_")
        enum_name = f"{model_name}_{member.name}"
        return sa.Enum(*member.items, name=enum_name)
    elif hasattr(api, "IntEnum") and isinstance(member, api.IntEnum):  # type: ignore
        return sa.SmallInteger()
    elif isinstance(member, FK_TYPES):
        value_type = resolve_member_types(member)
        if value_type is None:
            raise TypeError("Instance and Typed members must specify types")
        return py_type_to_sql_column(model, member, value_type, **kwargs)
    elif isinstance(member, Relation):
        # Relations are for backrefs
        item_type = member.validate_mode[-1]
        if item_type is None:
            raise TypeError("Relation members must specify types")

        # Resolve the item type
        value_type = resolve_member_types(item_type)
        if value_type is None:
            raise TypeError("Relation members must specify types")
        return None  # Relations are just syntactic sugar
    elif isinstance(member, (api.List, api.ContainerList, api.Tuple, api.Set)):
        item_type = member.validate_mode[-1]
        if item_type is None:
            raise TypeError("List, Set, and Tuple members must specify types")

        # Resolve the item type
        value_type = resolve_member_types(item_type)
        if value_type is None:
            raise TypeError("List, Set, and Tuple members must specify types")
        if issubclass(value_type[0], JSONModel):
            return sa.JSON(**kwargs)
        t = py_type_to_sql_column(model, member, value_type, **kwargs)
        if isinstance(t, tuple):
            t = t[0]  # Use only the value type
        return sa.ARRAY(t)
    elif isinstance(member, api.Bytes):
        return sa.LargeBinary(**kwargs)
    elif isinstance(member, api.Dict):
        return sa.JSON(**kwargs)
    raise NotImplementedError(
        f"A column for {member.name} of {model} could not be determined "
        f"automatically, please specify it manually by tagging it "
        f"with .tag(column=<sqlalchemy column>)"
    )


def create_table_column(model: Type["SQLModel"], member: Member) -> sa.Column:
    """Converts an Atom member into a sqlalchemy data type.

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
    get_column = getattr(member, "get_column", None)
    if get_column is not None:
        # Allow custom members to define the column programatically
        return get_column(model)

    # Copy the metadata as we modify it
    metadata = member.metadata.copy() if member.metadata else {}

    # If a column is specified use that
    if "column" in metadata:
        return metadata["column"]

    metadata.pop("store", None)
    column_name = metadata.pop("name", member.name)
    column_type = metadata.pop("type", None)
    for k in SERIALIZE_KWARGS:
        metadata.pop(k, None)

    # Extract column kwargs from member metadata
    kwargs = {}
    for k in COLUMN_KWARGS:
        if k in metadata:
            kwargs[k] = metadata.pop(k)

    # Set default nullable value
    if "nullable" not in kwargs:
        if "primary_key" in kwargs:
            kwargs["nullable"] = False
        elif isinstance(member, NON_NULL_MEMBERS):
            kwargs["nullable"] = False
        elif hasattr(member, "optional"):
            kwargs["nullable"] = member.optional  # type: ignore
        elif isinstance(member, Typed):
            optional = member.validate_mode[0] == Validate.OptionalTyped
            kwargs["nullable"] = optional
        elif isinstance(member, Instance):
            optional = member.validate_mode[0] == Validate.OptionalInstance
            kwargs["nullable"] = optional

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


def create_table(model: Type["SQLModel"], metadata: sa.MetaData) -> sa.Table:
    """Create an sqlalchemy table by inspecting the Model and generating
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
    meta = getattr(model, "Meta", None)
    if meta:
        # Abstract field
        abstract = getattr(meta, "abstract", False)
        if abstract:
            raise NotImplementedError(
                f"Tables cannot be created for abstract models: {model}"
            )

        # Unique constraints
        unique_together = getattr(meta, "unique_together", None)
        if unique_together is not None:
            if not isinstance(unique_together, (tuple, list)):
                raise TypeError("Meta unique_together must be a tuple or list")
            if isinstance(unique_together[0], str):
                unique_together = [unique_together]
            for constraint in unique_together:
                if isinstance(constraint, (tuple, list)):
                    constraint = sa.UniqueConstraint(*constraint)
                args.append(constraint)

        # Check constraints
        constraints = getattr(meta, "constraints", None)
        if constraints is not None:
            if not isinstance(constraints, (tuple, list)):
                raise TypeError("Meta constraints must be a tuple or list")
            args.extend(constraints)

        # Composite indexes
        composite_indexes = getattr(meta, "composite_indexes", None)
        if composite_indexes is not None:
            if not isinstance(composite_indexes, (tuple, list)):
                raise TypeError("Meta composite_indexes must be a tuple or list")
            for index in composite_indexes:
                if not isinstance(index, (tuple, list)):
                    raise TypeError("Index must be a tuple or list")
                args.extend([schema.Index(*index)])

        # Validate get_latest_by
        get_latest_by = getattr(meta, "get_latest_by", None)
        if get_latest_by is not None:
            if isinstance(get_latest_by, str):
                get_latest_by = (get_latest_by,)
            if not all(isinstance(f, str) for f in get_latest_by):
                raise TypeError("Meta get_latest_by must be a str or tuple[str]")

    # Create table
    table = sa.Table(name, metadata, *args)

    # Hook up any database triggers defined
    triggers = getattr(meta, "triggers", None)
    if triggers is not None:
        if isinstance(triggers, dict):
            triggers = list(triggers.items())
        elif not isinstance(triggers, (tuple, list)):
            raise TypeError("Meta triggers must be a dict, tuple, or list")
        for event, trigger in triggers:
            # Allow triggers to be a lambda that generates one
            if not isinstance(trigger, sa.schema.DDL) and callable(trigger):
                trigger = trigger()
            sa.event.listen(table, event, trigger)

    return table


def model_latest_by_field(Model: Type["SQLModel"]) -> TupleType[str]:
    meta = getattr(Model, "Meta", None)
    get_latest_by = getattr(meta, "get_latest_by", None)
    if get_latest_by is None:
        raise TypeError(
            f"Model '{Model}' has no get_latest_by field defined in it's Meta"
        )
    if isinstance(get_latest_by, str):
        return (get_latest_by,)
    return get_latest_by


def reverse_order_clause(
    clause: Union[schema.Column, UnaryExpression]
) -> UnaryExpression:
    if isinstance(clause, schema.Column):
        # sql default is asc so reverse
        return clause.desc()
    if clause.modifier is desc_op:
        modifier = asc_op
    else:
        modifier = desc_op
    return UnaryExpression(
        clause.element, modifier=modifier, wraps_column_expression=False
    )


class SQLModelSerializer(ModelSerializer):
    """Uses sqlalchemy to lookup the model."""

    def flatten_object(self, obj: Model, scope: ScopeType) -> Any:
        """Serialize a model for entering into the database

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
        if isinstance(obj, SQLModel):
            return obj._id
        return type(obj).serializer.flatten_object(obj, scope)

    async def get_object_state(self, obj, state, scope):
        """Load the object state if needed. Since the __model__ is not saved
        to the db tables with SQL we know that if it's "probably" there
        because a query was used.
        """
        ModelType = obj.__class__
        if "__model__" in state:
            return state  # Joined already
        q = ModelType.objects.query(None, _id=state["_id"])
        return await ModelType.objects.fetchone(q)

    def _default_registry(self):
        """Add all sql and json models to the registry"""
        registry = JSONSerializer.instance().registry.copy()
        registry.update({m.__model__: m for m in find_sql_models()})
        return registry


class SQLModelManager(ModelManager):
    """Manages models via aiopg, aiomysql, or similar libraries supporting
    SQLAlchemy tables. It stores a table for each class and when accessed
    on a Model subclass it returns a table proxy binding.

    """

    #: Constraint naming convenctions
    conventions = Dict(default=CONSTRAINT_NAMING_CONVENTIONS)

    #: Metadata
    metadata = Instance(sa.MetaData)

    #: Table proxy cache
    proxies = Dict()

    #: Cache results.
    cache = Bool(True)

    def _default_metadata(self) -> sa.MetaData:
        binding = SQLBinding(manager=self)
        return sa.MetaData(binding, naming_convention=self.conventions)

    def create_tables(self) -> DictType[Type["SQLModel"], sa.Table]:
        """Create sqlalchemy tables for all registered SQLModels"""
        tables = {}
        for cls in find_sql_models():
            table = cls.__table__
            if table is None:
                table = self.create_table_and_restore_fn(cls)
            if not table.metadata.bind:
                table.metadata.bind = SQLBinding(manager=self, table=table)
            tables[cls] = table
        return tables

    def create_table_and_restore_fn(self, cls: Type["SQLModel"]) -> sa.Table:
        """Generate the sqlalchemy table and optimized restore function.
        This is done here to make sure that foreign and forwarded members
        are now resolved.

        """
        assert cls.__table__ is None
        table = cls.__table__ = create_table(cls, self.metadata)
        cls.__generated_restorestate__ = generate_sql_restorestate(cls)
        return table

    def __get__(
        self, obj: T, cls: Optional[Type[T]] = None
    ) -> Union["SQLTableProxy[T]", "SQLModelManager"]:
        """Retrieve the table for the requested object or class."""
        cls = cls or obj.__class__
        if not issubclass(cls, Model):
            return self  # Only return the client when used from a Model
        proxy = self.proxies.get(cls)
        if proxy is None:
            table = cls.__table__
            if table is None:
                table = self.create_table_and_restore_fn(cls)
            proxy = self.proxies[cls] = SQLTableProxy(table=table, model=cls)
        return proxy

    def _default_database(self):
        raise EnvironmentError(
            "No database engine has been set. Use "
            "SQLModelManager.instance().database = <db>"
        )


class ConnectionProxy(Atom):
    """An wapper for a connection to be used with async with syntax that
    does nothing but passes the existing connection when entered.

    """

    connection = Value()

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, tb):
        pass


class SQLTableProxy(Atom, Generic[T]):
    #: Table this is a proxy to
    table = Instance(sa.Table, optional=False)

    #: Model which owns the table
    model = ForwardSubclass(lambda: SQLModel)

    #: Cache of pk: obj using weakrefs
    cache = Typed(weakref.WeakValueDictionary, ())

    #: Key used to pull the connection out of filter kwargs
    connection_kwarg = Str("connection")

    #: Key used to pass the force restore option
    restore_kwarg = Str("force_restore")

    #: Reference to the aiomysql or aiopg Engine
    #: This is used to get a connection from the connection pool.
    @property
    def engine(self):
        """Retrieve the database engine."""
        db = self.table.bind.manager.database
        if isinstance(db, dict):
            return db[self.model.__database__]
        return db

    def connection(self, connection=None):
        """Create a new connection or the return given connection as an async
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

    def create_table(self, **kwargs):
        """A wrapper for create which catches the create queries then executes
        them
        """
        table = self.table
        table.bind.create(table, **kwargs)
        return table.bind.wait()

    def create_alter_foreign_keys(self, **kwargs):
        """Create any foreign keys that have use_alter=True"""
        table = self.table
        for constraint in self.table.foreign_key_constraints:
            if getattr(constraint, "use_alter", False):
                table.bind.create(constraint, **kwargs)
        return table.bind.wait()

    def drop_alter_foreign_keys(self, **kwargs):
        """Create any foreign keys that have use_alter=True"""
        table = self.table
        for constraint in self.table.foreign_key_constraints:
            if getattr(constraint, "use_alter", False):
                table.bind.drop(constraint, **kwargs)
        return table.bind.wait()

    def drop_table(self, **kwargs):
        table = self.table
        table.bind.drop(table, **kwargs)
        return table.bind.wait()

    async def execute(self, *args, **kwargs):
        connection = kwargs.pop(self.connection_kwarg, None)
        async with self.connection(connection) as conn:
            return await conn.execute(*args, **kwargs)

    async def fetchall(self, query: QueryType, connection=None):
        """Fetch all results for the query.

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

    async def fetchmany(self, query, size: Optional[int] = None, connection=None):
        """Fetch size results for the query.

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

    async def fetchone(self, query: QueryType, connection=None):
        """Fetch a single result for the query.

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

    async def scalar(self, query: QueryType, connection=None):
        """Fetch the scalar result for the query.

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

    async def get_or_create(self, **filters) -> TupleType[T, bool]:
        """Get or create a model matching the given criteria

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
        state = {
            k: v for k, v in filters.items() if "__" not in k and k != connection_kwarg
        }
        obj = self.model(**state)
        await obj.save(force_insert=True, connection=connection)
        return (obj, True)

    async def create(self, **state) -> T:
        """Create a and save model with the given state.

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
        obj = cast(T, self.model(**state))
        await obj.save(force_insert=True, connection=connection)
        return obj

    async def bulk_create(self, items: Sequence[T], connection=None) -> Sequence[T]:
        """Perform a bulk create from a sequence of models. This will
        populate the primary key of the items as needed but will not pull
        any fields that have not been defined. The restored flag will still
        be False.

        Parameters
        ----------
        items: Sequence[T]
            The list of items to create.
        connection: Connetion
            The connection to use (if None one from the pool will be used)

        Returns
        -------
        items: Sequence[T]
            The items passed in. Only postgres will populate the primary keys.

        """
        table = self.table
        values = [item.__prepare_state_for_db__() for item in items]
        if not values:
            return items
        async with self.connection(connection) as conn:
            # TODO: Properly detect?
            postgres = "aiopg" in conn.__class__.__module__
            if postgres:
                pk_column = table.c[self.model.__pk__]
                q = table.insert().returning(pk_column).values(values)
            else:
                # TODO: Get return value?
                q = table.insert().values(values)

            result = await conn.execute(q)
            if postgres:
                cache = self.cache
                for r, item in zip(await result.fetchall(), items):
                    # Don't overwrite if force inserting
                    if not item._id:
                        item._id = r[0]
                    cache[item._id] = item
            return items

    def __getattr__(self, name: str):
        """All other fields are delegated to the query set"""
        qs: SQLQuerySet[T] = SQLQuerySet(proxy=self)
        return getattr(qs, name)


class SQLQuerySet(Atom, Generic[T]):
    #: Proxy
    proxy = Instance(SQLTableProxy, optional=False)
    connection = Value()

    filter_clauses = List()
    related_clauses = Set()
    prefetch_clauses = Set()
    outer_join = Bool()
    order_clauses = List()
    groupby_clauses = List()
    distinct_clauses = Set()
    limit_count = Int()
    query_offset = Int()
    force_restore = Bool()

    def clone(self, **kwargs) -> "SQLQuerySet[T]":
        state: DictType[str, Any] = self.__getstate__()  # type: ignore
        state.update(kwargs)
        return self.__class__(**state)

    def query(self, query_type: str = "select", *columns, **kwargs):
        if kwargs:
            return self.filter(**kwargs).query(query_type)
        p = self.proxy
        from_table = p.table
        tables = {from_table}
        use_labels = bool(self.related_clauses)
        outer_join = self.outer_join
        existing_joins = set()
        for clause in self.related_clauses:
            model = p.model
            table = p.table

            # Walk the fk relations
            alias = []
            for part in clause.split("__"):
                m = model.members().get(part)
                assert m is not None, f"{model} has no field {part}"
                assert issubclass(model, SQLModel)

                alias.append(part)
                rel_model_types = resolve_member_types(m)
                assert rel_model_types is not None
                rel_model = rel_model_types[0]
                assert issubclass(rel_model, SQLModel)
                rel_table = rel_model.objects.table

                # For example:
                # class Job(SQLModel):
                #    status = Str()
                #    roles = Relation(Role)
                # class Role(SQLModel):
                #    name = Str()
                #    job = Instance(Job)
                if isinstance(m, Relation):
                    # Case when looking though the relation
                    # r = await Job.objects.filter(roles__name="foo")
                    backref = resolve_backref(model, m.through or rel_model)
                    rel_key = backref.name
                    self_key = model.__pk__
                else:
                    # Normal foreign key cases or select related
                    # r = await JobRole.objects.filter(job__status="live")
                    rel_key = rel_model.__pk__
                    self_key = m.name

                # Handled renamed fields
                if self_key in model.__renamed_fields__:
                    self_key = model.__renamed_fields__[self_key]
                if rel_key in rel_model.__renamed_fields__:
                    rel_key = rel_model.__renamed_fields__[rel_key]

                join_key = (model, self_key, rel_model, rel_key)

                # Avoid duplicate join, eg `select_related('a', 'a__b")`
                # would join on a twice.
                # TODO: Cannot join the same table twice, need to use alias
                # where `select_related('a__b', 'a__b")`
                if join_key not in existing_joins:
                    onclause = table.c[self_key] == rel_table.c[rel_key]
                    from_table = from_table.join(
                        rel_table, onclause=onclause, isouter=outer_join
                    )
                    existing_joins.add(join_key)

                tables.add(rel_table)
                model = rel_model
                table = rel_table

        if query_type == "select":
            q = sa.select(columns or tables, use_labels=use_labels).select_from(
                from_table
            )
        elif query_type == "delete":
            q = sa.delete(from_table)
        elif query_type == "update":
            q = sa.update(from_table)
        else:
            raise ValueError("Unsupported query type")

        if self.distinct_clauses:
            q = q.distinct(*self.distinct_clauses)

        if self.filter_clauses:
            if len(self.filter_clauses) == 1:
                q = q.where(self.filter_clauses[0])
            else:
                q = q.where(sa.and_(*self.filter_clauses))

        if self.order_clauses:
            q = q.order_by(*self.order_clauses)

        if self.limit_count:
            q = q.limit(self.limit_count)

        if self.query_offset:
            q = q.offset(self.query_offset)

        if self.groupby_clauses:
            q = q.group_by(*self.groupby_clauses)

        return q

    def select_related(
        self, *related: Sequence[str], outer_join: Optional[bool] = None
    ) -> "SQLQuerySet[T]":
        """Define related fields to join in the query.

        Parameters
        ----------
        args: List[str]
            List of related fields to join.
        outer_join: Bool
            If given set whether or not a left outer join is used.

        Returns
        -------
        query: SQLQuerySet
            A clone of this queryset with the related field terms added.

        """
        return self.clone(
            related_clauses=self.related_clauses | set(related),
            outer_join=self.outer_join if outer_join is None else outer_join,
        )

    def prefetch_related(self, *related: Sequence[str]) -> "SQLQuerySet[T]":
        """Define related fields to prefetch in a separate query.

        Parameters
        ----------
        args: List[str]
            List of related fields fetchs

        Returns
        -------
        query: SQLQuerySet
            A clone of this queryset with the prefetch fields added.

        """
        # Validate relations
        for r in related:
            assert resolve_relation(self.proxy.model, r)
        return self.clone(prefetch_clauses=self.prefetch_clauses | set(related))

    def order_by(self, *args, reverse: bool = False) -> "SQLQuerySet[T]":
        """Order the query by the given fields.

        Parameters
        ----------
        args: list[str or column]
            Fields to order by. A "-" prefix denotes decending.
        reverse: bool
            Reverse the order.

        Returns
        -------
        query: SQLQuerySet
            A clone of this queryset with the ordering terms added.

        """
        order_clauses = self.order_clauses[:]
        related_clauses = self.related_clauses.copy()
        model = self.proxy.model
        for arg in args:
            if isinstance(arg, str):
                # Convert django-style to sqlalchemy ordering column
                if arg[0] == "-":
                    field = arg[1:]
                    ascending = False
                else:
                    field = arg
                    ascending = True

                col, new_clauses = resolve_member_column(model, field)
                related_clauses.update(new_clauses)

                if ascending:
                    clause = col.asc()
                else:
                    clause = col.desc()
            else:
                clause = arg
            if reverse:
                clause = reverse_order_clause(clause)
            if clause not in order_clauses:
                order_clauses.append(clause)
        return self.clone(order_clauses=order_clauses, related_clauses=related_clauses)

    def distinct(self, *args) -> "SQLQuerySet[T]":
        """Apply distinct on the given column.

        Parameters
        ----------
        args: list[str or column]
            Fields that must be distinct.

        Returns
        -------
        query: SQLQuerySet
            A clone of this queryset with the distinct terms added.

        """
        distinct_clauses = self.distinct_clauses.copy()
        related_clauses = self.related_clauses.copy()
        model = self.proxy.model
        for arg in args:
            if isinstance(arg, str):
                # Convert name to sqlalchemy column
                clause, new_clauses = resolve_member_column(model, arg)
                related_clauses.update(new_clauses)
            else:
                clause = arg
            distinct_clauses.add(clause)
        return self.clone(
            distinct_clauses=distinct_clauses, related_clauses=related_clauses
        )

    def group_by(self, *args) -> "SQLQuerySet[T]":
        """Apply group by on the given column.

        Parameters
        ----------
        args: list[str or column]
            Fields that must be grouped by.

        Returns
        -------
        query: SQLQuerySet
            A clone of this queryset with the group by terms added.

        """
        groupby_clauses = self.groupby_clauses.copy()
        related_clauses = self.related_clauses.copy()
        model = self.proxy.model
        for arg in args:
            if isinstance(arg, str):
                # Convert name to sqlalchemy column
                clause, new_clauses = resolve_member_column(model, arg)
                related_clauses.update(new_clauses)
            else:
                clause = arg
            groupby_clauses.append(clause)
        return self.clone(
            groupby_clauses=groupby_clauses, related_clauses=related_clauses
        )

    def where_clause(self, k: str, v: Any, related_clauses: SetType[str]):
        """Create a where clause from a django-style parameter.
        This will modify the list of related clauses if a join occurs.

        Parameters
        ----------
        k: str
            The filter key, eg name__startswith
        v: object
            The value
        related_clauses: set[str]
            Set of related clauses needed

        Returns
        -------
        clause: sqlalchemy.sq.expression
            The filter clause

        """
        model = self.proxy.model
        op = "eq"
        if "__" in k:
            field, maybe_op = k.rsplit("__", 1)
            if maybe_op in QUERY_OPS:
                op = maybe_op
                k = field

        col, new_clauses = resolve_member_column(model, k)
        related_clauses.update(new_clauses)

        # Support lookups by model
        if isinstance(v, Model):
            v = v.serializer.flatten_object(v, scope={})
        elif isinstance(v, enum.Enum):
            v = v.value
        elif op in ("in", "notin"):
            # Flatten lists when using in or notin ops
            v = model.serializer.flatten(v, scope={})

        return getattr(col, QUERY_OPS[op])(v)

    def filter(self, *args, **kwargs: DictType[str, Any]) -> "SQLQuerySet[T]":
        """Filter the query by the given parameters. This accepts sqlalchemy
        filters by arguments and django-style parameters as kwargs.

        Parameters
        ----------
        args: List
            List of sqlalchemy filters or a dict of django style filters to or
        kwargs: Dict[str, object]
            Django style filters to use

        Returns
        -------
        query: SQLQuerySet
            A clone of this queryset with the filter terms added.

        """
        p = self.proxy
        filter_clauses = self.filter_clauses[:]
        related_clauses = self.related_clauses.copy()

        connection_kwarg = p.connection_kwarg
        restore_kwarg = p.restore_kwarg

        # Build filter
        for arg in args:
            if isinstance(arg, dict):
                or_clause = sa.or_(
                    *[self.where_clause(k, v, related_clauses) for k, v in arg.items()]
                )
                filter_clauses.append(or_clause)
            else:
                filter_clauses.append(arg)

        # Build the filter operations
        for k, v in kwargs.items():
            if k == connection_kwarg or k == restore_kwarg:
                continue
            filter_clauses.append(self.where_clause(k, v, related_clauses))

        return self.clone(
            connection=kwargs.get(connection_kwarg, self.connection),
            force_restore=kwargs.get(restore_kwarg, self.force_restore),
            filter_clauses=filter_clauses,
            related_clauses=related_clauses,
        )

    def exclude(self, *args, **kwargs: DictType[str, Any]) -> "SQLQuerySet[T]":
        """Exclude results matching the given parameters by wrapping each
        clause in a NOT expression. This accepts sqlalchemy filters by
        arguments and django-style parameters as kwargs.

        Parameters
        ----------
        args: List
            List of sqlalchemy filters or a dict of django style filters to or
        kwargs: Dict[str, object]
            Django style filters to use

        Returns
        -------
        query: SQLQuerySet
            A clone of this queryset with the excluded filter terms added.

        """
        p = self.proxy
        filter_clauses = self.filter_clauses[:]
        related_clauses = self.related_clauses.copy()

        connection_kwarg = p.connection_kwarg
        restore_kwarg = p.restore_kwarg

        # Build filter
        for arg in args:
            if isinstance(arg, dict):
                or_clause = sa.or_(
                    *[self.where_clause(k, v, related_clauses) for k, v in arg.items()]
                )
                filter_clauses.append(sa.not_(or_clause))
            else:
                filter_clauses.append(sa.not_(arg))

        # Build the filter operations
        for k, v in kwargs.items():
            if k == connection_kwarg or k == restore_kwarg:
                continue
            clause = self.where_clause(k, v, related_clauses)
            filter_clauses.append(sa.not_(clause))

        return self.clone(
            connection=kwargs.get(connection_kwarg, self.connection),
            force_restore=kwargs.get(restore_kwarg, self.force_restore),
            filter_clauses=filter_clauses,
            related_clauses=related_clauses,
        )

    def __getitem__(self, key: Union[int, slice]) -> "SQLQuerySet[T]":
        if isinstance(key, slice):
            offset = key.start or 0
            limit = key.stop - key.start if key.stop else 0
        elif isinstance(key, int):
            limit = 1
            offset = key
        else:
            raise TypeError("Invalid key")
        if offset < 0:
            raise ValueError("Cannot use a negative offset")
        if limit < 0:
            raise ValueError("Cannot use a negative limit")
        return self.clone(limit_count=limit, query_offset=offset)

    def limit(self, limit: int) -> "SQLQuerySet[T]":
        return self.clone(limit_count=limit)

    def offset(self, offset: int) -> "SQLQuerySet[T]":
        return self.clone(query_offset=offset)

    # -------------------------------------------------------------------------
    # Query execution API
    # -------------------------------------------------------------------------
    async def values(
        self,
        *args,
        distinct: bool = False,
        flat: bool = False,
        group_by: Optional[Sequence[Union[str, sa.Column]]] = None,
    ) -> Sequence[Any]:
        """Returns the results as a list of dict instead of models.

        Parameters
        ----------
        args: List[str or column]
            List of columns to select
        distinct: Bool
            Return only distinct rows
        flat: Bool
            Requires exactly one arg and will flatten the result into a single
            list of values.
        group_by: List[str or column]
            Optional Columns to group by

        Returns
        -------
        results: List
            List of results depending on the parameters described above

        """
        if flat and len(args) != 1:
            raise ValueError("Values with flat=True can only have one param")
        if args:
            model = self.proxy.model
            columns = []
            for col in args:
                if isinstance(col, str):
                    col, _ = resolve_member_column(model, col)
                columns.append(col)
            q = self.query("select", *columns)
        else:
            q = self.query("select")
        if group_by is not None:
            q = q.group_by(group_by)
        if distinct:
            q = q.distinct()
        cursor = await self.proxy.fetchall(q, connection=self.connection)
        if flat:
            return [row[0] for row in cursor]
        return cursor

    async def count(self, *args, **kwargs) -> int:
        if args or kwargs:
            return await self.filter(*args, **kwargs).count()
        subq = self.query("select").alias("subquery")
        q = sa.func.count().select().select_from(subq)
        return await self.proxy.scalar(q, connection=self.connection)

    def max(self, *columns):
        return self.aggregate(*columns, func=sa.func.max)

    def min(self, *columns):
        return self.aggregate(*columns, func=sa.func.min)

    def mode(self, *columns):
        return self.aggregate(*columns, func=sa.func.mode)

    def sum(self, *columns):
        return self.aggregate(*columns, func=sa.func.sum)

    def aggregate(self, *args, func=None):
        model = self.proxy.model
        columns = []
        for col in args:
            if isinstance(col, str):
                col, _ = resolve_member_column(model, col)
            columns.append(func(col) if func is not None else col)
        subq = self.query("select").alias("subquery")
        q = sa.select(columns).select_from(subq)
        return self.proxy.fetchone(q, connection=self.connection)

    async def exists(self, *args, **kwargs) -> bool:
        if args or kwargs:
            return await self.filter(*args, **kwargs).exists()
        q = sa.exists(self.query("select")).select()
        return await self.proxy.scalar(q, connection=self.connection)

    async def delete(self, *args, **kwargs):
        if args or kwargs:
            return await self.filter(*args, **kwargs).delete()
        q = self.query("delete")
        return await self.proxy.execute(q, connection=self.connection)

    async def update(self, **values):
        """Perform an update of the given values."""
        # Translate any renamed fields back to the database value
        for py_name, db_name in self.proxy.model.__renamed_fields__.items():
            if py_name in values:
                values[db_name] = values.pop(py_name)
        q = self.query("update").values(**values)
        return await self.proxy.execute(q, connection=self.connection)

    def __await__(self):
        # So await Model.objects.filter() works
        f = asyncio.ensure_future(self.all())
        yield from f
        return f.result()

    async def all(self, *args, **kwargs) -> Sequence[T]:
        """Get the all results matching the query. This will force restore the
        items even if it was in the cache.

        Returns
        -------
        results: list[Model]
            The models entry matching the query

        """
        if args or kwargs:
            return await self.filter(*args, **kwargs).all()
        cache = await self.prefetch()
        q = self.query("select")
        restore = self.proxy.model.restore
        cursor = await self.proxy.fetchall(q, connection=self.connection)
        force = self.force_restore
        return [
            cast(T, await restore(row, force=force, prefetched=cache)) for row in cursor
        ]

    async def get(self, *args, **kwargs) -> Optional[T]:
        """Get the first result matching the query. Unlike django this will
        NOT raise an error if multiple objects would be returned or an entry
        does not exist. This will force restore the item even if it was in the
        cache.

        Returns
        -------
        model: Optional[Model]
            The first entry matching the query

        """
        if args or kwargs:
            return await self.filter(*args, **kwargs).get()
        q = self.query("select")
        row = await self.proxy.fetchone(q, connection=self.connection)
        if row is None:
            return None
        cache = await self.prefetch()
        model = self.proxy.model
        force = self.force_restore
        return cast(T, await model.restore(row, force=force, prefetched=cache))

    async def prefetch(self) -> Optional[DictType[Any, StateType]]:
        """Perform a prefetch lookup and populate the cache."""
        if not self.prefetch_clauses:
            return None

        # Cache is a mapping of this model's pk to related member field values
        cache: DictType[Any, StateType] = {}

        model = self.proxy.model
        sub_query = self.query("select", model.objects.table.c[model.__pk__])

        # Perform a query for each related field
        for field in self.prefetch_clauses:
            #: TDOO: This only works with a single relation
            m, RelModel, ref_member, rel_col = resolve_relation(model, field)
            results = await RelModel.objects.filter(
                rel_col.in_(sub_query), connection=self.connection
            )

            # Group the results by the this models pk
            # Eg if Email.attachments is a relation to Attachments
            # This will group by the Email value
            if isinstance(m, Relation):
                # Get list of items
                for r in results:
                    pk = ref_member.get_slot(r)._id
                    prefetched_state = cache.get(pk)
                    if prefetched_state is None:
                        prefetched_state = cache[pk] = {field: []}
                    relation_values = prefetched_state.get(field)
                    if relation_values is None:
                        relation_values = prefetched_state[field] = []
                    relation_values.append(r)
            else:
                for r in results:
                    pk = ref_member.get_slot(r)._id
                    prefetched_state = cache.get(pk)
                    if prefetched_state is None:
                        prefetched_state = cache[pk] = {}
                    prefetched_state[field] = r
        return cache

    async def earliest(self, *fields: str) -> Optional[T]:
        if not fields:
            fields = model_latest_by_field(self.proxy.model)
        return await self.order_by(*fields).first()

    async def latest(self, *fields: str) -> Optional[T]:
        if not fields:
            fields = model_latest_by_field(self.proxy.model)
        return await self.order_by(*fields, reverse=True).first()

    async def first(self) -> Optional[T]:
        return await self.get()

    async def last(self) -> Optional[T]:
        if not self.order_clauses:
            raise ValueError("Using last on an unordered query is not supported")
        order_clauses = [reverse_order_clause(clause) for clause in self.order_clauses]
        return await self.clone(order_clauses=order_clauses).get()


class SQLBinding(Atom):
    #: Model Manager
    manager = Instance(SQLModelManager)

    #: The queue
    queue = ContainerList()

    engine = property(lambda s: s)

    @property
    def name(self):
        return self.dialect.name

    @property
    def dialect(self):
        """Get the dialect of the database."""
        db = self.manager.database
        if isinstance(db, dict):
            db = db["default"]
        return db.dialect

    def schema_for_object(self, obj):
        return obj.schema

    def contextual_connect(self, **kwargs):
        return self

    def connect(self, **kwargs):
        return self

    def execution_options(self, **kw):
        return self

    def compiler(self, statement, parameters, **kwargs):
        return self.dialect.compiler(statement, parameters, engine=self, **kwargs)

    def create(self, entity, **kwargs):
        node = ddl.SchemaGenerator(self.dialect, self, **kwargs)
        node.traverse_single(entity)

    def drop(self, entity, **kwargs):
        node = ddl.SchemaDropper(self.dialect, self, **kwargs)
        node.traverse_single(entity)

    def _run_ddl_visitor(self, visitorcallable, element, connection=None, **kwargs):
        visitorcallable(self.dialect, self, **kwargs).traverse_single(element)

    def _run_visitor(self, visitorcallable, element, connection=None, **kwargs):
        node = visitorcallable(self.dialect, self, **kwargs)
        node.traverse_single(element)

    def execute(self, object_, *multiparams, **params):
        self.queue.append((object_, multiparams, params))

    async def wait(self):
        db = self.manager.database
        if isinstance(db, dict):
            engine = db["default"]
        else:
            engine = db
        result = None
        async with engine.acquire() as conn:
            try:
                while self.queue:
                    op, args, kwargs = self.queue.pop(0)
                    result = await conn.execute(op, args)
            finally:
                self.queue = []  # Wipe queue on error
        return result


async def get_cached_model(cls: Type[T], pk: Any, state: StateType) -> Optional[T]:
    """Retrieve a model from the cache using the given pk. If the cached
    object does not exist attempt to restore it from the state otherwise create
    a model that has not been loaded and only contains the id.

    Parameters
    ----------
    cls: Type[SQLModel]
        The class to lookup.
    pk: Any
        The primary key to look for.
    state: StateType
        The state from a join query.

    Returns
    -------
    obj: Optional[SQLModel]
        If the pk is not None an instance of cls.

    """
    if cls.__joined_pk__ in state and state[cls.__joined_pk__]:
        return await cls.restore(state)  # Restore from joined row result
    if not pk:
        return None
    cache = cls.objects.cache
    obj = cache.get(pk)
    if obj is not None:
        return obj  # item is already in the cache
    # Create an unloaded model
    obj = cls.__new__(cls)
    cache[pk] = obj
    obj._id = pk
    return obj


def generate_sql_restorestate(cls: Type["SQLModel"]) -> RestoreStateFn:
    """Generate an optimized restore function for the SQL model. The generated
    function creates "inline" dict key lookups for the table columns that
    may have been joined or renamed. This avoids having to do this at runtime.

    """
    template = [
        "async def __restorestate__(self, state, scope=None):",
        "if '__model__' in state and state['__model__'] != self.__model__:",
        "    name = state['__model__']",
        "    raise ValueError(",
        "        f'Trying to use {name} state for {self.__model__} object'",
        "    )",
        "scope = scope or {}",
        "if '__ref__' in state and state['__ref__'] is not None:",
        "    scope[state['__ref__']] = self",
    ]

    on_error = cls.__on_error__
    default_unflatten = cls.serializer.unflatten
    setters = []
    excluded = {"__model__", "__ref__", "__restored__"}
    for f, m in cls.members().items():
        if f in excluded:
            continue
        meta = m.metadata or {}
        order = meta.get("setstate_order", 1000)

        # Allow  tagging a custom unflatten fn
        unflatten = meta.get("unflatten", default_unflatten)

        setters.append((order, f, m, unflatten))
    setters.sort(key=lambda it: it[0])

    namespace: DictType[str, Any] = {
        "default_unflatten": default_unflatten,
        "get_cached_model": get_cached_model,
    }

    # The state dict may have data from multiple tables that have been joined
    # together. This handles that case.
    table_name = cls.__model__
    for order, f, m, unflatten in setters:
        if m.metadata is not None:
            col = m.metadata.get("name", f)
        else:
            col = f
        k = f"{table_name}_{col}"
        # Since f, col, and k are potentially an untrusted input, make sure they are
        # valid python identifiers to prevent unintended code being generated.
        if not f.isidentifier():
            raise ValueError(f"Field '{f}' cannot be used for code generation")

        # TODO: Do proper column name validation
        if not k.replace(".", "_").isidentifier():
            raise ValueError(f"Key '{k}' cannot be used for code generation")

        if not col.isidentifier():
            raise ValueError(f"Renamed '{col}' cannot be used for code generation")

        # TODO: Is there a better way to check for multiple keys?
        if f in cls.__renamed_fields__:
            # Make sure renamed fields are checked for
            template.append(f"if '{col}' in state or '{k}' in state or '{f}' in state:")
            template.append(
                f"    v = state['{col}' if '{col}' in state else ("
                f"'{k}' if '{k}' in state else '{f}')]"
            )
        elif col in cls.__fields__ and not isinstance(m, Relation):
            # Expression to retrieve the value
            # Always check the joined type first
            template.append(f"if '{f}' in state or '{k}' in state:")
            template.append(f"    v = state['{k}' if '{k}' in state else '{f}']")
        else:
            template.append(f"if '{f}' in state:")
            template.append(f"    v = state['{f}']")

        # If a custom unflatten is not provided use the member type information
        # to pick the most efficient way to restore the value
        if unflatten is default_unflatten:
            RelModel = None
            if isinstance(m, FK_TYPES):
                types = resolve_member_types(m)
                if types and len(types) == 1 and issubclass(types[0], Model):
                    RelModel = types[0]

            if RelModel is not None:
                # TODO: This is fine for Typed members but not Instance..
                # as it may need to be a subclass
                namespace[f"rel_model_{f}"] = RelModel

                if issubclass(RelModel, JSONModel):
                    obj = f"await rel_model_{f}.restore(v)"
                else:
                    obj = f"await get_cached_model(rel_model_{f}, v, state)"

                # Only convert if the object has not already been restored
                expr = "\n            ".join(
                    [
                        f"if isinstance(v, rel_model_{f}):",
                        f"    self.{f} = v",
                        "else:",
                        f"    self.{f} = {obj}",
                    ]
                )

            elif is_primitive_member(m):
                expr = f"self.{f} = v"
            else:
                expr = f"self.{f} = await default_unflatten(v, scope)"
        else:
            # Use provided unflatten function
            namespace[f"unflatten_{f}"] = unflatten
            if asyncio.iscoroutinefunction(unflatten):
                expr = f"self.{f} = await unflatten_{f}(v, scope)"
            else:
                expr = f"self.{f} = unflatten_{f}(v, scope)"

        if on_error == "raise":
            template.append(f"    {expr}")
        else:
            if on_error == "log":
                handler = f"self.__log_restore_error__(e, '{f}', state, scope)"
            else:
                handler = "pass"
            template.extend(
                [
                    "    try:",
                    f"        {expr}",
                    "    except Exception as e:",
                    f"        {handler}",
                ]
            )

    # Update restored state
    template.append("self.__restored__ = True")
    source = "\n    ".join(template)
    # print("\n----------------------------------------\n")
    # print(cls)
    # print(source)
    # print("\n----------------------------------------\n")
    return generate_function(source, namespace, "__restorestate__")


class SQLMeta(ModelMeta):
    """Both the pk and _id are aliases to the primary key"""

    def __new__(meta, name, bases, dct):
        cls = ModelMeta.__new__(meta, name, bases, dct)

        members = cls.members()

        # If a member tagged with primary_key=True is defined,
        # on this class, use that as the primary key and reassign
        # the _id member to alias the new primary key.
        pk: Member = cls._id
        for name, m in members.items():
            if name == "_id":
                continue
            if m.metadata and m.metadata.get("primary_key"):
                if pk.name != "_id" and m.name != pk.name:
                    raise NotImplementedError(
                        "Using multiple primary keys is not yet supported. "
                        f"Both {pk.name} and {m.name} are marked as primary."
                    )
                pk = m

        if pk is not cls._id:
            # Workaround member index generation issue
            # TODO: Remove this
            old_index = cls._id.index
            if old_index > 0 and pk.index != old_index:
                pk.set_index(old_index)

            # Reassign the _id field to the primary key member.
            cls._id = members["_id"] = pk

        # Ensure proper pk name is fields list
        if pk.name != "_id" and "_id" in cls.__fields__:
            cls.__fields__.remove("_id")
        if pk.name not in cls.__fields__:
            cls.__fields__.insert(0, pk.name)

        # Check that the atom member indexes are still valid after
        # reassinging to avoid a bug in the past.
        member_indices = set()
        for name, m in members.items():
            if name == "_id":
                continue  # The _id is an alias
            assert m.index not in member_indices
            member_indices.add(m.index)

        # Set to the sqlalchemy Table
        cls.__table__ = None

        # Will be set to the table model by manager, not done here to avoid
        # import errors that may occur
        cls.__backrefs__ = set()

        # If a Meta class is defined check it's validity and if extending
        # do not inherit the abstract attribute
        Meta = dct.get("Meta", None)
        if Meta is not None:
            for f in dir(Meta):
                if f.startswith("_"):
                    continue
                if f not in VALID_META_FIELDS:
                    raise TypeError(f"{f} is not a valid Meta field on {cls}.")

            db_table = getattr(Meta, "db_table", None)
            if db_table:
                cls.__model__ = db_table

            db_name = getattr(Meta, "db_name", None)
            if db_name:
                cls.__database__ = db_name

        # If this inherited from an abstract model but didn't specify
        # Meta info make the subclass not abstract unless it was redefined
        base_meta = getattr(cls, "Meta", None)
        if base_meta and getattr(base_meta, "abstract", None):
            if not Meta:

                class Meta(base_meta):
                    abstract = False

                cls.Meta = Meta
            elif getattr(Meta, "abstract", None) is None:
                Meta.abstract = False

        # Set the pk name after table is set
        cls.__pk__ = (pk.metadata or {}).get("name", pk.name)
        cls.__joined_pk__ = f"{cls.__model__}_{cls.__pk__}"

        # Create a set of fields to remove from state before saving to the db
        # this removes Relation instances and several needed for json
        excluded_fields = cls.__excluded_fields__ = {
            "__model__",
            "__ref__",
            "__restored__",
        }
        if cls.__pk__ != "_id":
            excluded_fields.add("_id")
        for name, member in cls.members().items():
            if isinstance(member, Relation):
                excluded_fields.add(name)

        # Cache the mapping of any renamed fields
        renamed_fields = cls.__renamed_fields__ = {}
        for old_name, member in cls.members().items():
            if old_name in excluded_fields:
                continue  # Ignore excluded fields
            if member.metadata:
                new_name = member.metadata.get("name")
                if new_name is not None:
                    renamed_fields[old_name] = new_name

        return cls


class SQLModel(Model, metaclass=SQLMeta):
    """A model that can be saved and restored to and from a database supported
    by sqlalchemy.

    """

    #: Primary key field name
    __pk__: ClassVar[str]

    #: Table name and primary key
    __joined_pk__: ClassVar[str]

    #: Models which link back to this
    __backrefs__: ClassVar[SetType[TupleType[Type[Model], Member]]]

    #: List of fields which have been tagged with a different column name
    #: Mapping is class attr -> database column name.
    __renamed_fields__: ClassVar[DictType[str, str]]

    #: Set of fields to exclude from the database
    __excluded_fields__: ClassVar[SetType[str]]

    #: Reference to the sqlalchemy table backing this model
    __table__: ClassVar[Optional[sa.Table]]

    #: Database name. If the `database` field of the manager is a dict
    #: This field will be used to determine which engine to use.
    __database__: ClassVar[str] = "default"

    #: Use SQL serializer
    serializer = SQLModelSerializer.instance()

    #: Use SQL object manager
    objects = SQLModelManager.instance()

    #: ID of this object in the database. Subclasses can redefine this as needed
    _id = Typed(int).tag(primary_key=True)

    @classmethod
    async def restore(
        cls: Type[T],
        state: StateType,
        force: Optional[bool] = None,
        prefetched: Optional[DictType[Any, StateType]] = None,
        **kwargs: Any,
    ) -> T:
        """Restore an object from the database using the primary key. Save
        a ref in the table's object cache.  If force is True, update
        the cache if it exists.

        Parameters
        ----------
        state: Mapping[str, Any]
            A mapping of field name to value. May contain result of a join (eg
            state of multiple models prefexed with the table name).
        force: Optional[bool]
            Whether to force calling restorestate. This is used to to avoid
            restoring cached objects.
        prefetched: Optional[dict]
            A mapping of prefetched related values. If present the objects
            primary key is looked up and added to the state.

        Returns
        -------
        model: SQLModel
            The restored or cached model.

        """
        if cls.__joined_pk__ in state:
            # When sqlalchemy does a join the key will have a prefix
            # of the database name
            pk = state[cls.__joined_pk__]
        else:
            pk = state[cls.__pk__]

        # Note make sure this always occurs to force table creation
        cache = cls.objects.cache
        if pk is not None:
            # Check if this is in the cache
            obj = cache.get(pk)
        else:
            obj = None

        if obj is None:
            # Create and cache it
            obj = cls.__new__(cls)

            # Do not place empty pk in cache
            if pk is not None:
                cache[pk] = obj
            restore = True
        else:
            # Check the default for force reloading
            if force is None:
                force = not cls.objects.table.bind.manager.cache

            # Note that if force is false and the object was restored
            # (ie from another query) the object in the cache is reused
            # and any (potentially new) data in the state is discarded.
            restore = force or not obj.__restored__

        if restore:
            # Merge any prefetched relation members into the restore state
            # so the base class's restore method can find them.
            if prefetched is not None:
                prefetched_state = prefetched.get(pk)
                if prefetched_state is not None:
                    state = dict(state)  # Convert row proxy
                    state.update(prefetched_state)

            # This ideally should only be done if created
            await obj.__restorestate__(state)

        return obj

    async def load(
        self: T,
        connection=None,
        reload: bool = False,
        fields: Optional[Sequence[str]] = None,
    ):
        """Alias to load this object from the database

        Parameters
        ----------
        connection: Connection
            The connection instance to use in a transaction
        reload: Bool
            If True force reloading the state even if the state has
            already been loaded.
        fields: Sequence[str]
            Optional list of field names to load. Use this to refresh
            specific fields from the database.

        """
        skip = self.__restored__ and not reload and not fields
        if skip or not self._id:
            return  # Already loaded or won't do anything
        db = self.objects
        t = db.table
        if fields is not None:
            renamed = self.__renamed_fields__
            columns = (t.c[renamed.get(f, f)] for f in fields)
            q = sa.select(columns).select_from(t)
        else:
            q = t.select()
        q = q.where(t.c[self.__pk__] == self._id)
        state = await db.fetchone(q, connection=connection)
        await self.__restorestate__(state)

    def __prepare_state_for_db__(self):
        """Get the state that should be saved into the database"""
        state = self.__getstate__()

        # Remove any fields are in the state but should not go into the db
        for f in self.__excluded_fields__:
            state.pop(f, None)

        # Replace any renamed fields
        for py_name, db_name in self.__renamed_fields__.items():
            state[db_name] = state.pop(py_name)

        if not self._id:
            # Postgres errors if using None for the pk
            state.pop(self.__pk__, None)

        return state

    async def save(
        self: T,
        force_insert: bool = False,
        force_update: bool = False,
        update_fields: Optional[Sequence[str]] = None,
        connection=None,
    ):
        """Alias to save this object to the database

        Parameters
        ----------
        force_insert: Bool
            Ensure that save performs an insert
        force_update: Bool
            Ensure that save performs an update
        update_fields: Iterable[str]
            If given, only update the given fields
        connection: Connection
            The connection instance to use in a transaction

        Returns
        -------
        result: Value
            Update or save result

        """
        if force_insert and force_update:
            raise ValueError("Cannot use force_insert and force_update together")

        db = self.objects
        table = db.table
        state = self.__prepare_state_for_db__()
        async with db.connection(connection) as conn:
            if force_update or (self._id and not force_insert):
                # If update fields was given, only pass those
                if update_fields is not None:
                    # Replace any update fields with the appropriate name
                    renamed = self.__renamed_fields__
                    update_fields = [renamed.get(f, f) for f in update_fields]

                    # Replace update fields with only those given
                    state = {f: state[f] for f in update_fields}

                q = (
                    table.update()
                    .where(table.c[self.__pk__] == self._id)
                    .values(**state)
                )
                r = await conn.execute(q)
                if not r.rowcount:
                    log.warning(
                        f'Did not update "{self}", either no rows with '
                        f"pk={self._id} exist or it has not changed."
                    )
            else:
                q = table.insert().values(**state)
                r = await conn.execute(q)

                # Don't overwrite if force inserting
                if not self._id:
                    if hasattr(r, "lastrowid"):
                        self._id = r.lastrowid  # MySQL
                    else:
                        self._id = await r.scalar()  # Postgres

                # Save a ref to the object in the model cache
                db.cache[self._id] = self
            self.__restored__ = True
            return r

    async def delete(self: T, connection=None):
        """Alias to delete this object in the database"""
        pk = self._id
        if not pk:
            return
        db = self.objects
        table = db.table  # type: sa.Table
        q = table.delete().where(table.c[self.__pk__] == pk)
        async with db.connection(connection) as conn:
            r = await conn.execute(q)
            if not r.rowcount:
                log.warning(
                    f'Did not delete "{self}", no rows with ' f"pk={self._id} exist."
                )
            del db.cache[pk]
            del self._id
            return r
