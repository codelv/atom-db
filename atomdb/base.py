"""
Copyright (c) 2018-2022, Jairus Martin.

Distributed under the terms of the MIT License.

The full license is in the file LICENSE.text, distributed with this software.

Created on Jun 12, 2018
"""

import asyncio
import enum
import logging
from base64 import b64decode, b64encode
from collections.abc import MutableMapping
from datetime import date, datetime, time
from decimal import Decimal
from pprint import pformat
from random import getrandbits
from typing import Any, Callable, ClassVar
from typing import Dict as DictType
from typing import List as ListType
from typing import Optional
from typing import Tuple as TupleType
from typing import Type, TypeVar
from uuid import UUID

from atom.api import (
    Atom,
    AtomMeta,
    Bool,
    Coerced,
    Dict,
    Float,
    Instance,
    Int,
    List,
    Member,
    Property,
    Str,
    Typed,
    Value,
    set_default,
)

T = TypeVar("T")
M = TypeVar("M", bound="Model")
ScopeType = DictType[int, Any]
StateType = DictType[str, Any]
GetStateFn = Callable[[M, Optional[ScopeType]], StateType]
RestoreStateFn = Callable[[M, StateType, Optional[ScopeType]], None]
log = logging.getLogger("atomdb")


def find_subclasses(cls: Type[T]) -> ListType[Type[T]]:
    """Finds subclasses of the given class"""
    classes = []
    for subclass in cls.__subclasses__():
        classes.append(subclass)
        classes.extend(find_subclasses(subclass))
    return classes


def is_db_field(m: Member) -> bool:
    """Check if the member should be saved into the database.  Any member that
    does not start with an underscore, is not a Property, and is not tagged
    with `store=False` is considered to be field to save into the database.

    Parameters
    ----------
    m: Member
        The atom member to check.

    Returns
    -------
    result: bool
        Whether the member should be saved into the database.

    """
    metadata = m.metadata
    default = not m.name.startswith("_")
    if metadata is not None:
        return metadata.get("store", default)
    if isinstance(m, Property):
        return False  # Users can override this by tagging it with store=True
    return default


def is_primitive_member(m: Member) -> Optional[bool]:
    """Check if the member can be serialized without calling flatten. If the
    member references a field that is not yet resolved it returns None
    indicating that it cannot determine whether it is primitive yet.

    Parameters
    ----------
    m: Member
        The atom member to check.

    Returns
    -------
    result: Optional[bool]
        Whether the member is a primitive type that can be intrinsicly
        converted.

    """
    if isinstance(m, (Bool, Str, Int, Float)):
        return True
    if hasattr(m, "resolve"):
        # These cannot be resolved until their dependencies are available
        return None
    if isinstance(m, (List, Typed, Instance, Dict, Coerced)):
        try:
            types = resolve_member_types(m, resolve=False)
        except UnresolvableError:
            return None
        if types is None:
            return False  # Value can be any type
        if types and all(t in (int, float, bool, str) for t in types):
            return True
    return False


def resolve_member_types(
    member: Member, resolve: bool = True
) -> Optional[TupleType[type, ...]]:
    """Determine the validation types specified on a member.

    Parameters
    ----------
    member: Member
        The member to retrieve the type from
    resolve: bool
        Whether to resolve "Forward" members.
    Returns
    -------
    types: Optional[Tuple[Model|Member|type, ..]]
        The member types. If types is `None` then the member does not do any
        type validation.

    Raises
    ------
    UnresolveableError
        If `resolve=False` and the member has a nested forwarded member this
        will raise an UnresolvableError with the unresolved member.

    """
    # TODO: This should really use the validate mode...
    if hasattr(member, "resolve"):
        if not resolve:
            raise UnresolvableError(member)  # Do not resolve now
        types = member.resolve()  # type: ignore
    elif isinstance(member, Coerced):
        types = member.validate_mode[-1][0]
    else:
        types = member.validate_mode[-1]
    if types is None:
        return None
    if isinstance(types, tuple):
        # Dict may have an member in the types list, so walk the types
        # and resolve all of those.
        resolved: ListType[type] = []
        for t in types:
            if isinstance(t, Member):
                r = resolve_member_types(t, resolve)
                if r is None:
                    # TODO: Think about whether this is correct to bail out here
                    return None
                resolved.extend(r)
            else:
                resolved.append(t)
        return tuple(resolved)
    if isinstance(types, Member):
        # Follow the chain. For example if the member is defined
        # as `List(Tuple(float)))` lookup the types of the nested Tuple().
        return resolve_member_types(types, resolve)
    if isinstance(types, str):
        return None  # Custom validation method
    return (types,)


class UnresolvableError(Exception):
    """Error raised when a Forwarded Member cannot be resolved at the time
    when the resolve_member_types is called.

    """

    def __init__(self, member):
        self.member = member
        super().__init__(f"Cannot resolve {member}")


class ModelSerializer(Atom):
    """Handles serializing and deserializing of Model subclasses. It
    will automatically save and restore references where present.

    """

    #: Hold one instance per subclass for easy reuse
    _instances: ClassVar[DictType[Type["ModelSerializer"], "ModelSerializer"]] = {}

    #: Store all registered models
    registry = Dict()

    #: Mapping of type name to coercer function
    coercers = Dict(
        default={
            "datetime.date": lambda v, scope: date(**v),
            "datetime.datetime": lambda v, scope: datetime(**v),
            "datetime.time": lambda v, scope: time(**v),
            "bytes": lambda v, scope: b64decode(v["bytes"]),
            "decimal": lambda v, scope: Decimal(v["value"]),
            "uuid": lambda v, scope: UUID(v["id"]),
        }
    )

    @classmethod
    def instance(cls: Type["ModelSerializer"]) -> "ModelSerializer":
        if cls not in ModelSerializer._instances:
            ModelSerializer._instances[cls] = cls()
        return ModelSerializer._instances[cls]

    def flatten(self, v: Any, scope: Optional[ScopeType] = None) -> Any:
        """Convert Model objects to a dict

        Parameters
        ----------
        v: Object
            The object to flatten
        scope: Dict
            The scope of references available for circular lookups

        Returns
        -------
        result: Object
            The flattened object

        """
        flatten = self.flatten
        scope = scope or {}

        # Handle circular reference
        if isinstance(v, Model):
            return v.serializer.flatten_object(v, scope)
        elif isinstance(v, (list, tuple, set)):
            return [flatten(item, scope) for item in v]
        elif isinstance(v, (dict, MutableMapping)):
            return {k: flatten(item, scope) for k, item in v.items()}
        elif isinstance(v, enum.Enum):
            return v.value
        # TODO: Handle other object types
        return v

    def flatten_object(self, obj: "Model", scope: ScopeType) -> Any:
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
        raise NotImplementedError

    async def unflatten(self, v: Any, scope: Optional[ScopeType] = None) -> Any:
        """Convert dict or list to Models

        Parameters
        ----------
        v: Dict or List
            The object(s) to unflatten
        scope: Dict
            The scope of references available for circular lookups

        Returns
        -------
        result: Object
            The unflattened object

        """
        if isinstance(v, dict):
            unflatten = self.unflatten
            # Circular reference
            if scope and "__ref__" in v:
                ref = v["__ref__"]
                if ref in scope:
                    return scope[ref]

            # Create the object
            if "__model__" in v:
                cls = self.registry[v["__model__"]]
                return await cls.serializer.unflatten_object(cls, v, scope)

            # Convert py types
            if "__py__" in v:
                py_type = v.pop("__py__")
                coercer = self.coercers.get(py_type)
                if coercer:
                    if asyncio.iscoroutinefunction(coercer):
                        return await coercer(v, scope)
                    return coercer(v, scope)
                elif py_type == "set" or py_type == "atomset":
                    return {await unflatten(i) for i in v["values"]}
                elif py_type == "tuple":
                    return tuple([await unflatten(i) for i in v["values"]])
            return {k: await unflatten(i, scope) for k, i in v.items()}
        elif isinstance(v, list):
            unflatten = self.unflatten
            return [await unflatten(item, scope) for item in v]
        return v

    async def unflatten_object(
        self, cls: Type["Model"], state: StateType, scope: ScopeType
    ) -> Optional["Model"]:
        """Restore the object for the given class, state, and scope.
        If a reference is given the scope should be updated with the newly
        created object using the given ref.

        Parameters
        ----------
        cls: Class
            The type of object expected
        state: Dict
            The state of the object to restore

        Returns
        -------
        result: object or None
            A the newly created object (or an existing object if using a cache)
            or None if this object does not exist in the database.
        """
        _id = state.get("_id")

        # Get the object for this id, retrieve from cache if needed
        obj, created = await self.get_or_create(cls, state, scope)

        # Lookup the object if needed
        if created and _id is not None:
            # If a new object was created lookup the state for that object
            state = await self.get_object_state(obj, state, scope)
            if state is None:
                return None

        # Child objects may have circular references to this object
        # so we must update the scope with this reference to handle this
        # before restoring any children
        if scope and "__ref__" in state:
            scope[state["__ref__"]] = obj

        # If not restoring from cache update the state
        if created:
            await obj.__restorestate__(state, scope)
        return obj

    async def get_or_create(
        self, cls: Type["Model"], state: Any, scope: ScopeType
    ) -> TupleType["Model", bool]:
        """Get a cached object for this _id or create a new one. Subclasses
        should override this as needed to provide object caching if desired.

        Parameters
        ----------
        cls: Class
            The type of object expected
        state: Dict
            Unflattened state of object to restore
        scope: Dict
            Scope of objects available when flattened

        Returns
        -------
        result: Tuple[object, bool]
            A tuple of the object and a flag stating if it was created or not.

        """
        return (cls.__new__(cls), True)

    async def get_object_state(self, obj: "Model", state: Any, scope: ScopeType) -> Any:
        """Lookup the state needed to restore the given object id and class.

        Parameters
        ----------
        obj: Model
            The object created by `get_or_create`
        state: Dict
            Unflattened state of object to restore
        scope: Dict
            Scope of objects available when flattened

        Returns
        -------
        result: Any
            The model state needed to restore this object

        """
        raise NotImplementedError


class ModelManager(Atom):
    """A descriptor so you can use this somewhat like Django's models.
    Assuming your using motor.

    Examples
    --------
    MyModel.objects.find_one({'_id':'someid})

    """

    #: Stores instances of each class so we can easily reuse them if desired
    _instances: ClassVar[DictType[Type["ModelManager"], "ModelManager"]] = {}

    @classmethod
    def instance(cls) -> "ModelManager":
        if cls not in ModelManager._instances:
            ModelManager._instances[cls] = cls()
        return ModelManager._instances[cls]

    #: Used to access the database
    database = Value()

    def _default_database(self) -> Any:
        raise NotImplementedError

    def __get__(self, obj: T, cls: Optional[Type[T]] = None):
        """Handle objects from the class that oType[wns the manager. Subclasses
        should override this as needed.

        """
        raise NotImplementedError


def generate_getstate(cls: Type["Model"]) -> GetStateFn:
    """Generate an optimized __getstate__ function for the given model.

    Parameters
    ----------
    cls: Type[Model]
        The clase to generate a getstate function for.

    Returns
    -------
    result: GetStateFn
        A function optimized to generate the state for the given model class.

    """
    template = [
        "def __getstate__(self, scope=None):",
        "scope = scope or {}",
        "scope[self.__ref__] = self",
        "state = {",
    ]
    default_flatten = cls.serializer.flatten
    members = cls.members()
    namespace = {
        "default_flatten": default_flatten,
    }
    for f in cls.__fields__:
        # Since f is potentially an untrusted input, make sure it is a valid
        # python identifier to prevent unintended code being generated.
        if not f.isidentifier():
            raise ValueError(f"Field '{f}' cannot be used for code generation")
        m = members[f]
        meta = m.metadata or {}
        flatten = meta.get("flatten", default_flatten)
        if flatten is default_flatten:
            if is_primitive_member(m):
                expr = f"self.{f}"
            else:
                expr = f"default_flatten(self.{f}, scope)"
        else:
            namespace[f"flatten_{f}"] = flatten
            expr = f"flatten_{f}(self.{f}, scope)"
        template.append(f'    "{f}": {expr},')

    template.append('    "__model__": self.__model__,')
    template.append('    "__ref__": self.__ref__,')
    template.append("}")
    if "_id" in members:
        template.append("if self._id:")
        template.append('    state["_id"] = self._id')
    template.append("return state")
    source = "\n    ".join(template)
    return generate_function(source, namespace, "__getstate__")


def generate_restorestate(cls: Type["Model"]) -> RestoreStateFn:
    """Generate an optimized __restorestate__ function for the given model.

    Parameters
    ----------
    cls: Type[Model]
        The clase to generate a getstate function for.

    Returns
    -------
    result: RestoreStateFn
        A function optimized to restore the state for the given model class.

    """
    # Python must do some caching because using key in state and state[key]
    # seems to be faster than using get
    template = [
        "async def __restorestate__(self, state, scope=None):",
        "if '__model__' in state and state['__model__'] != self.__model__:",
        "    name = state['__model__']",
        "    raise ValueError(",
        "        f'Trying to use {name} state for {self.__model__} object'",
        "    )",
        "if '__ref__' in state and state['__ref__'] is not None:",
        "    scope = scope or {}",
        "    scope[state['__ref__']] = self",
    ]

    default_unflatten = cls.serializer.unflatten
    members = cls.members()
    excluded = (
        "__ref__",
        "__restored__",
    )
    setters = []
    for f, m in members.items():
        if f in excluded:
            continue
        meta = m.metadata or {}
        order = meta.get("setstate_order", 1000)

        # Allow  tagging a custom unflatten fn
        unflatten = meta.get("unflatten", default_unflatten)

        setters.append((order, f, m, unflatten))
    setters.sort(key=lambda it: it[0])

    on_error = cls.__on_error__

    namespace: DictType[str, Any] = {
        "default_unflatten": default_unflatten,
    }
    for order, f, m, unflatten in setters:
        # Since f is potentially an untrusted input, make sure it is a valid
        # python identifier to prevent unintended code being generated.
        if not f.isidentifier():
            raise ValueError(f"Field '{f}' cannot be used for code generation")

        template.append(f"if '{f}' in state:")

        # Determine the expresion to unflatten the value
        if unflatten is default_unflatten:
            RelModel = None
            # If the member is typed we can shortcut looking up the __model__
            # type from the state and restore it directly.
            # Note that this does not work for instances.
            if isinstance(m, Typed):
                types = resolve_member_types(m, resolve=False)
                if types and len(types) == 1 and issubclass(types[0], Model):
                    RelModel = types[0]
            if RelModel is not None:
                namespace[f"rel_model_{f}"] = RelModel
                expr = f"await rel_model_{f}.restore(state['{f}'])"
            elif is_primitive_member(m):
                # Direct assignment
                expr = f"state['{f}']"
            else:
                # Default flatten
                expr = f"await default_unflatten(state['{f}'], scope)"
        else:
            namespace[f"unflatten_{f}"] = unflatten
            if asyncio.iscoroutinefunction(unflatten):
                expr = f"await unflatten_{f}(state['{f}'], scope)"
            else:
                expr = f"unflatten_{f}(state['{f}'], scope)"

        # Do the assignment
        if on_error == "raise":
            template.append(f"    self.{f} = {expr}")
        else:
            if on_error == "log":
                handler = f"self.__log_restore_error__(e, '{f}', state, scope)"
            else:
                handler = "pass"
            template.extend(
                [
                    "    try:",
                    f"        self.{f} = {expr}",
                    "    except Exception as e:",
                    f"        {handler}",
                ]
            )

    # Update restored state
    template.append("self.__restored__ = True")
    source = "\n    ".join(template)
    return generate_function(source, namespace, "__restorestate__")


def generate_function(
    source: str,
    namespace: DictType[str, Any],
    fn_name: str,
) -> Callable[..., Any]:
    """Generate an optimized function

    Parameters
    ----------
    source: str
        The function source code
    namespaced: dict
        Namespace available to the function
    fn_name: str
        The name of the generated function.

    Returns
    -------
    fn: function
        The function generated.

    """
    # print(source)
    try:
        assert source.startswith(f"def {fn_name}") or source.startswith(
            f"async def {fn_name}"
        )
        code = compile(source, __name__, "exec", optimize=1)
    except Exception as e:
        raise RuntimeError(f"Could not generate code: {e}:\n{source}")

    result: DictType[str, Any] = {}
    exec(code, namespace, result)

    # Optimize global access
    fn = result[fn_name]
    return fn


class ModelMeta(AtomMeta):
    def __new__(meta, name, bases, dct):
        cls = AtomMeta.__new__(meta, name, bases, dct)

        # Fields that are saved in the db. By default it uses all atom members
        # that don't start with an underscore and are not taged with store.
        if "__fields__" not in dct:
            cls.__fields__ = [
                name for name, m in cls.members().items() if is_db_field(m)
            ]

        # Model name used so the serializer knows what class to recreate
        # when restoring
        if "__model__" not in dct:
            cls.__model__ = f"{cls.__module__}.{cls.__name__}"

        # Generate optimized get and restore functions
        # Some general testing indicates this improves getstate by about 2x
        # and restorestate by about 20% but it depends on the model.
        if "__generated_getstate__" not in dct:
            cls.__generated_getstate__ = generate_getstate(cls)

        if "__generated_restorestate__" not in dct:
            cls.__generated_restorestate__ = generate_restorestate(cls)

        return cls


class Model(Atom, metaclass=ModelMeta):
    """An atom model that can be serialized and deserialized to and from
    a database.

    """

    # --------------------------------------------------------------------------
    # Class attributes
    # --------------------------------------------------------------------------
    __slots__ = "__weakref__"

    #: List of database field member names
    __fields__: ClassVar[ListType[str]]

    #: Table name used when saving into the database
    __model__: ClassVar[str]

    #: Error handling
    __on_error__: ClassVar[str] = "log"  # "ignore" or "raise"

    # --------------------------------------------------------------------------
    # Internal model members
    # --------------------------------------------------------------------------

    #: A unique ID used to handle cyclical serialization and deserialization
    __ref__ = Int(factory=lambda: getrandbits(32))

    #: Flag to indicate if this model has been restored or saved
    __restored__ = Bool().tag(store=False)

    # --------------------------------------------------------------------------
    # Serialization API
    # --------------------------------------------------------------------------

    #: Handles encoding and decoding. Subclasses should redefine this to a
    #: subclass of ModelSerializer
    serializer: ClassVar[ModelSerializer] = ModelSerializer.instance()

    #: Optimized serialize functions. These are generated by the metaclass.
    __generated_getstate__: ClassVar[GetStateFn]
    __generated_restorestate__: ClassVar[RestoreStateFn]

    def __getstate__(self, scope: Optional[ScopeType] = None) -> StateType:
        """Get the serialized model state. By default this delegates to an
        optimized function generated by the ModelMeta class.

        Parameters
        ----------
        scope: Optionl[ScopeType
            The scope to lookup circular references.

        Returns
        -------
        state: StateType
            The state of the object.

        """
        return self.__generated_getstate__(scope)

    async def __restorestate__(
        self, state: StateType, scope: Optional[ScopeType] = None
    ):
        """Restore an object from the a state from the database. This is
        async as it will lookup any referenced objects from the DB.

        State is restored by calling setattr(k, v) for every item in the state
        that has an associated atom member.  Members can be tagged with a
        `setstate_order=<number>` to define the order of setattr calls. Errors
        from setattr are caught and logged instead of raised.

        Parameters
        ----------
        state: Dict
            A dictionary of state keys and values
        scope: Dict or None
            A namespace to use to resolve any possible circular references.
            The __ref__ value is used as the keys.

        """
        await self.__generated_restorestate__(state, scope)  # type: ignore

    def __log_restore_error__(
        self, e: Exception, k: str, state: StateType, scope: Optional[ScopeType]
    ):
        """Log details when restoring a member fails. This typically only will
        occur if the state has data from an old model after a schema change.

        """
        obj = state.get(k)
        log.warning(
            f"Error loading state:"
            f"{self.__model__}.{k} = {pformat(obj)}:"
            f"\nRef: {self.__ref__}"
            f"\nScope: {pformat(scope)}"
            f"\nState: {pformat(state)}"
            f"\n{e}"
        )

    # --------------------------------------------------------------------------
    # Database API
    # --------------------------------------------------------------------------

    #: Handles database access. Subclasses should redefine this.
    objects: ClassVar[ModelManager] = ModelManager()

    @classmethod
    async def restore(cls: Type[M], state: StateType, **kwargs: Any) -> M:
        """Restore an object from the database state"""
        obj = cls.__new__(cls)
        await obj.__restorestate__(state)
        return obj

    async def load(self):
        """Alias to load this object from the database"""
        raise NotImplementedError

    async def save(self):
        """Alias to delete this object to the database"""
        raise NotImplementedError

    async def delete(self):
        """Alias to delete this object in the database"""
        raise NotImplementedError


class JSONSerializer(ModelSerializer):
    def flatten(self, v: Any, scope: Optional[ScopeType] = None):
        """Flatten date, datetime, time, decimal, and bytes as a dict with
        a __py__ field and arguments to reconstruct it. Also see the coercers

        """
        if isinstance(v, (date, datetime, time)):
            # This is inefficient space wise but still allows queries
            s: DictType[str, Any] = {
                "__py__": f"{v.__class__.__module__}.{v.__class__.__name__}"
            }
            if isinstance(v, (date, datetime)):
                s.update({"year": v.year, "month": v.month, "day": v.day})
            if isinstance(v, (time, datetime)):
                s.update(
                    {
                        "hour": v.hour,
                        "minute": v.minute,
                        "second": v.second,
                        "microsecond": v.microsecond,
                        # TODO: Timezones
                    }
                )
            return s
        if isinstance(v, bytes):
            return {"__py__": "bytes", "bytes": b64encode(v).decode()}
        if isinstance(v, Decimal):
            return {"__py__": "decimal", "value": str(v)}
        if isinstance(v, UUID):
            return {"__py__": "uuid", "id": str(v)}
        if isinstance(v, (tuple, set)):
            flatten = self.flatten
            type_name = v.__class__.__name__
            return {"__py__": type_name, "values": [flatten(it) for it in v]}
        return super().flatten(v, scope)

    def flatten_object(self, obj: Model, scope: ScopeType) -> DictType[str, Any]:
        """Flatten to just json but add in keys to know how to restore it."""
        ref = obj.__ref__
        if ref in scope:
            return {"__ref__": ref, "__model__": obj.__model__}
        else:
            scope[ref] = obj
        return obj.__getstate__(scope)

    async def get_object_state(self, obj: Any, state: StateType, scope: ScopeType):
        """State should be contained in the dict"""
        return state

    def _default_registry(self) -> DictType[str, Type[Model]]:
        return {m.__model__: m for m in find_subclasses(JSONModel)}


class JSONModel(Model):
    """A simple model that can be serialized to json. Useful for embedding
    within other models.

    """

    serializer = JSONSerializer.instance()
    __restored__ = set_default(True)  # type: ignore
