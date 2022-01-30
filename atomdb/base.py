"""
Copyright (c) 2018-2021, Jairus Martin.

Distributed under the terms of the MIT License.

The full license is in the file LICENSE.text, distributed with this software.

Created on Jun 12, 2018

@author: jrm
"""
import os
import logging
import traceback
from typing import Dict as DictType
from typing import List as ListType
from typing import Tuple as TupleType
from typing import Any, ClassVar, Generic, Type, TypeVar, Optional, Union, Callable
from collections.abc import MutableMapping
from random import getrandbits
from pprint import pformat
from base64 import b64encode, b64decode
from datetime import date, time, datetime
from decimal import Decimal
from uuid import UUID
from atom.api import (
    Atom,
    AtomMeta,
    Member,
    Property,
    Instance,
    Dict,
    Str,
    Coerced,
    Value,
    Typed,
    Bytes,
    Bool,
    Int,
    Float,
    set_default,
)

T = TypeVar("T")
M = TypeVar("M", bound="Model")
ScopeType = DictType[Union[str, bytes], Any]
StateType = DictType[str, Any]
logger = logging.getLogger("atomdb")
GetStateFn = Callable[[M, Optional[ScopeType]], StateType]
RestoreStateFn = Callable[[M, StateType, Optional[ScopeType]], None]


def find_subclasses(cls: Type[T]) -> ListType[Type[T]]:
    """Finds subclasses of the given class"""
    classes = []
    for subclass in cls.__subclasses__():
        classes.append(subclass)
        classes.extend(find_subclasses(subclass))
    return classes


def is_db_field(m: Member) -> bool:
    """Check if the member should be saved into the database.  Any member that
    does not start with an underscore and is not tagged with `store=False`
    is considered to be field to save into the database.

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
    return default


def is_primitive_member(m: Member) -> bool:
    """Check if the member can be serialized without calling flatten.

    Parameters
    ----------
    m: Member
        The atom member to check.

    Returns
    -------
    result: bool
        Whether the member is a primiative type that can be intrinsicly
        converted.

    """
    if isinstance(m, (Bool, Str, Int, Float)):
        return True
    # TODO: Handle more such as List(str), etc..
    return False


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
            "datetime.date": lambda s: date(**s),
            "datetime.datetime": lambda s: datetime(**s),
            "datetime.time": lambda s: time(**s),
            "bytes": lambda s: b64decode(s["bytes"]),
            "decimal": lambda s: Decimal(s["value"]),
            "uuid": lambda s: UUID(s["id"]),
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
        unflatten = self.unflatten
        scope = scope or {}
        if isinstance(v, dict):
            # Circular reference
            ref = v.get("__ref__")
            if ref is not None and ref in scope:
                return scope[ref]

            # Create the object
            name = v.get("__model__")
            if name is not None:
                cls = self.registry[name]
                return await cls.serializer.unflatten_object(cls, v, scope)

            # Convert py types
            py_type = v.pop("__py__", None)
            if py_type:
                coercer = self.coercers.get(py_type)
                if coercer:
                    return coercer(v)

            return {k: await unflatten(i, scope) for k, i in v.items()}
        elif isinstance(v, (list, tuple)):
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
        ref = state.get("__ref__")

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
        if ref is not None:
            scope[ref] = obj

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


def generate_getstate(cls: Type["Model"], include_defaults: bool = True) -> GetStateFn:
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
    if include_defaults:
        template.append('    "__model__": self.__model__,')
        template.append('    "__ref__": self.__ref__,')

    default_flatten = cls.serializer.flatten
    members = cls.members()
    namespace = {}
    for f in cls.__fields__:
        # Since f is potentially an untrusted input, make sure it is a valid
        # python identifier to prevent unintended code being generated.
        if not f.isidentifier():
            raise ValueError(f"Field '{f}' cannot be used for code generation")
        m = members[f]
        meta = m.metadata or {}
        flatten = meta.get("flatten", default_flatten)
        if flatten is default_flatten and is_primitive_member(m):
            template.append(f'    "{f}": self.{f},')
        else:
            namespace[f"flatten_{f}"] = flatten
            template.append(
                f'    "{f}": flatten_{f}(self.{f}, scope),',
            )

    template.append("}")

    if include_defaults:
        template.append('if self._id is not None:')
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
    cls.__model__
    on_error = cls.__on_error__
    template = [
        "async def __restorestate__(self, state, scope=None):",
        "if '__model__' in state and state['__model__'] != self.__model__:",
        "    name = state['__model__']",
        "    raise ValueError(",
        "        f'Trying to use {name} state for {self.__model__} object'",
        "    )",
        "scope = scope or {}",
        # Python must do some caching because this seems to be faster than using get
        "if '__ref__' in state and state['__ref__'] is not None:",
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

        setters.append((order, f, unflatten))
    setters.sort(key=lambda it: it[0])

    on_error = cls.__on_error__

    namespace = {
        "default_unflatten": default_unflatten,
    }
    for order, f, unflatten in setters:
        # Since f is potentially an untrusted input, make sure it is a valid
        # python identifier to prevent unintended code being generated.
        if not f.isidentifier():
            raise ValueError(f"Field '{f}' cannot be used for code generation")
        m = members[f]
        template.append(f"if '{f}' in state:")
        if unflatten is default_unflatten:
            if is_primitive_member(m):
                # Direct assignment
                expr = f"self.{f} = state['{f}']"
            else:
                # Default flatten
                expr = f"self.{f} = await default_unflatten(state['{f}'], scope)"
        else:
            namespace[f"unflatten_{f}"] = unflatten
            expr = f"self.{f} = await unflatten_{f}(state['{f}'], scope)"

        if on_error == "raise":
            template.append(f"    {expr}")
        else:
            if on_error == "log":
                handler = f"self.__log_restore_error__(e, '{f}', state, scope)"
            else:
                handler = "pass"
            template.extend(
                [
                    f"    try:",
                    f"        {expr}",
                    f"    except Exception as e:",
                    f"        {handler}",
                ]
            )

    # Update restored state
    template.append("self.__restored__ = True")
    source = "\n    ".join(template)
    return generate_function(source, namespace, "__restorestate__")


def generate_function(source: str, namespace: DictType[str, Any], fn_name: str):
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

    # TODO: Use byteplay to rewrite globals to load fast

    result: DictType[str, Any] = {}
    exec(code, namespace, result)
    return result[fn_name]  # type: ignore


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
    __on_error__: ClassVar[str] = "log"  # "drop" or "raise"

    # --------------------------------------------------------------------------
    # Internal model members
    # --------------------------------------------------------------------------

    #: ID of this object in the database. Subclasses can redefine this as needed
    _id = Bytes()  # type: Any

    #: A unique ID used to handle cyclical serialization and deserialization
    __ref__ = Bytes(factory=lambda: b"%0x" % getrandbits(30 * 4))  # type: Any

    #: Flag to indicate if this model has been restored or saved
    __restored__ = Bool().tag(store=False)

    #: State set when restored from the database. This should be updated
    #: upon successful save and never modified
    #:__state__ = Typed(dict).tag(store=False)

    # --------------------------------------------------------------------------
    # Serialization API
    # --------------------------------------------------------------------------

    #: Handles encoding and decoding. Subclasses should redefine this to a
    #: subclass of ModelSerializer

    serializer: ModelSerializer = ModelSerializer.instance()

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
        self, e: Exception, k: str, state: StateType, scope: ScopeType
    ):
        """Log details when restoring a member fails. This typically only will
        occur if the state has data from an old model after a schema change.

        """
        obj = state.get(k)
        logger.debug(
            f"Error loading state:"
            f"{self.__model__}.{k} = {pformat(obj)}:"
            f"\nSelf: {self.__ref__}: {scope.get(self.__ref__)}"
            f"\nScope: {pformat(scope)}"
            f"\nState: {pformat(state)}"
            f"\n{e}"
        )

    # --------------------------------------------------------------------------
    # Database API
    # --------------------------------------------------------------------------

    #: Handles database access. Subclasses should redefine this.
    objects: ModelManager = ModelManager()

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
        return super().flatten(v, scope)

    def flatten_object(self, obj: Model, scope: ScopeType) -> DictType[str, Any]:
        """Flatten to just json but add in keys to know how to restore it."""
        ref = obj.__ref__
        if ref in scope:
            return {"__ref__": ref, "__model__": obj.__model__}
        else:
            scope[ref] = obj
        state = obj.__getstate__(scope)
        _id = state.get("_id")
        if _id:
            return {"_id": _id, "__ref__": ref, "__model__": state["__model__"]}
        return state

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

    #: JSON cannot encode bytes
    _id = Str()
    __ref__ = Str(factory=lambda: (b"%0x" % getrandbits(30 * 4)).decode())
    __restored__ = set_default(True)  # type: ignore
