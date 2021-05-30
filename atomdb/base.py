"""
Copyright (c) 2018-2020, Jairus Martin.

Distributed under the terms of the MIT License.

The full license is in the file LICENSE.text, distributed with this software.

Created on Jun 12, 2018

@author: jrm
"""
import os
import logging
import traceback
from atom.api import (
    Atom, Property, Instance, Dict, Str, Coerced, Value, Typed, Bytes, Bool,
    set_default
)
from atom.atom import AtomMeta
from collections.abc import MutableMapping
from random import getrandbits
from pprint import pformat
from base64 import b64encode, b64decode
from datetime import date, time, datetime
from decimal import Decimal
from uuid import UUID


logger = logging.getLogger('atomdb')


def find_subclasses(cls):
    """ Finds subclasses of the given class"""
    classes = []
    for subclass in cls.__subclasses__():
        classes.append(subclass)
        classes.extend(find_subclasses(subclass))
    return classes


class ModelSerializer(Atom):
    """ Handles serializing and deserializing of Model subclasses. It
    will automatically save and restore references where present.

    """
    #: Hold one instance per subclass for easy reuse
    _instances = {}

    #: Store all registered models
    registry = Dict()

    #: Mapping of type name to coercer function
    coercers = Dict(default={
        'datetime.date': lambda s: date(**s),
        'datetime.datetime': lambda s: datetime(**s),
        'datetime.time': lambda s: time(**s),
        'bytes': lambda s: b64decode(s['bytes']),
        'decimal': lambda s: Decimal(s['value']),
        'uuid': lambda s: UUID(s['id']),
    })

    @classmethod
    def instance(cls):
        if cls not in ModelSerializer._instances:
            ModelSerializer._instances[cls] = cls()
        return ModelSerializer._instances[cls]

    def flatten(self, v, scope=None):
        """ Convert Model objects to a dict

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
        raise NotImplementedError


    async def unflatten(self, v, scope=None):
        """ Convert dict or list to Models

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
            ref = v.get('__ref__')
            if ref is not None and ref in scope:
                return scope[ref]

            # Create the object
            name = v.get('__model__')
            if name is not None:
                cls = self.registry[name]
                return await cls.serializer.unflatten_object(cls, v, scope)

            # Convert py types
            py_type = v.pop('__py__', None)
            if py_type:
                coercer = self.coercers.get(py_type)
                if coercer:
                    return coercer(v)

            return {k: await unflatten(i, scope) for k, i in v.items()}
        elif isinstance(v, (list, tuple)):
            return [await unflatten(item, scope) for item in v]
        return v

    async def unflatten_object(self, cls, state, scope):
        """ Restore the object for the given class, state, and scope.
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
        _id = state.get('_id')
        ref = state.get('__ref__')

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

    async def get_or_create(self, cls, state, scope):
        """ Get a cached object for this _id or create a new one. Subclasses
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

    async def get_object_state(self, obj,  state, scope):
        """ Lookup the state needed to restore the given object id and class.

        Parameters
        ----------
        obj: Object
            The object created by `get_or_create`
        state: Dict
            Unflattened state of object to restore
        scope: Dict
            Scope of objects available when flattened

        Returns
        -------
        result: Dict
            The model state needed to restore this object

        """
        raise NotImplementedError


class ModelManager(Atom):
    """ A descriptor so you can use this somewhat like Django's models.
    Assuming your using motor.

    Examples
    --------
    MyModel.objects.find_one({'_id':'someid})

    """

    #: Stores instances of each class so we can easily reuse them if desired
    _instances = {}

    @classmethod
    def instance(cls):
        if cls not in ModelManager._instances:
            ModelManager._instances[cls] = cls()
        return ModelManager._instances[cls]

    #: Used to access the database
    database = Value()

    def _default_database(self):
        raise NotImplementedError

    def __get__(self, obj, cls=None):
        """ Handle objects from the class that owns the manager. Subclasses
        should override this as needed.

        """
        raise NotImplementedError


class ModelMeta(AtomMeta):
    def __new__(meta, name, bases, dct):
        cls = AtomMeta.__new__(meta, name, bases, dct)

        # Fields that are saved in the db. By default it uses all atom members
        # that don't start with an underscore and are not taged with store.
        if '__fields__' not in dct:
            cls.__fields__ = tuple((
                m.name for m in cls.members().values()
                if (m.metadata or {}).get('store', not m.name.startswith("_"))
            ))

        # Model name used so the serializer knows what class to recreate
        # when restoring
        if '__model__' not in dct:
            cls.__model__ = f'{cls.__module__}.{cls.__name__}'
        return cls


class Model(Atom, metaclass=ModelMeta):
    """ An atom model that can be serialized and deserialized to and from
    a database.

    """
    __slots__ = '__weakref__'

    #: ID of this object in the database. Subclasses can redefine this as needed
    _id = Bytes()

    #: A unique ID used to handle cyclical serialization and deserialization
    __ref__ = Bytes(factory=lambda: b'%0x' % getrandbits(30 * 4))

    #: Flag to indicate if this model has been restored or saved
    __restored__ = Bool().tag(store=False)

    #: State set when restored from the database. This should be updated
    #: upon successful save and never modified
    #:__state__ = Typed(dict).tag(store=False)

    # ==========================================================================
    # Serialization API
    # ==========================================================================

    #: Handles encoding and decoding. Subclasses should redefine this to a
    #: subclass of ModelSerializer
    serializer = None

    def __getstate__(self, scope=None):
        default_flatten = self.serializer.flatten

        scope = scope or {}

        # ID for circular references
        ref = self.__ref__
        scope[ref] = self

        state = {
            '__model__': self.__model__,
            '__ref__': ref,
        }
        if self._id is not None:
            state['_id'] = self._id

        members = self.members()
        for f in self.__fields__:
            m = members[f]
            meta = m.metadata or {}
            flatten = meta.get('flatten', default_flatten)
            state[f] = flatten(getattr(self, f), scope)

        return state

    async def __restorestate__(self, state, scope=None):
        """ Restore an object from the a state from the database. This is
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
        name = state.get('__model__', self.__model__)
        if name != self.__model__:
            raise ValueError(f"Trying to use {name} state for "
                             f"{self.__model__} object")
        scope = scope or {}
        ref = state.get('__ref__')
        if ref is not None:
            scope[ref] = self
        members = self.members()

        # Order the keys by the members 'setstate_order' if given
        valid_keys = []
        default_unflatten = self.serializer.unflatten
        for k in state.keys():
            m = members.get(k)
            if m is not None:
                meta = m.metadata or {}
                order = meta.get('setstate_order', 1000)

                # Allow  tagging a custom unflatten fn
                unflatten = meta.get('unflatten', default_unflatten)

                valid_keys.append((order, k, unflatten))
        valid_keys.sort(key=lambda it: it[0])

        # Save initial database state
        # self.__state__ = dict(state)

        for order, k, unflatten in valid_keys:
            try:
                v = state[k]
                obj = await unflatten(v, scope)
                setattr(self, k, obj)
            except Exception as e:
                exc = traceback.format_exc()
                logger.error(
                    f"Error loading state:"
                    f"{self.__model__}.{k} = {pformat(obj)}:"
                    f"\nSelf: {ref}: {scope.get(ref)}"
                    f"\nValue: {pformat(v)}"
                    f"\nScope: {pformat(scope)}"
                    f"\nState: {pformat(state)}"
                    f"\n{exc}")

        # Update restored state
        self.__restored__ = True

    # ==========================================================================
    # Database API
    # ==========================================================================

    #: Handles database access. Subclasses should redefine this.
    objects = None

    @classmethod
    async def restore(cls, state):
        """ Restore an object from the database state """
        obj = cls.__new__(cls)
        await obj.__restorestate__(state)
        return obj

    async def load(self):
        """ Alias to load this object from the database """
        raise NotImplementedError

    async def save(self):
        """ Alias to delete this object to the database """
        raise NotImplementedError

    async def delete(self):
        """ Alias to delete this object in the database """
        raise NotImplementedError


class JSONSerializer(ModelSerializer):

    def flatten(self, v, scope=None):
        """ Flatten date, datetime, time, decimal, and bytes as a dict with
        a __py__ field and arguments to reconstruct it. Also see the coercers

        """
        if isinstance(v, (date, datetime, time)):
            # This is inefficient space wise but still allows queries
            s = {'__py__': f'{v.__class__.__module__}.{v.__class__.__name__}'}
            if isinstance(v, (date, datetime)):
                s.update({
                    'year': v.year,
                    'month': v.month,
                    'day': v.day
                })
            if isinstance(v, (time, datetime)):
                s.update({
                    'hour': v.hour,
                    'minute': v.minute,
                    'second': v.second,
                    'microsecond': v.microsecond,
                    # TODO: Timezones
                })
            return s
        if isinstance(v, bytes):
            return {'__py__': 'bytes', 'bytes': b64encode(v).decode()}
        if isinstance(v, Decimal):
            return {'__py__': 'decimal', 'value': str(v)}
        if isinstance(v, UUID):
            return {'__py__': 'uuid', 'id': str(v)}
        return super().flatten(v, scope)

    def flatten_object(self, obj, scope):
        """ Flatten to just json but add in keys to know how to restore it.

        """
        ref = obj.__ref__
        if ref in scope:
            return {'__ref__': ref, '__model__': obj.__model__}
        else:
            scope[ref] = obj
        state = obj.__getstate__(scope)
        _id = state.get("_id")
        return {'_id': _id,
                '__ref__': ref,
                '__model__': state['__model__']} if _id else state

    async def get_object_state(self, obj,  state, scope):
        """ State should be contained in the dict

        """
        return state

    def _default_registry(self):
        return {m.__model__: m for m in find_subclasses(JSONModel)}


class JSONModel(Model):
    """ A simple model that can be serialized to json. Useful for embedding
    within other objects.

    """
    serializer = JSONSerializer.instance()

    #: JSON cannot encode bytes
    _id = Str()
    __ref__ = Str(factory=lambda: (b'%0x' % getrandbits(30 * 4)).decode())
    __restored__ = set_default(True)
