"""
Copyright (c) 2018-2020, Jairus Martin.

Distributed under the terms of the MIT License.

The full license is in the file LICENSE.text, distributed with this software.

Created on Jun 12, 2018

@author: jrm
"""
import bson
import weakref
from atom.api import Atom, Instance, Value, Dict, Typed
from .base import (
    ModelManager, ModelSerializer, Model, find_subclasses, JSONSerializer
)


class NoSQLModelSerializer(ModelSerializer):
    """ Handles serializing and deserializing of Model subclasses. It
    will automatically save and restore references where present.

    """
    async def get_or_create(self, cls, state, scope):
        """ Restore an object from the database. If the object is cached,
        use that instead.
        """
        # Check if this is in the cache
        pk = state.get('_id')
        cache = cls.objects.cache
        obj = cache.get(pk)
        if obj is None:
            # Create and cache it
            obj = cls.__new__(cls)
            if pk:
                cache[pk] = obj

            # This ideally should only be done if created
            return (obj, True)
        return (obj, False)

    async def get_object_state(self, obj, state, scope):
        ModelType = obj.__class__
        return await ModelType.objects.find_one({'_id': state['_id']})

    def flatten_object(self, obj, scope):
        ref = obj.__ref__
        if ref in scope:
            return {'__ref__': ref, '__model__': obj.__model__}
        scope[ref] = obj
        state = obj.__getstate__(scope)
        _id = state.get("_id")
        if _id is None:
            return state
        return {'_id': _id, '__ref__': ref, '__model__': obj.__model__}

    def _default_registry(self):
        """ Add all nosql and json models to the registry
        """
        registry = JSONSerializer.instance().registry.copy()
        registry.update({m.__model__: m for m in find_subclasses(NoSQLModel)})
        return registry


class NoSQLDatabaseProxy(Atom):
    """ A proxy to the collection which holds a cache of model objects.

    """
    #: Object cache
    cache = Typed(weakref.WeakValueDictionary, ())

    #: Database handle
    table = Value()

    def __getattr__(self, name):
        return getattr(self.table, name)


class NoSQLModelManager(ModelManager):
    """ A descriptor so you can use this somewhat like Django's models.
    Assuming your using motor or txmongo.

    Examples
    --------
    MyModel.objects.find_one({'_id':'someid})

    """

    #: Table proxy cache
    proxies = Dict()

    def __get__(self, obj, cls=None):
        """ Handle objects from the class that owns the manager """
        cls = cls or obj.__class__
        if not issubclass(cls, Model):
            return self  # Only return the collection when used from a Model
        proxy = self.proxies.get(cls)
        if proxy is None:
            proxy = self.proxies[cls] = NoSQLDatabaseProxy(
                table=self.database[cls.__model__])
        return proxy

    def _default_database(self):
        raise EnvironmentError("No database has been set. Use "
                               "NoSQLModelManager.instance().database = <db>")


class NoSQLModel(Model):
    """ An atom model that can be serialized and deserialized to and from
    MongoDB.

    """

    #: ID of this object in the database
    _id = Instance(bson.ObjectId)

    #: Handles encoding and decoding
    serializer = NoSQLModelSerializer.instance()

    #: Handles database access
    objects = NoSQLModelManager.instance()

    @classmethod
    async def restore(cls, state, force=False):
        """ Restore an object from the database. If the object is cached,
        use that instead.
        """
        pk = state['_id']

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

    async def load(self):
        """ Alias to load this object from the database """
        pk = self._id
        if self.__restored__ or pk is None:
            return # Already loaded or nothing to load
        state = await self.objects.find_one({'_id': pk})
        if state is not None:
            await self.__restorestate__(state)

    async def save(self):
        """ Alias to delete this object to the database """
        db = self.objects
        state = self.__getstate__()
        if self._id is None:
            r = await db.insert_one(state)
            self._id = r.inserted_id
            db.cache[self._id] = self
        else:
            r = await db.replace_one({'_id': self._id}, state, upsert=True)
        self.__restored__ = True
        return r

    async def delete(self):
        """ Alias to delete this object in the database """
        db = self.objects
        pk = self._id
        if pk:
            r = await db.delete_one({'_id': pk})
            del db.cache[pk]
            del self._id
            return r
