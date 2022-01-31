import pytest
from atom.api import *
from atomdb.base import (
    Model, ModelManager, ModelSerializer,
    is_db_field, is_primitive_member
)


class AbstractModel(Model):
    objects = ModelManager.instance()
    serializer = ModelSerializer.instance()

    rating = Int()


class Dummy(Model):
    _private = Int()
    computed = Int().tag(store=False)
    id = Int()
    enabled = Bool()
    string = Bool()
    list_of_int = List(int)
    list_of_str = List(str)
    list_of_any = List()
    tuple_of_any = Tuple()
    tuple_of_number = Tuple((float, int))
    tuple_of_int_or_model = Tuple((int, Model))
    set_of_any = Tuple()
    set_of_number = Set(float)
    set_of_model = Set(AbstractModel)
    meta = Dict()
    typed_int = Typed(int)
    typed_dict = Typed(dict)
    instance_of_model = Instance(AbstractModel)


@pytest.mark.asyncio
async def test_manager():
    mgr = ModelManager.instance()

    # Not implemented for abstract manager
    with pytest.raises(NotImplementedError):
        mgr.database

    # Not implemented for abstract manager
    with pytest.raises(NotImplementedError):
        AbstractModel.objects


@pytest.mark.asyncio
async def test_serializer():
    m = AbstractModel()
    ser = ModelSerializer.instance()
    with pytest.raises(NotImplementedError):
        await ser.get_object_state(m, {}, {})

    with pytest.raises(NotImplementedError):
        await ser.flatten_object(m, {})


@pytest.mark.asyncio
async def test_model():
    m = AbstractModel()

    # Not implemented for abstract models
    with pytest.raises(NotImplementedError):
        await m.load()

    with pytest.raises(NotImplementedError):
        await m.save()

    with pytest.raises(NotImplementedError):
        await m.delete()

    with pytest.raises(ValueError):
        state = {"__model__": "not.this.Model"}
        await AbstractModel.restore(state)

    # Old state fields do not blow up
    state = m.__getstate__()
    state["removed_field"] = "no-longer-exists"
    state["rating"] = 3.5  # Type changed
    obj = await AbstractModel.restore(state)


@pytest.mark.parametrize('attr, expected', (
    ('id', True),
    ('_private', False),
    ('computed', False),
))
def test_is_db_field(attr, expected):
    member = Dummy.members()[attr]
    assert is_db_field(member) == expected


@pytest.mark.parametrize('attr, expected', (
    ('id', True),
    ('enabled', True),
    ('string', True),
    ('list_of_int', True),
    ('list_of_any', False),
    ('list_of_str', True),
    ('tuple_of_any', False),
    ('tuple_of_number', True),
    ('tuple_of_int_or_model', False),
    ('set_of_any', False),
    ('set_of_number', True),
    ('set_of_model', False),
    ('typed_int', True),
    ('typed_dict', False),
    ('instance_of_model', False),
    ('meta', False),
))
def test_is_primitive_member(attr, expected):
    member = Dummy.members()[attr]
    assert is_primitive_member(member) == expected
