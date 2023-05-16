import logging

import pytest
from atom.api import (
    Bool,
    Coerced,
    Dict,
    ForwardInstance,
    ForwardTyped,
    Instance,
    Int,
    List,
    Property,
    Set,
    Tuple,
    Typed,
)

from atomdb.base import (
    Model,
    ModelManager,
    ModelSerializer,
    generate_function,
    is_db_field,
    is_primitive_member,
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
    list_of_tuple = List(Tuple())
    list_of_tuple_of_float = List(Tuple(float))
    tuple_of_any = Tuple()
    tuple_of_number = Tuple((float, int))
    tuple_of_int_or_model = Tuple((int, Model))
    tuple_of_forwarded = Tuple(ForwardTyped(lambda: NotYetDefined))
    set_of_any = Tuple()
    set_of_number = Set(float)
    set_of_model = Set(AbstractModel)
    dict_of_any = Dict()
    dict_of_str_any = Dict(str)
    dict_of_str_int = Dict(str, int)
    typed_int = Typed(int)
    typed_dict = Typed(dict)
    instance_of_model = Instance(AbstractModel)
    forwarded_instance = ForwardInstance(lambda: NotYetDefined)
    coerced_int = Coerced(int)
    prop = Property(lambda self: True)
    tagged_prop = Property(lambda self: 0).tag(store=True)


class NotYetDefined:
    pass


async def test_manager():
    mgr = ModelManager.instance()

    # Not implemented for abstract manager
    with pytest.raises(NotImplementedError):
        mgr.database

    # Not implemented for abstract manager
    with pytest.raises(NotImplementedError):
        AbstractModel.objects


async def test_serializer():
    m = AbstractModel()
    ser = ModelSerializer.instance()
    with pytest.raises(NotImplementedError):
        await ser.get_object_state(m, {}, {})

    with pytest.raises(NotImplementedError):
        await ser.flatten_object(m, {})


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
    assert obj.rating == 0


@pytest.mark.parametrize(
    "attr, expected",
    (
        ("id", True),
        ("_private", False),
        ("computed", False),
        ("prop", False),
        ("tagged_prop", True),
    ),
)
def test_is_db_field(attr, expected):
    member = Dummy.members()[attr]
    assert is_db_field(member) == expected


@pytest.mark.parametrize(
    "attr, expected",
    (
        ("id", True),
        ("enabled", True),
        ("string", True),
        ("list_of_int", True),
        ("list_of_any", False),
        ("list_of_str", True),
        ("list_of_tuple", False),
        ("list_of_tuple_of_float", True),
        ("tuple_of_any", False),
        ("tuple_of_number", False),
        ("tuple_of_int_or_model", False),
        ("tuple_of_forwarded", False),
        ("set_of_any", False),
        ("set_of_number", False),
        ("set_of_model", False),
        ("typed_int", True),
        ("typed_dict", False),
        ("instance_of_model", False),
        ("forwarded_instance", None),
        ("dict_of_any", False),
        ("dict_of_str_any", False),
        ("dict_of_str_int", True),
        ("coerced_int", True),
        ("prop", False),
        ("tagged_prop", False),
    ),
)
def test_is_primitive_member(attr, expected):
    member = Dummy.members()[attr]
    assert is_primitive_member(member) == expected


def test_gen_fn():
    fn = generate_function(
        "\n".join(("def foo(v):", "    return str(v)")),
        {"str": str},
        "foo",
    )
    assert callable(fn)
    assert fn(1) == "1"

    # Not a fn
    with pytest.raises(RuntimeError):
        generate_function('__import__("os").path.exists()', {}, "__import__")


async def test_on_error_raise():
    """When __on_error__ is raise any old data in the state will make the
    restore fail.
    """

    class A(Model):
        __on_error__ = "raise"
        value = Int()

    with pytest.raises(TypeError):
        await A.restore({"value": "str"})


async def test_on_error_ignore():
    """When __on_error__ is "ignore" and setattr fails the error is discarded"""

    class B(Model):
        __on_error__ = "ignore"
        old_field = Int()
        new_field = Int()

    b = await B.restore({"old_field": "str", "new_field": 1})
    assert b.old_field == 0
    assert b.new_field == 1


async def test_on_error_log(caplog):
    """When __on_error__ is "log" (the default) and setattr fails the error
    is logged.
    """

    class C(Model):
        old_field = Int()
        new_field = Int()

    with caplog.at_level(logging.DEBUG):
        c = await C.restore({"old_field": "str", "new_field": 1})
        assert c.old_field == 0
        assert c.new_field == 1
        assert "Error loading state:" in caplog.text
        assert f"{C.__model__}.old_field" in caplog.text
        assert "object must be of type 'int'" in caplog.text
