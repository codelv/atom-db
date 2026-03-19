import json
import uuid
from datetime import date, datetime, time
from decimal import Decimal

import pytest
from atom.api import (
    Bool,
    Bytes,
    ForwardInstance,
    Instance,
    Int,
    List,
    Range,
    Set,
    Str,
    Tuple,
    Typed,
)

from atomdb.base import JSONModel, RestoreError

# Set default to raise
JSONModel.__on_error__ = "raise"


class Dates(JSONModel):
    d = Instance(date)
    t = Instance(time)
    dt = Instance(datetime)


class Options(JSONModel):
    a = Bool()
    b = Str()
    c = Range(0, 10)


class User(JSONModel):
    options = Instance(Options)


class File(JSONModel):
    id = Instance(uuid.UUID, factory=uuid.uuid4)
    name = Str()
    data = Bytes()


class Page(JSONModel):
    files = List(File)
    created = Instance(datetime).tag(
        flatten=lambda d, scope: d.timestamp(),
        unflatten=lambda v, scope: datetime.fromtimestamp(v) if v else None,
    )


class Tree(JSONModel):
    name = Str()
    related = ForwardInstance(lambda: Tree)


class Amount(JSONModel):
    total = Instance(Decimal)


class ImageExtra(JSONModel):
    name = Str()
    enabled = Bool()


class Image(JSONModel):
    tags = Set(str)
    extras = Set(ImageExtra)


class Point(JSONModel):
    position = Tuple(float)


class Position(JSONModel):
    x = Int()
    y = Int()


class Rectangle(JSONModel):
    height = Range(low=0, value=1)
    width = Range(low=0, value=1)
    center = Typed(Position)


async def test_json_dates():
    now = datetime.now()
    obj = Dates(d=now.date(), t=now.time(), dt=now)

    state = obj.__getstate__()
    data = json.dumps(state)
    r = await Dates.restore(json.loads(data))
    assert r.d == obj.d and r.t == obj.t and r.dt == r.dt


async def test_json_decimal():
    d = Decimal("3.9")
    obj = Amount(total=d)
    state = obj.__getstate__()
    data = json.dumps(state)
    r = await Amount.restore(json.loads(data))
    assert r.total == d


async def test_json_nested():
    obj = User(options=Options(a=True, b="Yes", c=1))
    state = obj.__getstate__()
    data = json.dumps(state)
    r = await User.restore(json.loads(data))
    assert (
        r.options.a == obj.options.a
        and r.options.b == obj.options.b
        and r.options.c == obj.options.c
    )


async def test_json_bytes():
    obj = File(name="test.png", data=b"abc")
    state = obj.__getstate__()
    data = json.dumps(state)
    r = await File.restore(json.loads(data))
    assert r.name == obj.name and r.data == obj.data


async def test_json_list():
    f1 = File(name="test.png", data=b"abc")
    f2 = File(name="blueberry.jpg", data=b"123")
    now = datetime.now()
    obj = Page(files=[f1, f2], created=now)
    state = obj.__getstate__()
    assert isinstance(state["created"], float)  # Make sure conversion occurred
    data = json.dumps(state)
    r = await Page.restore(json.loads(data))
    assert r.created == now
    assert len(r.files) == 2
    assert r.files[0].name == f1.name and r.files[0].data == f1.data
    assert r.files[1].name == f2.name and r.files[1].data == f2.data
    assert r.files[1].id == f2.id


async def test_json_set():
    obj = Image(tags={"cat", "dog"})
    state = obj.__getstate__()
    data = json.dumps(state)
    r = await Image.restore(json.loads(data))
    assert r.tags == {"cat", "dog"}


async def test_json_set_nested():
    crop = ImageExtra(name="crop", enabled=True)
    obj = Image(extras={crop})
    state = obj.__getstate__()
    data = json.dumps(state)
    r = await Image.restore(json.loads(data))
    assert len(r.extras) == 1
    for it in r.extras:
        assert it.name == crop.name
        assert it.enabled == crop.enabled


async def test_json_tuple():
    obj = Point(position=(2.1, 3.3))
    state = obj.__getstate__()
    data = json.dumps(state)
    r = await Point.restore(json.loads(data))
    assert r.position == (2.1, 3.3)


async def test_json_cyclical():
    b = Tree(name="b")
    a = Tree(name="a", related=b)

    # Create a cyclical ref
    b.related = a

    obj = a
    state = obj.__getstate__()
    data = json.dumps(state)
    print(data)
    r = await Tree.restore(json.loads(data))
    assert r.name == "a"
    assert r.related.name == b.name
    assert r.related.related == r


async def test_json_optional_typed_model():
    # Test save & restore of optional typed member with no value
    rect = Rectangle(width=1, height=2)
    state = JSONModel.serializer.flatten(rect)
    rect = await JSONModel.serializer.unflatten(state)
    assert rect.center is None
    # Now set the value and make sure it's restored'
    rect.center = Position(x=10, y=20)
    state = JSONModel.serializer.flatten(rect)
    rect = await JSONModel.serializer.unflatten(state)
    assert rect.center.x == 10 and rect.center.y == 20


async def test_json_restore_error():
    # Test save & restore of optional typed member with no value
    rect = Rectangle(width=1, height=2)
    state = JSONModel.serializer.flatten(rect)
    state["width"] = "Not an int"
    with pytest.raises(RestoreError) as exc_info:
        await JSONModel.serializer.unflatten(state)
    restore_error = exc_info.value
    assert (
        restore_error.field == "width"
        and restore_error.cls is Rectangle
        and isinstance(restore_error.exc, TypeError)
    )
