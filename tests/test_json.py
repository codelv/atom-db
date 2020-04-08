import pytest
import json
from datetime import date, time, datetime
from atomdb.base import JSONModel
from atom.api import *

class Dates(JSONModel):
    d = Instance(date)
    t = Instance(time)
    dt = Instance(datetime)


class Options(JSONModel):
    a = Bool()
    b = Unicode()

class User(JSONModel):
    options = Instance(Options)

class File(JSONModel):
    name = Str()
    data = Bytes()

class Page(JSONModel):
    files = List(File)

class Tree(JSONModel):
    name = Str()
    related = ForwardInstance(lambda: Tree)

@pytest.mark.asyncio
async def test_json_dates():
    now = datetime.now()
    obj = Dates(d=now.date(), t=now.time(), dt=now)

    state = obj.__getstate__()
    data = json.dumps(state)
    r = await Dates.restore(json.loads(data))
    assert r.d == obj.d and r.t == obj.t and r.dt == r.dt


@pytest.mark.asyncio
async def test_json_nested():
    obj = User(options=Options(a=True, b="Yes"))
    state = obj.__getstate__()
    data = json.dumps(state)
    r = await User.restore(json.loads(data))
    assert r.options.a == obj.options.a and r.options.b == obj.options.b


@pytest.mark.asyncio
async def test_json_bytes():
    obj = File(name="test.png", data=b'abc')
    state = obj.__getstate__()
    data = json.dumps(state)
    r = await File.restore(json.loads(data))
    assert r.name == obj.name and r.data == obj.data


@pytest.mark.asyncio
async def test_json_list():
    f1 = File(name="test.png", data=b'abc')
    f2 = File(name="blueberry.jpg", data=b'123')
    obj = Page(files=[f1, f2])
    state = obj.__getstate__()
    data = json.dumps(state)
    r = await Page.restore(json.loads(data))
    assert len(r.files) == 2
    assert r.files[0].name == f1.name and r.files[0].data == f1.data
    assert r.files[1].name == f2.name and r.files[1].data == f2.data


@pytest.mark.asyncio
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
    assert r.name == 'a'
    assert r.related.name == b.name
    assert r.related.related == r
