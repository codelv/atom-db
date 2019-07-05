import pytest
import json
from datetime import date, time, datetime
from atomdb.base import JSONModel
from atom.api import *


@pytest.mark.asyncio
async def test_json_dates():

    class Test(JSONModel):
        d = Instance(date)
        t = Instance(time)
        dt = Instance(datetime)

    now = datetime.now()
    obj = Test(d=now.date(), t=now.time(), dt=now)

    state = obj.__getstate__()
    data = json.dumps(state)
    r = await Test.restore(json.loads(data))
    assert r.d == obj.d and r.t == obj.t and r.dt == r.dt



@pytest.mark.asyncio
async def test_json_nested():
    class Options(JSONModel):
        a = Bool()
        b = Unicode()

    class Test(JSONModel):
        options = Instance(Options)

    obj = Test(options=Options(a=True, b="Yes"))
    state = obj.__getstate__()
    data = json.dumps(state)
    r = await Test.restore(json.loads(data))
    assert r.options.a == obj.options.a and r.options.b == obj.options.b
