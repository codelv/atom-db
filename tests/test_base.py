import pytest
from atomdb.base import Model, ModelManager, ModelSerializer


class AbstractModel(Model):
    objects = ModelManager.instance()
    serializer = ModelSerializer.instance()


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


@pytest.mark.asyncio
async def test_model():
    m = AbstractModel()

    # Not implemented for abstract models
    with pytest.raises(NotImplementedError):
        await m.save()

    with pytest.raises(NotImplementedError):
        await m.delete()

    with pytest.raises(ValueError):
        state = {'__model__': 'not.this.Model'}
        await AbstractModel.restore(state)

    # Old state fields do no blow up
    state = m.__getstate__()
    state['old_field'] = 'no-longer-exists'
    obj = await AbstractModel.restore(state)
