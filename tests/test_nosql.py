import os
import random
from pprint import pprint

import pytest
from atom.api import Atom, Bool, Dict, Enum, ForwardInstance, Instance, List, Str

try:
    from motor.motor_asyncio import AsyncIOMotorClient

    from atomdb.nosql import NoSQLModel, NoSQLModelManager
except ImportError:
    pytest.skip("mongo/motor is not available", allow_module_level=True)


class User(NoSQLModel):
    name = Str()
    email = Str()
    active = Bool()
    settings = Dict()

    def _default_settings(self):
        return {"font-size": 16, "theme": "maroon"}


class Image(NoSQLModel):
    name = Str()
    path = Str()


class Page(NoSQLModel):
    title = Str()
    status = Enum("preview", "live")
    body = Str()
    author = Instance(User)
    images = List(Image)
    related = List(ForwardInstance(lambda: Page))


class Comment(NoSQLModel):
    page = Instance(Page)
    author = Instance(User)
    status = Enum("pending", "approved")
    body = Str()
    reply_to = ForwardInstance(lambda: Comment)


@pytest.fixture
def db(event_loop):
    MONGO_URL = os.environ.get("MONGO_URL", None)
    if MONGO_URL:
        client = AsyncIOMotorClient(MONGO_URL, io_loop=event_loop)
    else:
        client = AsyncIOMotorClient(io_loop=event_loop)
    db = client.enaml_web_test_db
    mgr = NoSQLModelManager.instance()
    mgr.database = db
    mgr.proxies = {}  # Flush the db cache
    yield db


async def test_db_manager(db):
    mgr = NoSQLModelManager.instance()

    # Check non-model access, it should not return the collection
    class NotModel(Atom):
        objects = mgr

    assert NotModel.objects == mgr

    # Now change it
    del mgr.database
    with pytest.raises(EnvironmentError):
        await User.objects.find().to_list(length=10)

    # And restore
    mgr.database = db
    await User.objects.find().to_list(length=10)


async def test_simple_save_restore_delete(db):
    await User.objects.drop()

    # Save
    user = User(name="name", email="name@ex.com", active=True)
    await user.save()
    assert user._id is not None

    # Restore
    state = await User.objects.find_one({"name": user.name})
    assert state

    u = await User.restore(state)
    assert u is user  # No cached
    assert u._id == user._id
    assert u.name == user.name
    assert u.email == user.email
    assert u.active == user.active

    # Update
    user.active = False
    await user.save()

    state = await User.objects.find_one({"name": user.name})
    assert state
    u = await User.restore(state)
    assert not u.active

    # Create second user
    another_user = User(name="other", email="other@ex.com", active=True)
    await another_user.save()

    # Delete
    await user.delete()
    state = await User.objects.find_one({"name": user.name})
    assert not state

    # Make sure second user still exists
    state = await User.objects.find_one({"name": another_user.name})
    assert state


async def test_nested_save_restore(db):
    await Image.objects.drop()
    await User.objects.drop()
    await Page.objects.drop()
    await Comment.objects.drop()

    authors = [User(name=f"User{i}", active=True) for i in range(2)]
    for a in authors:
        await a.save()

    images = [Image(name=f"Img{i}", path=f"/app/{i}") for i in range(10)]

    # Only save the first few, it should serialize the others
    for i in range(3):
        await images[i].save()

    pages = [
        Page(
            title=f"Page{i}",
            body=f"Content{i}",
            author=author,
            images=[random.choice(images) for j in range(random.randint(0, 2))],
            status=random.choice(Page.status.items),
        )
        for i in range(4)
        for author in authors
    ]
    for p in pages:
        await p.save()

        # Generate comments
        comments = []
        for i in range(random.randint(1, 10)):
            commentor = User(name=f"User{i}")
            await commentor.save()
            comment = Comment(
                author=commentor,
                page=p,
                status=random.choice(Comment.status.items),
                reply_to=random.choice([None] + comments),
                body=f"Body{i}",
            )
            comments.append(comment)
            await comment.save()

    for p in pages:
        # Find in db
        state = await Page.objects.find_one(
            {"author._id": p.author._id, "title": p.title}
        )
        assert state, f"Couldnt find page by {p.title} by {p.author.name}"
        r = await Page.restore(state)
        assert p._id == r._id
        assert p.author._id == r.author._id
        assert p.title == r.title
        assert p.body == r.body
        for img_1, img_2 in zip(p.images, r.images):
            assert img_1.path == img_2.path

        async for state in Comment.objects.find({"page._id": p._id}):
            comment = await Comment.restore(state)
            assert comment.page._id == p._id
            async for state in Comment.objects.find({"reply_to._id": comment._id}):
                reply = await Comment.restore(state)
                assert reply.page._id == p._id


async def test_circular(db):
    # Test that a circular reference is properly stored as a reference
    # and doesn't create an infinite loop
    await Page.objects.drop()

    p = Page(title="Home", body="HomeBody")
    related_page = Page(title="Other", body="OtherBody", related=[p])

    # Create a circular reference
    p.related = [related_page]
    await p.save()

    # Make sure it restores properly
    state = await Page.objects.find_one({"_id": p._id})
    pprint(state)
    r = await Page.restore(state)
    assert r.title == p.title
    assert r.related[0].title == related_page.title
    assert r.related[0].related[0] == r


async def test_load(db):
    # That an object can be loaded by setting the ID and calling load.
    await User.objects.drop()

    authors = [User(name=f"User{i}", active=True) for i in range(2)]
    for a in authors:
        await a.save()

    user = User(_id=authors[0]._id)
    assert not user.name and not user.__restored__

    # Load should do nothing if already restored (which is faked here)
    user.__restored__ = True
    await user.load()
    assert not user.name

    # Now ensure a normal load works
    user.__restored__ = False
    await user.load()
    assert user.name == authors[0].name
