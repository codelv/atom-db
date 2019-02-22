import pytest
import atomdb.nosql
from atom.api import *
from atomdb.nosql import NoSQLModel
from faker import Faker
from motor.motor_asyncio import AsyncIOMotorClient
from pprint import pprint


faker = Faker()


class User(NoSQLModel):
    name = Unicode()
    email = Unicode()
    active = Bool()
    settings = Dict()

    def _default_settings(self):
        return {'font-size': 16, 'theme': 'maroon'}


class Image(NoSQLModel):
    name = Unicode()
    path = Unicode()


class Page(NoSQLModel):
    title = Unicode()
    status = Enum('preview', 'live')
    body = Unicode()
    author = Instance(User)
    images = List(Image)
    related = List(ForwardInstance(lambda: Page))


class Comment(NoSQLModel):
    page = Instance(Page)
    author = Instance(User)
    status = Enum('pending', 'approved')
    body = Unicode()
    reply_to = ForwardInstance(lambda: Comment)


@pytest.yield_fixture()
def db(event_loop):
    client = AsyncIOMotorClient(io_loop=event_loop)
    db = client.enaml_web_test_db
    atomdb.nosql.DEFAULT_DATABASE = db
    yield db


@pytest.mark.asyncio
async def test_db_manager(db):
    from atomdb.nosql import NoSQLModelManager
    mgr = NoSQLModelManager.instance()
    assert db == mgr.get_database()

    # Check non-model access, it should not return the collection
    class NotModel(Atom):
        objects = mgr
    assert NotModel.objects == mgr

    # Now change it
    atomdb.nosql.DEFAULT_DATABASE = None
    with pytest.raises(EnvironmentError):
        await User.objects.find().to_list(length=10)

    # And restore
    atomdb.nosql.DEFAULT_DATABASE = db
    await User.objects.find().to_list(length=10)


@pytest.mark.asyncio
async def test_simple_save_restore_delete(db):
    await User.objects.drop()

    # Save
    user = User(name=faker.name(), email=faker.email(), active=True)
    await user.save()
    assert user._id is not None

    # Restore
    state = await User.objects.find_one({'name': user.name})
    assert state

    u = await User.restore(state)
    assert u._id == user._id
    assert u.name == user.name
    assert u.email == user.email
    assert u.active == user.active

    # Update
    user.active = False
    await user.save()

    state = await User.objects.find_one({'name': user.name})
    assert state
    u = await User.restore(state)
    assert not u.active

    # Create second user
    another_user = User(name=faker.name(), email=faker.email(), active=True)
    await another_user.save()

    # Delete
    await user.delete()
    state = await User.objects.find_one({"name": user.name})
    assert not state

    # Make sure second user still exists
    state = await User.objects.find_one({"name": another_user.name})
    assert state


@pytest.mark.asyncio
async def test_nested_save_restore(db):
    await Image.objects.drop()
    await User.objects.drop()
    await Page.objects.drop()
    await Comment.objects.drop()

    authors = [
        User(name=faker.name(), active=True) for i in range(2)
    ]
    for a in authors:
        await a.save()

    images = [
        Image(name=faker.job(), path=faker.image_url()) for i in range(10)
    ]

    # Only save the first few, it should serialize the others
    for i in range(3):
        await images[i].save()

    pages = [
        Page(title=faker.catch_phrase(), body=faker.text(), author=author,
             images=[faker.random.choice(images) for j in range(faker.random.randint(0, 2))],
             status=faker.random.choice(Page.status.items))
        for i in range(4) for author in authors
    ]
    for p in pages:
        await p.save()

        # Generate comments
        comments = []
        for i in range(faker.random.randint(1, 10)):
            commentor = User(name=faker.name())
            await commentor.save()
            comment = Comment(author=commentor, page=p,
                              status=faker.random.choice(Comment.status.items),
                              reply_to=faker.random.choice([None]+comments),
                              body=faker.text())
            comments.append(comment)
            await comment.save()

    for p in pages:
        # Find in db
        state = await Page.objects.find_one({'author._id': p.author._id,
                                             'title': p.title})
        assert state, f'Couldnt find page by {p.title} by {p.author.name}'
        r = await Page.restore(state)
        assert p._id == r._id
        assert p.author._id == r.author._id
        assert p.title == r.title
        assert p.body == r.body
        for img_1, img_2 in zip(p.images, r.images):
            assert img_1.path == img_2.path

        async for state in Comment.objects.find({'page._id': p._id}):
            comment = await Comment.restore(state)
            assert comment.page._id == p._id
            async for state in Comment.objects.find({'reply_to._id':
                                                         comment._id}):
                reply = await Comment.restore(state)
                assert reply.page._id == p._id


@pytest.mark.asyncio
async def test_circular(db):
    # Test that a circular reference is properly stored as a reference
    # and doesn't create an infinite loop
    await Page.objects.drop()

    p = Page(title=faker.catch_phrase(), body=faker.text())
    related_page = Page(title=faker.catch_phrase(), body=faker.text(),
                        related=[p])

    # Create a circular reference
    p.related = [related_page]
    await p.save()

    # Make sure it restores properly
    state = await Page.objects.find_one({'_id': p._id})
    pprint(state)
    r = await Page.restore(state)
    assert r.title == p.title
    assert r.related[0].title == related_page.title
    assert r.related[0].related[0] == r
