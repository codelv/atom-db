import os
import re
import pytest
import random
import aiomysql
import atomdb.sql
import sqlalchemy as sa
from aiomysql.sa import create_engine
from atom.api import *
from atomdb.sql import SQLModel, SQLModelManager, Relation
from datetime import datetime, date, time
from faker import Faker
from pprint import pprint


faker = Faker()

if 'MYSQL_URL' not in os.environ:
    os.environ['MYSQL_URL'] = 'mysql://mysql:mysql@127.0.0.1:3306/test_atomdb'

DATABASE_URL = os.environ['MYSQL_URL']


class User(SQLModel):
    id = Typed(int).tag(primary_key=True)
    name = Unicode().tag(length=200)
    email = Unicode().tag(length=64)
    active = Bool()
    age = Int()
    hashed_password = Bytes()
    settings = Dict()


class Job(SQLModel):
    name = Unicode().tag(length=64)
    roles = Relation(lambda: JobRole)


class JobRole(SQLModel):
    name = Unicode().tag(length=64)
    job = Instance(Job)


class Image(SQLModel):
    name = Unicode().tag(length=100)
    path = Unicode().tag(length=200)
    metadata = Dict()


class Page(SQLModel):
    title = Str().tag(length=60)
    status = Enum('preview', 'live')
    body = Unicode().tag(type=sa.UnicodeText())
    author = Instance(User)
    images = List(Instance(Image))
    related = List(ForwardInstance(lambda: Page)).tag(nullable=True)
    rating = Float()
    visits = Long()
    date = Instance(date)
    last_updated = Instance(datetime)
    tags = List(str)

    # A bit verbose but provides a custom column specification
    data = Instance(object).tag(column=sa.Column('data', sa.LargeBinary()))


class Comment(SQLModel):
    page = Instance(Page)
    author = Instance(User)
    status = Enum('pending', 'approved')
    body = Unicode().tag(type=sa.UnicodeText())
    reply_to = ForwardInstance(lambda: Comment).tag(nullable=True)
    when = Instance(time)


def test_build_tables():
    # Trigger table creation
    SQLModelManager.instance().create_tables()


def test_custom_table_name():
    table_name = 'some_table.test'

    class Test(SQLModel):
        __model__ = table_name

    assert Test.objects.table.name == table_name


@pytest.fixture
async def db(event_loop):
    m = re.match(r'mysql://(.+):(.*)@(.+):(\d+)/(.+)', DATABASE_URL)
    assert m, "MYSQL_URL is an invalid format"
    user, pwd, host, port, db = m.groups()

    params = dict(
        host=host, port=int(port), user=user, password=pwd, loop=event_loop)

    # Create the DB
    async with aiomysql.connect(**params) as conn:
        async with conn.cursor() as c:
            # WARNING: Not safe
            await c.execute('DROP DATABASE IF EXISTS %s;' % db)
            await c.execute('CREATE DATABASE %s;' % db)

    async with create_engine(db=db, **params) as engine:
        atomdb.sql.DEFAULT_DATABASE = engine
        yield engine


@pytest.mark.asyncio
async def test_drop_create_table(db):
    try:
        await User.objects.drop()
    except Exception as e:
        if 'Unknown table' not in str(e):
            raise
    await User.objects.create()


@pytest.mark.asyncio
async def test_simple_save_restore_delete(db):
    try:
        await User.objects.drop()
    except Exception as e:
        if 'Unknown table' not in str(e):
            raise
    await User.objects.create()

    # Save
    user = User(name=faker.name(), email=faker.email(), active=True)
    await user.save()
    assert user._id is not None

    # Restore
    state = await User.objects.get(name=user.name)
    assert state

    u = await User.restore(state)
    assert u._id == user._id
    assert u.name == user.name
    assert u.email == user.email
    assert u.active == user.active

    # Update
    user.active = False
    await user.save()

    state = await User.objects.get(name=user.name)
    assert state
    u = await User.restore(state)
    assert not u.active

    # Create second user
    another_user = User(name=faker.name(), email=faker.email(), active=True)
    await another_user.save()

    # Delete
    await user.delete()
    state = await User.objects.get(name=user.name)
    assert not state

    # Make sure second user still exists
    state = await User.objects.get(name=another_user.name)
    assert state


@pytest.mark.asyncio
async def test_query(db):
    await User.objects.create()

    # Create second user
    for i in range(10):
        user = User(name=faker.name(), email=faker.email(), active=True)
        await user.save()

    for row in await User.objects.all():
        print(row)

    for row in await User.objects.filter(name=user.name):
        print(row)


@pytest.mark.asyncio
async def test_query_many_to_one(db):
    await Job.objects.create()
    await JobRole.objects.create()

    jobs = []

    for i in range(5):
        job = Job(name=faker.job())
        await job.save()
        jobs.append(job)

        for i in range(random.randint(1, 5)):
            role = JobRole(name=faker.bs(), job=job)
            await role.save()

    loaded = []
    q = Job.objects.table.join(JobRole.objects.table).select(use_labels=True)

    for row in await Job.objects.fetchall(q):
        #: TODO: combine the joins back up
        job = await Job.restore(row)
        for role in job.roles:
            assert role.job == job
        loaded.append(job)
    #assert len(jobs) == len(loaded)


@pytest.mark.skip(reason="Not implemented")
@pytest.mark.asyncio
async def test_nested_save_restore(db):
    await Image.objects.create()
    await User.objects.create()
    await Page.objects.create()
    await Comment.objects.create()

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


@pytest.mark.skip(reason="Not implemented")
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
