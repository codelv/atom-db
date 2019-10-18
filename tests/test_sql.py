import gc
import os
import re
import pytest
import random
import atomdb.sql
import sqlalchemy as sa
from atom.api import *
from atomdb.sql import SQLModel, SQLModelManager, Relation
from datetime import datetime, date, time
from faker import Faker
from pprint import pprint

from pymysql.err import IntegrityError

faker = Faker()

if 'DATABASE_URL' not in os.environ:
    os.environ['DATABASE_URL'] = 'mysql://mysql:mysql@127.0.0.1:3306/test_atomdb'

DATABASE_URL = os.environ['DATABASE_URL']


class User(SQLModel):
    id = Typed(int).tag(primary_key=True)
    name = Str().tag(length=200)
    email = Str().tag(length=64)
    active = Bool()
    age = Int()
    hashed_password = Bytes()
    settings = Dict()
    rating = Instance(float).tag(nullable=True)


class Job(SQLModel):
    name = Str().tag(length=64, unique=True)
    roles = Relation(lambda: JobRole)


class JobRole(SQLModel):
    name = Str().tag(length=64)
    job = Instance(Job)


class Image(SQLModel):
    name = Str().tag(length=100)
    path = Str().tag(length=200)
    metadata = Typed(dict).tag(nullable=True)
    alpha = Range(low=0, high=255)
    data = Instance(bytes).tag(nullable=True)

    # Maps to sa.ARRAY, must include the item_type tag
    size = Instance(tuple).tag(nullable=True, item_type=int)


class Page(SQLModel):
    title = Str().tag(length=60)
    status = Enum('preview', 'live')
    body = Str().tag(type=sa.UnicodeText())
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


class PageImage(SQLModel):
    # Example through table for job role
    page = Instance(Page).tag(nullable=False)
    image = Instance(Page).tag(nullable=False)

    class Meta:
        db_table = 'page_image_m2m'
        unique_together = ('page', 'image')


class Comment(SQLModel):
    page = Instance(Page)
    author = Instance(User)
    status = Enum('pending', 'approved')
    body = Str().tag(type=sa.UnicodeText())
    reply_to = ForwardInstance(lambda: Comment).tag(nullable=True)
    when = Instance(time)


class Email(SQLModel):
    to = Str().tag(length=120)
    from_ = Str().tag(name='from').tag(length=120)
    body = Str().tag(length=1024)


def test_build_tables():
    # Trigger table creation
    SQLModelManager.instance().create_tables()


def test_custom_table_name():
    table_name = 'some_table.test'

    class Test(SQLModel):
        __model__ = table_name

    assert Test.objects.table.name == table_name


async def reset_tables(*models):
    for Model in models:
        try:
            await Model.objects.drop_table()
        except Exception as e:
            if 'Unknown table' not in str(e):
                raise
        await Model.objects.create_table()


@pytest.fixture
async def db(event_loop):
    m = re.match(r'(.+)://(.+):(.*)@(.+):(\d+)/(.+)', DATABASE_URL)
    assert m, "MYSQL_URL is an invalid format"
    schema, user, pwd, host, port, db = m.groups()

    if schema == 'mysql':
        from aiomysql.sa import create_engine
        from aiomysql import connect
    elif schema == 'postgresql':
        from aiopg.sa import create_engine
        from aiopg import connect
    else:
        raise ValueError("Unsupported database schema: %s" % schema)

    params = dict(
        host=host, port=int(port), user=user, password=pwd, loop=event_loop)

    if schema == 'mysql':
        params['autocommit'] =True

    # Create the DB
    async with connect(**params) as conn:
        async with conn.cursor() as c:
            # WARNING: Not safe
            await c.execute('DROP DATABASE IF EXISTS %s;' % db)
            await c.execute('CREATE DATABASE %s;' % db)

    async with create_engine(db=db, **params) as engine:
        mgr = SQLModelManager.instance()
        mgr.database = engine
        yield engine


def test_query_ops_valid():
    """ Test that operators are all valid """
    from sqlalchemy.sql.expression import ColumnElement
    from atomdb.sql import QUERY_OPS
    for k, v in QUERY_OPS.items():
        assert hasattr(ColumnElement, v)


@pytest.mark.asyncio
async def test_drop_create_table(db):
    try:
        await User.objects.drop_table()
    except Exception as e:
        if 'Unknown table' not in str(e):
            raise
    await User.objects.create_table()


@pytest.mark.asyncio
async def test_simple_save_restore_delete(db):
    await reset_tables(User)

    # Save
    user = User(name=faker.name(), email=faker.email(), active=True)
    await user.save()
    assert user._id is not None

    # Restore
    u = await User.objects.get(name=user.name)
    assert u
    assert u._id == user._id
    assert u.name == user.name
    assert u.email == user.email
    assert u.active == user.active

    # Update
    user.active = False
    await user.save()

    u = await User.objects.get(name=user.name)
    assert u
    assert not u.active

    # Create second user
    another_user = User(name=faker.name(), email=faker.email(), active=True)
    await another_user.save()

    # Delete
    await user.delete()
    assert await User.objects.get(name=user.name) is None

    # Make sure second user still exists
    assert await User.objects.get(name=another_user.name) is not None


@pytest.mark.asyncio
async def test_query(db):
    await reset_tables(User)

    # Create second user
    for i in range(10):
        user = User(name=faker.name(), email=faker.email(), age=20, active=True)
        await user.save()

    for user in await User.objects.all():
        print(user)

    for user in await User.objects.filter(name=user.name):
        print(user)

    # Delete one
    await User.objects.delete(name=user.name)
    assert len(await User.objects.all()) == 9

    # Delete them all
    await User.objects.delete(active=True)
    assert len(await User.objects.all()) == 0


@pytest.mark.asyncio
async def test_query_related(db):
    await reset_tables(User, Job, JobRole)

    job = await Job.objects.create(name=faker.job())
    job1 = await Job.objects.create(name=faker.job())
    job2 = await Job.objects.create(name=faker.job())

    role = await JobRole.objects.create(job=job, name=faker.job())
    role1 = await JobRole.objects.create(job=job1, name=faker.job())
    role2 = await JobRole.objects.create(job=job2, name=faker.job())

    roles = await JobRole.objects.filter(job__name__in=[job.name, job2.name])
    assert len(roles) == 2

    roles = await JobRole.objects.filter(job__name=job2.name)
    assert len(roles) == 1

    roles = await JobRole.objects.filter(job__name__not='none of the above')
    assert len(roles) == 3

    # Cant do multiple joins
    with pytest.raises(NotImplementedError):
        roles = await JobRole.objects.get(job__name__other=1)


@pytest.mark.asyncio
async def test_get_or_create(db):
    await reset_tables(User, Job, JobRole)

    name = faker.name()
    email = faker.email()

    user, created = await User.objects.get_or_create(
        name=name, email=email)
    assert created
    assert user._id and user.name == name and user.email == user.email

    u, created = await User.objects.get_or_create(
        name=user.name, email=user.email)
    assert u._id == user._id
    assert not created and user.name == name and user.email == user.email

    # Test passing model
    job, created = await Job.objects.get_or_create(name=faker.job())
    assert job and created

    role, created = await JobRole.objects.get_or_create(
        job=job, name=faker.job())
    assert role and created and role.job._id == job._id

    role_check, created = await JobRole.objects.get_or_create(
        job=job, name=role.name)
    assert role_check._id == role._id and not created


@pytest.mark.asyncio
async def test_create(db):
    await reset_tables(Job, JobRole)

    job = await Job.objects.create(name=faker.job())
    assert job and job._id

    # DB should enforce unique ness
    with pytest.raises(IntegrityError):
        same_job = await Job.objects.create(name=job.name)


@pytest.mark.asyncio
async def test_transaction_rollback(db):
    await reset_tables(Job, JobRole)

    with pytest.raises(ValueError):
        async with Job.objects.connection() as conn:
            trans = await conn.begin()
            try:
                # Must pass in the connection parameter for transactions
                job = await Job.objects.create(name=faker.job(), connection=conn)
                assert job._id is not None
                for i in range(3):
                    role = await JobRole.objects.create(job=job, name=faker.job(),
                                                connection=conn)
                    assert role._id is not None
                complete = True
                raise ValueError("Oh crap, I didn't want to do that")
            except:
                await trans.rollback()
                rollback = True
                raise
            else:
                rollback = False
                await trans.commit()

    assert complete and rollback
    assert len(await Job.objects.all()) == 0
    assert len(await JobRole.objects.all()) == 0


@pytest.mark.asyncio
async def test_transaction_commit(db):
    await reset_tables(Job, JobRole)

    async with Job.objects.connection() as conn:
        trans = await conn.begin()
        try:
            # Must pass in the connection parameter for transactions
            job = await Job.objects.create(name=faker.job(), connection=conn)
            assert job._id is not None
            for i in range(3):
                role = await JobRole.objects.create(job=job, name=faker.job(),
                                            connection=conn)
                assert role._id is not None
        except:
            await trans.rollback()
            raise
        else:
            await trans.commit()

    assert len(await Job.objects.all()) == 1
    assert len(await JobRole.objects.all()) == 3


@pytest.mark.asyncio
async def test_filters(db):
    await reset_tables(User)

    user, created = await User.objects.get_or_create(
        name=faker.name(), email=faker.email(), age=21, active=True)
    assert created

    user2, created = await User.objects.get_or_create(
        name=faker.name(), email=faker.email(), age=48, active=False,
        rating=10.0)
    assert created

    # Startswith
    u = await User.objects.get(name__startswith=user.name[0])
    assert u.name == user.name
    assert u is user # Now cached

    # In query
    users = await User.objects.filter(name__in=[user.name, user2.name])
    assert len(users) == 2

    # Test use of count
    assert await User.objects.count(name__in=[user.name, user2.name]) == 2

    # Is query
    users = await User.objects.filter(active__is=False)
    assert len(users) == 1 and users[0].active == False
    assert users[0] is user2 # Now cached

    # Not query
    users = await User.objects.filter(rating__isnot=None)
    assert len(users) == 1 and users[0].rating is not None

    # Lt query
    users = await User.objects.filter(age__lt=30)
    assert len(users) == 1 and users[0].age == user.age

    # Not supported
    with pytest.raises(ValueError):
        users = await User.objects.filter(age__xor=1)

    # Missing op
    with pytest.raises(ValueError):
        users = await User.objects.filter(age__=1)


@pytest.mark.asyncio
async def test_column_rename(db):
    """ Columns can be tagged with custom names. Verify that it works.

    """
    await reset_tables(Email)

    e = Email(from_=faker.email(), to=faker.email(), body=faker.job())
    await e.save()

    # Check without use labels
    table = Email.objects.table
    q = table.select().where(table.c.to==e.to)
    row = await Email.objects.fetchone(q)
    assert row['from'] == e.from_, 'Column rename failed'

    # Check with use labels
    q = table.select(use_labels=True).where(table.c.to==e.to)
    row = await Email.objects.fetchone(q)
    assert row[f'{table.name}_from'] == e.from_, 'Column rename failed'

    # Restoring a renamed column needs to work
    restored = await Email.objects.get(to=e.to)
    restored.from_ == e.from_


@pytest.mark.asyncio
async def test_query_many_to_one(db):
    await reset_tables(Job, JobRole)

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

    print(q)

    r = await Job.objects.execute(q)
    assert r.returns_rows

    for row in await JobRole.objects.fetchall(q):
        #: TODO: combine the joins back up
        role = await JobRole.restore(row)

        # Job should be restored from the cache
        assert role.job is not None
        #for role in job.roles:
        #    assert role.job == job
        loaded.append(job)

    assert len(await Job.objects.fetchmany(q, size=2)) == 2

    # Make sure they pull from cache
    roles = await JobRole.objects.all()
    for role in roles:
        assert role.job is not None

    # Clear cache and ensure it doesn't pull from cache now
    Job.objects.cache.clear()
    JobRole.objects.cache.clear()

    roles = await JobRole.objects.all()
    for role in roles:
        assert role.job is None


@pytest.mark.asyncio
async def test_save_errors(db):
    await reset_tables(User)

    u = User()
    with pytest.raises(ValueError):
        # Cant do both
        await u.save(force_insert=True, force_update=True)

    # Updating unsaved will not work
    r = await u.save(force_update=True)
    assert r.rowcount == 0


@pytest.mark.asyncio
async def test_object_caching(db):
    await reset_tables(Email)

    e = Email(from_=faker.email(), to=faker.email(), body=faker.job())
    await e.save()
    pk = e._id
    aref = Email.objects.cache.get(pk)
    assert aref is e, 'Cached object is invalid'

    # Delete
    del e
    del aref

    gc.collect()

    # Make sure cache was cleaned up
    aref = Email.objects.cache.get(pk)
    assert aref is None, 'Cached object was not released'


def test_invalid_meta_field():
    with pytest.raises(TypeError):
        class TestTable(SQLModel):
            id = Int().tag(primary_key=True)

            class Meta:
                # table_name is invalid, use db_table
                table_name = 'use db_table'


def test_invalid_multiple_pk():
    with pytest.raises(NotImplementedError):
        class TestTable(SQLModel):
            id = Int().tag(primary_key=True)
            id2 = Int().tag(primary_key=True)


def test_abstract_tables():

    class AbstractUser(SQLModel):
        name = Str().tag(length=60)

        class Meta:
            abstract = True

    class CustomUser(AbstractUser):
        data = Dict()

    class CustomUserWithMeta(AbstractUser):
        data = Dict()
        class Meta:
            db_table = 'custom_user'

    class AbstractCustomUser(AbstractUser):
        data = Dict()
        class Meta(AbstractUser.Meta):
            db_table = 'custom_user2'

    class CustomUser2(AbstractCustomUser):
        pass

    class CustomUser3(AbstractCustomUser):
        class Meta(AbstractCustomUser.Meta):
            abstract = False

    # Attempts to invoke create_table on abstract models should fail
    with pytest.raises(NotImplementedError):
        AbstractUser.objects

    # Subclasses of abstract models become concrete so this is ok
    assert CustomUser.objects

    # Subclasses of abstract models become with Meta concrete so this is ok
    assert CustomUserWithMeta.objects

    # Subclasses that inherit Meta, inherit Meta :)
    with pytest.raises(NotImplementedError):
        AbstractCustomUser.objects  # Abstract is inherited in this case

    # This is okay too
    CustomUser2.objects
    CustomUser3.objects

