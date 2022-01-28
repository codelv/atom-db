import gc
import os
import re
import pytest
import random

from decimal import Decimal
from datetime import datetime, date, time, timedelta
from faker import Faker

faker = Faker()

if "DATABASE_URL" not in os.environ:
    os.environ[
        "DATABASE_URL"
    ] = "postgres://postgres:postgres@127.0.0.1:5432/test_atomdb"

DATABASE_URL = os.environ["DATABASE_URL"]

try:
    import atomdb.sql
    import sqlalchemy as sa
    from atom.api import *
    from atomdb.sql import JSONModel, SQLModel, SQLModelManager, Relation

    if DATABASE_URL.startswith("mysql"):
        from pymysql.err import IntegrityError
    else:
        from psycopg2.errors import UniqueViolation as IntegrityError
except ImportError as e:
    pytest.skip("aiomysql and aiopg not available", allow_module_level=True)

IS_MYSQL = DATABASE_URL.startswith("mysql")


class AbstractUser(SQLModel):
    email = Str().tag(length=64)
    hashed_password = Bytes()

    class Meta:
        abstract = True


class User(AbstractUser):
    id = Typed(int).tag(primary_key=True, name="user_id")
    name = Str().tag(length=200)
    active = Bool()
    age = Int()
    settings = Dict()
    rating = Instance(float).tag(nullable=True)


class Job(SQLModel):
    name = Str().tag(length=64, unique=True)
    roles = Relation(lambda: JobRole)
    duration = Instance(timedelta)


class JobRole(SQLModel):
    name = Str().tag(length=64)
    default = Bool()
    job = Instance(Job)
    tasks = Relation(lambda: JobTask)

    check_one_default = sa.schema.DDL(
        """
        CREATE OR REPLACE FUNCTION check_one_default() RETURNS TRIGGER
        LANGUAGE plpgsql
        AS $$
        BEGIN
            IF EXISTS (SELECT * from "test_sql.JobRole"
                       WHERE "default" = true AND "job" = NEW."job") THEN
                RAISE EXCEPTION 'A default aleady exists';
            END IF;
            RETURN NEW;
        END;
        $$;"""
    )

    trigger = sa.schema.DDL(
        """
        CREATE CONSTRAINT TRIGGER check_default_role AFTER INSERT OR UPDATE
        OF "default" ON "test_sql.JobRole"
        FOR EACH ROW EXECUTE PROCEDURE check_one_default();"""
    )

    class Meta:
        triggers = [
            (
                "after_create",
                lambda: JobRole.check_one_default.execute_if(dialect="postgresql"),
            ),
            ("after_create", lambda: JobRole.trigger.execute_if(dialect="postgresql")),
        ]


class JobTask(SQLModel):
    id = Int().tag(primary_key=True)
    role = Instance(JobRole)
    desc = Str().tag(length=20)


class ImageInfo(JSONModel):
    depth = Int()


class Image(SQLModel):
    name = Str().tag(length=100)
    path = Str().tag(length=200)
    metadata = Typed(dict).tag(nullable=True)
    alpha = Range(low=0, high=255)
    data = Instance(bytes).tag(nullable=True)

    # Maps to sa.ARRAY, must include the item_type tag
    size = Instance(tuple).tag(nullable=True, item_type=int)

    #: Maps to sa.JSON
    info = Instance(ImageInfo, ())


class BigInt(Int):
    """Custom member which defines the sa column type using get_column method"""

    def get_column(self, model):
        return sa.Column(self.name, sa.BigInteger())


class Page(SQLModel):
    title = Str().tag(length=60)
    status = Enum("preview", "live")
    body = Str().tag(type=sa.UnicodeText())
    author = Instance(User)
    if DATABASE_URL.startswith("postgres"):
        images = List(Instance(Image))
        related = List(ForwardInstance(lambda: Page)).tag(nullable=True)
        tags = List(str)

    visits = BigInt()
    date = Instance(date)
    last_updated = Instance(datetime)
    rating = Instance(Decimal)
    ranking = Float().tag(name="order")

    # A bit verbose but provides a custom column specification
    data = Instance(object).tag(column=sa.Column("data", sa.LargeBinary()))


class PageImage(SQLModel):
    # Example through table for job role
    page = Instance(Page).tag(nullable=False)
    image = Instance(Page).tag(nullable=False)

    class Meta:
        db_table = "page_image_m2m"
        unique_together = ("page", "image")


class Comment(SQLModel):
    page = Instance(Page)
    author = Instance(User)
    status = Enum("pending", "approved")
    body = Str().tag(type=sa.UnicodeText())
    reply_to = ForwardInstance(lambda: Comment).tag(nullable=True)
    when = Instance(time)


class Email(SQLModel):
    id = Int().tag(name="email_id", primary_key=True)
    to = Str().tag(length=120)
    from_ = Str().tag(name="from").tag(length=120)
    body = Str().tag(length=1024)


class Ticket(SQLModel):
    code = Str().tag(length=64, primary_key=True)
    desc = Str().tag(length=500)


class ImportedTicket(Ticket):
    meta = Dict()


class Document(SQLModel):
    uuid = Str().tag(length=64, primary_key=True)


class Project(SQLModel):
    doc = Instance(Document)


def test_build_tables():
    # Trigger table creation
    SQLModelManager.instance().create_tables()


def test_custom_table_name():
    table_name = "some_table.test"

    class Test(SQLModel):
        __model__ = table_name

    assert Test.objects.table.name == table_name


def test_sanity_pk_and_fields():
    class A(SQLModel):
        foo = Str()

    assert A.__pk__ == '_id'
    assert A.__fields__ == ['_id', 'foo']


def test_sanity_pk_override():
    class A(SQLModel):
        id = Int().tag(primary_key=True)
        foo = Str()

    assert A._id is A.id
    assert A.__pk__ == 'id'
    assert A.__fields__ == ['id', 'foo']


def test_sanity_pk_renamed():
    class A(SQLModel):
        id = Int().tag(primary_key=True, name="table_id")
        foo = Str()

    assert A._id is A.id
    assert A.__pk__ == 'table_id'
    assert A.__fields__ == ['id', 'foo']


def test_sanity_relation_exluded():
    class Child(SQLModel):
        pass

    class Parent(SQLModel):
        children = Relation(lambda: Child)

    assert "children" in Parent.__excluded_fields__


@pytest.mark.asyncio
async def test_sanity_flatten_unflatten():

    async def unflatten_date(v: str, scope=None):
        return datetime.strptime(v, "%Y-%m-%d").date()

    def flatten_date(v: date, scope=None):
        return v.strftime("%Y-%m-%d")

    class TableOfUnformattedGarbage(SQLModel):
        created = Instance(date).tag(
            type=sa.String(length=10),
            flatten=flatten_date,
            unflatten=unflatten_date)

    r = await TableOfUnformattedGarbage.restore({
        '__model__': TableOfUnformattedGarbage.__model__,
        '_id': 1,
        'created': '2020-10-28',
    })
    assert r.created == date(2020, 10, 28)
    assert r.__getstate__()['created'] == '2020-10-28'


def test_sanity_renamed_fields():
    class A(SQLModel):
        some_field = Str().tag(name="SomeField")

    class B(SQLModel):
        other_field = Str()

    A.__renamed_fields__ == {"some_field": "SomeField"}
    B.__renamed_fields__ == {"some_field": "SomeField"}


def test_table_subclass():
    # Test that a non-abstract table can be subclassed
    class Base(SQLModel):
        class Meta:
            abstract = True

    class A(Base):
        id = Int().tag(primary_key=True)

        class Meta:
            db_table = "test_a"

    assert A.__pk__ == 'id'
    assert A.__model__ == "test_a"
    assert A.__fields__ == ['id']

    class B(A):
        extra_col = Dict()

        class Meta:
            db_table = "test_b"

    assert B.__pk__ == 'id'
    assert B.__model__ == "test_b"
    assert B.__fields__ == ['id', 'extra_col']


async def reset_tables(*models):
    for Model in models:
        try:
            await Model.objects.drop_table()
        except Exception as e:
            msg = str(e)
            if not ("Unknown table" in msg or "does not exist" in msg):
                raise  # Unexpected error
        await Model.objects.create_table()


@pytest.fixture
async def db(event_loop):

    if DATABASE_URL.startswith("sqlite"):
        m = re.match(r"(.+)://(.+)", DATABASE_URL)
        assert m, "DATABASE_URL is an invalid format"
        schema, db = m.groups()
        params = dict(database=db)
    else:
        m = re.match(r"(.+)://(.+):(.*)@(.+):(\d+)/(.+)", DATABASE_URL)
        assert m, "DATABASE_URL is an invalid format"
        schema, user, pwd, host, port, db = m.groups()
        params = dict(host=host, port=int(port), user=user, password=pwd)

    if schema == "mysql":
        from aiomysql.sa import create_engine
        from aiomysql import connect
    elif schema == "postgres":
        from aiopg.sa import create_engine
        from aiopg import connect
    elif schema == "sqlite":
        from aiosqlite import connect

        create_engine = connect
    else:
        raise ValueError("Unsupported database schema: %s" % schema)

    if schema == "mysql":
        params["autocommit"] = True

    params["loop"] = event_loop

    if schema == "sqlite":
        if os.path.exists(db):
            os.remove(db)
    else:
        if schema == "postgres":
            params["database"] = "postgres"

        async with connect(**params) as conn:
            async with conn.cursor() as c:
                # WARNING: Not safe
                await c.execute("DROP DATABASE IF EXISTS %s;" % db)
                await c.execute("CREATE DATABASE %s;" % db)

    if schema == "mysql":
        params["db"] = db
    elif schema == "postgres":
        params["database"] = db

    async with create_engine(**params) as engine:
        mgr = SQLModelManager.instance()
        mgr.database = {"default": engine}
        yield engine


def test_query_ops_valid():
    """Test that operators are all valid"""
    from sqlalchemy.sql.expression import ColumnElement
    from atomdb.sql import QUERY_OPS

    for k, v in QUERY_OPS.items():
        assert hasattr(ColumnElement, v)


@pytest.mark.asyncio
async def test_drop_create_table(db):
    await reset_tables(User)


@pytest.mark.asyncio
async def test_simple_save_restore_delete(db):
    await reset_tables(User)

    # Save
    user = User(name=faker.name(), email=faker.email(), active=True)
    await user.delete()  # Deleting unsaved item does nothing
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
        user = User(
            name=f"name-{i}", email="email-{i}@example.com", age=20, active=True
        )
        await user.save()

    for user in await User.objects.all():
        print(user)

    for user in await User.objects.filter(name=user.name):
        print(user)

    assert await User.objects.filter(name=user.name).exists()
    assert not await User.objects.filter(name="I DO NOT EXIST").exists()

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
    assert await JobRole.objects.count(job__name__in=[job.name, job2.name]) == 2

    roles = await JobRole.objects.filter(job__name=job2.name)
    assert len(roles) == 1
    assert await JobRole.objects.count(job__name=job2.name) == 1

    roles = await JobRole.objects.filter(job=job2)
    assert len(roles) == 1 and roles[0] == role2

    roles = await JobRole.objects.filter(job__in=[job2])
    assert len(roles) == 1 and roles[0] == role2

    roles = await JobRole.objects.filter(job__name__not="none of the above")
    assert len(roles) == 3

    # Cant do multiple joins
    with pytest.raises(ValueError):
        roles = await JobRole.objects.get(job__name__other=1)


@pytest.mark.asyncio
async def test_query_related_reverse(db):
    await reset_tables(User, Job, JobRole)

    job = await Job.objects.create(name=faker.job())
    job1 = await Job.objects.create(name=faker.job())
    job2 = await Job.objects.create(name=faker.job())

    role = await JobRole.objects.create(job=job, name=faker.job())
    role1 = await JobRole.objects.create(job=job1, name=faker.job())
    role2 = await JobRole.objects.create(job=job2, name=faker.job())

    jobs = await Job.objects.filter(roles__name=role1.name)
    assert jobs == [job1]

    jobs = await Job.objects.filter(roles__in=[role, role2])
    assert jobs == [job, job2] or jobs == [job2, job]

    assert await Job.objects.filter(roles__in=[role, role2]).count() == 2


@pytest.mark.asyncio
async def test_query_order_by(db):
    await reset_tables(User)
    # Create second user
    users = []
    for i in range(3):
        user = User(name=faker.name(), email=faker.email(), age=i, active=True)
        await user.save()
        users.append(user)

    users.sort(key=lambda it: it.name)
    assert await User.objects.order_by("name").all() == users

    users.reverse()
    assert await User.objects.order_by("-name").all() == users


@pytest.mark.asyncio
async def test_query_limit(db):
    await reset_tables(User)
    # Create second user
    users = []
    for i in range(3):
        user = User(name=faker.name(), email=faker.email(), age=i, active=True)
        await user.save()
        users.append(user)

    assert len(await User.objects.limit(2).all()) == 2
    assert len(await User.objects.offset(2).all()) == 1

    assert len(await User.objects.filter()[1:2].all()) == 1
    assert len(await User.objects.filter()[1:].all()) == 2
    assert len(await User.objects.filter()[0].all()) == 1

    # Keys must be integers
    with pytest.raises(TypeError):
        User.objects.filter()[1.2]

    with pytest.raises(TypeError):
        User.objects.filter()[1.2:3]

    # No negative offests
    with pytest.raises(ValueError):
        User.objects.filter()[-1]

    # No negative limits
    with pytest.raises(ValueError):
        User.objects.filter()[0:-1]


@pytest.mark.asyncio
async def test_query_pk(db):
    await reset_tables(Ticket)
    t = await Ticket.objects.create(code="special")
    assert await Ticket.objects.get(code="special") is t


@pytest.mark.asyncio
async def test_query_subclassed_pk(db):
    await reset_tables(ImportedTicket)
    t = await ImportedTicket.objects.create(
        code="special",
        meta={"source": "db"}
    )
    assert await ImportedTicket.objects.get(code="special") is t


@pytest.mark.asyncio
async def test_query_renamed_pk(db):
    await reset_tables(Email)
    email = await Email.objects.create(
        to="bob@example.com",
        from_="alice@example.com",
        body="Hello ;)")
    assert await Email.objects.get(id=email._id) is email


@pytest.mark.asyncio
async def test_query_bad_column_name(db):
    await reset_tables(Ticket)
    t = await Ticket.objects.create(code="special")
    with pytest.raises(ValueError):
        await Ticket.objects.get(unknown="special")


@pytest.mark.asyncio
async def test_query_select_related(db):
    await reset_tables(Job, JobRole)
    # Create second user
    job = await Job.objects.create(name=faker.job())
    await JobRole.objects.create(job=job, name=faker.job())
    await JobRole.objects.create(name=faker.job())
    del job

    Job.objects.cache.clear()
    JobRole.objects.cache.clear()

    # Without select related it only has the Job with it's pk
    roles = await JobRole.objects.all()
    assert len(roles) == 2 and roles[0].job.__restored__ is False
    del roles

    # TODO: Shouldn't have to do this here...
    Job.objects.cache.clear()
    JobRole.objects.cache.clear()

    # With select related the job is fully loaded
    # since the second role does not set a job it is excluded due to the
    # default inner join
    roles = await JobRole.objects.select_related("job").all()
    assert len(roles) == 1 and roles[0].job.__restored__ is True

    # Using outer join includes related fields that are null
    roles = await JobRole.objects.select_related("job", outer_join=True).all()
    assert len(roles) == 2
    assert roles[0].job.__restored__ is True
    assert roles[1].job is None


@pytest.mark.asyncio
async def test_query_values(db):
    await reset_tables(User)
    # Create second user
    users = []
    user = User(name="Bob", email=faker.email(), age=40, active=True)
    await user.save()

    user1 = User(name="Jack", email=faker.email(), age=30, active=False)
    await user1.save()

    user2 = User(name="Bob", email=faker.email(), age=20, active=False)
    await user2.save()

    vals = await User.objects.filter(active=True).values()
    assert len(vals) == 1 and vals[0]["email"] == user.email

    assert await User.objects.order_by("name").values("name", distinct=True) == [
        ("Bob",),
        ("Jack",),
    ]

    assert await User.objects.order_by("age").values("age", flat=True) == [20, 30, 40]

    assert await User.objects.filter(active=True).values("age", flat=True) == [40]

    # Cannot use flat with multiple values
    with pytest.raises(ValueError):
        await User.objects.values("name", "age", flat=True)


@pytest.mark.skipif(IS_MYSQL, reason="Distinct and count doesn't work")
@pytest.mark.asyncio
async def test_query_distinct(db):
    await reset_tables(User)
    # Create second user
    users = []
    user = User(name="Bob", email=faker.email(), age=40, active=True)
    await user.save()

    user1 = User(name="Jack", email=faker.email(), age=30, active=False)
    await user1.save()

    user2 = User(name="Bob", email=faker.email(), age=20, active=False)
    await user2.save()

    num_names = await User.objects.distinct("name").count()
    assert num_names == 2
    distinct_names = (
        await User.objects.distinct("name").order_by("name").values("name", flat=True)
    )
    assert distinct_names == ["Bob", "Jack"]

    num_ages = await User.objects.distinct("age").count()
    assert num_ages == 3
    num_ages = await User.objects.filter(age__gt=25).distinct("age").count()
    assert num_ages == 2


@pytest.mark.asyncio
async def test_get_or_create(db):
    await reset_tables(User, Job, JobRole)

    name = faker.name()
    email = faker.email()

    user, created = await User.objects.get_or_create(name=name, email=email)
    assert created
    assert user._id and user.name == name and user.email == user.email

    u, created = await User.objects.get_or_create(name=user.name, email=user.email)
    assert u._id == user._id
    assert not created and user.name == name and user.email == user.email

    # Test passing model
    job, created = await Job.objects.get_or_create(name=faker.job())
    assert job and created

    role, created = await JobRole.objects.get_or_create(job=job, name=faker.job())
    assert role and created and role.job._id == job._id

    role_check, created = await JobRole.objects.get_or_create(job=job, name=role.name)
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
                    role = await JobRole.objects.create(
                        job=job, name=faker.job(), connection=conn
                    )
                    assert role._id is not None
                complete = True
                raise ValueError("Oh crap, I didn't want to do that")
            except Exception as e:
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
                role = await JobRole.objects.create(
                    job=job, name=faker.job(), connection=conn
                )
                assert role._id is not None
        except:
            await trans.rollback()
            raise
        else:
            await trans.commit()

    assert len(await Job.objects.all()) == 1
    assert len(await JobRole.objects.all()) == 3


@pytest.mark.asyncio
async def test_transaction_delete(db):
    await reset_tables(User)

    name = faker.name()
    async with User.objects.connection() as conn:
        trans = await conn.begin()
        try:
            # Must pass in the connection parameter for transactions
            user = await User.objects.create(
                name=name, email=faker.email(), age=20, active=True, connection=conn
            )
            assert user._id is not None
            await User.objects.delete(name=name, connection=conn)
        except:
            await trans.rollback()
            raise
        else:
            await trans.commit()

    assert not await User.objects.exists(name=name)


@pytest.mark.asyncio
async def test_filters(db):
    await reset_tables(User)

    user, created = await User.objects.get_or_create(
        name=faker.name(), email=faker.email(), age=21, active=True
    )
    assert created

    user2, created = await User.objects.get_or_create(
        name=faker.name(), email=faker.email(), age=48, active=False, rating=10.0
    )
    assert created

    # Startswith
    u = await User.objects.get(name__startswith=user.name[0])
    assert u.name == user.name
    assert u is user  # Now cached

    # In query
    users = await User.objects.filter(name__in=[user.name, user2.name])
    assert len(users) == 2

    # Test use of count
    assert await User.objects.count(name__in=[user.name, user2.name]) == 2

    # Is query
    users = await User.objects.filter(active__is=False)
    assert len(users) == 1 and users[0].active is False
    assert users[0] is user2  # Now cached

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

    # Invalid name
    with pytest.raises(ValueError):
        users = await User.objects.filter(does_not_exist=True)


@pytest.mark.asyncio
async def test_update(db):
    await reset_tables(User)
    # Create second user
    users = []
    user = User(name="Bob", email=faker.email(), age=40, active=True)
    await user.save()

    user1 = User(name="Jack", email=faker.email(), age=30, active=False)
    await user1.save()

    user2 = User(name="Bob", email=faker.email(), age=20, active=False)
    await user2.save()

    assert await User.objects.filter(age=20).exists()
    r = await User.objects.filter(age=20).update(age=25)
    assert not await User.objects.filter(age=20).exists()

    assert await User.objects.filter(active=False).exists()
    r = await User.objects.update(active=True)
    assert not await User.objects.filter(active=False).exists()

    assert await User.objects.filter(active=False).count() == 0
    r = await User.objects.filter(name="Bob").update(active=False)
    assert await User.objects.filter(active=False).count() == 2


@pytest.mark.asyncio
async def test_update_renamed(db):
    """Make sure update takes account for any renamed columns"""
    await reset_tables(User, Page)

    await Page.objects.create(title="Test1", status="live", ranking=100)
    await Page.objects.create(title="Test2", status="live", ranking=100)
    await Page.objects.create(title="Test3", status="live", ranking=1)

    assert await Page.objects.filter(ranking=100).count() == 2
    await Page.objects.filter(ranking=100).update(ranking=3)
    assert await Page.objects.filter(ranking=100).count() == 0
    assert await Page.objects.filter(ranking=3).count() == 2


@pytest.mark.asyncio
async def test_column_rename(db):
    """Columns can be tagged with custom names. Verify that it works."""
    await reset_tables(Email)

    e = Email(from_=faker.email(), to=faker.email(), body=faker.job())
    await e.save()

    # Check without use labels
    table = Email.objects.table
    q = table.select().where(table.c.to == e.to)
    row = await Email.objects.fetchone(q)
    assert row["from"] == e.from_, "Column rename failed"

    # Check with use labels
    q = table.select(use_labels=True).where(table.c.to == e.to)
    row = await Email.objects.fetchone(q)
    assert row[f"{table.name}_from"] == e.from_, "Column rename failed"

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
        assert role.job.__restored__ is True
        # for role in job.roles:
        #    assert role.job == job
        loaded.append(job)

    assert len(await Job.objects.fetchmany(q, size=2)) == 2

    # Make sure they pull from cache
    roles = await JobRole.objects.all()
    for role in roles:
        assert role.job is not None
        assert role.job.__restored__ is True

    # Clear cache and ensure it doesn't pull from cache now
    Job.objects.cache.clear()
    JobRole.objects.cache.clear()

    roles = await JobRole.objects.all()
    used = set()
    for role in roles:
        assert role.job is not None
        if role.job not in used:
            assert role.job.__restored__ is False
            used.add(role.job)
        await role.job.load()
        assert role.job.__restored__ is True


@pytest.mark.asyncio
async def test_query_multiple_joins(db):
    await reset_tables(Job, JobRole, JobTask)

    ceo = await Job.objects.create(name="CEO")
    cfo = await Job.objects.create(name="CFO")
    swe = await Job.objects.create(name="SWE")

    ceo_role = await JobRole.objects.create(name="CEO", job=ceo)
    cfo_role = await JobRole.objects.create(name="CFO", job=cfo)
    swe_role = await JobRole.objects.create(name="SWE", job=swe)

    code_task = await JobTask.objects.create(desc="Code", role=swe_role)
    hire_task = await JobTask.objects.create(desc="Hire", role=ceo_role)
    fire_task = await JobTask.objects.create(desc="Fire", role=ceo_role)
    account_task = await JobTask.objects.create(desc="Account", role=cfo_role)

    jobs = await Job.objects.filter(roles__tasks__desc="Fire")
    assert jobs == [ceo]

    jobs = await Job.objects.order_by("name").filter(
        roles__tasks__desc__notin=["Hire", "Fire"])
    assert jobs == [cfo, swe]



@pytest.mark.asyncio
async def test_save_update_fields(db):
    """Test that using save with update_fields only updates the fields
    specified

    """
    await reset_tables(User)
    await reset_tables(Page)

    page = await Page.objects.create(
        title="Test", body="This is only a test", status="live"
    )
    assert page.visits == 0
    page.visits += 1
    page.body = "New body"
    await page.save(update_fields=["visits"])

    del Page.objects.cache[page._id]
    page = await Page.objects.get(title="Test")

    # This field should not be saved
    assert page.body == "This is only a test"
    # But this should be saved
    assert page.visits == 1


@pytest.mark.asyncio
async def test_load_fields(db):
    """Test that using load with fields only loads the given field"""
    await reset_tables(User)
    await reset_tables(Page)

    page = await Page.objects.create(
        title="Test", body="This is only a test", status="live"
    )
    assert page.visits == 0

    # Update outside the orm
    t = Page.objects.table
    q = t.update().where(t.c._id == page._id).values(visits=1288821, title="New title")
    async with Page.objects.connection() as conn:
        await conn.execute(q)

    page.body = "This has changed"

    # Reload the visits
    await page.load(fields=["visits"])

    # This should be the only field that updates
    assert page.visits == 1288821

    # This should not change
    assert page.title == "Test"
    assert page.body == "This has changed"

    # Reload the title
    await page.load(fields=["title", "visits"])
    assert page.title == "New title"


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
    assert aref is e, "Cached object is invalid"

    # Delete
    del e
    del aref

    gc.collect()

    # Make sure cache was cleaned up
    aref = Email.objects.cache.get(pk)
    assert aref is None, "Cached object was not released"


@pytest.mark.asyncio
async def test_fk_custom_type(db):
    await reset_tables(Document, Project)
    doc = await Document.objects.create(uuid="foo")
    project = await Project.objects.create(doc=doc)
    col = Project.objects.table.columns['doc']
    assert isinstance(col.type, sa.String)


def test_invalid_meta_field():
    with pytest.raises(TypeError):

        class TestTable(SQLModel):
            id = Int().tag(primary_key=True)

            class Meta:
                # table_name is invalid, use db_table
                table_name = "use db_table"


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
            db_table = "custom_user"

    class AbstractCustomUser(AbstractUser):
        data = Dict()

        class Meta(AbstractUser.Meta):
            db_table = "custom_user2"

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
