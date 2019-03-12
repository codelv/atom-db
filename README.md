[![Build Status](https://travis-ci.org/codelv/atom-db.svg?branch=master)](https://travis-ci.org/codelv/atom-db)
[![codecov](https://codecov.io/gh/codelv/atom-db/branch/master/graph/badge.svg)](https://codecov.io/gh/codelv/atom-db)

atom-db is a database abstraction layer for the
[atom](https://github.com/nucleic/atom) framework. This package provides api's for
seamlessly saving and restoring atom objects from json based document databases
and (coming soon) SQL databases supported by sqlalchemy.


### Why?

The main reason for building this is to make it easier have database integration
with [enaml](https://github.com/nucleic/enaml) applications.  Without this,
a separate framework is needed to define database models, which is a
duplication of work.

This was originally a part of [enaml-web](https://github.com/codelv/enaml-web)
but has been pulled out to a separate package.


### Structure

The design is based somewhat on django. Using `Model.objects` retrieves a
manager for that type of object which can be used to create queries. No
restriction is imposed on what type of manager is used, leaving that to
whichever database library is preferred (ex motor, txmongo, sqlalchemy,...).

In addition to `Model.objects` a serializer is added to each class as
`Model.serializer` which is used to serialize and deserialize the objects
to and from the database.


### Example using MongoDB and motor

Just define models using atom members, but subclass the NoSQLModel.

```python

from atom.api import Unicode, Int, Instance, List
from atomdb.nosql import NoSQLModel, NoSQLModelManager
from motor.motor_asyncio import AsyncIOMotorClient

# Set DB
client = AsyncIOMotorClient()
mgr = NoSQLModelManager.instance()
mgr.database = client.test_db


class Group(NoSQLModel):
    name = Unicode()

class User(NoSQLModel):
    name = Unicode()
    age = Int()
    groups = List(Group)


```

Then we can create an instance and save it. It will perform an upsert or replace
the existing entry.

```python

admins = Group(name="Admins")
await admins.save()

# It will save admins using it's ObjectID
bob = User(name="Bob", age=32, groups=[admins])
await bob.save()

tom = User(name="Tom", age=34, groups=[admins])
await tom.save()

```

To fetch from the DB each model has a `ModelManager` called `objects` that will
simply return the collection for the model type. For example.

```python

# Fetch from db, you can use any MongoDB queries here
state = await User.objects.find_one({'name': "James"})
if state:
    james = await User.restore(state)

# etc...
```

Restoring is async because it will automatically fetch any related objects
(ex the groups in this case). It saves objects using the ObjectID when present.

And finally you can either delete using queries on the manager directly or
call on the object.

```python
await tom.delete()
assert not await User.objects.find_one({'name': "Tom"})

```

You can exclude members from being saved to the DB by tagging them
with `.tag(store=False)`.


### SQL with aiomysql / aiopg

Just define models using atom members, but subclass the SQLModel.

Tag members with information needed for sqlalchemy tables, ex
`Str().tag(length=40)` will make a `sa.String(40)`.
See https://docs.sqlalchemy.org/en/latest/core/type_basics.html. Tagging with
`store=False` will make the member be excluded from the db.

atomdb will attempt to determine the proper column type, but if you need more
control, you can tag the member to specify the column type with
`type=sa.<type>` or specify the full column definition with
`column=sa.Column(...)`.  See the tests for examples.


#### Table creation / dropping

Once your tables are defined as atom models, create and drop tables using the
async wrappers on top of sqlalchemy's engine.

```python

from atomdb.sql import SQLModel, SQLModelManager

# Call create_tables to create sqlalchemy tables. This does NOT write them to
# the db but ensures that all ForeignKey relations are created
SQLModelManager.instance().create_tables()

# Now actually drop/create for each of your models

# Drop the table for this model (will raise sqlalchemy's error if it doesn't exist)
await User.objects.drop()

# Create the user table
await User.objects.create()


```


#### ORM like queries

Only very basic ORM style queries are implemented for common use cases. These
are `get`, `get_or_create`, `filter`, and `all`. These all accept
"django style" queries using `<name>=<value>` or `<name>__<op>=<value>`.

For example:

```python

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

# In query
users = await User.objects.filter(name__in=[user.name, user2.name])
assert len(users) == 2

# Is query
users = await User.objects.filter(active__is=False)
assert len(users) == 1 and users[0].active == False

```

See [sqlachemy's ColumnElement](https://docs.sqlalchemy.org/en/latest/core/sqlelement.html?highlight=column#sqlalchemy.sql.expression.ColumnElement)
for which queries can be used in this way.  Also the tests check that these
actually work as intended.


#### Advanced / raw queries

For more advanced queries using joins, etc.. you must build the query with
sqlalchemy then execute it. The `sa.Table` for an atom model can be retrieved
using `Model.objects.table` on which you can use select, where, etc... to build
up whatever query you need.

Then use `fetchall`, `fetchone`, `fetchmany`, or `execute` to do the query.

These methods do NOT return an object but the row from the database so they
must manually be restored.

When joining you'll usually want to pass `use_labels=True`.  For example:

```python

q = Job.objects.table.join(JobRole.objects.table).select(use_labels=True)

for row in await Job.objects.fetchall(q):
    # Restore each manually, it handles pulling out the fields that are it's own
    job = await Job.restore(row)
    role = await JobRole.restore(row)

```

Depending on the relationships, you may need to then post-process these so they
can be accessed in a more pythonic way. This is trade off between complexity
and ease of use.


### Contributing

This is early in development and may have issues. Pull requests,
feature requests, are welcome!
