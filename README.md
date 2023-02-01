[![status](https://github.com/codelv/atom-db/actions/workflows/ci.yml/badge.svg)](https://github.com/codelv/atom-db/actions)
[![codecov](https://codecov.io/gh/codelv/atom-db/branch/master/graph/badge.svg)](https://codecov.io/gh/codelv/atom-db)

atom-db is a database abstraction layer for the
[atom](https://github.com/nucleic/atom) framework. This package provides api's for
seamlessly saving and restoring atom objects from json based document databases
and SQL databases supported by sqlalchemy.


The main reason for building this is to make it easier have database integration
with [enaml](https://github.com/nucleic/enaml) applications so a separate
framework is not needed to define database models.

This was originally a part of [enaml-web](https://github.com/codelv/enaml-web)
but has been pulled out to a separate package.


### Overview

- Supports MySQL and Postgres
- Uses django like queries or raw sqlalchemy queries
- Works with alembic database migrations
- Supports MongoDB using motor

### Structure

The design is based somewhat on django.

There is a "manager" called `Model.objects` to do queries on the database table
created for each subclass.

Serialization and deserialization is done with `Model.serializer`.

> Note: As of 0.3.11 serialization can be customizer per member by tagging the
member with a `flatten` or `unflatten` which should be a async callable which
accepts the value and scope.

Each `Model` has async `save`, `delete`, and `restore` methods to interact with
the database. This can be customized if needed using
`__restorestate__` and `__getstate__`.


# MySQL and Postgres support

You can use atom-db to save and restore atom subclasses to MySQL and Postgres.

Just define models using atom members, but subclass the `SQLModel` and atom-db
will convert the builtin atom members of your model to sqlalchemy table columns
and create a `sqlalchemy.Table` for your model.


### Customizing table creation

To customize how table columns are created you can tag members with information
needed for sqlalchemy columns, ex `Str().tag(length=40)` will make a `sa.String(40)`.
See https://docs.sqlalchemy.org/en/latest/core/type_basics.html. Tagging any
member with `store=False` will make the member be excluded from the db.

atomdb will attempt to determine the proper column type, but if you need more
control, you can tag the member to specify the column type with
`type=sa.<type>` or specify the full column definition with
`column=sa.Column(...)`.

If you have a custom member, you can define a `def get_column(self, model)`
or `def get_column_type(self, model)` method to create the table column for the
given model.


##### Primary keys

You can tag a member with `primary_key=True` to make it the pk. If no member
is tagged with `primary_key` it will create and use `_id` as the primary key.
The`_id` member will be always alias to the actual primary key. Use the `__pk__`
attribute of the class to get the name of the primary key member.

##### Table metadata

Like in Django a nested `Meta` class  can be added to specify the `db_name`,
`unique_together`, `composite indexes` and `constraints`.

If no `db_name` is specified on a Meta class, the table name defaults the what
is set in the `__model__` member. This defaults to the qualname of the class,
eg `myapp.SomeModel`.


```python

class SomeModel(SQLModel):
    # ...

    class Meta:
        db_table = 'custom_table_name'

```



`composite indexes` must be a list of each composite index description. See [sqlachemy's Index](https://docs.sqlalchemy.org/en/14/core/constraints.html#sqlalchemy.schema.Index) for index tuple description.

First element is the index name. If `None`, the name will be auto-generated according the convention. Followings elements are columns table. Order of columns matters.

```python

class SomeModel(SQLModel):
    # ...

    class Meta:
        composite_indexes = [(None, 'a', 'b'), ('ìndex_b_c', 'b', 'c')]

```

In the exemple above, two composite indexes are created. The first is on columns 'a' and 'b' with name `ix_SomeModel_a_b`. The second one on columns 'b' and 'c' with name `ìndex_b_c`.


##### Table creation / dropping

Once your tables are defined as atom models, create and drop tables using
 `create_table` and `drop_table` of `Model.objects` respectively For example:

```python

from atomdb.sql import SQLModel, SQLModelManager

# Call create_tables to create sqlalchemy tables. This does NOT write them to
# the db but ensures that all ForeignKey relations are created
mgr = SQLModelManager.instance()
mgr.create_tables()

# Now actually drop/create for each of your models

# Drop the table for this model (will raise sqlalchemy's error if it doesn't exist)
await User.objects.drop_table()

# Create the user table
await User.objects.create_table()


```

The `mgr.create_tables()` method will create the sqlalchemy tables for each
imported SQLModel subclass (anything in the manager's `registry` dict). This
should be called after all of your models are imported so sqlalchemy can
properly setup any foreign key relations.

The manager also has a `metadata` member which holds the `sqlalchemy.MetaData`
needed for migrations.

Once the tables are created, they are accessible via `Model.objects.table`.

> Note: The sqlachemy table is also assigned to the `__table__` attribute of
each model class, however this will not be defined until the manager has
created it.


#### Database setup

Before accessing the DB you must assign a "database engine" to the manager's
`database` member.

> Note: As of `0.6.2` you can also specify this as a dictionary to use multiple
databases.

```python
import os
import re
from aiomysql.sa import create_engine
from atomdb.sql import SQLModelManager

DATABASE_URL = os.environ.get('MYSQL_URL')

# Parse the DB url
m = re.match(r'mysql://(.+):(.*)@(.+):(\d+)/(.+)', DATABASE_URL)
user, pwd, host, port, db = m.groups()

# Create the engine
engine = await create_engine(
    db=db, user=user, password=pwd, host=host, port=port)

# Assign it to the manager
mgr = SQLModelManager.instance()
mgr.database = engine


```

This engine will then be used by the manager to execute queries.  You can
retrieve the database engine from any Model by using `Model.objects.engine`.


###### Multiple database

If you need to use more than one database it looks like this.

```python

# Multiple databases
mgr = SQLModelManager.instance()
mgr.database = {
    'default': await create_engine(**default_db_params),
    'other': await create_engine(**other_db_params),
}

```

To specify which database is used either using the `__database__` class field
or specify it as the `db_name` on the model Meta.

```python

class ExternalData(SQLModel):

    # ... fields
    class Meta:
        db_name = 'other'
        db_table = 'external_data'


```


#### Django style queries

Only very basic ORM style queries are implemented for common use cases. These
are `get`, `get_or_create`, `filter`, and `all`. These all accept
"django style" queries using `<name>=<value>` or `<name>__<op>=<value>`.

For example:

```python

john, created = await User.objects.get_or_create(
        name="John Doe", email="jon@example.com", age=21, active=True)
assert created

jane, created = await User.objects.get_or_create(
        name="Jane Doe", email="jane@example.com", age=48, active=False,
        rating=10.0)
assert created

# Startswith
u = await User.objects.get(name__startswith="John")
assert u.name == john.name

# In query
users = await User.objects.filter(name__in=[john.name, jane.name])
assert len(users) == 2

# Is query
users = await User.objects.filter(active__is=False)
assert len(users) == 1 and users[0].active == False

```

See [sqlachemy's ColumnElement](https://docs.sqlalchemy.org/en/latest/core/sqlelement.html?highlight=column#sqlalchemy.sql.expression.ColumnElement)
for which queries can be used in this way.  Also the tests check that these
actually work as intended.

> Note: As of `0.4.0` you can pass sqlalchemy filters as non-keyword arguments
directly to the filter method.


###### Caching, select related, and prefetch related

Foreign key relations can automatically be loaded using `select_related` and
`prefetch_related`. Select related will perform a
join while prefetch related does a separate query.

Each Model has a cache available at `Model.objects.cache` which uses weakrefs to
ensure the same object is returned each time. You can manually prefetch objects
and atom-db will pull them from it's internal cache when restoring objects.

For example with a simple many to one relationship like this:

```python

class Category(SQLModel):
    name = Str()
    products = Relation(lambda: Product)

class Product(SQLModel):
    title = Str()
    category = Typed(Category)

category = await Category.objects.create(name="PCB")
await Product.objects.create(title="Stepper driver", category=category)

```

Use select related to load the product's category foreign key automatically.

```python
# In this case the category of each product will automatically be loaded
products = await Product.objects.select_related('category').filter(title__icontains="driver")
# The __restored__ flag can be used check if the model has been loaded
assert products[0].category.name == "PCB"
```

> If a foreign key relation is NOT in the cache or in the state from a joined row
it will create an "unloaded" model with only the primary key populated. In this
case the `__restored__` flag will be set to `False`.

From the other direction use prefetch related.

```python
category = await Category.objects.prefetch_related('products').get(name="PCB")
assert category.products[0].title == "Stepper driver"
```

> Note: prefetch_related does not apply a limit. If the query has a lot of rows
this may be a problem.

Alternatively you can prefetch the related objects and they will be
automatically pulled from the internal cache (eg `TheModel.objects.cache`).

```python
all_categories = await Category.objects.all()
products = await Product.objects.filter(title__icontains="driver")
assert products[0].category in all_categories
```


#### Advanced / raw sqlalchemy queries

For more advanced queries using joins, etc.. you must build the query with
sqlalchemy then execute it. The `sa.Table` for an atom model can be retrieved
using `Model.objects.table` on which you can use select, where, etc... to build
up whatever query you need.

Then use `fetchall`, `fetchone`, `fetchmany`, or `execute` to do these queries.

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


### Connections and Transactions

A connection can be retrieved using `Model.objects.connection()` and used
like normal aiomysql / aiopg connection. A transaction is done in the same way
as defined in the docs for those libraries eg.

```python

async with Job.objects.connection() as conn:
    trans = await conn.begin()
    try:
        # Do your queries here and pass the `connection` to each
        job, created = await Job.objects.get_or_create(connection=conn, **state)
    except:
        await trans.rollback()
        raise
    else:
        await trans.commit()

```

When using a transaction you need to pass the active connection to
each call or it will use a different connection outside of the transaction!

The connection argument is removed from the filters/state. If your model happens
to have a member named `connection` you can rename the connection argument by
with `Model.object.connection_kwarg = 'connection_'` or whatever name you like.

### Migrations

Migrations work using [alembic](https://alembic.sqlalchemy.org/en/latest/autogenerate.html). The metadata needed
to autogenerate migrations can be retrieved from `SQLModelManager.instance().metadata` so add the following
in your alembic's env.py:

```python
# Import your db models first
from myapp.models import *

from atomdb.sql import SQLModelManager
manager = SQLModelManager.instance()
manager.create_tables()  # Create sa tables
target_metadata = manager.metadata

```

The rest is handled by alembic.


> Note: As of 0.4.1 the constraint naming conventions can be set using
manager.constraints, this must be done before any tables are imported.



# NoSQL support

You can also use atom-db to save and restore atom subclasses to MongoDB.

The NoSQL version is very basic as mongo is much more relaxed. No restriction
is imposed on what type of manager is used, leaving that to whichever database
library is preferred but it's tested (and currently used) with [motor](https://motor.readthedocs.io/en/stable/)
and [tornado](https://www.tornadoweb.org/en/stable/index.html).

Just define models using atom members, but subclass the `NoSQLModel`.

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


## Contributing

This is currently used in a few projects but not considered mature by
any means.

Pull requests and feature requests are welcome!
