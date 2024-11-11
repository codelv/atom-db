# 0.8.1

- Rebase changes from  0.7.10 and 0.7.11
- Rework Relation & RelatedList so the return value is still a list instance
- Flatten builtin enum.Enum types to their value
- Add first, last, earliest, latest to QuerySet and support Meta get_latest_by
- Support using `_id` as alias to the primary key when doing filtering

# 0.8.0

- **breaking** Make `Relation` return a RelatedList with `save` and `load` methods.
- Don't rewrite bytecode
- Pass onclause when using join to prevent sqlalchemy from picking incorrect relation
- Fix select_related with duplicate joins (eg `select_related('a', 'a__b')`)
- Change Enum database name to include the table name
- Add builtin set / tuple support for JSONModel

# 0.7.11

- Support doing an or filter using django style queries by passing a dict arg

# 0.7.10

- Add group by


# 0.7.9

- Fix error with 3.11

# 0.7.8

- Fix problem with flatten/unflatten creating invalid columns.

# 0.7.7

- Return from bulk_create if values list is empty
- Fix problem with order_by not having __bool__ defined

# 0.7.6

- Change internal data types to set to speed up building queries and allow caching
- Add a `bulk_create` method to the model manager.
- Add `py.typed` to package

# 0.7.5

- Fix a bug preventing select_related on multiple fields from working
- Add django style `exclude` to filter

# 0.7.4

- Do not save Property members by default
- Annotate Model `objects` and `serializer` with `ClassVar`
- Change import sorting and cleanup errors found with flake8

# 0.7.3

- Revert force restore items from direct query even if in the cache.
Now queries can accept a `force_restore=True` to do this.
See https://en.wikipedia.org/wiki/Isolation_%28database_systems%29

# 0.7.2

- Support prefetching of one to one "Related" members.
- Remove _id field in base Model and JSONModel as it has no meaning there

# 0.7.1

- Always force restore items from direct query even if in the cache
- Make prefetch use parent queries connection

# 0.7.0

- Use generated functions to speed up save and restore
- BREAKING: To save memory (by avoiding overriding members) the `_id` and `__ref__` fields
   were changed to an `Int`.

# 0.6.4

- Fix queries joining through multiple tables
- Add initial implementation of prefetch_related

# 0.6.3

- Add workaround for subclassed pk handling

# 0.6.2

- Add support for using multiple databases
- Fix non-abstract subclasses throwing multiple primary key error.
- Make update work with renamed fields

# 0.6.1

- Merge `composite_indexes` and typing branches

# 0.6.0

- Add type hints
- Drop python 3.6 support
- Fix bug with renamed fields excluded fields

# 0.5.8

- Add `composite_indexes` to Model Meta.

# 0.5.7

- Add `distinct` to queryset

# 0.5.6

- Add `outer_join` to queryset to allow using a left outer join with select related

# 0.5.5

- Add builtin JSON serializer for `UUID`


# 0.5.4

- Add builtin JSON serializer for `Decimal`

# 0.5.3

- Add field types for `Decimal` and `timedelta`
- Fix bug with enum field name on postgres
- Fix array field with instance child types
- Add support for `update_fields` on save to only fields specified
- Add support for `fields` on load to only load fields specified


# 0.5.2

- Add support for table database `triggers`. See https://docs.sqlalchemy.org/en/14/core/ddl.html
- Fix bug in create_table where errors are not raised

# 0.5.1

- Add `update` method using `Model.objects.update(**values)`

# 0.5.0

- Replace usage of Unicode with Str to support atom 0.6.0

# 0.4.1

- Change order by to use `-` as desc instead of `~`
- Add default constraint naming conventions https://alembic.sqlalchemy.org/en/latest/naming.html#the-importance-of-naming-constraints
- Allow setting a `constraints` list on the Model `Meta` class
- Fix issue with `connection` arg not working properly when filtering

# 0.4.0

- Refactor SQL queries so they can be chained
    ex `Model.objects.filter(name="Something").filter(age__gt=18)`
- Add `order_by`, `limit`, and `offset`, slicing, and `exists`
- Support filtering using django-style reverse foreign key lookups,
    ex `Model.objects.filter(groups_in=[group1, group2])`
- Refactor count to support counting over joins

# 0.3.11

- Let a member be tagged with a custom `flatten` function

# 0.3.10

- Fix bug in SQLModel load using `_id` which is not a valid field

# 0.3.9

- Let a member be tagged with a custom `unflatten` function

# 0.3.8
- Properly restore JSONModel instances that have no `__model__` in the dict
when migrated from a dict or regular JSON field.

# 0.3.7

- Add a `__restored__` member to models and a `load` method so foreign keys
do not restore as None if not in the cache.
- Update to support atom 0.5.0


# 0.3.6

- Add `cache` option to SQLModelManager to determine if restoring
should always be done even if the object is in the cache.

# 0.3.5

- Set column type to json if the type is a JSONModel subclass

# 0.3.4

- Fix bug when saving using a generated id

# 0.3.3

- Change __setstate__ to __restorestate__ to not conflict with normal pickleing

# 0.3.2

- Support lookups on foreign key fields
- Add ability to specify `get_column` and `get_column_type` to let `atom.catom.Member`
    subclasses use custom sql columns if needed.

# 0.3.1

- Support lookups using renamed column fields

# 0.3.0

- The create and drop have been renamed to `create_table` and `drop_table` respectively.
- Add a shortcut `SomeModel.object.create(**state)` method
- Allow passing a db connection to manager methods (
    get, get_or_create, filter, delete, etc...) to better support transactions


# 0.2.4

- Fix the nosql serialization registry not being loaded properly

# 0.2.3

- Fix packaging issue with 0.2.2

# 0.2.2

- Fix bug with fk types
- Allow passing Model instances as filter parameters #8 by @youpsla

# 0.2.1

- Add a JSONModel that simply can be serialized and restored using JSON.

# 0.2.1

- Add ability to set an SQL model as `abstract` so that no sqlalchemy table is
created.

# 0.1.0

- Initial release
