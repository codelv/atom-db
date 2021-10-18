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
