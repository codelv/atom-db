
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
