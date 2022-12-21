import pytest
from test_sql import Image, Page, db, reset_tables

assert db  # fix flake8


@pytest.mark.benchmark(group="sql-create")
def test_benchmark_create(db, event_loop, benchmark):
    event_loop.run_until_complete(reset_tables(Image))

    async def create_images():
        for i in range(100):
            await Image.objects.create(
                name=f"Image {i}",
                path=f"/media/some/path/{i}",
                alpha=i % 255,
                # size=(320, 240),
                data=b"12345678",
                metadata={"tag": "sunset"},
            )

    def run():
        event_loop.run_until_complete(create_images())

    benchmark(run)


def prepare_benchmark(event_loop, n: int):
    event_loop.run_until_complete(reset_tables(Image))
    images = []

    async def make():
        images.append(
            await Image.objects.create(
                name=f"Image {i}",
                path=f"/media/some/path/{i}",
                alpha=i % 255,
                # size=(320, 240),
                data=b"12345678",
                metadata={"tag": "sunset"},
            )
        )

    for i in range(n):
        event_loop.run_until_complete(make())

    return images


@pytest.mark.benchmark(group="sql-get")
def test_benchmark_get(db, event_loop, benchmark):
    """Do a normal get query where the item is not in the cache"""
    prepare_benchmark(event_loop, n=1)
    Image.objects.cache.clear()

    async def task():
        image = await Image.objects.get(name="Image 0")
        assert image.name == "Image 0"

    benchmark(lambda: event_loop.run_until_complete(task()))


@pytest.mark.benchmark(group="sql-get")
def test_benchmark_get_cached(db, event_loop, benchmark):
    """Do a normal get query where the item is in the cache"""
    images = prepare_benchmark(event_loop, n=1)
    assert images

    async def task():
        image = await Image.objects.get(name="Image 0")
        assert image.name == "Image 0"

    benchmark(lambda: event_loop.run_until_complete(task()))


@pytest.mark.benchmark(group="sql-get")
def test_benchmark_get_raw(db, event_loop, benchmark):
    """Do a prebuilt get query with restoring without cache"""
    prepare_benchmark(event_loop, n=1)
    q = Image.objects.filter(name="Image 0").query("select")
    Image.objects.cache.clear()

    async def task():
        async with Image.objects.connection() as conn:
            cursor = await conn.execute(q)
            row = await cursor.fetchone()
            image = await Image.restore(row)
            assert image.name == "Image 0"

    benchmark(lambda: event_loop.run_until_complete(task()))


@pytest.mark.benchmark(group="sql-get")
def test_benchmark_get_raw_str(db, event_loop, benchmark):
    """Do a prebuilt get query with restoring without cache"""
    images = prepare_benchmark(event_loop, n=1)
    assert images
    q = str(Image.objects.filter(name="Image 0").query("select"))

    async def task():
        async with Image.objects.connection() as conn:
            cursor = await conn.execute(q, {"name_1": "Image 0"})
            row = await cursor.fetchone()
            image = await Image.restore(row)
            assert image.name == "Image 0"

    benchmark(lambda: event_loop.run_until_complete(task()))


@pytest.mark.benchmark(group="sql-get")
def test_benchmark_get_raw_cached(db, event_loop, benchmark):
    """Do a prebuilt get query with restoring with cache"""
    images = prepare_benchmark(event_loop, n=1)
    assert images
    q = Image.objects.filter(name="Image 0").query("select")
    # Image.objects.cache.clear()

    async def task():
        async with Image.objects.connection() as conn:
            cursor = await conn.execute(q)
            row = await cursor.fetchone()
            image = await Image.restore(row)
            assert image.name == "Image 0"

    benchmark(lambda: event_loop.run_until_complete(task()))


@pytest.mark.benchmark(group="sql-get")
def test_benchmark_get_raw_row(db, event_loop, benchmark):
    """Do a prebuilt get query without restoring"""
    prepare_benchmark(event_loop, n=1)
    q = Image.objects.filter(name="Image 0").query("select")

    async def task():
        async with Image.objects.connection() as conn:
            cursor = await conn.execute(q)
            row = await cursor.fetchone()
            assert row["name"] == "Image 0"
            # No restore

    benchmark(lambda: event_loop.run_until_complete(task()))


@pytest.mark.benchmark(group="sql-filter")
def test_benchmark_filter(db, event_loop, benchmark):
    """Do a filter query where no items are in the cache"""
    prepare_benchmark(event_loop, n=1000)
    Image.objects.cache.clear()

    async def task():
        results = await Image.objects.filter(alpha__ne=0)
        assert len(results) == 996

    benchmark(lambda: event_loop.run_until_complete(task()))


@pytest.mark.benchmark(group="sql-filter")
def test_benchmark_filter_cached(db, event_loop, benchmark):
    """Do a filter query where all items are in the cache"""
    images = prepare_benchmark(event_loop, n=1000)
    assert images

    async def task():
        results = await Image.objects.filter(alpha__ne=0)
        assert len(results) == 996

    benchmark(lambda: event_loop.run_until_complete(task()))


@pytest.mark.benchmark(group="sql-filter")
def test_benchmark_filter_raw(db, event_loop, benchmark):
    """Do a raw filter query where no items are in the cache"""
    prepare_benchmark(event_loop, n=1000)
    Image.objects.cache.clear()
    q = Image.objects.filter(alpha__ne=0).query("select")

    async def task():
        async with Image.objects.connection() as conn:
            cursor = await conn.execute(q)
            results = [await Image.restore(row) for row in await cursor.fetchall()]
            assert len(results) == 996

    benchmark(lambda: event_loop.run_until_complete(task()))


@pytest.mark.benchmark(group="sql-filter")
def test_benchmark_filter_raw_cached(db, event_loop, benchmark):
    """Do a raw filter query where all items are in the cache"""
    images = prepare_benchmark(event_loop, n=1000)
    assert images
    # Image.objects.cache.clear()
    q = Image.objects.filter(alpha__ne=0).query("select")

    async def task():
        async with Image.objects.connection() as conn:
            cursor = await conn.execute(q)
            results = [await Image.restore(row) for row in await cursor.fetchall()]
            assert len(results) == 996

    benchmark(lambda: event_loop.run_until_complete(task()))


@pytest.mark.benchmark(group="sql-filter")
def test_benchmark_filter_raw_row(db, event_loop, benchmark):
    """Do a raw filter query without restoring item from rows"""
    prepare_benchmark(event_loop, n=1000)
    # Image.objects.cache.clear()
    q = Image.objects.filter(alpha__ne=0).query("select")

    async def task():
        async with Image.objects.connection() as conn:
            cursor = await conn.execute(q)
            # No restore
            results = [row for row in await cursor.fetchall()]
            assert len(results) == 996

    benchmark(lambda: event_loop.run_until_complete(task()))


@pytest.mark.benchmark(group="sql-build-query")
def test_benchmark_filter_related_query(db, benchmark):
    def query():
        Page.objects.filter(author__name="Tom", status="live")

    benchmark(query)


@pytest.mark.benchmark(group="sql-build-query")
def test_benchmark_filter_query(db, benchmark):
    def query():
        Page.objects.filter(status="live")

    benchmark(query)


@pytest.mark.benchmark(group="sql-build-query")
def test_benchmark_filter_query_ordered(db, benchmark):
    def query():
        Page.objects.filter(status="live").order_by("last_updated")

    benchmark(query)
