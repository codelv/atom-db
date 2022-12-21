from datetime import datetime

import pytest
from atom.api import Bool, Dict, Float, Int, List, Str, Typed

from atomdb.base import JSONModel, Model

NOW = datetime.now()

flat_state = dict(
    title="This is a test",
    desc="""Lorem ipsum dolor sit amet, consectetur adipiscing elit,
    sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
    Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris
    nisi ut aliquip ex ea commodo consequat.
    Duis aute irure dolor in reprehenderit in voluptate velit esse cillum
    """,
    enabled=True,
    rating=8.3,
    sku=4567899,
    datetime=NOW.timestamp(),
    tags=["electronics", "laptop"],
    meta={"views": 0},
)

nested_state = {
    "__model__": "test_benchmark.Page",
    "blocks": [
        {"__model__": "test_benchmark.HeadingBlock", "text": "Main", "width": 0},
        {
            "__model__": "test_benchmark.MarkdownBlock",
            "content": "![Home](/)",
            "width": 0,
        },
    ],
    "enabled": True,
    "settings": {
        "__model__": "test_benchmark.PageSettings",
        "count": 100,
        "meta": "Lorem ipsum dolor",
    },
    "title": "Hello world",
}


class Product(Model):
    title = Str()
    desc = Str()
    enabled = Bool()
    rating = Float()
    sku = Int()
    tags = List(str)
    meta = Dict()
    created = Typed(datetime, factory=datetime.now).tag(
        flatten=lambda v, scope: v.timestamp() if v else None,
        unflatten=lambda v, scope: datetime.fromtimestamp(v) if v else None,
    )


class Block(JSONModel):
    width = Int()


class HeadingBlock(Block):
    text = Str()


class MarkdownBlock(Block):
    content = Str()


class PageSettings(JSONModel):
    meta = Str()
    count = Int()


class Page(JSONModel):
    title = Str()
    enabled = Bool()
    blocks = List(Block)
    settings = Typed(PageSettings, ())


@pytest.mark.benchmark(group="base")
def test_serialize_flat(benchmark):
    product = Product(
        title="This is a test",
        desc="""Lorem ipsum dolor sit amet, consectetur adipiscing elit,
        sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
        Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris
        nisi ut aliquip ex ea commodo consequat.
        Duis aute irure dolor in reprehenderit in voluptate velit esse cillum
        """,
        enabled=True,
        rating=8.3,
        created=NOW,
        sku=4567899,
        tags=["electronics", "laptop"],
        meta={"views": 0},
    )
    benchmark(product.__getstate__)


@pytest.mark.benchmark(group="base")
def test_restore_flat(benchmark, event_loop):
    def run():
        event_loop.run_until_complete(Product.restore(flat_state))

    benchmark(run)


@pytest.mark.benchmark(group="base")
def test_serialize_nested(benchmark):
    page = Page(
        title="Hello world",
        enabled=True,
        blocks=[HeadingBlock(text="Main"), MarkdownBlock(content="![Home](/)")],
        settings=PageSettings(meta="Lorem ipsum dolor", count=100),
    )
    benchmark(page.__getstate__)


@pytest.mark.benchmark(group="base")
def test_restore_nested(benchmark, event_loop):
    def run():
        event_loop.run_until_complete(Page.restore(nested_state))

    benchmark(run)
