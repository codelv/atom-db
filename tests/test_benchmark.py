import dis
import pytest
from atom.api import Float, Int, Str, List, Bool, Dict
from atomdb.base import Model

state = dict(
    title="This is a test",
    desc="""Lorem ipsum dolor sit amet, consectetur adipiscing elit,
    sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
    Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris
    nisi ut aliquip ex ea commodo consequat.
    Duis aute irure dolor in reprehenderit in voluptate velit esse cillum
    """,
    enabled=True,
    rating=8.3,
    tags=["electronics", "laptop"],
    meta={"views": 0},
)


class Product(Model):
    title = Str()
    desc = Str()
    enabled = Bool()
    rating = Float()
    tags = List(str)
    meta = Dict()


def test_serialize(benchmark):
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
        tags=["electronics", "laptop"],
        meta={"views": 0},
    )
    benchmark(product.__getstate__)


def test_restore(benchmark, event_loop):
    def run():
        event_loop.run_until_complete(Product.restore(state))

    benchmark(run)
