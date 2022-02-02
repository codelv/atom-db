import dis
import pytest
from datetime import datetime
from atom.api import Float, Int, Str, List, Bool, Dict, Typed
from atomdb.base import Model

NOW = datetime.now()

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
    datetime=NOW.timestamp(),
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
    created = Typed(datetime, factory=datetime.now).tag(
        flatten=lambda v, scope: v.timestamp() if v else None,
        unflatten=lambda v, scope: datetime.fromtimestamp(v) if v else None,
    )


@pytest.mark.benchmark(group="base")
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
        created=NOW,
        tags=["electronics", "laptop"],
        meta={"views": 0},
    )
    benchmark(product.__getstate__)


@pytest.mark.benchmark(group="base")
def test_restore(benchmark, event_loop):
    def run():
        event_loop.run_until_complete(Product.restore(state))

    benchmark(run)
