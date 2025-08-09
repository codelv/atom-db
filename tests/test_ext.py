import gc

import pytest
from atom.api import Atom, ContainerList, List

from atomdb.ext import atomlist_owner


def test_atomclist_owner():
    class A(Atom):
        items = List()

    class B(Atom):
        items = ContainerList()

    # atomlist
    a = A()
    assert atomlist_owner(a.items) is a

    # atomclist
    b = B()
    assert atomlist_owner(b.items) is b

    # deleted owner
    items = b.items
    del b
    gc.collect()
    assert atomlist_owner(items) is None

    # invalid type
    with pytest.raises(TypeError):
        atomlist_owner([])
