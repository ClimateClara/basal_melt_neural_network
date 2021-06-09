import pytest
from basal_melt_neural_networks.dummy_module import dummy_foo


def test_dummy():
    assert dummy_foo(4) == (4 + 4)
