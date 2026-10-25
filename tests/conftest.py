"""Shared sample data."""
import pytest
from latedeliveries.data import make_orders


@pytest.fixture
def orders():
    """Small deterministic sample."""
    return make_orders(n=300)
