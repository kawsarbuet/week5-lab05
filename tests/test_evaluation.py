"""Hand-checked cost and threshold tests."""
import numpy as np
import pytest
from latedeliveries.evaluation import choose_threshold, order_cost


def test_hand_calculated_cost():
    # Three vouchers ($12), one missed late ($15) => $27.
    assert order_cost([1, 0, 1, 0, 0], [1, 1, 0, 1, 0]) == 27


@pytest.mark.parametrize("truth,prob,expected", [
    ([0, 0, 0], [0.2, 0.4, 0.6], 0.95),
    ([1, 1, 1], [0.2, 0.4, 0.6], 0.20),
])
def test_threshold_extremes(truth, prob, expected):
    assert choose_threshold(truth, prob) == pytest.approx(expected)


def test_probability_shape_validation():
    with pytest.raises(ValueError):
        choose_threshold([0, 1], np.array([0.5]))
