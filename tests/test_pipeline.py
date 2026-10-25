"""Tests for leak prevention and preprocessing."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from latedeliveries.pipeline import FEATURES, SEED, build_pipeline, split_orders


@pytest.mark.parametrize("excluded", ["late", "order_id", "prep_minutes", "customer_rating", "courier_months", "vehicle"])
def test_excluded_features(excluded):
    assert excluded not in FEATURES


def test_repeatable_stratified_split(orders):
    a = split_orders(orders)
    b = split_orders(orders)
    assert a[0].index.equals(b[0].index)
    assert a[1].index.equals(b[1].index)
    assert abs(a[3].mean() - orders.late.mean()) < 0.03
    assert SEED == 881


def test_missing_numeric_and_unseen_weather(orders):
    x_train, _, y_train, _ = split_orders(orders)
    model = build_pipeline(LogisticRegression(max_iter=1000)).fit(x_train, y_train)
    sample = x_train.iloc[:1].copy()
    sample.loc[:, "distance_km"] = np.nan
    sample.loc[:, "weather"] = "hail"
    prob = model.predict_proba(sample)[0, 1]
    assert 0 <= prob <= 1
