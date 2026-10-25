"""Saved model preserves probability predictions."""

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from latedeliveries.pipeline import build_pipeline, split_orders


def test_roundtrip(tmp_path, orders):
    x_train, x_test, y_train, _ = split_orders(orders)
    pipeline = build_pipeline(LogisticRegression(max_iter=1000)).fit(x_train, y_train)
    expected = pipeline.predict_proba(x_test)
    path = tmp_path / "model.joblib"
    joblib.dump(pipeline, path)
    actual = joblib.load(path).predict_proba(x_test)
    np.testing.assert_allclose(actual, expected)
