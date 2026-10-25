"""Reproducible training, model selection, one test evaluation, and artifacts."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.tree import DecisionTreeClassifier

from latedeliveries.data import make_orders
from latedeliveries.evaluation import THRESHOLDS, choose_threshold, order_cost, score_predictions
from latedeliveries.pipeline import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, SEED, build_pipeline, split_orders

ROOT = Path(__file__).resolve().parents[2]


def main():
    """Create every required saved output without tuning against the test set."""
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    data = make_orders()
    x_train, x_test, y_train, y_test = split_orders(data)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    candidates = {
        "Dummy": DummyClassifier(strategy="prior"),
        "Logistic": LogisticRegression(max_iter=1000, random_state=SEED),
        "Balanced logistic": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        "Decision tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
        "Random forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=SEED, n_jobs=1),
    }
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}
    rows = []
    for name, estimator in candidates.items():
        scores = cross_validate(build_pipeline(estimator), x_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        row = {"model": name}
        for metric in scoring:
            row[f"{metric}_mean"] = float(scores[f"test_{metric}"].mean())
            row[f"{metric}_std"] = float(scores[f"test_{metric}"].std())
        rows.append(row)
    pd.DataFrame(rows).to_csv(ROOT / "reports/cv_results.csv", index=False)

    # Select a transparent probabilistic model with stable CV ROC-AUC, then tune it.
    base = build_pipeline(LogisticRegression(max_iter=1000, random_state=SEED))
    untuned_auc = float(cross_validate(base, x_train, y_train, cv=cv, scoring="roc_auc")["test_score"].mean())
    grid = GridSearchCV(
        base,
        {"model__C": [0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]},
        cv=cv, scoring="roc_auc", n_jobs=1,
    )
    grid.fit(x_train, y_train)
    tuned = grid.best_estimator_
    # Only training folds contribute to threshold selection.
    oof_prob = cross_val_predict(tuned, x_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = choose_threshold(y_train, oof_prob)
    threshold_costs = [{"threshold": float(t), "cost": order_cost(y_train, oof_prob >= t)} for t in THRESHOLDS]
    (ROOT / "reports/threshold_costs.json").write_text(json.dumps(threshold_costs, indent=2))

    # Final test is used exactly once, after selection and threshold are frozen.
    tuned.fit(x_train, y_train)
    test_prob = tuned.predict_proba(x_test)[:, 1]
    flagged = test_prob >= threshold
    results = {
        "seed": SEED, "train_size": len(y_train), "test_size": len(y_test),
        "late_rate": float(data["late"].mean()),
        "chosen_model": "Logistic regression", "tuning_metric": "roc_auc",
        "untuned_cv_roc_auc": untuned_auc, "tuned_cv_roc_auc": float(grid.best_score_),
        "cv_roc_auc_spread": float(pd.DataFrame(rows).set_index("model").loc["Logistic", "roc_auc_std"]),
        "best_params": grid.best_params_, "threshold": threshold,
        "confusion_matrix": confusion_matrix(y_test, flagged).tolist(),
        "metrics": score_predictions(y_test, test_prob, threshold),
        "costs": {
            "flag_nobody": order_cost(y_test, np.zeros(len(y_test), dtype=bool)),
            "flag_everybody": order_cost(y_test, np.ones(len(y_test), dtype=bool)),
            "model": order_cost(y_test, flagged),
        },
    }
    (ROOT / "reports/metrics.json").write_text(json.dumps(results, indent=2))
    joblib.dump(tuned, ROOT / "models/late_orders.joblib")
    info = {
        "scikit_learn_version": sklearn.__version__, "seed": SEED,
        "threshold": threshold, "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES, "features": FEATURES,
        "best_settings": grid.best_params_, "test_results": results,
    }
    (ROOT / "models/model_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
