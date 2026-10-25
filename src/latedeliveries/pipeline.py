"""Leak-free features, split and preprocessing pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 881
NUMERIC_FEATURES = ["distance_km", "items", "hour"]
CATEGORICAL_FEATURES = ["weather", "day_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def split_orders(frame):
    """Stratified split; keep test isolated until final evaluation."""
    return train_test_split(
        frame[FEATURES],
        frame["late"],
        test_size=0.25,
        stratify=frame["late"],
        random_state=SEED,
    )


def build_pipeline(model):
    """Fit preprocessing only on each training fold."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])
