import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heartml import config

ENGINEERED_FEATURES = ["hr_reserve_pct", "st_per_hr", "bp_stage2"]


class ClinicalFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        predicted_max = (220 - X["age"]).clip(lower=1)
        X["hr_reserve_pct"] = X["thalach"] / predicted_max
        X["st_per_hr"] = X["oldpeak"] / X["thalach"].clip(lower=1) * 100
        X["bp_stage2"] = np.where(X["trestbps"].isna(), np.nan, (X["trestbps"] >= 140).astype(float))
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else config.FEATURES
        return np.array(base + ENGINEERED_FEATURES, dtype=object)


def build_preprocessor():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    binary = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer([
        ("num", numeric, config.NUMERIC_FEATURES + ["hr_reserve_pct", "st_per_hr"]),
        ("bin", binary, config.BINARY_FEATURES + ["bp_stage2"]),
        ("cat", categorical, config.CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("clinical", ClinicalFeatures()),
        ("columns", columns),
    ])


def build_pipeline(estimator):
    return Pipeline([
        ("prep", build_preprocessor()),
        ("model", estimator),
    ])


def feature_names(pipeline):
    return list(pipeline.named_steps["prep"].named_steps["columns"].get_feature_names_out())
