import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from heartml import config
from heartml.features import build_pipeline


def make_raw(n=120, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(30, 77, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.integers(95, 190, n),
        "chol": rng.integers(130, 400, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(90, 200, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.uniform(0, 5, n).round(1),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n),
    })
    risk = (df["cp"] == 4).astype(int) + df["exang"] + (df["oldpeak"] > 2).astype(int) + (df["thalach"] < 140)
    df["num"] = np.where(risk >= 2, rng.integers(1, 5, n), 0)
    df.loc[rng.choice(n, 4, replace=False), "ca"] = np.nan
    df.loc[rng.choice(n, 2, replace=False), "thal"] = np.nan
    return df


@pytest.fixture
def raw_df():
    return make_raw()


@pytest.fixture
def clean_df(raw_df):
    from heartml.data import clean
    return clean(raw_df)


@pytest.fixture
def trained_model_dir(tmp_path, clean_df):
    pipe = build_pipeline(LogisticRegression(max_iter=500))
    pipe.fit(clean_df[config.FEATURES], clean_df[config.TARGET])
    joblib.dump(pipe, tmp_path / "model.joblib")
    (tmp_path / "metadata.json").write_text(json.dumps({"model_name": "test_logreg", "threshold": 0.5}))
    return tmp_path


@pytest.fixture
def sample_patient():
    return json.loads((config.ROOT_DIR / "sample_input.json").read_text())
