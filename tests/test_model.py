import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from heartml import config
from heartml.features import build_pipeline, feature_names
from heartml.predict import HeartModel
from heartml.train import importance_frame, holdout_metrics


@pytest.mark.parametrize("estimator", [
    LogisticRegression(max_iter=500),
    RandomForestClassifier(n_estimators=20, random_state=0),
])
def test_pipeline_trains_and_predicts_probabilities(clean_df, estimator):
    X, y = clean_df[config.FEATURES], clean_df[config.TARGET]
    pipe = build_pipeline(estimator).fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert len(importance_frame(pipe)) == len(feature_names(pipe))


def test_pipeline_learns_signal(clean_df):
    X, y = clean_df[config.FEATURES], clean_df[config.TARGET]
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    assert (pipe.predict(X) == y).mean() > 0.7


def test_test_metrics_keys_and_ranges():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.6, 0.4, 0.9])
    m = holdout_metrics(y, proba)
    assert set(m) == {"test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"}
    assert m["test_accuracy"] == 0.5
    assert m["test_roc_auc"] == 0.75


def test_saved_model_roundtrip(trained_model_dir, sample_patient):
    model = HeartModel(trained_model_dir / "model.joblib")
    out = model.predict([sample_patient])[0]
    assert model.name == "test_logreg"
    assert out["prediction"] in (0, 1)
    assert 0.5 <= out["confidence"] <= 1.0
    assert out["confidence"] == pytest.approx(max(out["probability_disease"], 1 - out["probability_disease"]),
                                              abs=1e-4)


def test_model_tolerates_missing_optional_fields(trained_model_dir, sample_patient):
    sample_patient["ca"] = None
    sample_patient["thal"] = None
    out = HeartModel(trained_model_dir / "model.joblib").predict([sample_patient])[0]
    assert 0 <= out["probability_disease"] <= 1
