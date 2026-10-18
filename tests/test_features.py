import numpy as np
import pytest

from heartml import config
from heartml.features import ENGINEERED_FEATURES, ClinicalFeatures, build_preprocessor


def test_clinical_features_values(clean_df):
    out = ClinicalFeatures().fit_transform(clean_df[config.FEATURES])
    for col in ENGINEERED_FEATURES:
        assert col in out.columns
    row = clean_df.iloc[0]
    assert out["hr_reserve_pct"].iloc[0] == pytest.approx(row["thalach"] / (220 - row["age"]))
    assert set(out["bp_stage2"].dropna().unique()) <= {0.0, 1.0}


def test_clinical_features_does_not_mutate_input(clean_df):
    X = clean_df[config.FEATURES]
    before = X.copy()
    ClinicalFeatures().fit_transform(X)
    assert X.equals(before)


def test_preprocessor_output_is_dense_and_complete(clean_df):
    prep = build_preprocessor()
    Xt = prep.fit_transform(clean_df[config.FEATURES])
    assert Xt.shape[0] == len(clean_df)
    assert not np.isnan(Xt).any()


def test_preprocessor_handles_unseen_category(clean_df):
    prep = build_preprocessor().fit(clean_df[config.FEATURES])
    row = clean_df[config.FEATURES].head(1).copy()
    row["thal"] = 99
    Xt = prep.transform(row)
    assert Xt.shape[1] == prep.transform(clean_df[config.FEATURES].head(1)).shape[1]


def test_numeric_columns_are_scaled(clean_df):
    prep = build_preprocessor()
    Xt = prep.fit_transform(clean_df[config.FEATURES])
    n_numeric = len(config.NUMERIC_FEATURES) + 2
    means = Xt[:, :n_numeric].mean(axis=0)
    assert np.allclose(means, 0, atol=1e-6)
