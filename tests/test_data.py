import numpy as np
import pandas as pd

from heartml import config
from heartml.data import clean, load_raw, missing_report


def test_clean_creates_binary_target(raw_df):
    out = clean(raw_df)
    assert config.TARGET in out.columns
    assert "num" not in out.columns
    assert set(out[config.TARGET].unique()) <= {0, 1}
    assert out[config.TARGET].sum() == (raw_df["num"] > 0).sum()


def test_clean_keeps_feature_order(raw_df):
    out = clean(raw_df)
    assert list(out.columns) == config.FEATURES + [config.TARGET]


def test_clean_removes_duplicates(raw_df):
    doubled = pd.concat([raw_df, raw_df.head(10)], ignore_index=True)
    assert len(clean(doubled)) == len(clean(raw_df))


def test_out_of_range_values_become_nan(raw_df):
    raw_df.loc[0, "chol"] = 0
    raw_df.loc[1, "trestbps"] = 999
    out = clean(raw_df)
    assert np.isnan(out.loc[0, "chol"])
    assert np.isnan(out.loc[1, "trestbps"])


def test_question_marks_parsed_as_missing(tmp_path):
    line = "63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,?,6.0,0\n"
    path = tmp_path / "raw.data"
    path.write_text(line * 3)
    df = load_raw(path)
    assert df["ca"].isna().all()
    assert df.shape == (3, len(config.RAW_COLUMNS))


def test_missing_report_lists_only_missing(clean_df):
    report = missing_report(clean_df)
    assert set(report.index) <= {"ca", "thal"}
    assert (report > 0).all()
