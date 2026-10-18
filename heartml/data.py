import argparse
import logging

import numpy as np
import pandas as pd
import requests

from heartml import config

log = logging.getLogger(__name__)


def download_raw(force=False):
    if config.RAW_FILE.exists() and not force:
        log.info("Raw data already present at %s", config.RAW_FILE)
        return
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s", config.DATA_URL)
    resp = requests.get(config.DATA_URL, timeout=30)
    resp.raise_for_status()
    config.RAW_FILE.write_bytes(resp.content)
    log.info("Saved %d bytes to %s", len(resp.content), config.RAW_FILE)


def load_raw(path=config.RAW_FILE):
    return pd.read_csv(path, header=None, names=config.RAW_COLUMNS, na_values="?")


def clean(df):
    df = df.copy().apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["num"])
    df[config.TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"]).drop_duplicates().reset_index(drop=True)

    for col, (lo, hi) in config.VALID_RANGES.items():
        bad = ~df[col].between(lo, hi) & df[col].notna()
        if bad.any():
            log.warning("%s: %d out-of-range values set to NaN", col, int(bad.sum()))
            df.loc[bad, col] = np.nan

    return df[config.FEATURES + [config.TARGET]]


def missing_report(df):
    return df.isna().sum().loc[lambda s: s > 0]


def main():
    parser = argparse.ArgumentParser(description="Download and clean the UCI heart disease data")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    download_raw(force=args.force)
    raw = load_raw()
    df = clean(raw)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_FILE, index=False)

    log.info("Raw shape %s -> clean shape %s", raw.shape, df.shape)
    log.info("Missing values after cleaning:\n%s", missing_report(df).to_string() or "none")
    log.info("Class balance:\n%s", df[config.TARGET].value_counts().to_string())
    log.info("Clean data written to %s", config.CLEAN_FILE)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
