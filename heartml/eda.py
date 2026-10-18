import json
import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from heartml import config

plt.switch_backend("Agg")

log = logging.getLogger(__name__)

LABELS = {0: "No disease", 1: "Disease"}
PALETTE = {"No disease": "#4C72B0", "Disease": "#C44E52"}


def _save(fig, name):
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log.info("Saved %s", path)
    return path


def plot_class_balance(df):
    counts = df[config.TARGET].map(LABELS).value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(counts.index, counts.values, color=[PALETTE[k] for k in counts.index])
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 2, f"{val} ({val / len(df):.0%})", ha="center")
    ax.set_title("Class balance")
    ax.set_ylabel("Patients")
    return _save(fig, "class_balance.png")


def plot_missing(df):
    miss = df.isna().sum()
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(5, 3))
    if miss.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        ax.axis("off")
    else:
        miss.plot.barh(ax=ax, color="#8172B2")
        ax.set_xlabel("Missing count")
    ax.set_title("Missing values per feature")
    return _save(fig, "missing_values.png")


def plot_numeric_distributions(df):
    data = df.assign(label=df[config.TARGET].map(LABELS))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flat, config.NUMERIC_FEATURES):
        sns.histplot(data=data, x=col, hue="label", hue_order=list(PALETTE), kde=True, palette=PALETTE, ax=ax,
                     stat="density", common_norm=False, alpha=0.45)
        ax.set_title(col)
    axes.flat[-1].axis("off")
    fig.suptitle("Numeric feature distributions by outcome", fontsize=14)
    return _save(fig, "numeric_histograms.png")


def plot_categorical(df):
    cols = config.BINARY_FEATURES + config.CATEGORICAL_FEATURES
    data = df.assign(label=df[config.TARGET].map(LABELS))
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flat, cols):
        rate = data.groupby(col)[config.TARGET].mean()
        sns.countplot(data=data, x=col, hue="label", hue_order=list(PALETTE), palette=PALETTE, ax=ax)
        ax.set_title(f"{col}  (disease rate: " + ", ".join(f"{int(k)}={v:.0%}" for k, v in rate.items()) + ")",
                     fontsize=8)
        ax.legend(fontsize=7)
    fig.suptitle("Categorical features vs outcome", fontsize=14)
    return _save(fig, "categorical_counts.png")


def plot_correlation(df):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation heatmap")
    return _save(fig, "correlation_heatmap.png")


def plot_boxplots(df):
    data = df.assign(label=df[config.TARGET].map(LABELS))
    melted = data.melt(id_vars="label", value_vars=config.NUMERIC_FEATURES)
    g = sns.catplot(data=melted, x="label", y="value", col="variable", kind="box", sharey=False,
                    palette=PALETTE, hue="label", height=3.5, aspect=0.7)
    g.set_titles("{col_name}")
    g.set_xlabels("")
    path = config.FIGURES_DIR / "numeric_boxplots.png"
    g.savefig(path, dpi=120)
    plt.close(g.figure)
    return path


def summary(df):
    target_corr = df.corr(numeric_only=True)[config.TARGET].drop(config.TARGET).sort_values()
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "class_counts": {LABELS[k]: int(v) for k, v in df[config.TARGET].value_counts().items()},
        "missing": {k: int(v) for k, v in df.isna().sum().items() if v},
        "correlation_with_target": target_corr.round(3).to_dict(),
        "describe": df[config.NUMERIC_FEATURES].describe().round(2).to_dict(),
    }


def main():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.CLEAN_FILE)
    sns.set_theme(style="whitegrid")
    plot_class_balance(df)
    plot_missing(df)
    plot_numeric_distributions(df)
    plot_boxplots(df)
    plot_categorical(df)
    plot_correlation(df)
    out = config.FIGURES_DIR.parent / "eda_summary.json"
    out.write_text(json.dumps(summary(df), indent=2))
    log.info("EDA summary written to %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
