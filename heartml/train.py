import argparse
import json
import logging
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from heartml import __version__, config
from heartml.features import build_pipeline, feature_names

plt.switch_backend("Agg")
log = logging.getLogger(__name__)

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

CANDIDATES = {
    "logistic_regression": (
        LogisticRegression(max_iter=2000, solver="liblinear", random_state=config.RANDOM_STATE),
        {
            "model__C": [0.01, 0.1, 0.3, 1.0, 3.0],
            "model__penalty": ["l1", "l2"],
            "model__class_weight": [None, "balanced"],
        },
    ),
    "random_forest": (
        RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1),
        {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 4, 8],
            "model__min_samples_leaf": [1, 3, 5],
            "model__max_features": ["sqrt", 0.5],
        },
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=config.RANDOM_STATE),
        {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [2, 3],
            "model__subsample": [0.8, 1.0],
        },
    ),
}

QUICK_GRIDS = {
    "logistic_regression": {"model__C": [0.1, 1.0]},
    "random_forest": {"model__n_estimators": [100], "model__max_depth": [4]},
    "gradient_boosting": {"model__n_estimators": [50]},
}


def load_dataset(path=config.CLEAN_FILE):
    df = pd.read_csv(path)
    return df[config.FEATURES], df[config.TARGET]


def split(X, y):
    return train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)


def holdout_metrics(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "test_accuracy": accuracy_score(y_true, pred),
        "test_precision": precision_score(y_true, pred, zero_division=0),
        "test_recall": recall_score(y_true, pred, zero_division=0),
        "test_f1": f1_score(y_true, pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y_true, proba),
    }


def cv_summary(search):
    i = search.best_index_
    res = search.cv_results_
    out = {}
    for name in SCORING:
        out[f"cv_{name}_mean"] = float(res[f"mean_test_{name}"][i])
        out[f"cv_{name}_std"] = float(res[f"std_test_{name}"][i])
    return out


def log_grid_children(name, search):
    res = search.cv_results_
    for i, params in enumerate(res["params"]):
        with mlflow.start_run(run_name=f"{name}-cand-{i:02d}", nested=True):
            mlflow.log_params({k.replace("model__", ""): v for k, v in params.items()})
            mlflow.log_metrics({f"cv_{m}_mean": float(res[f"mean_test_{m}"][i]) for m in SCORING})
            mlflow.set_tag("rank_roc_auc", int(res["rank_test_roc_auc"][i]))


def importance_frame(pipeline):
    model = pipeline.named_steps["model"]
    names = feature_names(pipeline)
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        values = np.abs(model.coef_).ravel()
    return pd.DataFrame({"feature": names, "importance": values}).sort_values("importance", ascending=False)


def save_plots(name, pipeline, X_test, y_test, outdir):
    paths = []

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test, display_labels=["No disease", "Disease"],
                                          cmap="Blues", ax=ax)
    ax.set_title(f"{name} - confusion matrix")
    paths.append(outdir / "confusion_matrix.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(f"{name} - ROC curve")
    paths.append(outdir / "roc_curve.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)

    imp = importance_frame(pipeline).head(15)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(imp["feature"][::-1], imp["importance"][::-1], color="#55A868")
    ax.set_title(f"{name} - top features")
    paths.append(outdir / "feature_importance.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)

    imp.to_csv(outdir / "feature_importance.csv", index=False)
    paths.append(outdir / "feature_importance.csv")
    return paths


def train_one(name, X_train, X_test, y_train, y_test, quick=False):
    estimator, grid = CANDIDATES[name]
    if quick:
        grid = QUICK_GRIDS[name]
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    search = GridSearchCV(build_pipeline(estimator), grid, scoring=SCORING, refit="roc_auc", cv=cv, n_jobs=-1)

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"model_family": name, "stage": "tuning", "quick": str(quick)})
        mlflow.log_params({"model": name, "cv_folds": config.CV_FOLDS, "test_size": config.TEST_SIZE,
                           "n_train": len(X_train), "n_test": len(X_test),
                           "grid_size": int(np.prod([len(v) for v in grid.values()]))})
        search.fit(X_train, y_train)
        log_grid_children(name, search)

        best = search.best_estimator_
        proba = best.predict_proba(X_test)[:, 1]
        metrics = {**cv_summary(search), **holdout_metrics(y_test, proba)}

        mlflow.log_params({f"best_{k.replace('model__', '')}": v for k, v in search.best_params_.items()})
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for p in save_plots(name, best, X_test, y_test, tmp):
                mlflow.log_artifact(str(p), artifact_path="plots")
            report = classification_report(y_test, (proba >= 0.5).astype(int),
                                           target_names=["No disease", "Disease"])
            (tmp / "classification_report.txt").write_text(report)
            mlflow.log_artifact(str(tmp / "classification_report.txt"))
            pd.DataFrame(search.cv_results_).to_csv(tmp / "cv_results.csv", index=False)
            mlflow.log_artifact(str(tmp / "cv_results.csv"))

        signature = infer_signature(X_test, proba)
        mlflow.sklearn.log_model(best, artifact_path="model", signature=signature,
                                 input_example=X_test.head(3))

        log.info("%-20s cv_roc_auc=%.3f test_roc_auc=%.3f test_acc=%.3f", name,
                 metrics["cv_roc_auc_mean"], metrics["test_roc_auc"], metrics["test_accuracy"])
        return {"name": name, "run_id": run.info.run_id, "pipeline": best,
                "best_params": search.best_params_, "metrics": metrics}


def save_best(best, X_train):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipeline"], config.MODELS_DIR / "model.joblib")
    meta = {
        "model_name": best["name"],
        "mlflow_run_id": best["run_id"],
        "package_version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "features": config.FEATURES,
        "threshold": 0.5,
        "best_params": {k.replace("model__", ""): v for k, v in best["best_params"].items()},
        "metrics": {k: round(v, 4) for k, v in best["metrics"].items()},
        "n_train": int(len(X_train)),
    }
    config.METADATA_FILE.write_text(json.dumps(meta, indent=2, default=str))
    log.info("Best model %s saved to %s", best["name"], config.MODELS_DIR / "model.joblib")
    return meta


def main():
    parser = argparse.ArgumentParser(description="Train and track heart disease classifiers")
    parser.add_argument("--models", nargs="+", default=list(CANDIDATES), choices=list(CANDIDATES))
    parser.add_argument("--quick", action="store_true", help="tiny grids for CI smoke runs")
    args = parser.parse_args()

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split(X, y)
    log.info("Train %d rows, test %d rows, positive rate %.2f", len(X_train), len(X_test), y.mean())

    results = [train_one(m, X_train, X_test, y_train, y_test, quick=args.quick) for m in args.models]
    best = max(results, key=lambda r: r["metrics"]["cv_roc_auc_mean"])

    with mlflow.start_run(run_name="model-selection"):
        mlflow.set_tag("stage", "selection")
        mlflow.log_param("selected_model", best["name"])
        mlflow.log_param("selection_metric", "cv_roc_auc_mean")
        table = pd.DataFrame([{"model": r["name"], **r["metrics"]} for r in results]).round(4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model_comparison.csv"
            table.to_csv(path, index=False)
            mlflow.log_artifact(str(path))
        mlflow.log_metrics({f"{r['name']}_cv_roc_auc": r["metrics"]["cv_roc_auc_mean"] for r in results})

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.FIGURES_DIR.parent / "model_comparison.csv", index=False)
    print(table.to_string(index=False))
    save_best(best, X_train)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
