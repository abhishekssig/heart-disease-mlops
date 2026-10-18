import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

RAW_FILE = RAW_DIR / "processed.cleveland.data"
CLEAN_FILE = PROCESSED_DIR / "heart_clean.csv"
MODEL_FILE = Path(os.getenv("MODEL_PATH", MODELS_DIR / "model.joblib"))
METADATA_FILE = MODELS_DIR / "metadata.json"

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
TARGET = "target"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "ca", "thal"]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

VALID_RANGES = {
    "age": (18, 100),
    "trestbps": (70, 220),
    "chol": (100, 600),
    "thalach": (60, 220),
    "oldpeak": (0.0, 7.0),
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "heart-disease-classifier")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", (ROOT_DIR / "mlruns").as_uri())
