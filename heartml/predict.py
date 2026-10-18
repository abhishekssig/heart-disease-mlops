import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from heartml import config


class HeartModel:
    def __init__(self, path=config.MODEL_FILE, threshold=0.5):
        self.path = Path(path)
        self.pipeline = joblib.load(self.path)
        meta_path = self.path.with_name("metadata.json")
        self.metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.threshold = self.metadata.get("threshold", threshold)

    @property
    def name(self):
        return self.metadata.get("model_name", type(self.pipeline.named_steps["model"]).__name__)

    def predict(self, records):
        frame = pd.DataFrame(records, columns=config.FEATURES).astype(float)
        proba = self.pipeline.predict_proba(frame)[:, 1]
        results = []
        for p in proba:
            label = int(p >= self.threshold)
            results.append({
                "prediction": label,
                "label": "Heart disease" if label else "No heart disease",
                "probability_disease": round(float(p), 4),
                "confidence": round(float(p if label else 1 - p), 4),
            })
        return results


def main():
    parser = argparse.ArgumentParser(description="Run the saved model on a JSON file of patient records")
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", type=Path, default=config.MODEL_FILE)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    if isinstance(payload, dict) and "patients" in payload:
        records = payload["patients"]
    elif isinstance(payload, list):
        records = payload
    else:
        records = [payload]
    model = HeartModel(args.model)
    for rec, res in zip(records, model.predict(records)):
        print(json.dumps({"input": rec, "output": res}))


if __name__ == "__main__":
    main()
