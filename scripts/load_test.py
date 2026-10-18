import argparse
import json
import random
import time
import urllib.request


def patient():
    return {
        "age": random.randint(29, 77),
        "sex": random.randint(0, 1),
        "cp": random.randint(1, 4),
        "trestbps": random.randint(94, 200),
        "chol": random.randint(126, 564),
        "fbs": random.randint(0, 1),
        "restecg": random.randint(0, 2),
        "thalach": random.randint(71, 202),
        "exang": random.randint(0, 1),
        "oldpeak": round(random.uniform(0, 6.2), 1),
        "slope": random.randint(1, 3),
        "ca": random.randint(0, 3),
        "thal": random.choice([3, 6, 7]),
    }


def main():
    parser = argparse.ArgumentParser(description="Send random traffic to the API so dashboards have data")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("-n", type=int, default=300)
    parser.add_argument("--delay", type=float, default=0.1)
    args = parser.parse_args()

    ok = bad = 0
    for i in range(args.n):
        body = patient()
        if random.random() < 0.05:
            body["age"] = 300
        req = urllib.request.Request(f"{args.url}/predict", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=5)
            ok += 1
        except Exception:
            bad += 1
        time.sleep(args.delay)
    print(f"sent {args.n}: ok={ok} rejected/failed={bad}")


if __name__ == "__main__":
    main()
