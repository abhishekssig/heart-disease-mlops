#!/usr/bin/env bash
set -euo pipefail

URL=${1:-http://localhost:8000}
cd "$(dirname "$0")/.."

echo "GET  $URL/health";       curl -sf "$URL/health"; echo
echo "GET  $URL/model-info";   curl -sf "$URL/model-info" | head -c 300; echo
echo "POST $URL/predict";      curl -sf -X POST "$URL/predict" -H "Content-Type: application/json" -d @sample_input.json; echo
echo "POST $URL/predict/batch"; curl -sf -X POST "$URL/predict/batch" -H "Content-Type: application/json" -d @sample_batch.json; echo
echo "POST $URL/predict (invalid, expect 422)"
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$URL/predict" -H "Content-Type: application/json" -d '{"age": 500}')
echo "$code"; [ "$code" = "422" ]
