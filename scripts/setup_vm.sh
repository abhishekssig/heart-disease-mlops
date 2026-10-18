#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
if [ ! -d .venv ]; then
  $PY -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo
echo "== tool check =="
python --version
for t in git docker kubectl minikube helm; do
  if command -v "$t" >/dev/null 2>&1; then
    echo "$t: $($t version --client 2>/dev/null | head -1 || $t --version | head -1)"
  else
    echo "$t: not installed"
  fi
done
if command -v docker >/dev/null 2>&1; then
  docker ps >/dev/null 2>&1 && echo "docker daemon: reachable" || echo "docker daemon: NOT reachable (try: sudo usermod -aG docker \$USER)"
fi
sudo -n true 2>/dev/null && echo "sudo: available" || echo "sudo: needs password or not allowed"
