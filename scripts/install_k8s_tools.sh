#!/usr/bin/env bash
set -euo pipefail

ARCH=amd64
BIN=${BIN:-$HOME/.local/bin}
mkdir -p "$BIN"

if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
  echo "docker installed - log out and back in (or run: newgrp docker) before continuing"
fi

if ! command -v kubectl >/dev/null 2>&1; then
  KVER=$(curl -fsSL https://dl.k8s.io/release/stable.txt)
  curl -fsSLo "$BIN/kubectl" "https://dl.k8s.io/release/${KVER}/bin/linux/${ARCH}/kubectl"
  chmod +x "$BIN/kubectl"
fi

if ! command -v minikube >/dev/null 2>&1; then
  curl -fsSLo "$BIN/minikube" "https://storage.googleapis.com/minikube/releases/latest/minikube-linux-${ARCH}"
  chmod +x "$BIN/minikube"
fi

case ":$PATH:" in
  *":$BIN:"*) ;;
  *) echo "export PATH=\"$BIN:\$PATH\"" >> "$HOME/.bashrc"; echo "added $BIN to PATH in ~/.bashrc" ;;
esac

"$BIN/kubectl" version --client
"$BIN/minikube" version
