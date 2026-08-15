#!/usr/bin/env bash
# Download the Open Bandit Dataset (spec section 6.4).
#
#   ./scripts/download_obd.sh sample   # 10k rows per (policy, campaign), for tests/CI
#   ./scripts/download_obd.sh full     # ~26M rows, ~413 MB zip, for reported results
#
# Everything lands under data/, which is gitignored.
set -euo pipefail

MODE="${1:-sample}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT/data"
mkdir -p "$DATA_DIR"

case "$MODE" in
  sample)
    TARGET="$DATA_DIR/obd_sample"
    if [ -d "$TARGET" ]; then
      echo "sample already present at $TARGET"
      exit 0
    fi
    git clone --depth 1 https://github.com/st-tech/zr-obp "$DATA_DIR/zr-obp"
    mv "$DATA_DIR/zr-obp/obd" "$TARGET"
    rm -rf "$DATA_DIR/zr-obp"
    echo "sample OBD at $TARGET"
    ;;
  full)
    TARGET="$DATA_DIR/obd_full"
    ZIP="$DATA_DIR/open_bandit_dataset.zip"
    if [ -d "$TARGET" ]; then
      echo "full dataset already present at $TARGET"
      exit 0
    fi
    curl -fL --retry 3 -o "$ZIP" https://research.zozo.com/data_release/open_bandit_dataset.zip
    mkdir -p "$TARGET"
    unzip -q "$ZIP" -d "$TARGET"
    rm -f "$ZIP"
    echo "full OBD at $TARGET"
    ;;
  *)
    echo "usage: $0 [sample|full]" >&2
    exit 2
    ;;
esac
