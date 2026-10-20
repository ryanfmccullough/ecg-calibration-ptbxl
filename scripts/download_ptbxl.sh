#!/usr/bin/env bash
# Fetch PTB-XL 1.0.3 from PhysioNet into <target>/ptbxl (about 1.7 GB).
set -euo pipefail
target="${1:-data}"
mkdir -p "$target"
cd "$target"
wget -r -N -c -np -nH --cut-dirs=3 -q --show-progress \
  https://physionet.org/files/ptb-xl/1.0.3/
mv -n "1.0.3" ptbxl 2>/dev/null || true
echo "PTB-XL is at $target/ptbxl"
