#!/usr/bin/env bash
# Fetch PTB-XL 1.0.3 into <target>/ptbxl (about 1.7 GB). Safe to re-run; both
# paths resume and skip files that are already complete.
#
# Prefers PhysioNet's public AWS S3 mirror when the aws CLI is installed
# (parallel, typically tens of MB/s). Falls back to a recursive wget from
# physionet.org, which works but is heavily rate-limited (often tens of KB/s).
set -euo pipefail
target="${1:-data}"
dest="$target/ptbxl"
mkdir -p "$dest"

if command -v aws >/dev/null 2>&1; then
  aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.3/ "$dest"
else
  echo "aws CLI not found; falling back to wget from physionet.org (slow)" >&2
  (
    cd "$dest"
    # -nH --cut-dirs=3 strips files/ptb-xl/1.0.3/ so the dataset lands directly
    # here; -R drops the directory-listing pages wget uses to recurse.
    wget -r -N -c -np -nH --cut-dirs=3 -R "index.html*" -q --show-progress \
      https://physionet.org/files/ptb-xl/1.0.3/
  )
fi
echo "PTB-XL is at $dest"
