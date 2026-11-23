#!/bin/bash
# Build ccvc_dapn_oem.zip for Kaggle from the committed state of this repository (spec 6, 12).
#
#   bash tools/make_kaggle_package_dapn.sh [out_dir]          # default out_dir: dist/
#
# - refuses to build with uncommitted changes to tracked files (the zip must be a commit);
# - runs tools/check_dapn_diff.py and tools/verify_origin.py --steps 1 first;
# - zip = git archive HEAD (origin files + DAPN files, so --method ccvc_origin also works) + PACKAGE_COMMIT,
#   top folder CCVC-DAPN/ -> unzip into /kaggle/working gives /kaggle/working/CCVC-DAPN
#   (the origin package uses /kaggle/working/CCVC-Origin: both can live in one notebook).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-dist}
PY=${PYTHON:-python}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "[package] ERROR: commit your changes first"; git status --short; exit 2; }
$PY tools/check_dapn_diff.py
$PY tools/verify_origin.py --steps 1
COMMIT=$(git rev-parse HEAD)
mkdir -p "$OUT"
TMP=$(mktemp -d)
echo "$COMMIT" > "$TMP/PACKAGE_COMMIT"
ZIP="$OUT/ccvc_dapn_oem.zip"
rm -f "$ZIP"
git archive --format=zip --prefix=CCVC-DAPN/ --add-file="$TMP/PACKAGE_COMMIT" -o "$ZIP" HEAD
rm -rf "$TMP"
for f in tools/run_diag_origin.sh tools/summarize_diag_origin.py; do      # diagnostics 0a/0b of the origin runs
    unzip -l "$ZIP" | grep -q "CCVC-DAPN/$f\$" || { echo "[package] ERROR: $f missing from the zip"; exit 2; }
done
echo "[package] $ZIP  ($(du -h "$ZIP" | cut -f1), commit ${COMMIT:0:7}, $(unzip -l "$ZIP" | tail -1 | awk '{print $2}') files)"
echo "[package] Kaggle: unzip -q ccvc_dapn_oem.zip -d /kaggle/working && cd /kaggle/working/CCVC-DAPN && bash run_dapn.sh split 1/4 --dryrun"
echo "[package] origin diagnostics 0a/0b: cd /kaggle/working/CCVC-DAPN && bash tools/run_diag_origin.sh --quick  (then without --quick)"
