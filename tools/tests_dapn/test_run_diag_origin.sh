#!/bin/bash
# CPU tests of tools/run_diag_origin.sh + tools/summarize_diag_origin.py (spec "chay chan doan origin", section 5).
#   bash tools/tests_dapn/test_run_diag_origin.sh <work_dir>      (work_dir is wiped first; outside the repo)
# Builds two short origin runs (run_baseline.sh, split 1/4 and 1/8, synthetic data), lays them out like Kaggle
# inputs (<work>/kaggle_input/...), then checks auto-discovery, --quick, the full run (6 separate outputs, summary
# tables A-C, inputs untouched) and the failure cases. Exit code = number of failed checks.
set -u
W=${1:?usage: test_run_diag_origin.sh <work_dir>}
cd "$(dirname "$0")/../.." || exit 1
PY=${PYTHON:-python}
export PYTHONIOENCODING=utf-8
rm -rf "$W"; mkdir -p "$W"
W=$(cd "$W" && pwd)
FAILS=0; PASSES=0
pass() { echo "PASS  $*"; PASSES=$((PASSES+1)); }
fail() { echo "FAIL  $*"; FAILS=$((FAILS+1)); }
check() { local name=$1; shift; if "$@"; then pass "$name"; else fail "$name"; fi; }
section() { echo; echo "=== $* ==="; }

KIN="$W/kaggle_input"                         # plays /kaggle/input
NB="$KIN/ccvc-origin-output"                  # output version of the origin notebook
section "0. synthetic data + two short origin runs (run_baseline.sh 1/4, 1/8, 1 epoch, CPU)"
$PY tools/tests/make_fake_oem.py --out "$KIN/oem-dataset" --n_train 16 --n_val 4 --size 96 > "$W/fake.log" 2>&1
check "fake data: 4 val images" test "$(ls "$KIN/oem-dataset/oem/images/val" | wc -l)" -eq 4
export SEARCH_ROOT="$KIN/oem-dataset" DEVICE=cpu NUM_WORKERS=0 RESTORE_DIR=none EVAL_EVERY=1 EPOCHS=1 \
       EXTRA="--crop_size 64 --log_every 2" OUT_ROOT="$W/work/exp/oem" PART_ROOT="$W/work/partitions/oem"
for s in 1/4 1/8; do
    bash run_baseline.sh split $s > "$W/base_${s/\//_}.log" 2>&1
    check "run_baseline.sh split $s exit 0" test $? -eq 0
done
unset SEARCH_ROOT DEVICE NUM_WORKERS RESTORE_DIR EVAL_EVERY EPOCHS EXTRA OUT_ROOT PART_ROOT
mkdir -p "$NB/exp" "$NB/partitions"
cp -r "$W/work/exp/oem" "$NB/exp/oem"; cp -r "$W/work/partitions/oem" "$NB/partitions/oem"
rm -rf "$W/work"                              # config.json val_list now points nowhere, as in a new notebook
R14="$NB/exp/oem/1_4_s0/ccvc_aug/seed0"; R18="$NB/exp/oem/1_8_s0/ccvc_aug/seed0"
check "origin runs laid out as Kaggle input (final.pth, config.json, final_eval.json)" \
    test -f "$R14/final.pth" -a -f "$R18/final.pth" -a -f "$R14/final_eval.json" -a -f "$R18/final_eval.json"
snap() { (cd "$KIN" && find . -type f | sort | xargs sha256sum); }
snap > "$W/input_before.txt"

export DIAG_SEARCH_ROOT="$KIN" DIAG_TEST_DEVICE=cpu DIAG_TEST_N_VAL=4 DIAG_NUM_WORKERS=0 DIAG_MIN_FREE_GB=0.01 \
       DIAG_TEST_MAX_IMAGES=4 DIAG_TEST_DIAG_ARGS="--k 8 --iters 5 --sample 3000"
run() { local out=$1; shift; bash tools/run_diag_origin.sh --out-root "$out" "$@" > "$W/last.log" 2>&1; }

section "1. --quick: auto-discovered paths, preflight, diagnose 1/4 branch 1 only"
run "$W/out_quick" --quick; rc=$?
check "--quick exit 0" test $rc -eq 0
check "paths found: run 1/4, val.txt (no config path -> search), data root" bash -c \
    "grep -qE 'run 1/4 .*1_4_s0.ccvc_aug.seed0 .*found' '$W/last.log' && grep -qE 'val.txt .*partitions.oem.val.txt .*under' '$W/last.log' && grep -qE 'data root .*oem-dataset.oem ' '$W/last.log'"
check "preflight lines: CUDA warning, 4 val lines, files present, checksum 'không kiểm được', verify_origin PASS" bash -c \
    "grep -q 'WARNING: no CUDA' '$W/last.log' && grep -q 'val.txt: 4 lines' '$W/last.log' && grep -q 'every image/label' '$W/last.log' && grep -q 'không kiểm được' '$W/last.log' && grep -q '1 integrity *PASS' '$W/last.log'"
check "--quick output in _quick/, no eval_ckpt run" bash -c \
    "test -f '$W/out_quick/_quick/1_4/diagnose_b1/diagnose_b1.json' && ! grep -q 'eval_ckpt.py' '$W/last.log' && test ! -e '$W/out_quick/1_4'"

section "2. full run: 6 steps, 6 separate output directories, summary A-C"
run "$W/out" ; rc=$?
cp "$W/last.log" "$W/full.log"
check "full run exit 0" test $rc -eq 0
check "order: eval 1/4, eval 1/8, diag 1/4 b1, diag 1/8 b1, diag 1/4 b2, diag 1/8 b2" bash -c \
    "test \"\$(grep -o '^--- \[[^]]*\] .*' '$W/full.log' | sed 's/^--- \[[^]]*\] //' | tr '\n' '|')\" = 'eval_ckpt 1/4|eval_ckpt 1/8|diagnose 1/4 branch 1|diagnose 1/8 branch 1|diagnose 1/4 branch 2|diagnose 1/8 branch 2|'"
check "6 outputs in 6 directories" bash -c "for s in 1_4 1_8; do test -f '$W/out/'\$s/eval_ckpt/eval_ckpt.json -a -f '$W/out/'\$s/eval_ckpt/eval_ckpt_confusion_b1.csv \
    -a -f '$W/out/'\$s/diagnose_b1/diagnose_b1.json -a -f '$W/out/'\$s/diagnose_b2/diagnose_b2_per_class.csv || exit 1; done"
check "outputs belong to their run (split of each json) and eval matches final_eval.json" $PY - "$W/out" <<'EOF'
import json, sys
d = sys.argv[1]
for s in ('1_4', '1_8'):
    e = json.load(open('%s/%s/eval_ckpt/eval_ckpt.json' % (d, s)))
    assert e['split'] == s and e['checkpoint_kind'] == 'origin' and e['check_vs_final_eval']['ok'], e['check_vs_final_eval']
    assert abs(e['agree_wrong_ratio'] + e['agree_right_ratio'] + e['conflict_ratio'] - 1) < 1e-6
    for b in (1, 2):
        g = json.load(open('%s/%s/diagnose_b%d/diagnose_b%d.json' % (d, s, b, b)))
        assert g['split'] == s and g['branch'] == b and g['settings']['mixed_threshold'] == 0.6
        assert g['settings']['boundary_radius_px_full_res'] == 3 and g['settings']['tau'] == 0.1
EOF
check "run_log.txt: start/end, exit code per step, OK vs final_eval" bash -c \
    "grep -q 'start ' '$W/out/run_log.txt' && grep -q 'end .* 0 step(s) failed' '$W/out/run_log.txt' && test \$(grep -c 'exit 0 ' '$W/out/run_log.txt') -eq 6 && test \$(grep -c 'vs final_eval.json: OK' '$W/out/run_log.txt') -ge 2"
check "summary.md: tables A, B, C (2 splits x 2 branches), no missing cell, sums OK" bash -c \
    "grep -q 'Bảng A' '$W/out/summary.md' && grep -q 'Bảng B' '$W/out/summary.md' && test \$(grep -c '^### split' '$W/out/summary.md') -eq 4 && ! grep -q 'chưa có' '$W/out/summary.md' && grep -q '1.000000 (OK)' '$W/out/summary.md' && grep -q '| Building |' '$W/out/summary.md'"
check "summary.csv: rows for A, B, C" $PY - "$W/out/summary.csv" <<'EOF'
import csv, sys
r = list(csv.DictReader(open(sys.argv[1], encoding='utf-8')))
t = {x['table'] for x in r}
assert t == {'A', 'B', 'C'}, t
assert any(x['metric'] == 'agree_wrong.in_pure_cluster_of_predicted_class' and x['branch'] == '2' and x['split'] == '1_8' for x in r)
EOF
snap > "$W/input_after.txt"
check "nothing written under the (read-only) input tree" cmp -s "$W/input_before.txt" "$W/input_after.txt"

section "3. preflight failures stop before any step (exit 2)"
mv "$R18/final_eval.json" "$W/fe.bak"
run "$W/out_e1"; rc=$?
check "missing final_eval.json -> exit 2, named" bash -c "test $rc -eq 2 && grep -q 'missing final_eval.json' '$W/last.log' && ! grep -q '^--- \[' '$W/last.log'"
mv "$W/fe.bak" "$R18/final_eval.json"
DIAG_TEST_N_VAL=384 run "$W/out_e2"; rc=$?
check "val.txt line count != expected -> exit 2, actual count printed" bash -c "test $rc -eq 2 && grep -q 'val.txt has 4 lines, expected 384' '$W/last.log'"
mkdir -p "$KIN/copy-same/1_4"; cp -p "$R14"/{final.pth,config.json,final_eval.json} "$KIN/copy-same/1_4/"
run "$W/out_e3" --only eval --splits 1/4; rc=$?
check "identical copy of the 1/4 run (same final.pth) counts once -> runs" bash -c "test $rc -eq 0 && grep -q '2 copies of the same final.pth' '$W/last.log'"
printf 'x' >> "$KIN/copy-same/1_4/final.pth"
run "$W/out_e4"; rc=$?
check "two different final.pth for 1/4 -> exit 2, both listed, asks for --run-1-4" bash -c \
    "test $rc -eq 2 && grep -q 'found 2' '$W/last.log' && grep -q 'copy-same' '$W/last.log' && grep -q 'pass --run-1-4' '$W/last.log'"
run "$W/out_e5" --run-1-4 "$R14" --only eval --splits 1/4; rc=$?
check "... and --run-1-4 resolves it" test $rc -eq 0
rm -rf "$KIN/copy-same"
V1=$(ls "$KIN/oem-dataset/oem/images/val" | head -n 1)
mv "$KIN/oem-dataset/oem/images/val/$V1" "$W/img.bak"
run "$W/out_e6"; rc=$?
check "missing val image -> exit 2, named" bash -c "test $rc -eq 2 && grep -q 'missing under' '$W/last.log' && grep -q '$V1' '$W/last.log'"
mv "$W/img.bak" "$KIN/oem-dataset/oem/images/val/$V1"

section "4. one step fails midway: the others still run, summary says 'chưa có'"
cp -p "$R18/final.pth" "$W/ck.bak"; head -c 1000 "$W/ck.bak" > "$R18/final.pth"
run "$W/out_e7"; rc=$?
check "corrupted 1/8 checkpoint -> exit 4" test $rc -eq 4
check "1/8 steps fail (exit != 0), 1/4 steps succeed" bash -c \
    "grep -E '^  eval_ckpt 1/8 +exit [1-9]' '$W/last.log' >/dev/null && grep -E '^  diagnose 1/8 branch 2 +exit [1-9]' '$W/last.log' >/dev/null && test \$(grep -cE '^  (eval_ckpt|diagnose) 1/4.*exit 0 ' '$W/last.log') -eq 3"
check "1/4 outputs written, summary.md has 'chưa có (xem run_log)' for 1/8" bash -c \
    "test -f '$W/out_e7/1_4/diagnose_b2/diagnose_b2.json' && grep -q 'chưa có (xem run_log)' '$W/out_e7/summary.md' && test \$(grep -c 'nhánh' '$W/out_e7/summary.md') -eq 4"
cp -p "$W/ck.bak" "$R18/final.pth"
snap > "$W/input_after2.txt"
check "input tree identical after the failure cases (restored)" cmp -s "$W/input_before.txt" "$W/input_after2.txt"

section "4b. one split per session: --split 1/4 with only the 1/4 run attached, later --split 1/8, then merge"
mv "$NB/exp/oem/1_8_s0" "$W/r18.away"                       # session 1: the 1/8 run is not attached
run "$W/s14" --split 1/4 --quick; rc=$?
check "session 1/4 --quick (1/8 run absent): exit 0, diagnose 1/4 in _quick/" bash -c \
    "test $rc -eq 0 && test -f '$W/s14/_quick/1_4/diagnose_b1/diagnose_b1.json' && ! grep -q 'run 1/8' '$W/last.log'"
run "$W/s14" --split 1/4; rc=$?
check "session 1/4: exit 0, 3 steps (eval, diagnose b1, b2), only 1_4/ written" bash -c \
    "test $rc -eq 0 && test \$(grep -cE '^  (eval_ckpt|diagnose) 1/4.*exit 0 ' '$W/last.log') -eq 3 && test ! -e '$W/s14/1_8' && test -f '$W/s14/1_4/diagnose_b2/diagnose_b2.json'"
check "session 1/4: summary.md has only the 1/4 column" bash -c \
    "grep -q '^| chỉ số | 1/4 |\$' '$W/s14/summary.md' && ! grep -q 'chưa có' '$W/s14/summary.md'"
mv "$W/r18.away" "$NB/exp/oem/1_8_s0"
mv "$NB/exp/oem/1_4_s0" "$W/r14.away"                       # session 2: only the 1/8 run attached
run "$W/s18" --split 1/8 --quick; rc=$?
check "session 1/8 --quick (1/4 run absent): exit 0, diagnose 1/8 in _quick/" bash -c \
    "test $rc -eq 0 && test -f '$W/s18/_quick/1_8/diagnose_b1/diagnose_b1.json' && grep -q 'quick: diagnose 1/8 branch 1' '$W/last.log'"
run "$W/s18" --split 1/8; rc=$?
check "session 1/8: exit 0, 3 steps, only 1_8/ written" bash -c \
    "test $rc -eq 0 && test \$(grep -cE '^  (eval_ckpt|diagnose) 1/8.*exit 0 ' '$W/last.log') -eq 3 && test ! -e '$W/s18/1_4'"
mv "$W/r14.away" "$NB/exp/oem/1_4_s0"
mkdir -p "$W/merged" && cp -r "$W/s14/1_4" "$W/s18/1_8" "$W/merged/"
$PY tools/summarize_diag_origin.py summary --out-root "$W/merged" > /dev/null 2>&1
check "merge of the two sessions: one summary with 1/4 and 1/8, nothing missing" bash -c \
    "grep -q '^| chỉ số | 1/4 | 1/8 |\$' '$W/merged/summary.md' && ! grep -q 'chưa có' '$W/merged/summary.md' && test \$(grep -c '^### split' '$W/merged/summary.md') -eq 4"
snap > "$W/input_after3.txt"
check "input tree identical after the per-split sessions" cmp -s "$W/input_before.txt" "$W/input_after3.txt"

section "5. origin untouched, DAPN markers, diagnose unit test"
$PY tools/verify_origin.py --steps 1 > "$W/vo.log" 2>&1; check "verify_origin step 1 PASS" test $? -eq 0
$PY tools/check_dapn_diff.py > "$W/cd.log" 2>&1; check "check_dapn_diff PASS" test $? -eq 0
$PY tools/tests_dapn/test_diagnose.py > "$W/td.log" 2>&1
check "test_diagnose.py: 0 failed" grep -q "test_diagnose: 0 failed" "$W/td.log"

echo
echo "test_run_diag_origin: $PASSES passed, $FAILS failed  (logs in $W)"
exit $FAILS
