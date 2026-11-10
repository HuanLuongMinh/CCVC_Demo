#!/bin/bash
# CPU test suite for the OEM reproduction (spec 8.1 + plan additions). Synthetic data, tiny crop, few epochs.
#   bash tools/tests/run_cpu_tests.sh <work_dir>
# Prints PASS/FAIL per check and a summary; exit code = number of failed checks.
set -u
W=${1:?usage: run_cpu_tests.sh <work_dir>}
cd "$(dirname "$0")/../.." || exit 1
PY=${PYTHON:-python}
export PYTHONIOENCODING=utf-8
rm -rf "$W"; mkdir -p "$W/input"
FAILS=0; PASSES=0
pass() { echo "PASS  $*"; PASSES=$((PASSES+1)); }
fail() { echo "FAIL  $*"; FAILS=$((FAILS+1)); }
check() { local name=$1; shift; if "$@"; then pass "$name"; else fail "$name"; fi; }
section() { echo; echo "=== $* ==="; }

export SEARCH_ROOT="$W/input" OUT_ROOT="$W/work/exp/oem" PART_ROOT="$W/work/partitions/oem" DEVICE=cpu \
       NUM_WORKERS=0 RESTORE_DIR=none EVAL_EVERY=1 EPOCHS=1 EXTRA="--crop_size 64 --log_every 2"
run() { bash run_baseline.sh > "$W/last_run.log" 2>&1; local rc=$?; grep -v "Warning\|warn(\|super().__init__" "$W/last_run.log" >> "$W/all_runs.log"; return $rc; }
R() { echo "$OUT_ROOT/$1"; }     # run dir helper

section "1. synthetic data + fake weights"
$PY tools/tests/make_fake_oem.py --out "$W/input" --n_train 16 --n_val 4 --size 192 2>&1 | grep -v Warn
check "fake data: 16 train / 4 val tiles" test "$(ls "$W/input/oem/images/train" | wc -l)" -eq 16 -a "$(ls "$W/input/oem/label/val" | wc -l)" -eq 4

section "2. SPLIT=1_8 then SPLIT=1_4 (ccvc_aug): separate dirs, nested, disjoint"
SPLIT=1_8 run; check "1_8 exit 0" test $? -eq 0
SPLIT=1_4 run; check "1_4 exit 0" test $? -eq 0
check "two run dirs with final_eval.json" test -f "$(R 1_8_s0/ccvc_aug/seed0)/final_eval.json" -a -f "$(R 1_4_s0/ccvc_aug/seed0)/final_eval.json"
check "labeled(1_8) subset of labeled(1_4); labeled/unlabeled disjoint; union = train" $PY - "$PART_ROOT" <<'EOF'
import sys, os
P = sys.argv[1]
rd = lambda s, f: set(open(os.path.join(P, s, f)).read().split('\n')) - {''}
l8, u8, l4, u4 = rd('1_8_s0', 'labeled.txt'), rd('1_8_s0', 'unlabeled.txt'), rd('1_4_s0', 'labeled.txt'), rd('1_4_s0', 'unlabeled.txt')
assert len(l8) == 2 and len(l4) == 4, (len(l8), len(l4))
assert l8 < l4 and not (l8 & u8) and not (l4 & u4) and (l8 | u8) == (l4 | u4) and len(l8 | u8) == 16
EOF

section "3. SPLIT=full with FULL_UNLABELED=same and none"
SPLIT=full FULL_UNLABELED=same run; check "full/same exit 0" test $? -eq 0
SPLIT=full FULL_UNLABELED=none run; check "full/none exit 0" test $? -eq 0
check "full/none switched to sup_pure" grep -q "replaced by sup_pure" "$W/last_run.log"
check "full_same_s0 and full_none_s0 kept apart" test -f "$(R full_same_s0/ccvc_aug/seed0)/final_eval.json" -a -f "$(R full_none_s0/sup_pure/seed0)/final_eval.json"
check "full/same and full/none: same iterations" $PY - "$(R full_same_s0/ccvc_aug/seed0)" "$(R full_none_s0/sup_pure/seed0)" <<'EOF'
import json, sys
a, b = (json.load(open(d + '/config.json')) for d in sys.argv[1:])
assert a['total_iters'] == b['total_iters'] and a['n_labeled'] == b['n_labeled'] == 16, (a['total_iters'], b['total_iters'])
assert b['n_unlabeled'] == 0 and a['n_unlabeled'] == 16
EOF

section "4. several splits in one call -> error"
SPLIT="1_4 1_8" run; rc=$?; check "SPLIT='1_4 1_8' rejected (exit 2)" test $rc -eq 2
check "message says one split" grep -q "exactly ONE split" "$W/last_run.log"
SPLIT="1_4,1_8" run; check "SPLIT='1_4,1_8' rejected" test $? -eq 2
SPLIT="" run; check "SPLIT='' rejected" test $? -eq 2
SPLIT="1_3" run; check "SPLIT='1_3' rejected" test $? -eq 2

section "5. finished split is skipped; FORCE=1 renames, never deletes"
SPLIT=1_8 run; check "rerun 1_8 exit 0" test $? -eq 0
check "says already finished" grep -q "đã chạy xong" "$W/last_run.log"
SPLIT=1_8 FORCE=1 run; check "FORCE=1 run exit 0" test $? -eq 0
check "old dir renamed with final_eval.json inside" bash -c "ls -d '$(R 1_8_s0/ccvc_aug)'/seed0_old_* >/dev/null && test -f \"\$(ls -d '$(R 1_8_s0/ccvc_aug)'/seed0_old_* | head -1)/final_eval.json\""
check "new run finished" test -f "$(R 1_8_s0/ccvc_aug/seed0)/final_eval.json"

section "6/12. loss components: ccvc_aug, sup, sup_pure"
check "ccvc_aug: L_con and L_dis added, w 2/2/1" grep -q "\[iter 0\] L_total .* + L_con .*(raw .*, w_con 2) + L_dis .*(raw .*, w_dis 1)$" "$(R 1_4_s0/ccvc_aug/seed0)/log.txt"
SPLIT=1_8 METHOD=sup run; check "sup exit 0" test $? -eq 0
check "sup: L_con = 0 and L_dis = 0, not added" grep -q "\[iter 0\] L_total .* + L_con 0.0000 (raw .*, w_con 0) \[not added\] + L_dis 0.0000 (raw .*, w_dis 0) \[not added\]" "$(R 1_8_s0/sup/seed0)/log.txt"
check "sup: L_total = L_sup" $PY - "$(R 1_8_s0/sup/seed0)/log.txt" <<'EOF'
import re, sys
for line in open(sys.argv[1], encoding='utf-8'):
    m = re.search(r'\[iter \d+\] L_total ([\d.]+) = L_sup ([\d.]+)', line)
    if m: assert m.group(1) == m.group(2), line
EOF
SPLIT=1_8 METHOD=sup_pure run; check "sup_pure exit 0" test $? -eq 0
check "sup_pure: same iterations as sup, no unlabeled batch" $PY - "$(R 1_8_s0/sup/seed0)" "$(R 1_8_s0/sup_pure/seed0)" <<'EOF'
import json, sys
a, b = (json.load(open(d + '/config.json')) for d in sys.argv[1:])
assert a['total_iters'] == b['total_iters'] and b['effective_batch_unlabeled'] == 0 and b['sup_pure'], (a['total_iters'], b['total_iters'])
EOF

section "7. TRIAL=1 TRIAL_ITERS=5 -> trial_report.json"
SPLIT=1_4 TRIAL=1 TRIAL_ITERS=5 EPOCHS=80 EVAL_EVERY=5 run; check "trial exit 0" test $? -eq 0
check "trial_report fields + projection formula; no checkpoint left" $PY - "$(R 1_4_s0/ccvc_aug/seed0)/trial" <<'EOF'
import json, math, os, sys
d = sys.argv[1]; r = json.load(open(d + '/trial_report.json'))
for k in ['n_train', 'n_labeled', 'n_unlabeled', 'n_val', 'gpus', 'batch_per_gpu_per_loader', 'effective_batch_labeled',
          'amp', 'eval_mode', 's_per_iter_mean', 's_per_iter_median', 'memory_per_gpu', 'eval_time_s', 'miou_b1',
          'miou_b2', 'miou_ave', 'projection']:
    assert k in r, k
p = r['projection']
assert p['iters_per_epoch'] == math.ceil(r['n_unlabeled'] / r['effective_batch_unlabeled']) == 6, p
assert p['total_iters'] == 6 * 80 and p['n_evals'] == 16
h = (p['total_iters'] * r['s_per_iter_median'] + p['n_evals'] * r['eval_time_s']) / 3600
assert abs(h - p['total_hours']) < 1e-6 and p['kaggle_sessions'] == max(1, math.ceil(h / p['session_hours']))
assert not any(f.endswith('.pth') for f in os.listdir(d)), os.listdir(d)
assert os.path.exists(d + '/vis') and os.path.exists(d + '/plots/miou_curve.png')
EOF

section "8. resume: session limit, continuity, blocked config change, interrupted save"
SPLIT=n6 EPOCHS=3 MAX_SESSION_HOURS=0.00001 run; rc=$?
D=$(R n6_s0/ccvc_aug/seed0)
check "session limit -> exit 3 after epoch 1" test $rc -eq 3
check "last.pth present, no .tmp, progress epochs_done=1" $PY - "$D" <<'EOF'
import json, os, sys
d = sys.argv[1]
assert os.path.exists(d + '/last.pth') and not os.path.exists(d + '/last.pth.tmp')
assert json.load(open(d + '/progress.json'))['epochs_done'] == 1
EOF
check "stop message printed" grep -q "STOPPED FOR RESUME" "$W/last_run.log"
cp "$D/last.pth" "$W/last_epoch1.pth"
head -c 1000 "$D/last.pth" > "$D/last.pth.tmp"    # a save killed half-way leaves only a partial .tmp
SPLIT=n6 EPOCHS=3 MAX_SESSION_HOURS=0.00001 run; rc=$?
check "resume #1 (with a stale partial .tmp) -> exit 3 after epoch 2" test $rc -eq 3
check "log says RESUME from epoch 1" grep -q "RESUME from epoch 1" "$W/last_run.log"
SPLIT=n6 EPOCHS=4 run; rc=$?
check "changed EPOCHS -> resume blocked (non-zero)" test $rc -ne 0
check "message lists the differing key" grep -q "epochs: saved=3 now=4" "$W/last_run.log"
SPLIT=n6 EPOCHS=3 MAX_SESSION_HOURS=0 run; rc=$?
check "resume #2 -> finished (exit 0)" test $rc -eq 0
check "iterations continuous, lr follows the poly schedule across resumes" $PY - "$D" <<'EOF'
import csv, json, sys
d = sys.argv[1]
cfg = json.load(open(d + '/config.json'))
rows = list(csv.DictReader(open(d + '/train_log.csv')))
its = [int(r['iteration']) for r in rows]
assert its == sorted(set(its)) and its[-1] == cfg['total_iters'], its
T = cfg['total_iters']
for r in rows:   # lr set after step i (0-based) = base * (1 - i/T)^0.9 * mult; logged after step it-1
    i = int(r['iteration']) - 1
    exp = 0.001 * (1 - i / T) ** 0.9 * 1.0
    assert abs(float(r['lr_backbone']) - exp) < 1e-6 * max(exp, 1e-9) + 1e-9, (r['iteration'], r['lr_backbone'], exp)
ev = list(csv.DictReader(open(d + '/eval_log.csv')))
assert [int(r['epoch']) for r in ev] == [0, 1, 2], [r['epoch'] for r in ev]
assert len(cfg['resumes']) == 2
EOF
check "the copy of the interrupted run's last.pth still loads" $PY -c "import torch,sys; ck=torch.load(sys.argv[1], map_location='cpu', weights_only=False); assert ck['epoch']==0" "$W/last_epoch1.pth"

section "9/13. pretrained weight check"
check "fake deep-stem weights accepted (100%)" bash -c "$PY -m model.backbone.check_pretrained --path '$W/input/model_lab/resnet101.pth' | grep -q 'matched .*(100.0%'"
for b in torchvision missing_key wrong_shape no_stem extra_key; do
    $PY -m model.backbone.check_pretrained --path "$W/input/bad_weights/$b.pth" > "$W/w_$b.log" 2>&1
    check "$b.pth rejected" test $? -eq 1
done
check "torchvision: lists conv1.0.weight missing + conv1.weight unexpected" bash -c "grep -q 'conv1.0.weight' '$W/w_torchvision.log' && grep -q '^    conv1.weight' '$W/w_torchvision.log'"
check "missing_key: names layer3.5.conv2.weight" grep -q "layer3.5.conv2.weight" "$W/w_missing_key.log"
check "wrong_shape: names layer2.0.bn1.weight with shapes" grep -q "layer2.0.bn1.weight: expected (128,), file has (7,)" "$W/w_wrong_shape.log"
check "no_stem: names conv1.6.weight as deep-stem" grep -q "conv1.6.weight: expected (128, 64, 3, 3), file has None" "$W/w_no_stem.log"
check "extra_key: names foo.bar" grep -q "foo.bar" "$W/w_extra_key.log"
mkdir -p "$W/tv_model_lab" && cp "$W/input/bad_weights/torchvision.pth" "$W/tv_model_lab/resnet101.pth"
SPLIT=1_8 SEED=7 PRETRAINED_DIR="$W/tv_model_lab" run; check "run_baseline stops on torchvision weights (exit 2)" test $? -eq 2
check "no run dir created for it" test ! -d "$(R 1_8_s0/ccvc_aug/seed7)"

section "10. collect_results.py"
$PY tools/collect_results.py --out_root "$OUT_ROOT" > "$W/collect.log" 2>&1
N_FINAL=$(find "$OUT_ROOT" -name final_eval.json -not -path "*_old_*" -not -path "*/trial/*" | wc -l)
check "results_summary.csv has one row per finished run ($N_FINAL)" test "$(($(wc -l < "$OUT_ROOT/results_summary.csv") - 1))" -eq "$N_FINAL"
cat "$W/collect.log"

section "11. panels and plots"
check "5-column panels: one per evaluation, same 4 val images" $PY - "$(R n6_s0/ccvc_aug/seed0)" <<'EOF'
import json, os, sys
d = sys.argv[1]
assert sorted(os.listdir(d + '/vis')) == ['epoch_001_iter_000005.png', 'epoch_002_iter_000010.png', 'epoch_003_iter_000015.png'], os.listdir(d + '/vis')
assert len(json.load(open(d + '/config.json'))['vis_ids']) == 4
EOF
check "plot_run.py rebuilds the 6 plots" bash -c "rm -rf '$(R n6_s0/ccvc_aug/seed0)/plots' && $PY tools/plot_run.py '$(R n6_s0/ccvc_aug/seed0)' | grep -q 'confidence_reliability.png' && test \$(ls '$(R n6_s0/ccvc_aug/seed0)/plots' | wc -l) -eq 6"
check "pseudo_label_dist.csv: baseline row + 2 rows per eval; % sums to 100" $PY - "$(R n6_s0/ccvc_aug/seed0)" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1] + '/pseudo_label_dist.csv')))
assert rows[0]['branch'] == 'labeled_gt' and len(rows) == 1 + 2 * 3, len(rows)
for r in rows:
    s = sum(float(v) for k, v in r.items() if k.startswith('pct_'))
    assert abs(s - 100) < 1e-3, s
EOF
check "confidence_reliability.csv: one row per eval, ratios in [0,1]" $PY - "$(R n6_s0/ccvc_aug/seed0)" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1] + '/confidence_reliability.csv')))
assert len(rows) == 3 and all(0 <= float(r['conf_ratio']) <= 1 for r in rows)
EOF

section "14. restore from a previous session's output"
PREV="$W/prev_input/notebook-out"; mkdir -p "$PREV"
cp -r "$W/work/exp" "$W/work/partitions" "$PREV/"
mkdir -p "$PREV/exp/oem/n6_s0/ccvc_aug/seed1"
cp "$W/last_epoch1.pth" "$PREV/exp/oem/n6_s0/ccvc_aug/seed1/last.pth"
cp "$(R n6_s0/ccvc_aug/seed0)/config.json" "$PREV/exp/oem/n6_s0/ccvc_aug/seed1/"
echo '{"epochs_done": 1, "epochs": 3}' > "$PREV/exp/oem/n6_s0/ccvc_aug/seed1/progress.json"
NEW="$W/work2"
$PY tools/restore_output.py --restore_dir auto --search_root "$W/prev_input" --out_root "$NEW/exp/oem" --part_root "$NEW/partitions/oem" > "$W/restore1.log" 2>&1
check "restore into an empty working dir (exit 0)" test $? -eq 0
check "finished runs and the in-progress run restored, trial/old skipped" $PY - "$NEW/exp/oem" <<'EOF'
import os, sys
r = sys.argv[1]
assert os.path.exists(r + '/1_4_s0/ccvc_aug/seed0/final_eval.json')
assert os.path.exists(r + '/n6_s0/ccvc_aug/seed1/last.pth')
assert not os.path.exists(r + '/1_4_s0/ccvc_aug/seed0/trial')
assert not any('_old_' in d for d in os.listdir(r + '/1_8_s0/ccvc_aug'))
EOF
check "partitions restored" test -f "$NEW/partitions/oem/1_8_s0/labeled.txt" -a -f "$NEW/partitions/oem/val.txt"
echo '{"epochs_done": 2, "epochs": 3}' > "$NEW/exp/oem/n6_s0/ccvc_aug/seed1/progress.json"
$PY tools/restore_output.py --restore_dir auto --search_root "$W/prev_input" --out_root "$NEW/exp/oem" --part_root "$NEW/partitions/oem" > "$W/restore2.log" 2>&1
check "local run further along is kept" grep -q "keep local n6_s0/ccvc_aug/seed1" "$W/restore2.log"
echo "images/train/x.tif label/train/x.tif" >> "$NEW/partitions/oem/1_8_s0/labeled.txt"
$PY tools/restore_output.py --restore_dir auto --search_root "$W/prev_input" --out_root "$NEW/exp/oem" --part_root "$NEW/partitions/oem" > "$W/restore3.log" 2>&1
check "differing partition stops the restore (exit 2)" test $? -eq 2
check "RESTORE_DIR=none does nothing" bash -c "$PY tools/restore_output.py --restore_dir none --out_root '$NEW/exp/oem' --part_root '$NEW/partitions/oem' | grep -q 'nothing to restore'"

section "15. CC mask of the CutMix pass (--cc_mask_mixed)"
# With REAL_PRETRAINED_DIR (real UniMatch resnet101.pth) the predictions vary in space, so the masks of the
# pasted and the own image differ inside the box and the check must tell both behaviours apart.
REQ=""; PRE=""
if [[ -n "${REAL_PRETRAINED_DIR:-}" ]]; then REQ="--require_differ"; PRE="$REAL_PRETRAINED_DIR"; fi
for mode in True False; do
    expect=mixed; [[ $mode == False ]] && expect=upstream
    # gamma = 0 so that every conflicting pixel is CC; dump iterations 10-14, when the (randomly initialised)
    # heads already give image-dependent predictions; every dump must be correct, at least one must discriminate
    SPLIT=1_4 SEED=5 TRIAL=1 TRIAL_ITERS=15 EPOCHS=3 PRETRAINED_DIR="$PRE" EXTRA="--crop_size 128 --log_every 2 --conf_threshold 0.0 --cc_mask_mixed $mode --debug_dump_cc $W/cc_$mode --debug_dump_cc_iter 10" run
    check "aug run with --cc_mask_mixed $mode (exit 0)" test $? -eq 0
    $PY tools/tests/check_cc_mask.py "$W"/cc_${mode}_iter*.pth --expect $expect $REQ > "$W/cc_$mode.log" 2>&1
    check "CC mask follows the $expect behaviour inside the CutMix box" test $? -eq 0
    cat "$W/cc_$mode.log"
done

section "16. command line: bash run_baseline.sh split 1/8 [--dryrun] [options]"
cli() { bash run_baseline.sh "$@" > "$W/last_run.log" 2>&1; local rc=$?; grep -v "Warning\|warn(\|super().__init__" "$W/last_run.log" >> "$W/all_runs.log"; return $rc; }
cli; check "no split -> exit 2" test $? -eq 2
check "message: missing split" grep -q "missing split" "$W/last_run.log"
cli split 1/4 1/8; check "split 1/4 1/8 -> exit 2" test $? -eq 2
check "message: only ONE split" grep -q "only ONE split" "$W/last_run.log"
cli split 1/3; check "split 1/3 -> exit 2" test $? -eq 2
cli split 1/8 --bogus; check "unknown option -> exit 2" test $? -eq 2
cli split; check "split without value -> exit 2" test $? -eq 2
cli -h; check "-h -> exit 0 + usage" bash -c "test $? -eq 0 && grep -q 'Usage: bash run_baseline.sh split' '$W/last_run.log'"
cli split 1/8 --dryrun --trial-iters 2 --epochs 2 --eval-every 1 --seed 3 --extra "--crop_size 64 --log_every 2"
check "split 1/8 --dryrun (exit 0)" test $? -eq 0
check "dry run -> 1_8_s0/ccvc_aug/seed3/trial/trial_report.json" test -f "$(R 1_8_s0/ccvc_aug/seed3/trial)/trial_report.json"
check "results printed at the end (TRIAL + summary table)" bash -c "grep -q 'RESULTS ' '$W/last_run.log' && grep -q 'TRIAL  mIoU b1' '$W/last_run.log' && grep -q 'finished run(s)' '$W/last_run.log'"
cli split 1/8 --method sup_pure --epochs 1 --eval-every 1 --extra "--crop_size 64 --log_every 2"
check "split 1/8 --method sup_pure (exit 0)" test $? -eq 0
cli split 1/8 --method sup_pure --epochs 1
check "finished split -> skipped, results shown" bash -c "test $? -eq 0 && grep -q 'đã chạy xong' '$W/last_run.log' && grep -q 'FINISHED' '$W/last_run.log'"

echo
echo "================ $PASSES passed, $FAILS failed ================"
exit $FAILS
