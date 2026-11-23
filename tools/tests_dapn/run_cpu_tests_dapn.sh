#!/bin/bash
# CPU test suite of CCVC-DAPN (spec 9.1 tests 1-18 + DAPN on both sub-networks; stage 2: 13, 17, dry-run
# comparison with the baseline (spec 7), collect_results_dapn.py).
#   bash tools/tests_dapn/run_cpu_tests_dapn.sh <work_dir>        (work_dir is wiped first; outside the repo)
# Env: DAPN_ORIG_DIR=<dir with the original dynamic_anchor.py / dapg_loss.py> (default: downloaded, pinned commit)
# Prints PASS/FAIL per check and a summary; exit code = number of failed checks. Commit before running
# (test 18 packages HEAD).
set -u
W=${1:?usage: run_cpu_tests_dapn.sh <work_dir>}
cd "$(dirname "$0")/../.." || exit 1
REPO=$(pwd)
PY=${PYTHON:-python}
export PYTHONIOENCODING=utf-8
rm -rf "$W"; mkdir -p "$W/input" "$W/hook"
W=$(cd "$W" && pwd)
FAILS=0; PASSES=0
pass() { echo "PASS  $*"; PASSES=$((PASSES+1)); }
fail() { echo "FAIL  $*"; FAILS=$((FAILS+1)); }
check() { local name=$1; shift; if "$@"; then pass "$name"; else fail "$name"; fi; }
section() { echo; echo "=== $* ==="; }
filt() { grep -v "Warning\|warn(\|super().__init__\|oneDNN\|Consider using tensor.detach" ; }

export SEARCH_ROOT="$W/input" DEVICE=cpu NUM_WORKERS=0 RESTORE_DIR=none EVAL_EVERY=1 EPOCHS=1 \
       EXTRA="--crop_size 64 --log_every 2"
O_OUT="$W/work/exp/oem"; O_PART="$W/work/partitions/oem"
D_OUT="$W/work/exp/oem_dapn"; D_PART="$W/work/partitions/oem_dapn"
base() { OUT_ROOT=$O_OUT PART_ROOT=$O_PART bash run_baseline.sh "$@" > "$W/last.log" 2>&1; local rc=$?; filt < "$W/last.log" >> "$W/all_runs.log"; return $rc; }
dapn() { OUT_ROOT=$D_OUT PART_ROOT=$D_PART bash run_dapn.sh "$@" > "$W/last.log" 2>&1; local rc=$?; filt < "$W/last.log" >> "$W/all_runs.log"; return $rc; }

section "0. synthetic data"
$PY tools/tests/make_fake_oem.py --out "$W/input" --n_train 16 --n_val 4 --size 96 2>&1 | filt
check "fake data: 16 train / 4 val" test "$(ls "$W/input/oem/images/train" | wc -l)" -eq 16

section "1. verify_origin step 1 (integrity); one byte changed in a copy -> FAIL"
$PY tools/verify_origin.py --steps 1 > "$W/v1.log" 2>&1; check "integrity PASS on this repository" test $? -eq 0
mkdir -p "$W/tamper" && git archive HEAD | tar -x -C "$W/tamper"
printf ' ' >> "$W/tamper/util/utils.py"
$PY "$W/tamper/tools/verify_origin.py" --steps 1 > "$W/v1t.log" 2>&1; rc=$?
check "modified origin file -> FAIL (exit 1)" test $rc -eq 1
check "the modified file is named" grep -q "modified util/utils.py" "$W/v1t.log"
rm "$W/tamper/CCVC_aug.py"
$PY "$W/tamper/tools/verify_origin.py" --steps 1 > "$W/v1m.log" 2>&1
check "missing origin file -> FAIL and named" grep -q "missing  CCVC_aug.py" "$W/v1m.log"

section "2/4. golden (from tag ccvc-origin): CCVC_aug.py and CCVC_aug_dapn.py --use_dapg False"
$PY tools/verify_origin.py --steps 2,3 --work "$W/verify" > "$W/v23.log" 2>&1
cat "$W/v23.log" | grep -E "^\s+[23] "
check "step 2: CCVC_aug.py == golden (losses of 5 iterations + weights)" grep -qE "^\s+2 behaviour\s+PASS" "$W/v23.log"
check "step 3: CCVC_aug_dapn.py --use_dapg False == golden" grep -qE "^\s+3 superset\s+PASS" "$W/v23.log"

section "5. initialisation: ccvc_dapn and CCVC_aug.py, same seed -> origin part bit-identical"
$PY tools/golden_capture.py --script CCVC_aug.py --work "$W/verify" --at init --out "$W/init_o.json" > /dev/null 2>&1
$PY tools/golden_capture.py --script CCVC_aug_dapn.py --work "$W/verify" --at init --out "$W/init_d.json" > /dev/null 2>&1
check "origin-part checksum identical; DAPN model holds da1 + da2" $PY - "$W/init_o.json" "$W/init_d.json" <<'EOF'
import json, sys
o, d = (json.load(open(p)) for p in sys.argv[1:])
assert o['init_sha256'] == d['init_sha256'] and o['init_keys'] == d['init_keys'], (o['init_sha256'], d['init_sha256'])
assert o['da_keys'] == [] and d['da_keys'] == ['da1', 'da2'], d['da_keys']
EOF

section "6. check_dapn_diff"
$PY tools/check_dapn_diff.py > /dev/null; check "CCVC_aug_dapn.py: every difference marked [DAPN]" test $? -eq 0
sed 's/^            loss_con = loss_con \* args.w_con$/            loss_con = loss_con * args.w_con  # changed/' CCVC_aug_dapn.py > "$W/bad_dapn.py"
$PY tools/check_dapn_diff.py --dapn "$W/bad_dapn.py" > /dev/null; check "one unmarked line -> FAIL" test $? -eq 1

section "7. DAPN copy == original DAPN files (main_final 33ad638)"
$PY tools/tests_dapn/test_dapn_equivalence.py ${DAPN_ORIG_DIR:+--orig_dir "$DAPN_ORIG_DIR"} > "$W/eq.log" 2>&1
cat "$W/eq.log" | tail -1; check "assign / proto / quality / loss / gradients within 1e-6" test $? -eq 0 -a -n "$(grep '^PASS' "$W/eq.log")"

section "8-12. unit tests (presets, both sub-networks, gradients, optimizer, masks, fp32, evaluation)"
$PY tools/tests_dapn/test_dapn_unit.py 2>&1 | filt | tee "$W/unit.log" | grep -E "^(PASS|FAIL)"
check "test_dapn_unit.py: 0 failed" grep -q "test_dapn_unit: 0 failed" "$W/unit.log"

section "3. origin path imports no DAPN file (sys.modules at exit of every python process)"
cat > "$W/hook/sitecustomize.py" <<'EOF'
import atexit, os, sys
_p = os.environ.get('MODLOG')
if _p:
    def _dump():
        with open(_p, 'a') as f:
            f.write(os.path.basename(sys.argv[0]) + ' :: ' + ' '.join(sorted(sys.modules)) + '\n')
    atexit.register(_dump)
EOF
# module names only (the repository folder CCVC-DAPN may appear in paths, not in module names)
dapn_mods() { sed 's/^.* :: //' "$1" | tr ' ' '\n' | grep -iE "dapn" | sort -u | tr '\n' ' '; }
PYTHONPATH="$W/hook" MODLOG="$W/mods_baseline.txt" base split 1/4 --dryrun --trial-iters 2
check "run_baseline.sh --dryrun exit 0" test $? -eq 0
check "run_baseline.sh: training process ran (module log written)" grep -q "^CCVC_aug.py :: " "$W/mods_baseline.txt"
check "run_baseline.sh: no *dapn* module imported [$(dapn_mods "$W/mods_baseline.txt")]" test -z "$(dapn_mods "$W/mods_baseline.txt")"
PYTHONPATH="$W/hook" MODLOG="$W/mods_origin.txt" dapn split 1/4 --method ccvc_origin --dryrun --trial-iters 2
check "run_dapn.sh --method ccvc_origin --dryrun exit 0" test $? -eq 0
check "ccvc_origin: training script is CCVC_aug.py" bash -c "grep -q '^CCVC_aug.py :: ' '$W/mods_origin.txt'"
check "ccvc_origin: CCVC_aug.py process imports no *dapn* module" bash -c "! grep '^CCVC_aug.py :: ' '$W/mods_origin.txt' | sed 's/^.* :: //' | grep -qi dapn"
PYTHONPATH="$W/hook" MODLOG="$W/mods_dapn.txt" dapn split 1/4 --dryrun --trial-iters 2
check "control: ccvc_dapn process does import model.dapn / valid_dapn / run_utils_dapn" bash -c \
    "grep '^CCVC_aug_dapn.py :: ' '$W/mods_dapn.txt' | grep -q 'model.dapn ' && grep '^CCVC_aug_dapn.py :: ' '$W/mods_dapn.txt' | grep -q 'valid_dapn' && grep '^CCVC_aug_dapn.py :: ' '$W/mods_dapn.txt' | grep -q 'util.run_utils_dapn'"

section "full runs (1 epoch): run_baseline.sh ccvc_aug and run_dapn.sh ccvc_dapn, split 1/4"
base split 1/4; check "run_baseline.sh split 1/4 exit 0" test $? -eq 0
dapn split 1/4; check "run_dapn.sh split 1/4 (ccvc_dapn) exit 0" test $? -eq 0
RO="$O_OUT/1_4_s0/ccvc_aug/seed0"; RD="$D_OUT/1_4_s0/ccvc_dapn/seed0"
check "DAPN on both sub-networks end to end: final.pth has da1.* and da2.*; train_log has b1 and b2 columns filled" \
    $PY - "$RD" <<'EOF'
import csv, sys, torch
d = sys.argv[1]
sd = torch.load(d + '/final.pth', map_location='cpu')
assert any(k.startswith('da1.') for k in sd) and any(k.startswith('da2.') for k in sd)
rows = list(csv.DictReader(open(d + '/train_log.csv')))
for r in rows:
    for k in ('intra_b1', 'intra_b2', 'n_valid_proto_b1', 'n_valid_proto_b2', 'L_dapg_b1', 'L_dapg_b2', 'L_dapg_l', 'L_dapg_u', 'cc_ratio_loss'):
        assert r[k] != '' and float(r[k]) >= 0, (k, r[k])
    assert float(r['L_dapg_u']) > 0 and float(r['L_dapg_b2']) > 0
EOF
check "DAPN outputs: eval_log agreement columns, final_eval agreement metrics, confusion CSVs, config keys" \
    $PY - "$RD" <<'EOF'
import csv, json, os, sys
d = sys.argv[1]
r = list(csv.DictReader(open(d + '/eval_log.csv')))[-1]
s = float(r['agree_wrong_ratio']) + float(r['agree_right_ratio']) + float(r['conflict_ratio'])
assert abs(s - 1) < 1e-4, s
assert 'agree_wrong_Building' in r
f = json.load(open(d + '/final_eval.json'))
assert f['code_version'] == 'dapn' and 'agree_wrong_ratio' in f and len(f['agree_wrong_class']) == 8
assert os.path.exists(d + '/confusion_b1.csv') and os.path.exists(d + '/confusion_agree_wrong.csv')
c = json.load(open(d + '/config.json'))
for k, v in {'code_version': 'dapn', 'use_dapg': True, 'dapg_on': 'lu', 'dapg_branches': 'both', 'w_dapg': 0.2,
             'dapg_pos': 'feat', 'origin_commit': 'b44dc5707d98e12d8b015697c4f2dd66019c5ebb',
             'dapn_commit': '33ad6387068f78240e8b04f1dbddbc8dd6dc5d6c', 'dapn_branch': 'main_final',
             'verify_origin': 'PASS', 'da_max_groups': 64, 'da_temperature': 0.1}.items():
    assert c.get(k) == v, (k, c.get(k))
assert c['deviations_from_dapn'] and c['ccvc_dapn_commit']
EOF

section "stage-2 measurements run by the same command right after training (tools/post_run_dapn.py)"
cp "$W/all_runs.log" "$W/after_full.log"
check "ccvc_dapn run: diagnose_b1.json and diagnose_b2.json written after training" test -f "$RD/diagnose_b1.json" -a -f "$RD/diagnose_b2.json"
check "baseline (run_baseline.sh 1/4, same seed): eval_ckpt.json + diagnose_b1/b2.json written" \
    test -f "$RO/eval_ckpt.json" -a -f "$RO/diagnose_b1.json" -a -f "$RO/diagnose_b2.json"
check "results_summary.csv in exp/oem_dapn with the origin and the DAPN row" $PY - "$D_OUT/results_summary.csv" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
k = {(r['code_version'], r['split'], r['method']) for r in rows}
assert ('origin', '1_4', 'ccvc_aug') in k and ('dapn', '1_4', 'ccvc_dapn') in k, k
o = [r for r in rows if r['code_version'] == 'origin'][0]
assert o['agree_wrong_ratio'] != '' and o['diag_b1_purity_agree_wrong'] != '' and o['diag_b2_purity_agree_wrong'] != ''
EOF
check "summary table printed, analysis complete, exit 0" bash -c "grep -aq 'ANALYSIS SUMMARY' '$W/after_full.log' && grep -aq 'analysis complete' '$W/after_full.log'"
dapn split 1/4; rc=$?
check "same command again: exit 0, every step already done (nothing recomputed)" bash -c "test $rc -eq 0 && test \$(grep -ac 'already done' '$W/last.log') -ge 5 && ! grep -aq 'estimated' '$W/last.log'"
rm "$RD/diagnose_b2.json"
SESSION_START=$(( $(date +%s) - 12 * 3600 )) dapn split 1/4; rc=$?
check "no session time left: step not started, message to rerun, exit 0" bash -c "test $rc -eq 0 && grep -aq 'NOT STARTED' '$W/last.log' && grep -aq 'analysis INCOMPLETE' '$W/last.log' && test ! -f '$RD/diagnose_b2.json'"
dapn split 1/4; rc=$?
check "next call of the same command finishes the missing step only" bash -c "test $rc -eq 0 && test -f '$RD/diagnose_b2.json' && test \$(grep -ac 'estimated' '$W/last.log') -eq 1"
dapn split 1/4 --no-analysis; rc=$?
check "--no-analysis: skipped" bash -c "test $rc -eq 0 && grep -aq 'analysis skipped' '$W/last.log'"

section "12. eval_ckpt.py: origin and DAPN checkpoints reproduce final_eval.json"
$PY tools/eval_ckpt.py --ckpt "$RO/final.pth" --device cpu --num-workers 0 > "$W/ec_o.log" 2>&1
check "origin checkpoint: three mIoU = final_eval.json (+-0.01)" test $? -eq 0
check "origin checkpoint recognised as origin" grep -q "origin checkpoint" "$W/ec_o.log"
check "origin: eval_ckpt.json written next to the checkpoint, agreement metrics sum to 1" $PY - "$RO/eval_ckpt.json" <<'EOF'
import json, sys
e = json.load(open(sys.argv[1]))
assert e["checkpoint_kind"] == "origin" and abs(e["agree_wrong_ratio"] + e["agree_right_ratio"] + e["conflict_ratio"] - 1) < 1e-9
EOF
$PY tools/eval_ckpt.py --ckpt "$RD/final.pth" --device cpu --num-workers 0 --out "$W/eval_dapn.json" > "$W/ec_d.log" 2>&1
check "DAPN checkpoint: three mIoU = final_eval.json (+-0.01)" test $? -eq 0
check "DAPN checkpoint: da1, da2 ignored at evaluation" grep -q "dapn (da1, da2 ignored" "$W/ec_d.log"
check "eval_ckpt agreement metrics = final_eval.json of the DAPN run" $PY - "$W/eval_dapn.json" "$RD/final_eval.json" <<'EOF'
import json, sys
e, f = (json.load(open(p)) for p in sys.argv[1:])
for k in ('agree_wrong_ratio', 'agree_right_ratio', 'conflict_ratio', 'err_b1'):
    assert abs(e[k] - f[k]) < 1e-6, (k, e[k], f[k])
assert abs(e['agree_wrong_ratio'] + e['agree_right_ratio'] + e['conflict_ratio'] - 1) < 1e-9
EOF

section "13. diagnose_entanglement.py: constructed features, then origin and DAPN checkpoints"
$PY tools/tests_dapn/test_diagnose.py 2>&1 | filt | tee "$W/diag_unit.log" | grep -E "^(PASS|FAIL)"
check "test_diagnose.py: 0 failed (separable features -> purity ~ 1)" grep -q "test_diagnose: 0 failed" "$W/diag_unit.log"
for B in 1 2; do
    $PY tools/diagnose_entanglement.py --ckpt "$RO/final.pth" --branch $B --k 8 --iters 5 --sample 3000 \
        --device cpu --num-workers 0 > "$W/diag_o$B.log" 2>&1
    check "origin checkpoint, branch $B: diagnose_b$B.json + per_class.csv" test $? -eq 0 -a -f "$RO/diagnose_b$B.json" -a -f "$RO/diagnose_b${B}_per_class.csv"
done
$PY tools/diagnose_entanglement.py --ckpt "$RD/final.pth" --branch 1 --k 8 --iters 5 --sample 3000 \
    --device cpu --num-workers 0 > "$W/diag_d1.log" 2>&1
check "DAPN checkpoint, branch 1: runs (da1/da2 ignored)" bash -c "test -f '$RD/diagnose_b1.json' && grep -q 'dapn (da1, da2 ignored' '$W/diag_d1.log'"
check "diagnose outputs: 3 EM seeds, shares sum to 1, purity in [0,1], 8 classes per CSV" $PY - "$RO" <<'EOF'
import csv, json, sys
d = sys.argv[1]
for b in (1, 2):
    j = json.load(open('%s/diagnose_b%d.json' % (d, b)))
    o = j['overall']
    assert len(j['per_seed']) == 3 and j['branch'] == b and j['checkpoint_kind'] == 'origin'
    assert abs(sum(o[g + '.share']['mean'] for g in ('agree_right', 'agree_wrong', 'conflict')) - 1) < 1e-9
    assert 0 <= o['mean_purity_all_pixels']['mean'] <= 1
    rows = list(csv.DictReader(open('%s/diagnose_b%d_per_class.csv' % (d, b))))
    assert len({r['class'] for r in rows}) == 8
EOF

section "collect_results_dapn.py: origin + DAPN runs in one table"
$PY tools/collect_results_dapn.py --roots "$O_OUT" "$D_OUT" --out "$W/results_summary.csv" > "$W/collect.log" 2>&1
check "collect: exit 0" test $? -eq 0
check "collect: origin row (agree_* from eval_ckpt.json, diagnose columns) and DAPN row (agree_* from final_eval.json)" \
    $PY - "$W/results_summary.csv" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
o = [r for r in rows if r['code_version'] == 'origin' and r['split'] == '1_4' and r['method'] == 'ccvc_aug']
d = [r for r in rows if r['code_version'] == 'dapn' and r['split'] == '1_4' and r['method'] == 'ccvc_dapn']
assert len(o) == 1 and len(d) == 1, [(r['code_version'], r['split'], r['method']) for r in rows]
assert o[0]['agree_source'] == 'eval_ckpt' and o[0]['agree_wrong_ratio'] != '' and o[0]['diag_b1_purity_agree_wrong'] != ''
assert o[0]['diag_b2_purity_agree_right'] != ''
assert d[0]['agree_source'] == 'final_eval' and d[0]['dapg_branches'] == 'both' and d[0]['diag_b1_purity_agree_right'] != ''
EOF
check "collect: mean +- std over seeds printed" grep -q "mean +- std over seeds" "$W/collect.log"

section "14. output isolation and restore_output_dapn.py"
check "origin results only under exp/oem, DAPN results only under exp/oem_dapn" bash -c \
    "test -f '$RO/final_eval.json' -a -f '$RD/final_eval.json' -a ! -e '$O_OUT/1_4_s0/ccvc_dapn' -a ! -e '$D_OUT/1_4_s0/ccvc_aug'"
mkdir -p "$W/prev/out" && cp -r "$W/work/exp" "$W/work/partitions" "$W/prev/out/"
$PY tools/restore_output_dapn.py --restore_dir "$W/prev/out" --out_root "$W/restored/exp/oem_dapn" \
    --part_root "$W/restored/partitions/oem_dapn" > "$W/restore.log" 2>&1
check "restore: exit 0" test $? -eq 0
check "restore: DAPN run restored" test -f "$W/restored/exp/oem_dapn/1_4_s0/ccvc_dapn/seed0/final_eval.json"
check "restore: origin results NOT copied" test ! -e "$W/restored/exp/oem"
check "restore: warning about the origin results" grep -q "holds origin results" "$W/restore.log"

section "15. partitions vs the baseline run (--baseline-part-root)"
dapn split 1/4 --baseline-part-root "$O_PART" --dryrun --trial-iters 2
check "identical split -> run (exit 0)" test $? -eq 0
check "logged: identical to baseline" grep -q "partition check vs baseline: identical_to_baseline" "$W/last.log"
cp -r "$O_PART" "$W/base_bad"
sed -i '1d' "$W/base_bad/1_4_s0/labeled.txt"
dapn split 1/4 --baseline-part-root "$W/base_bad" --dryrun --trial-iters 2; rc=$?
check "one line different -> stop (exit 2)" test $rc -eq 2
check "message: split differs from the baseline run" grep -q "split differs from the baseline run" "$W/last.log"

section "16. resume ccvc_dapn (2 epochs, session limit after epoch 1)"
EPOCHS=2 MAX_SESSION_HOURS=0.000001 dapn split 1/8; rc=$?
R8="$D_OUT/1_8_s0/ccvc_dapn/seed0"
check "stopped for resume (exit 3), last.pth written" test $rc -eq 3 -a -f "$R8/last.pth"
check "last.pth holds da1.* and da2.*" $PY -c "
import sys, torch; sd = torch.load(sys.argv[1], map_location='cpu', weights_only=False)['model']
assert any(k.startswith('da1.') for k in sd) and any(k.startswith('da2.') for k in sd)" "$R8/last.pth"
EPOCHS=2 EXTRA="--crop_size 64 --log_every 2 --w_dapg 0.3" dapn split 1/8; rc=$?
check "w_dapg changed -> resume blocked" bash -c "test $rc -ne 0 && grep -q 'resume blocked: DAPN configuration differs' '$W/last.log' && grep -q 'w_dapg' '$W/last.log'"
EPOCHS=2 dapn split 1/8; rc=$?
check "same command -> resumes and finishes (exit 0)" test $rc -eq 0 -a -f "$R8/final_eval.json"
check "resumed at epoch 1 with da* restored from last.pth" grep -q "RESUME from epoch 1" "$R8/log.txt"
check "train_log.csv continues to the last iteration, no duplicates" $PY - "$R8" <<'EOF'
import csv, json, sys
it = [int(r['iteration']) for r in csv.DictReader(open(sys.argv[1] + '/train_log.csv'))]
total = json.load(open(sys.argv[1] + '/config.json'))['total_iters']
assert it == sorted(set(it)) and it[-1] == total, (it, total)
EOF

section "17. --dapg-pos c4: a few iterations end to end"
EXTRA="--crop_size 192 --log_every 2" dapn split 1/4 --dapg-pos c4 --dryrun --trial-iters 3 --no-baseline-trial
check "run_dapn.sh --dapg-pos c4 --dryrun exit 0" test $? -eq 0
RC4="$D_OUT/1_4_s0/ccvc_dapn_c4/seed0/trial"
check "c4: own run directory, config dapg_pos c4, DAPG on both branches (2048-d DA)" $PY - "$RC4" <<'EOF'
import csv, json, sys
d = sys.argv[1]
c = json.load(open(d + '/config.json'))
assert c['dapg_pos'] == 'c4' and c['dapg_branches'] == 'both'
r = list(csv.DictReader(open(d + '/train_log.csv')))[0]
assert float(r['L_dapg']) > 0 and r['L_dapg_b1'] != '' and r['L_dapg_b2'] != '', r
EOF

section "7. dry run vs baseline (trial_report.json: s/iter, memory, hours, % increase)"
dapn split 1/4 --dryrun --trial-iters 3
check "1/4 dry run exit 0" test $? -eq 0
check "1/4: compared with the origin dry run of run_baseline.sh (exp/oem/.../ccvc_aug/.../trial)" $PY - "$D_OUT/1_4_s0/ccvc_dapn/seed0/trial/trial_report.json" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))['dapn_vs_baseline']
assert c['baseline_method'] == 'ccvc_aug' and '/oem/1_4_s0/ccvc_aug/' in c['baseline_report'].replace(chr(92), '/')
assert c['s_per_iter_median']['dapn'] and c['s_per_iter_median']['baseline'] and c['s_per_iter_median']['increase_pct'] is not None
assert c['projected_hours_full_run']['increase_pct'] is not None and c['comparable']
EOF
check "comparison table printed" grep -q "DAPN vs baseline dry run" "$W/last.log"
dapn split 1/8 --dryrun --trial-iters 3
check "1/8 dry run (no baseline dry run yet) exit 0" test $? -eq 0
check "1/8: extra ccvc_dapn_off dry run made and used as the baseline" bash -c \
    "grep -q 'running the same dry run with --method ccvc_dapn_off' '$W/last.log' && test -f '$D_OUT/1_8_s0/ccvc_dapn_off/seed0/trial/trial_report.json'"
check "1/8: trial_report.json holds the comparison with ccvc_dapn_off" $PY - "$D_OUT/1_8_s0/ccvc_dapn/seed0/trial/trial_report.json" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))['dapn_vs_baseline']
assert c['baseline_method'] == 'ccvc_dapn_off' and c['s_per_iter_median']['increase_pct'] is not None
EOF

section "18. Kaggle package: unzip into an empty directory -> verify_origin PASS, run_dapn.sh --dryrun runs"
bash tools/make_kaggle_package_dapn.sh "$W/dist" > "$W/pkg.log" 2>&1; rc=$?
check "package built" test $rc -eq 0 -a -f "$W/dist/ccvc_dapn_oem.zip"
mkdir -p "$W/kaggle_working" && unzip -q "$W/dist/ccvc_dapn_oem.zip" -d "$W/kaggle_working"
check "zip contains origin + DAPN files, no .git" bash -c "test -f '$W/kaggle_working/CCVC-DAPN/CCVC_aug.py' -a -f '$W/kaggle_working/CCVC-DAPN/run_baseline.sh' -a -f '$W/kaggle_working/CCVC-DAPN/run_dapn.sh' -a -f '$W/kaggle_working/CCVC-DAPN/PACKAGE_COMMIT' -a ! -e '$W/kaggle_working/CCVC-DAPN/.git'"
(cd "$W/kaggle_working/CCVC-DAPN" && $PY tools/verify_origin.py --steps 1 > "$W/zip_v.log" 2>&1); check "unzipped: verify_origin step 1 PASS" test $? -eq 0
OUT_ROOT="$W/kaggle_working/exp/oem_dapn" PART_ROOT="$W/kaggle_working/partitions/oem_dapn" \
    bash "$W/kaggle_working/CCVC-DAPN/run_dapn.sh" split 1/4 --dryrun --trial-iters 2 > "$W/zip_run.log" 2>&1
check "unzipped: run_dapn.sh --dryrun exit 0" test $? -eq 0
check "unzipped run records the package commit" bash -c "grep -q '(zip)' '$W/kaggle_working/exp/oem_dapn/1_4_s0/ccvc_dapn/seed0/trial/config.json'"

echo
echo "==== run_cpu_tests_dapn: $PASSES passed, $FAILS failed ===="
exit $FAILS
