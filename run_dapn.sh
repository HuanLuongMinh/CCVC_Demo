#!/bin/bash
# CCVC + DAPN (Dynamic Anchor + DAPG loss on BOTH sub-networks) on OpenEarthMap - ONE split per call.
# Independent of run_baseline.sh (which stays the origin entry point, unchanged).
#
#   bash run_dapn.sh split 1/4 --dryrun                 # short trial: 300 iterations + 1 evaluation + trial_report.json
#   bash run_dapn.sh split 1/4                          # full run, method ccvc_dapn (default)
#   bash run_dapn.sh split 1/4 --method ccvc_origin     # CCVC origin (CCVC_aug.py, untouched) from this repository
#   bash run_dapn.sh -h                                 # every option
#
# Exit code: 0 = finished (or already finished / dry run done), 3 = stopped by --max-session-hours
# (run the SAME command in the next Kaggle session to resume), anything else = error.

# ===== CẤU HÌNH MẶC ĐỊNH — thường KHÔNG cần sửa: truyền bằng tham số dòng lệnh (bash run_dapn.sh -h) =====
SPLIT=${SPLIT-}                             # bắt buộc truyền `split <x>` (1/16 | 1/8 | 1/4 | 1/2 | full | n<số>)
TRIAL=${TRIAL:-0}                           # --dryrun
TRIAL_ITERS=${TRIAL_ITERS:-300}
METHOD=${METHOD:-ccvc_dapn}                 # ccvc_dapn | ccvc_dapn_l | ccvc_dapn_b1 | ccvc_dapn_off | ccvc_origin
W_DAPG=${W_DAPG:-0.2}                       # trọng số L_DAPG (giữ tỉ lệ CE : DAPG = 1 : 0,1 của DAPN vì w_CE = 2)
DAPG_POS=${DAPG_POS:-feat}                  # feat (304 kênh, mặc định) | c4 (layer4, ablation)
BASELINE_PART_ROOT=${BASELINE_PART_ROOT:-}  # partitions/oem của lần chạy baseline: đối chiếu sha256, khác -> dừng
SEED=${SEED:-0}
SPLIT_SEED=${SPLIT_SEED:-0}
FULL_UNLABELED=${FULL_UNLABELED:-same}
IMPORT_LABELED=${IMPORT_LABELED:-}
EPOCHS=${EPOCHS:-80}
EVAL_EVERY=${EVAL_EVERY:-5}
BATCH_SIZE=${BATCH_SIZE:-2}
LAMBDA_PRESET=${LAMBDA_PRESET:-blended}     # blended = 2/2/1 (như baseline)
DATA_ROOT=${DATA_ROOT:-}
PRETRAINED_DIR=${PRETRAINED_DIR:-}
OUT_ROOT=${OUT_ROOT:-/kaggle/working/exp/oem_dapn}            # origin: /kaggle/working/exp/oem
PART_ROOT=${PART_ROOT:-/kaggle/working/partitions/oem_dapn}   # origin: /kaggle/working/partitions/oem
RESTORE_DIR=${RESTORE_DIR:-auto}            # auto: chép exp/oem_dapn + partitions/oem_dapn của phiên trước | <dir> | none
MAX_SESSION_HOURS=${MAX_SESSION_HOURS:-11.25}
NUM_WORKERS=${NUM_WORKERS:-2}
FORCE=${FORCE:-0}
EXTRA=${EXTRA:-}
# =============================================
SEARCH_ROOT=${SEARCH_ROOT:-/kaggle/input}
DEVICE=${DEVICE:-cuda}
SESSION_START=${SESSION_START:-$(date +%s)}

set -uo pipefail
cd "$(dirname "$0")" || exit 2
PY=${PYTHON:-python}
die() { echo "[run_dapn] ERROR: $*" >&2; exit 2; }
log() { echo "[run_dapn] $*"; }

usage() {
    cat <<'EOF'
Usage: bash run_dapn.sh split <1/16|1/8|1/4|1/2|full|n<k>> [options]

  split <x>                   the ONE split of this run (1/4 and 1_4 both work)
  --dryrun                    short trial: --trial-iters iterations + 1 evaluation
  --trial-iters <n>           iterations of the dry run (default 300)
  --method <m>                ccvc_dapn (default: DAPG on labeled+unlabeled, da1 on branch 1 AND da2 on branch 2)
                              ccvc_dapn_l   DAPG on labeled images only (both branches)
                              ccvc_dapn_b1  DAPG on branch 1 only (ablation: one prototype set)
                              ccvc_dapn_off CCVC_aug_dapn.py with DAPG off (= origin; equivalence tests)
                              ccvc_origin   CCVC_aug.py of the origin, untouched (= run_baseline.sh)
  --w-dapg <x>                weight of L_DAPG (default 0.2)
  --dapg-pos <p>              feat (default) | c4
  --baseline-part-root <dir>  partitions/oem of the baseline run: labeled/unlabeled/val must be identical
  --lambda <p>                blended (2/2/1, default) | original (5/1/2) | readme_original (5/2/1)
  --seed <n> / --split-seed <n> / --epochs <n> / --eval-every <n> / --batch-size <n>
  --full-unlabeled <m>        split full only: same (default) | none
  --max-session-hours <h>     stop cleanly after the epoch running past h hours (default 11.25)
  --force                     run a finished split again (old directory renamed, not deleted)
  --import-labeled <file> / --data-root <dir> / --pretrained-dir <dir>
  --restore-dir <d>           auto (default) | none | <dir>
  --out-root <dir>            results (default /kaggle/working/exp/oem_dapn)
  --part-root <dir>           splits (default /kaggle/working/partitions/oem_dapn)
  --extra "<args>"            extra arguments for the training script
  -h, --help                  this help
EOF
}

# ---------- 0. command line ----------
need() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "$1 needs a value (bash run_dapn.sh -h)"; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        split|--split)
            need "$@"; SPLIT=$2; shift 2
            [[ $# -gt 0 && "$1" != -* ]] && die "only ONE split per call (got '$SPLIT' and '$1')"
            ;;
        --dryrun|--dry-run) TRIAL=1; shift ;;
        --trial-iters)      need "$@"; TRIAL_ITERS=$2; shift 2 ;;
        --method)           need "$@"; METHOD=$2; shift 2 ;;
        --w-dapg)           need "$@"; W_DAPG=$2; shift 2 ;;
        --dapg-pos)         need "$@"; DAPG_POS=$2; shift 2 ;;
        --baseline-part-root) need "$@"; BASELINE_PART_ROOT=$2; shift 2 ;;
        --full-unlabeled)   need "$@"; FULL_UNLABELED=$2; shift 2 ;;
        --lambda)           need "$@"; LAMBDA_PRESET=$2; shift 2 ;;
        --seed)             need "$@"; SEED=$2; shift 2 ;;
        --split-seed)       need "$@"; SPLIT_SEED=$2; shift 2 ;;
        --epochs)           need "$@"; EPOCHS=$2; shift 2 ;;
        --eval-every)       need "$@"; EVAL_EVERY=$2; shift 2 ;;
        --batch-size)       need "$@"; BATCH_SIZE=$2; shift 2 ;;
        --max-session-hours) need "$@"; MAX_SESSION_HOURS=$2; shift 2 ;;
        --force)            FORCE=1; shift ;;
        --import-labeled)   need "$@"; IMPORT_LABELED=$2; shift 2 ;;
        --data-root)        need "$@"; DATA_ROOT=$2; shift 2 ;;
        --pretrained-dir)   need "$@"; PRETRAINED_DIR=$2; shift 2 ;;
        --restore-dir)      need "$@"; RESTORE_DIR=$2; shift 2 ;;
        --out-root)         need "$@"; OUT_ROOT=$2; shift 2 ;;
        --part-root)        need "$@"; PART_ROOT=$2; shift 2 ;;
        --extra)            [[ $# -ge 2 ]] || die "--extra needs a value"; EXTRA=$2; shift 2 ;;
        -h|--help)          usage; exit 0 ;;
        *) die "unknown argument '$1' (bash run_dapn.sh -h)" ;;
    esac
done
SPLIT=${SPLIT//\//_}

# ---------- 1. check the settings ----------
[[ -n "$SPLIT" ]] || { usage; die "missing split: bash run_dapn.sh split 1/4 [--dryrun]"; }
[[ "$SPLIT" =~ [[:space:],\;] ]] && die "SPLIT='$SPLIT' holds several values: exactly ONE split per call"
[[ "$SPLIT" =~ ^(1_16|1_8|1_4|1_2|full|n[0-9]+)$ ]] || die "invalid split '$SPLIT' (1/16 | 1/8 | 1/4 | 1/2 | full | n<number>)"
[[ "$FULL_UNLABELED" =~ ^(same|none)$ ]] || die "--full-unlabeled must be same or none"
[[ "$SPLIT" == full && "$FULL_UNLABELED" == none ]] && die "split full with --full-unlabeled none has no unlabeled images: not a DAPN setting (use run_baseline.sh)"
[[ "$LAMBDA_PRESET" =~ ^(blended|original|readme_original)$ ]] || die "--lambda must be blended, original or readme_original"
[[ "$DAPG_POS" =~ ^(feat|c4)$ ]] || die "--dapg-pos must be feat or c4"
[[ "$W_DAPG" =~ ^[0-9]*\.?[0-9]+$ ]] || die "--w-dapg must be a number"
[[ "$TRIAL" =~ ^[01]$ ]] || die "TRIAL must be 0 or 1"

case "$LAMBDA_PRESET" in
    blended)         LAMBDA="--w_CE 2.0 --w_con 2.0 --w_dis 1.0" ;;
    original)        LAMBDA="--w_CE 5.0 --w_con 1.0 --w_dis 2.0" ;;
    readme_original) LAMBDA="--w_CE 5.0 --w_con 2.0 --w_dis 1.0" ;;
esac
# identical to run_baseline.sh (COMMON + ccvc_aug flags): DAPN changes nothing else
COMMON="--backbone resnet101 --crop_size 512 --base_lr 0.001 --lr_backbone 1.0 --lr_network 10.0 --mul_scheduler 0.9
        --mode_mapping else --conf_threshold 0.9 --conf_prob True --w_confident 2.0 --dropout 0.5 --use_SPL False
        $LAMBDA"
CCVC_AUG="--mode_criterion CE --cc_mask_mixed True --use_MLP True --use_norm True --use_dropout True --mode_confident vote_threshold"
DAPG_COMMON="--w_dapg $W_DAPG --dapg_pos $DAPG_POS"
case "$METHOD" in
    ccvc_origin)   SCRIPT=CCVC_aug.py;      CODE_VERSION=origin; DAPG="" ;;
    ccvc_dapn)     SCRIPT=CCVC_aug_dapn.py; CODE_VERSION=dapn;   DAPG="--use_dapg True --dapg_on lu --dapg_branches both $DAPG_COMMON" ;;
    ccvc_dapn_l)   SCRIPT=CCVC_aug_dapn.py; CODE_VERSION=dapn;   DAPG="--use_dapg True --dapg_on l --dapg_branches both $DAPG_COMMON" ;;
    ccvc_dapn_b1)  SCRIPT=CCVC_aug_dapn.py; CODE_VERSION=dapn;   DAPG="--use_dapg True --dapg_on lu --dapg_branches b1 $DAPG_COMMON" ;;
    ccvc_dapn_off) SCRIPT=CCVC_aug_dapn.py; CODE_VERSION=dapn;   DAPG="--use_dapg False" ;;
    *) die "--method '$METHOD' is not a preset (ccvc_dapn | ccvc_dapn_l | ccvc_dapn_b1 | ccvc_dapn_off | ccvc_origin)" ;;
esac
FLAGS="$CCVC_AUG $DAPG"

# ---------- 2. origin integrity (verify_origin step 1) ----------
VERIFY_LOG=$(mktemp)
$PY tools/verify_origin.py --steps 1 | tee "$VERIFY_LOG"
[[ ${PIPESTATUS[0]} -eq 0 ]] || { rm -f "$VERIFY_LOG"; die "verify_origin FAILED: an origin file was modified (see above); restore it (git checkout ccvc-origin -- <file>)"; }
rm -f "$VERIFY_LOG"
if [[ "$CODE_VERSION" == dapn ]]; then
    $PY tools/check_dapn_diff.py || die "CCVC_aug_dapn.py has changes outside [DAPN] blocks"
fi

# ---------- 3. environment ----------
for mod in tifffile matplotlib yaml; do
    $PY -c "import $mod" 2>/dev/null || { pkg=$mod; [[ $mod == yaml ]] && pkg=pyyaml; log "installing $pkg"; $PY -m pip install -q "$pkg" || die "pip install $pkg failed"; }
done
$PY tools/check_env.py --search_root "$SEARCH_ROOT" --work_dir "$(dirname "$(dirname "$OUT_ROOT")")"
if [[ "$DEVICE" == cuda ]]; then
    NGPU=$($PY tools/check_env.py --gpu_count) || die "PyTorch not importable"
    [[ "$NGPU" -ge 1 ]] || die "no GPU visible (Kaggle: Accelerator = GPU T4 x2)"
else
    NGPU=1
fi

# ---------- 4. restore the previous session's DAPN output (exp/oem_dapn only) ----------
$PY tools/restore_output_dapn.py --restore_dir "$RESTORE_DIR" --search_root "$SEARCH_ROOT" \
    --out_root "$OUT_ROOT" --part_root "$PART_ROOT" || die "restore failed (see above)"

# ---------- 5. pretrained ResNet-101 ----------
if [[ -n "$PRETRAINED_DIR" ]]; then
    PTH=$(find "$PRETRAINED_DIR" -name resnet101.pth 2>/dev/null | sort | head -n 1)
else
    ALL_PTH=$(find "$SEARCH_ROOT" -maxdepth 6 -name resnet101.pth 2>/dev/null | sort)
    PTH=$(grep -E 'model[-_]lab' <<< "$ALL_PTH" | head -n 1)
    [[ -n "$PTH" ]] || PTH=$(head -n 1 <<< "$ALL_PTH")
fi
[[ -n "$PTH" && -f "$PTH" ]] || die "resnet101.pth not found (${PRETRAINED_DIR:-under $SEARCH_ROOT}); add the Kaggle dataset model-lab or use --pretrained-dir"
mkdir -p pretrained
ln -sf "$(cd "$(dirname "$PTH")" && pwd)/$(basename "$PTH")" pretrained/resnet101.pth || cp "$PTH" pretrained/resnet101.pth
log "pretrained: $PTH"
$PY -m model.backbone.check_pretrained --path pretrained/resnet101.pth --arch resnet101 || die "pretrained weights rejected"

# ---------- 6. data check + split (same rule and split_seed as the origin) ----------
PREP_LOG=$(mktemp)
$PY tools/prepare_oem.py --split "$SPLIT" --split_seed "$SPLIT_SEED" --full_unlabeled "$FULL_UNLABELED" \
    ${IMPORT_LABELED:+--import_labeled "$IMPORT_LABELED"} ${DATA_ROOT:+--data_root "$DATA_ROOT"} \
    --search_root "$SEARCH_ROOT" --part_root "$PART_ROOT" | tee "$PREP_LOG"
[[ ${PIPESTATUS[0]} -eq 0 ]] || die "data check / split failed (see above)"
PART_DIR=$(grep '^PART_DIR=' "$PREP_LOG" | tail -n 1 | cut -d= -f2-)
rm -f "$PREP_LOG"
[[ -d "$PART_DIR" ]] || die "split directory not created"
FOUND_ROOT=$($PY tools/run_meta.py data_root "$PART_DIR/split_info.json")
PART_CHECK=not_checked
if [[ -n "$BASELINE_PART_ROOT" ]]; then
    $PY tools/verify_origin.py --steps 4 --part-root "$PART_ROOT" --baseline-part-root "$BASELINE_PART_ROOT" \
        --split-dir "$(basename "$PART_DIR")" || die "split differs from the baseline run ($BASELINE_PART_ROOT): stop"
    PART_CHECK=identical_to_baseline
fi

# ---------- 7. run directory: $OUT_ROOT/<split>_s<split_seed>/<method>/seed<seed> ----------
MDIR=$METHOD
[[ "$LAMBDA_PRESET" != blended ]] && MDIR="${MDIR}_${LAMBDA_PRESET}"
[[ "$CODE_VERSION" == dapn && "$METHOD" != ccvc_dapn_off && "$W_DAPG" != 0.2 ]] && MDIR="${MDIR}_w${W_DAPG}"
[[ "$CODE_VERSION" == dapn && "$METHOD" != ccvc_dapn_off && "$DAPG_POS" != feat ]] && MDIR="${MDIR}_${DAPG_POS}"
RUN_DIR="$OUT_ROOT/$(basename "$PART_DIR")/$MDIR/seed$SEED"
SAVE_DIR=$RUN_DIR
if [[ "$TRIAL" == 1 ]]; then
    SAVE_DIR="$RUN_DIR/trial"
    rm -rf "$SAVE_DIR"
    TRIAL_ARGS="--trial_iters $TRIAL_ITERS"
else
    TRIAL_ARGS=""
    if [[ -d "$RUN_DIR" && "$FORCE" == 1 ]]; then
        OLD="${RUN_DIR}_old_$(date +%Y%m%d_%H%M%S)"
        mv "$RUN_DIR" "$OLD" || die "cannot rename $RUN_DIR"
        log "--force: previous run directory renamed (not deleted) -> $OLD"
    elif [[ -f "$RUN_DIR/final_eval.json" ]]; then
        log "split này đã chạy xong: $RUN_DIR/final_eval.json exists (--force to run it again)"
        $PY tools/show_results.py --run_dir "$RUN_DIR" --out_root "$OUT_ROOT"
        exit 0
    fi
fi
mkdir -p "$SAVE_DIR"
cp "$PART_DIR/labeled.txt" "$PART_DIR/unlabeled.txt" "$SAVE_DIR/"
$PY tools/run_meta.py write "$PART_DIR/split_info.json" "$SAVE_DIR/run_meta.json" \
    method="$METHOD" lambda_preset="$LAMBDA_PRESET" part_dir="$PART_DIR" \
    labeled_class_dist="$PART_DIR/labeled_class_dist.json" run_dir="$RUN_DIR" \
    trial="$([[ $TRIAL == 1 ]] && echo true || echo false)" eval_every="$EVAL_EVERY" extra_args="$EXTRA" \
    code_version="$CODE_VERSION" entry_point=run_dapn.sh verify_origin=PASS partition_check="$PART_CHECK" \
    baseline_part_root="$BASELINE_PART_ROOT" \
    || die "cannot write run_meta.json"

NL=$(grep -c . "$PART_DIR/labeled.txt"); NU=$(grep -c . "$PART_DIR/unlabeled.txt"); NV=$(grep -c . "$PART_ROOT/val.txt")
log "================================================================"
log "split $SPLIT (split_seed $SPLIT_SEED) | method $METHOD ($SCRIPT, code $CODE_VERSION) | lambda $LAMBDA_PRESET | seed $SEED"
log "images: labeled $NL | unlabeled $NU | val $NV | GPUs $NGPU | batch/GPU $BATCH_SIZE"
case "$METHOD" in
    ccvc_origin)   log "DAPN: none (origin CCVC_aug.py)" ;;
    ccvc_dapn_off) log "DAPN: use_dapg False (equivalence check)" ;;
    ccvc_dapn_b1)  log "DAPN: use_dapg True | w_dapg $W_DAPG | dapg_on lu | branch 1 only (da1) | pos $DAPG_POS" ;;
    ccvc_dapn_l)   log "DAPN: use_dapg True | w_dapg $W_DAPG | dapg_on l (labeled only) | BOTH branches (da1 + da2) | pos $DAPG_POS" ;;
    ccvc_dapn)     log "DAPN: use_dapg True | w_dapg $W_DAPG | dapg_on lu | BOTH branches (da1 + da2) | pos $DAPG_POS" ;;
esac
log "partition check vs baseline: $PART_CHECK"
log "results -> $SAVE_DIR   (free disk $(df -h "$OUT_ROOT" 2>/dev/null | awk 'NR==2{print $4}'))"
[[ "$TRIAL" == 1 ]] && log "DRY RUN: $TRIAL_ITERS iterations + 1 evaluation"
log "================================================================"

# ---------- 8. train ----------
RUN_LOG="$SAVE_DIR/run_$(date +%Y%m%d_%H%M%S).log"
# shellcheck disable=SC2086
$PY "$SCRIPT" --config configs/oem.yaml --data_root "$FOUND_ROOT" \
    --labeled_id_path "$PART_DIR/labeled.txt" --unlabeled_id_path "$PART_DIR/unlabeled.txt" \
    --val_id_path "$PART_ROOT/val.txt" --save_path "$SAVE_DIR" --run_meta "$SAVE_DIR/run_meta.json" \
    --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS" --seed "$SEED" \
    --gpus "$NGPU" --device "$DEVICE" --eval_every "$EVAL_EVERY" \
    --max_session_hours "$MAX_SESSION_HOURS" --session_start "$SESSION_START" \
    $COMMON $FLAGS $TRIAL_ARGS $EXTRA 2>&1 | tee -a "$RUN_LOG"
RC=${PIPESTATUS[0]}

# ---------- 9. results ----------
[[ $RC -eq 0 || $RC -eq 3 ]] && $PY tools/show_results.py --run_dir "$SAVE_DIR" --out_root "$OUT_ROOT"
case $RC in
    0) log "finished: $SAVE_DIR" ;;
    3) log "stopped by MAX_SESSION_HOURS=$MAX_SESSION_HOURS: save /kaggle/working (Save Version), add it as input next session and run the SAME command" ;;
    *) log "training failed with exit code $RC (log: $RUN_LOG)" ;;
esac
exit $RC
