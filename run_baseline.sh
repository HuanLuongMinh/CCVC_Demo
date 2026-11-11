#!/bin/bash
# CCVC (CVPR 2023) on OpenEarthMap, Pascal VOC protocol - ONE split per call, everything from this file.
#
#   bash run_baseline.sh split 1/4 --dryrun      # short trial: 300 iterations + 1 evaluation + trial_report.json
#   bash run_baseline.sh split 1/4               # full run (80 epochs); next time: split 1/8, split 1/16, ...
#   bash run_baseline.sh split full --full-unlabeled none
#   bash run_baseline.sh -h                      # every option
#
# Exit code: 0 = finished (or already finished / dry run done), 3 = stopped by --max-session-hours
# (run the SAME command in the next Kaggle session to resume), anything else = error.

# ===== CẤU HÌNH MẶC ĐỊNH — thường KHÔNG cần sửa: truyền bằng tham số dòng lệnh (bash run_baseline.sh -h) =====
SPLIT=${SPLIT-}                             # để trống: bắt buộc truyền `split <x>` (1/16 | 1/8 | 1/4 | 1/2 | full | n<số>)
TRIAL=${TRIAL:-0}                           # --dryrun: 1 = chạy thử ngắn (TRIAL_ITERS iteration + 1 eval)
TRIAL_ITERS=${TRIAL_ITERS:-300}
METHOD=${METHOD:-ccvc_aug}                  # sup | sup_pure | ccr | cvc | ccvc | ccvc_aug
SEED=${SEED:-0}                             # seed huấn luyện
SPLIT_SEED=${SPLIT_SEED:-0}                 # seed chia dữ liệu — giữ nguyên giữa các lần chạy để split lồng nhau
FULL_UNLABELED=${FULL_UNLABELED:-same}      # chỉ dùng khi split full: same | none (none -> tự dùng sup_pure)
IMPORT_LABELED=${IMPORT_LABELED:-}          # file danh sách ảnh có nhãn có sẵn (tùy chọn)
EPOCHS=${EPOCHS:-80}                        # paper, VOC
EVAL_EVERY=${EVAL_EVERY:-5}                 # đánh giá mỗi N epoch (+ epoch cuối)
BATCH_SIZE=${BATCH_SIZE:-2}                 # mỗi GPU, mỗi loader
LAMBDA_PRESET=${LAMBDA_PRESET:-blended}     # blended = 2/2/1 | original = 5/1/2 | readme_original = 5/2/1 (w_CE/w_con/w_dis)
DATA_ROOT=${DATA_ROOT:-}                    # trống: tự tìm dưới /kaggle/input
PRETRAINED_DIR=${PRETRAINED_DIR:-}          # trống: tự tìm resnet101.pth dưới /kaggle/input (ưu tiên model_lab / model-lab)
OUT_ROOT=${OUT_ROOT:-/kaggle/working/exp/oem}
PART_ROOT=${PART_ROOT:-/kaggle/working/partitions/oem}
RESTORE_DIR=${RESTORE_DIR:-auto}            # auto: tự chép exp/ + partitions/ từ output phiên trước | <thư mục> | none
MAX_SESSION_HOURS=${MAX_SESSION_HOURS:-11.25}  # quá mốc: xong epoch hiện tại, lưu last.pth, thoát mã 3
NUM_WORKERS=${NUM_WORKERS:-2}
FORCE=${FORCE:-0}                           # 1 = chạy lại split đã xong (thư mục cũ được đổi tên, không xóa)
EXTRA=${EXTRA:-}                            # tham số ghi đè thêm cho script train
# =============================================
SEARCH_ROOT=${SEARCH_ROOT:-/kaggle/input}   # (tests) nơi tìm dữ liệu / trọng số / output phiên trước
DEVICE=${DEVICE:-cuda}                      # (tests) cpu
SESSION_START=${SESSION_START:-$(date +%s)} # giờ bắt đầu phiên (unix time) cho MAX_SESSION_HOURS

set -uo pipefail
cd "$(dirname "$0")" || exit 2
PY=${PYTHON:-python}
die() { echo "[run_baseline] ERROR: $*" >&2; exit 2; }
log() { echo "[run_baseline] $*"; }

usage() {
    cat <<'EOF'
Usage: bash run_baseline.sh split <1/16|1/8|1/4|1/2|full|n<k>> [options]

  split <x>                 the ONE split of this run (1/4 and 1_4 both work)
  --dryrun                  short trial: --trial-iters iterations + 1 evaluation, no checkpoint left
  --trial-iters <n>         iterations of the dry run (default 300)
  --method <m>              ccvc_aug (default) | sup_pure | sup | ccr | cvc | ccvc
  --full-unlabeled <m>      split full only: same (default) | none (-> sup_pure)
  --lambda <p>              blended (2/2/1, default) | original (5/1/2) | readme_original (5/2/1)
  --seed <n>                training seed (default 0)
  --split-seed <n>          data split seed (default 0; keep it for nested splits)
  --epochs <n>              default 80
  --eval-every <n>          evaluate every n epochs + last epoch (default 5)
  --batch-size <n>          images per GPU per loader (default 2)
  --max-session-hours <h>   stop cleanly after the epoch running past h hours (default 11.25)
  --force                   run a finished split again (old directory renamed, not deleted)
  --import-labeled <file>   use this list of labeled images
  --data-root <dir>         OEM root (default: found under /kaggle/input)
  --pretrained-dir <dir>    folder with resnet101.pth (default: found under /kaggle/input, model_lab/model-lab first)
  --restore-dir <d>         auto (default) | none | <dir> : previous session's output to restore
  --out-root <dir>          results (default /kaggle/working/exp/oem)
  --part-root <dir>         splits (default /kaggle/working/partitions/oem)
  --extra "<args>"          extra arguments for the training script
  -h, --help                this help
EOF
}

# ---------- 0. command line ----------
need() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "$1 needs a value (bash run_baseline.sh -h)"; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        split|--split)
            need "$@"; SPLIT=$2; shift 2
            [[ $# -gt 0 && "$1" != -* ]] && die "only ONE split per call (got '$SPLIT' and '$1'); run the command again for the next split"
            ;;
        --dryrun|--dry-run) TRIAL=1; shift ;;
        --trial-iters)      need "$@"; TRIAL_ITERS=$2; shift 2 ;;
        --method)           need "$@"; METHOD=$2; shift 2 ;;
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
        *) die "unknown argument '$1' (bash run_baseline.sh -h)" ;;
    esac
done
SPLIT=${SPLIT//\//_}                        # 1/4 -> 1_4

# ---------- 1. check the settings: exactly one split per call ----------
[[ -n "$SPLIT" ]] || { usage; die "missing split: bash run_baseline.sh split 1/4 [--dryrun]"; }
[[ "$SPLIT" =~ [[:space:],\;] ]] && die "SPLIT='$SPLIT' holds several values: exactly ONE split per call"
[[ "$SPLIT" =~ ^(1_16|1_8|1_4|1_2|full|n[0-9]+)$ ]] || die "invalid split '$SPLIT' (1/16 | 1/8 | 1/4 | 1/2 | full | n<number>)"
[[ "$FULL_UNLABELED" =~ ^(same|none)$ ]] || die "--full-unlabeled must be same or none"
[[ "$LAMBDA_PRESET" =~ ^(blended|original|readme_original)$ ]] || die "--lambda must be blended, original or readme_original"
[[ "$TRIAL" =~ ^[01]$ ]] || die "TRIAL must be 0 or 1"
if [[ "$SPLIT" == full && "$FULL_UNLABELED" == none && "$METHOD" != sup_pure ]]; then
    log "split full with --full-unlabeled none: no unlabeled images -> method $METHOD replaced by sup_pure"
    METHOD=sup_pure
fi

case "$LAMBDA_PRESET" in
    blended)         LAMBDA="--w_CE 2.0 --w_con 2.0 --w_dis 1.0" ;;
    original)        LAMBDA="--w_CE 5.0 --w_con 1.0 --w_dis 2.0" ;;
    readme_original) LAMBDA="--w_CE 5.0 --w_con 2.0 --w_dis 1.0" ;;
esac
# paper / upstream values shared by every method (README_OEM tables 3.1 and 3.2)
COMMON="--backbone resnet101 --crop_size 512 --base_lr 0.001 --lr_backbone 1.0 --lr_network 10.0 --mul_scheduler 0.9
        --mode_mapping else --conf_threshold 0.9 --conf_prob True --w_confident 2.0 --dropout 0.5 --use_SPL False
        $LAMBDA"
case "$METHOD" in
    sup)      SCRIPT=CCVC_no_aug.py; FLAGS="--use_con False --use_dis False --w_con 0 --w_dis 0 --use_MLP False --mode_confident no" ;;
    sup_pure) SCRIPT=CCVC_no_aug.py; FLAGS="--use_con False --use_dis False --w_con 0 --w_dis 0 --use_MLP False --mode_confident no --sup_pure True" ;;
    ccr)      SCRIPT=CCVC_no_aug.py; FLAGS="--use_con True --use_dis False --use_MLP False --mode_confident no" ;;
    cvc)      SCRIPT=CCVC_no_aug.py; FLAGS="--use_con True --use_dis True --use_MLP True --use_norm True --use_dropout True --mode_confident no" ;;
    ccvc)     SCRIPT=CCVC_no_aug.py; FLAGS="--use_con True --use_dis True --use_MLP True --use_norm True --use_dropout True --mode_confident vote_threshold" ;;
    ccvc_aug) SCRIPT=CCVC_aug.py;    FLAGS="--mode_criterion CE --cc_mask_mixed True --use_MLP True --use_norm True --use_dropout True --mode_confident vote_threshold" ;;
    *) die "--method '$METHOD' is not a preset (sup | sup_pure | ccr | cvc | ccvc | ccvc_aug)" ;;
esac

# ---------- 2. environment ----------
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

# ---------- 3. restore the previous session's output ----------
$PY tools/restore_output.py --restore_dir "$RESTORE_DIR" --search_root "$SEARCH_ROOT" \
    --out_root "$OUT_ROOT" --part_root "$PART_ROOT" || die "restore failed (see above)"

# ---------- 4. pretrained ResNet-101 (deep stem, UniMatch/CCVC) ----------
if [[ -n "$PRETRAINED_DIR" ]]; then
    PTH=$(find "$PRETRAINED_DIR" -name resnet101.pth 2>/dev/null | sort | head -n 1)
else
    # Kaggle mounts /kaggle/input/<slug>/ (old) or /kaggle/input/datasets/<user>/<slug>/ (new); slug model_lab or model-lab
    ALL_PTH=$(find "$SEARCH_ROOT" -maxdepth 6 -name resnet101.pth 2>/dev/null | sort)
    PTH=$(grep -E 'model[-_]lab' <<< "$ALL_PTH" | head -n 1)
    [[ -n "$PTH" ]] || PTH=$(head -n 1 <<< "$ALL_PTH")
fi
[[ -n "$PTH" && -f "$PTH" ]] || die "resnet101.pth not found (${PRETRAINED_DIR:-under $SEARCH_ROOT, e.g. $SEARCH_ROOT/datasets/<user>/model-lab}); add the Kaggle dataset model-lab or use --pretrained-dir"
mkdir -p pretrained
ln -sf "$(cd "$(dirname "$PTH")" && pwd)/$(basename "$PTH")" pretrained/resnet101.pth || cp "$PTH" pretrained/resnet101.pth
log "pretrained: $PTH"
$PY -m model.backbone.check_pretrained --path pretrained/resnet101.pth --arch resnet101 || die "pretrained weights rejected (see the key list above)"

# ---------- 5. data check + split ----------
PREP_LOG=$(mktemp)
$PY tools/prepare_oem.py --split "$SPLIT" --split_seed "$SPLIT_SEED" --full_unlabeled "$FULL_UNLABELED" \
    ${IMPORT_LABELED:+--import_labeled "$IMPORT_LABELED"} ${DATA_ROOT:+--data_root "$DATA_ROOT"} \
    --search_root "$SEARCH_ROOT" --part_root "$PART_ROOT" | tee "$PREP_LOG"
[[ ${PIPESTATUS[0]} -eq 0 ]] || die "data check / split failed (see above)"
PART_DIR=$(grep '^PART_DIR=' "$PREP_LOG" | tail -n 1 | cut -d= -f2-)
rm -f "$PREP_LOG"
[[ -d "$PART_DIR" ]] || die "split directory not created"
FOUND_ROOT=$($PY tools/run_meta.py data_root "$PART_DIR/split_info.json")

# ---------- 6. run directory ----------
MDIR=$METHOD
[[ "$LAMBDA_PRESET" != blended ]] && MDIR="${METHOD}_${LAMBDA_PRESET}"
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
    || die "cannot write run_meta.json"

NL=$(grep -c . "$PART_DIR/labeled.txt"); NU=$(grep -c . "$PART_DIR/unlabeled.txt"); NV=$(grep -c . "$PART_ROOT/val.txt")
log "================================================================"
log "split $SPLIT (split_seed $SPLIT_SEED) | method $METHOD ($SCRIPT) | lambda $LAMBDA_PRESET | seed $SEED"
log "images: labeled $NL | unlabeled $NU | val $NV | GPUs $NGPU | batch/GPU $BATCH_SIZE"
log "results -> $SAVE_DIR   (free disk $(df -h "$OUT_ROOT" 2>/dev/null | awk 'NR==2{print $4}'))"
[[ "$TRIAL" == 1 ]] && log "DRY RUN: $TRIAL_ITERS iterations + 1 evaluation"
log "================================================================"

# ---------- 7. train (separate process; mp.spawn inside for 2 GPUs) ----------
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

# ---------- 8. results ----------
[[ $RC -eq 0 || $RC -eq 3 ]] && $PY tools/show_results.py --run_dir "$SAVE_DIR" --out_root "$OUT_ROOT"
case $RC in
    0) log "finished: $SAVE_DIR" ;;
    3) log "stopped by MAX_SESSION_HOURS=$MAX_SESSION_HOURS: save /kaggle/working (Save Version), add it as input next session and run the SAME command" ;;
    *) log "training failed with exit code $RC (log: $RUN_LOG)" ;;
esac
exit $RC
