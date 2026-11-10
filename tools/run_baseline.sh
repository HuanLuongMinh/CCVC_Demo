#!/bin/bash
# CCVC (CVPR 2023) on OpenEarthMap, Pascal VOC protocol - ONE split per call.
#   bash tools/run_baseline.sh                       # uses SPLIT below
#   SPLIT=1_8 bash tools/run_baseline.sh             # any variable can be overridden from the command line
#   SPLIT=full TRIAL=1 bash tools/run_baseline.sh
# Exit code: 0 = finished (or already finished / trial done), 3 = stopped by MAX_SESSION_HOURS
# (run the same command in the next session to resume), anything else = error.

# ===== CẤU HÌNH — MỖI LẦN CHẠY MỘT SPLIT =====
SPLIT=${SPLIT-1_4}                          # 1_16 | 1_8 | 1_4 | 1_2 | full | n<số>
TRIAL=${TRIAL:-0}                           # 1 = chạy thử ngắn (TRIAL_ITERS iteration + 1 eval); 0 = chạy đầy đủ
TRIAL_ITERS=${TRIAL_ITERS:-300}
METHOD=${METHOD:-ccvc_aug}                  # sup | sup_pure | ccr | cvc | ccvc | ccvc_aug
SEED=${SEED:-0}                             # seed huấn luyện
SPLIT_SEED=${SPLIT_SEED:-0}                 # seed chia dữ liệu — giữ nguyên giữa các lần chạy để split lồng nhau
FULL_UNLABELED=${FULL_UNLABELED:-same}      # chỉ dùng khi SPLIT=full: same | none (none -> tự dùng sup_pure)
IMPORT_LABELED=${IMPORT_LABELED:-}          # file danh sách ảnh có nhãn có sẵn (tùy chọn)
EPOCHS=${EPOCHS:-80}                        # paper, VOC
EVAL_EVERY=${EVAL_EVERY:-5}                 # đánh giá mỗi N epoch (+ epoch cuối)
BATCH_SIZE=${BATCH_SIZE:-2}                 # mỗi GPU, mỗi loader
LAMBDA_PRESET=${LAMBDA_PRESET:-blended}     # blended = 2/2/1 | original = 5/1/2 | readme_original = 5/2/1 (w_CE/w_con/w_dis)
DATA_ROOT=${DATA_ROOT:-}                    # trống: tự tìm dưới /kaggle/input
PRETRAINED_DIR=${PRETRAINED_DIR:-}          # trống: tự tìm dưới /kaggle/input/model_lab*
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
SESSION_START=${SESSION_START:-$(date +%s)} # nên đặt ở cell đầu notebook để tính cả thời gian chuẩn bị

set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
REPO=$(pwd)
PY=${PYTHON:-python}
die() { echo "[run_baseline] ERROR: $*" >&2; exit 2; }
log() { echo "[run_baseline] $*"; }

# ---------- 1. check the settings: exactly one split per call ----------
[[ -n "$SPLIT" ]] || die "SPLIT is empty: set exactly one split (1_16 | 1_8 | 1_4 | 1_2 | full | n<number>)"
[[ "$SPLIT" =~ [[:space:],\;] ]] && die "SPLIT='$SPLIT' holds several values: exactly ONE split per call"
[[ "$SPLIT" =~ ^(1_16|1_8|1_4|1_2|full|n[0-9]+)$ ]] || die "invalid SPLIT='$SPLIT' (1_16 | 1_8 | 1_4 | 1_2 | full | n<number>)"
[[ "$FULL_UNLABELED" =~ ^(same|none)$ ]] || die "FULL_UNLABELED must be same or none"
[[ "$LAMBDA_PRESET" =~ ^(blended|original|readme_original)$ ]] || die "LAMBDA_PRESET must be blended, original or readme_original"
[[ "$TRIAL" =~ ^[01]$ ]] || die "TRIAL must be 0 or 1"
if [[ "$SPLIT" == full && "$FULL_UNLABELED" == none && "$METHOD" != sup_pure ]]; then
    log "SPLIT=full FULL_UNLABELED=none: no unlabeled images -> METHOD=$METHOD replaced by sup_pure"
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
    *) die "METHOD='$METHOD' is not a preset (sup | sup_pure | ccr | cvc | ccvc | ccvc_aug)" ;;
esac

# ---------- 2. environment ----------
for mod in tifffile matplotlib yaml; do
    $PY -c "import $mod" 2>/dev/null || { pkg=$mod; [[ $mod == yaml ]] && pkg=pyyaml; log "installing $pkg"; $PY -m pip install -q "$pkg" || die "pip install $pkg failed"; }
done
if [[ "$DEVICE" == cuda ]]; then
    NGPU=$($PY -c "import torch; print(torch.cuda.device_count())") || die "PyTorch not importable"
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
    PTH=$(find "$SEARCH_ROOT" -maxdepth 1 -name 'model_lab*' -print0 2>/dev/null | xargs -0 -r -I{} find {} -name resnet101.pth 2>/dev/null | sort | head -n 1)
fi
[[ -n "$PTH" && -f "$PTH" ]] || die "resnet101.pth not found (${PRETRAINED_DIR:-$SEARCH_ROOT/model_lab*}); add the Kaggle dataset model_lab or set PRETRAINED_DIR"
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
FOUND_ROOT=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['data_root'])" "$PART_DIR/split_info.json")

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
        log "FORCE=1: previous run directory renamed (not deleted) -> $OLD"
    elif [[ -f "$RUN_DIR/final_eval.json" ]]; then
        log "split này đã chạy xong: $RUN_DIR/final_eval.json exists (FORCE=1 to run it again)"
        exit 0
    fi
fi
mkdir -p "$SAVE_DIR"
cp "$PART_DIR/labeled.txt" "$PART_DIR/unlabeled.txt" "$SAVE_DIR/"

$PY - "$PART_DIR/split_info.json" "$SAVE_DIR/run_meta.json" <<EOF || die "cannot write run_meta.json"
import json, sys
info = json.load(open(sys.argv[1]))
info.update({'method': '$METHOD', 'lambda_preset': '$LAMBDA_PRESET', 'part_dir': '$PART_DIR',
             'labeled_class_dist': '$PART_DIR/labeled_class_dist.json', 'run_dir': '$RUN_DIR',
             'trial': $TRIAL == 1, 'eval_every': $EVAL_EVERY, 'extra_args': '''$EXTRA'''})
json.dump(info, open(sys.argv[2], 'w'), indent=2)
EOF

NL=$(grep -c . "$PART_DIR/labeled.txt"); NU=$(grep -c . "$PART_DIR/unlabeled.txt"); NV=$(grep -c . "$PART_ROOT/val.txt")
log "================================================================"
log "split $SPLIT (split_seed $SPLIT_SEED) | method $METHOD ($SCRIPT) | lambda $LAMBDA_PRESET | seed $SEED"
log "images: labeled $NL | unlabeled $NU | val $NV | GPUs $NGPU | batch/GPU $BATCH_SIZE"
log "results -> $SAVE_DIR   (free disk $(df -h "$OUT_ROOT" 2>/dev/null | awk 'NR==2{print $4}'))"
[[ "$TRIAL" == 1 ]] && log "TRIAL: $TRIAL_ITERS iterations + 1 evaluation"
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
case $RC in
    0) log "finished: $SAVE_DIR" ;;
    3) log "stopped by MAX_SESSION_HOURS=$MAX_SESSION_HOURS: save /kaggle/working and run the same command next session" ;;
    *) log "training failed with exit code $RC (log: $RUN_LOG)" ;;
esac
exit $RC
