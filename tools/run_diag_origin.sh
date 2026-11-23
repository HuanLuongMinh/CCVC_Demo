#!/bin/bash
# Diagnostics 0a/0b on the CCVC origin checkpoints of split 1/4 and 1/8 (Kaggle, one Save Version).
#
#   bash tools/run_diag_origin.sh [--quick] [--data-root DIR] [--part-root DIR] [--run-1-4 DIR] [--run-1-8 DIR]
#                                 [--out-root DIR] [--only eval|diagnose] [--branches "1 2"] [--splits "1_4 1_8"]
#
# 0a  tools/eval_ckpt.py              -> agree_wrong / agree_right / conflict, per class, confusion matrices
# 0b  tools/diagnose_entanglement.py  -> branch 1 and 2: purity, mixed clusters, boundary band
# Both scripts run unchanged with their default thresholds (--tol 0.01, --mixed 0.6, --boundary-radius 3,
# --k 64, tau 0.1); every output path is passed explicitly (--out / --out-dir), one directory per step:
#   <out-root>/<split>/eval_ckpt/   <out-root>/<split>/diagnose_b<n>/   + run_log.txt, summary.md, summary.csv
# Paths not given are searched under /kaggle/input (tools/summarize_diag_origin.py find).
# --quick: preflight + diagnose 1/4 branch 1 on 2 images (small K) into <out-root>/_quick/ (path check, 1-2 min).
# Exit code: 0 every step OK; 2 preflight failed (nothing ran); 4 at least one step failed (the others ran).
#
# Test-only environment: DIAG_SEARCH_ROOT, DIAG_TEST_DEVICE, DIAG_TEST_MAX_IMAGES, DIAG_TEST_DIAG_ARGS,
# DIAG_TEST_N_VAL, DIAG_MIN_FREE_GB, DIAG_NUM_WORKERS.
set -u
cd "$(dirname "$0")/.." || exit 1
REPO=$(pwd)
PY=${PYTHON:-python}
export PYTHONIOENCODING=utf-8

QUICK=0; DATA_ROOT=''; PART_ROOT=''; RUN_14=''; RUN_18=''; ONLY=''; BRANCHES='1 2'; SPLITS='1_4 1_8'
OUT_ROOT=/kaggle/working/exp/oem_dapn/diag_origin
SEARCH_ROOT=${DIAG_SEARCH_ROOT:-/kaggle/input}
usage() { sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; }
need() { [[ $# -ge 2 ]] || { echo "[diag_origin] ERROR: $1 needs a value"; exit 2; }; }
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)     QUICK=1; shift ;;
        --data-root) need "$@"; DATA_ROOT=$2; shift 2 ;;
        --part-root) need "$@"; PART_ROOT=$2; shift 2 ;;
        --run-1-4)   need "$@"; RUN_14=$2; shift 2 ;;
        --run-1-8)   need "$@"; RUN_18=$2; shift 2 ;;
        --out-root)  need "$@"; OUT_ROOT=$2; shift 2 ;;
        --only)      need "$@"; ONLY=$2; shift 2 ;;
        --branches)  need "$@"; BRANCHES=$2; shift 2 ;;
        --splits)    need "$@"; SPLITS=${2//\//_}; shift 2 ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "[diag_origin] ERROR: unknown argument '$1'"; usage; exit 2 ;;
    esac
done
[[ -z "$ONLY" || "$ONLY" == eval || "$ONLY" == diagnose ]] || { echo "[diag_origin] ERROR: --only eval|diagnose"; exit 2; }
for b in $BRANCHES; do [[ "$b" == 1 || "$b" == 2 ]] || { echo "[diag_origin] ERROR: --branches takes 1 and/or 2"; exit 2; }; done
for s in $SPLITS; do [[ "$s" == 1_4 || "$s" == 1_8 ]] || { echo "[diag_origin] ERROR: --splits takes 1_4 and/or 1_8"; exit 2; }; done
[[ $QUICK == 1 ]] && SPLITS=1_4
VAL_LIST=''
[[ -n "$PART_ROOT" ]] && VAL_LIST="$PART_ROOT/val.txt"

LOG_DIR=$OUT_ROOT; [[ $QUICK == 1 ]] && LOG_DIR="$OUT_ROOT/_quick"
mkdir -p "$LOG_DIR" || { echo "[diag_origin] ERROR: cannot create $LOG_DIR"; exit 2; }
LOG="$LOG_DIR/run_log.txt"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
COMMON_DEV=()
[[ -n "${DIAG_TEST_DEVICE:-}" ]] && COMMON_DEV+=(--device "$DIAG_TEST_DEVICE")
COMMON_DEV+=(--num-workers "${DIAG_NUM_WORKERS:-2}")

main() {
    echo "================================================================================"
    echo "[diag_origin] start $(ts) | repo $REPO ($(cat PACKAGE_COMMIT 2>/dev/null || git rev-parse --short HEAD 2>/dev/null || echo '?'))"
    echo "[diag_origin] out-root $OUT_ROOT | splits $SPLITS | branches $BRANCHES | only ${ONLY:-all} | quick $QUICK"

    # ---------------------------------------------------------------- paths
    local env_file="$LOG_DIR/paths.env"
    local find_args=(find --search-root "$SEARCH_ROOT" --splits "$SPLITS" --env-out "$env_file")
    [[ -n "$RUN_14" ]] && find_args+=(--run-1-4 "$RUN_14")
    [[ -n "$RUN_18" ]] && find_args+=(--run-1-8 "$RUN_18")
    [[ -n "$VAL_LIST" ]] && find_args+=(--val-list "$VAL_LIST")
    [[ -n "$DATA_ROOT" ]] && find_args+=(--data-root "$DATA_ROOT")
    $PY tools/summarize_diag_origin.py "${find_args[@]}" || { echo "[diag_origin] STOP: paths not found (exit 2)"; return 2; }
    declare -A RUN
    local VAL DATA k v
    while IFS='=' read -r k v; do
        case $k in RUN_*) RUN[${k#RUN_}]=$v ;; VAL) VAL=$v ;; DATA) DATA=$v ;; esac
    done < "$env_file"

    # ---------------------------------------------------------------- preflight
    echo "--- preflight"
    $PY -c "import torch; print('[preflight] torch', torch.__version__); import sys; \
c = torch.cuda.is_available(); \
print('[preflight] GPU: ' + ', '.join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())) if c else \
'[preflight] WARNING: no CUDA - runs on CPU, slow, AMP off (eval_ckpt may then differ from final_eval.json)')" \
        || { echo "[diag_origin] STOP: python/torch not usable (exit 2)"; return 2; }
    $PY tools/summarize_diag_origin.py preflight --env "$env_file" --n-val "${DIAG_TEST_N_VAL:-384}" \
        --out-root "$OUT_ROOT" --min-free-gb "${DIAG_MIN_FREE_GB:-1}" || { echo "[diag_origin] STOP: preflight failed (exit 2)"; return 2; }
    $PY tools/verify_origin.py --steps 1 || { echo "[diag_origin] STOP: verify_origin step 1 FAIL (exit 2)"; return 2; }

    # ---------------------------------------------------------------- steps
    local results=() nfail=0
    step() {   # step <label> <cmd...>
        local label=$1; shift
        local t0=$SECONDS rc
        echo
        echo "--- [$(ts)] $label"
        echo "    $*"
        "$@"
        rc=$?
        if [[ $rc -ne 0 ]]; then
            nfail=$((nfail+1))
            echo "[diag_origin] $label: exit $rc"
        fi
        results+=("$(printf '%-34s exit %-3s %5ds' "$label" "$rc" $((SECONDS-t0)))")
        return 0
    }
    local diag_extra=()
    [[ -n "${DIAG_TEST_MAX_IMAGES:-}" ]] && diag_extra+=(--max-images "$DIAG_TEST_MAX_IMAGES")
    [[ -n "${DIAG_TEST_DIAG_ARGS:-}" ]] && diag_extra+=($DIAG_TEST_DIAG_ARGS)
    local common=(--val-list "$VAL" --data-root "$DATA" "${COMMON_DEV[@]}")

    if [[ $QUICK == 1 ]]; then
        step "quick: diagnose 1/4 branch 1" $PY tools/diagnose_entanglement.py --ckpt "${RUN[1_4]}/final.pth" \
            --branch 1 "${common[@]}" --max-images 2 --k 8 --sample 2000 --n-seeds 1 \
            --out-dir "$LOG_DIR/1_4/diagnose_b1"
    else
        local s b ej
        if [[ "$ONLY" != diagnose ]]; then
            for s in $SPLITS; do
                mkdir -p "$OUT_ROOT/$s/eval_ckpt"
                ej="$OUT_ROOT/$s/eval_ckpt/eval_ckpt.json"
                rm -f "$ej"
                step "eval_ckpt ${s/_//}" $PY tools/eval_ckpt.py --ckpt "${RUN[$s]}/final.pth" "${common[@]}" --out "$ej"
                if [[ -f "$ej" ]]; then
                    $PY -c "import json,sys; c=json.load(open(sys.argv[1]))['check_vs_final_eval']; \
print('[diag_origin] eval_ckpt %s vs final_eval.json: %s, max |diff| %.4f (tol %.2f)' % (sys.argv[2], 'OK' if c['ok'] else 'MISMATCH', max(c['abs_diff'].values()), c['tol']) if c else '[diag_origin] no final_eval.json to compare')" "$ej" "${s/_//}"
                fi
            done
        fi
        if [[ "$ONLY" != eval ]]; then
            for b in $BRANCHES; do
                for s in $SPLITS; do
                    rm -f "$OUT_ROOT/$s/diagnose_b$b/diagnose_b$b.json"
                    step "diagnose ${s/_//} branch $b" $PY tools/diagnose_entanglement.py --ckpt "${RUN[$s]}/final.pth" \
                        --branch "$b" "${common[@]}" "${diag_extra[@]}" --out-dir "$OUT_ROOT/$s/diagnose_b$b"
                done
            done
        fi
        echo
        $PY tools/summarize_diag_origin.py summary --out-root "$OUT_ROOT" --splits "$SPLITS" || nfail=$((nfail+1))
    fi

    echo
    echo "[diag_origin] ---- steps"
    local r
    for r in "${results[@]}"; do echo "  $r"; done
    echo "[diag_origin] end $(ts) | $nfail step(s) failed | log $LOG"
    [[ $QUICK == 0 ]] && echo "[diag_origin] summary: $OUT_ROOT/summary.md"
    [[ $nfail -eq 0 ]] && return 0 || return 4
}

main 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
