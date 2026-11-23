# CCVC-DAPN — CCVC + Dynamic Anchor (DAPN) trên OpenEarthMap

Repo này là **bản độc lập**: CCVC origin (nguyên byte) + DAPN (Dynamic Anchor + DAPG loss, không boundary loss) gắn vào **cả hai mạng con** của CCVC. Mọi thay đổi DAPN nằm trong **file mới**; không file nào của origin bị sửa.

- Chạy: `bash run_dapn.sh split 1/4 [--dryrun] [--method ...]` (hướng dẫn Kaggle: [HUONG_DAN_CHAY_DAPN.md](HUONG_DAN_CHAY_DAPN.md)).
- Origin vẫn chạy như cũ: `bash run_baseline.sh split 1/4`.
- Spec: `document/spec-gan-DAPN-vao-CCVC-ban-giao-claude-code.md` (bản 2, 23/11/2026).

## 1. Nguồn

| Nguồn | Repo | Commit | Dùng |
|---|---|---|---|
| CCVC origin | `HuanLuongMinh/CCVC_Demo` | `b44dc57` | toàn bộ cây file, nguyên byte (tag `ccvc-origin`, `ORIGIN_SOURCE.json`) |
| DAPN | `HuanLuongMinh/DAPN-for-VHR-Remote-Sensing-Segmentation`, nhánh `main_final` | `33ad638` | `dynamic_anchor.py`, `dapg_loss.py` → `model/dapn/` (`DAPN_SOURCE.json`) |

Bản chép DAPN chỉ đổi: `mmcv BaseModule` → `torch.nn.Module` (bỏ `init_cfg`), bỏ import/registry mmcv-mmseg, bỏ khối `__main__`. Kiểm thử `tools/tests_dapn/test_dapn_equivalence.py`: cùng seed, cùng tensor → `assign`, `proto`, `quality`, loss, gradient trùng bản gốc (sai khác tối đa 0).

## 2. Tái dựng CCVC origin — ba mức

1. **Chạy origin ngay trong repo này:** `bash run_baseline.sh split 1/4` hoặc `bash run_dapn.sh split 1/4 --method ccvc_origin`. Cả hai chạy `CCVC_aug.py` origin; không file `_dapn` nào được import (kiểm thử bằng `sys.modules`).
2. **Dựng lại bản sạch từ tag:**
   ```bash
   git checkout ccvc-origin                       # hoặc: git worktree add ../ccvc_origin ccvc-origin
   git archive --format=zip --prefix=CCVC-Origin/ -o ccvc_origin.zip ccvc-origin   # gói zip của origin
   git checkout main
   ```
   Cây file của tag có tree hash `9df939c…`, trùng `b44dc57` của CCVC_Demo.
3. **Xác nhận bằng công cụ:** `python tools/verify_origin.py`

   | Bước | Kiểm tra |
   |---|---|
   | 1 integrity | sha256 của 78 file trong `ORIGIN_MANIFEST.sha256` |
   | 2 behaviour | `CCVC_aug.py` 5 iteration CPU, dữ liệu giả, seed cố định → loss từng iteration + checksum trọng số trùng `tests/golden/origin_cpu_5iter.json` (sinh từ worktree tag `ccvc-origin`) |
   | 3 superset | `CCVC_aug_dapn.py --use_dapg False` → trùng bước 2 |
   | 4 partitions | `--part-root A --baseline-part-root B`: `labeled.txt`, `unlabeled.txt`, `val.txt` trùng sha256 |

   `run_dapn.sh` chạy bước 1 trước mỗi lần chạy; có FAIL → dừng.

   File golden ghi phiên bản torch/numpy/python (hiện: torch 2.13.0+cpu, numpy 2.2.6, python 3.12.9). Với phiên bản khác, bước 2 báo `SKIP` và bước 3 so với lượt chạy `CCVC_aug.py` mới trong cùng môi trường. Sinh lại golden:
   ```bash
   git worktree add /tmp/wt_origin ccvc-origin
   python tools/golden_capture.py --script /tmp/wt_origin/CCVC_aug.py --work /tmp/golden_work --out tests/golden/origin_cpu_5iter.json
   git worktree remove /tmp/wt_origin
   ```

**Giới hạn.** Trên GPU (cuDNN, AMP, 2 GPU) chạy lại cùng code, cùng seed **không trùng bit** với lần trước; mIoU dao động trong mức nhiễu giữa các lần chạy (nhật ký: 2–3 điểm giữa các lần eval). "Tái dựng" nghĩa là cùng code, cùng cấu hình, cùng split; không phải cùng con số đến chữ số thập phân. So sánh có ý nghĩa cần nhiều seed.

## 3. DAPN trong CCVC

```
ảnh ─► nhánh 1 (DeepLabV3+) ─► f1 = concat[reduce(c1), ASPP(c4)]  (304 kênh) ─► da1 ─► DAPG_b1
   └─► nhánh 2 (DeepLabV3+) ─► f̄2 = mapping + dropout            (304 kênh) ─► da2 ─► DAPG_b2
```

- **Hai bộ prototype riêng**: `da1` chỉ thấy đặc trưng nhánh 1, `da2` chỉ thấy nhánh 2; gradient DAPG không đi chéo nhánh (kiểm thử).
- Ba lượt có gradient: ảnh có nhãn (`labeled_img_mask`), không nhãn strong + CutMix (`aug_unlabeled_ignore_img_mask_for_mix`), không nhãn raw (`ignore_img_mask`). Lượt weak (no_grad, sinh nhãn giả): không tính.
- Công thức (theo code DAPN):
  ```
  L_pass = ½[DAPG(f1, da1) + DAPG(f̄2, da2)]        (--dapg_branches both, mặc định) | DAPG(f1, da1) (b1)
  L^l = L_pass(có nhãn);  L^u = ½[L_pass(strong) + L_pass(raw)]
  L_DAPG = ½(L^l + L^u)                             (--dapg_on lu, mặc định) | L^l (l)
  loss = 2·L_sup + 2·L_con + 1·L_dis + w_dapg·L_DAPG,   w_dapg = 0.2
  DAPG = L_intra + 0.5·L_inter + 0.1·L_quality
  ```
- DAPG tính fp32 (autocast tắt). Pixel có nhãn hạ độ phân giải (nearest) = 255 bị loại; ít hơn `max_groups` pixel → bỏ qua lượt đó. Feature NaN/Inf hoặc DAPGLoss báo NaN → loss 0, cảnh báo, đếm `dapg_nan_skips`.
- `da1`, `da2` tạo sau `super().__init__`, trong `torch.random.fork_rng()` với seed `seed + 1000` → phần origin của mô hình khởi tạo **trùng bit** với `CCVC_aug.py` cùng seed (kiểm thử).

### Siêu tham số (spec 4.2)

| Tham số | Giá trị | Cờ |
|---|---|---|
| `max_groups` (C′) | 64 | `--da_max_groups` |
| `temperature` τ | 0.1 | `--da_temperature` |
| `num_iters` T | 3 | `--da_num_iters` |
| `init_method` | xavier | `--da_init` |
| `min_quality` | 0.1 | `--da_min_quality` |
| `use_quality_gate` / `use_mask_predictor` | True / False | cố định |
| `ema_decay` | 0.0 | `--da_ema_decay` |
| `DAPGLoss.margin` | 0.3 | `--dapg_margin` |
| `lambda_inter`, `lambda_quality` | 0.5, 0.1 | `--dapg_lambda_inter`, `--dapg_lambda_quality` |
| `w_dapg` | 0.2 | `--w-dapg` (run_dapn.sh) |
| vị trí | `feat` | `--dapg-pos feat\|c4` |

### Sai khác so với DAPN gốc

- Không có boundary-aware loss.
- Optimizer: `da1`, `da2` vào nhóm SGD "phần còn lại" của CCVC (lr ×10, momentum 0.9, wd 1e-4), không phải AdamW lr 6e-5.
- Mặt nạ pixel 255 (bản gốc dùng mọi pixel).
- Vị trí: đặc trưng discrepancy 304 kênh của CCVC (UNetFormer của DAPN dùng `before_fusion`).
- Áp lên ảnh không nhãn; mỗi mạng con một module DA.
- NaN → loss 0 cho lượt đó thay vì dừng.

### Không đổi so với origin

`vote_threshold_label_selection` và CPL giữ nguyên hành vi upstream: hàm bỏ qua `use_prob`, nên γ được so với **logit thô** trong loss (cột `cc_ratio_loss` đo đúng mặt nạ này; `cc_ratio` origin đo theo softmax). Lịch lr, số epoch, batch, crop, augmentation, λ = 2/2/1, cách eval, `main_metric = miou_b1 at the last epoch` giữ nguyên.

## 4. Preset (`run_dapn.sh --method`)

| `--method` | Script | `use_dapg` | `dapg_on` | `dapg_branches` | Mục đích |
|---|---|---|---|---|---|
| `ccvc_dapn` (mặc định) | `CCVC_aug_dapn.py` | True | `lu` | `both` (da1 + da2) | cấu hình chính |
| `ccvc_dapn_l` | `CCVC_aug_dapn.py` | True | `l` | `both` | DAPG chỉ ảnh có nhãn |
| `ccvc_dapn_b1` | `CCVC_aug_dapn.py` | True | `lu` | `b1` | ablation: một bộ prototype |
| `ccvc_dapn_off` | `CCVC_aug_dapn.py` | False | – | – | kiểm thử tương đương |
| `ccvc_origin` | `CCVC_aug.py` (origin) | – | – | – | chạy lại baseline |

Kết quả: `/kaggle/working/exp/oem_dapn/<split>_s<split_seed>/<method>/seed<seed>/` (origin: `exp/oem`). Phân vùng: `/kaggle/working/partitions/oem_dapn` (cùng quy tắc `tools/prepare_oem.py`, cùng `split_seed` → cùng tập ảnh có nhãn; `--baseline-part-root` để đối chiếu sha256).

## 5. Đầu ra mới

- `train_log.csv` (mỗi 50 iter): `L_dapg`, `L_dapg_w`, `L_dapg_l`, `L_dapg_u`, `L_dapg_b1/b2`, `intra/inter/quality/n_valid_proto/q_mean_b{1,2}`, `cc_ratio_loss`, `dapg_nan_skips`.
- `eval_log.csv`, `final_eval.json`: `agree_wrong_ratio` = #(ŷ1 = ŷ2 ≠ GT)/#pixel hợp lệ, `agree_right_ratio`, `conflict_ratio`, `err_b1`, `agree_wrong_<lớp>`.
- Epoch cuối: `confusion_b1.csv`, `confusion_agree_wrong.csv` (8×8, hàng = GT).
- `config.json`: `code_version`, commit repo này / origin / DAPN, mọi tham số DA/DAPG, `deviations_from_dapn`, kết quả `verify_origin`. Resume bị chặn khi tham số DAPN khác.
- `tools/eval_ckpt.py --ckpt <run>/final.pth`: chạy lại đánh giá (cả checkpoint origin) → `eval_ckpt.json` có số đo đồng thuận; ba mIoU phải khớp `final_eval.json` ±0,01.
- `tools/diagnose_entanglement.py --ckpt <run>/final.pth --branch 1|2` (RQ0/RQ1, không train): đặc trưng 304 kênh của nhánh trên val (1/4 độ phân giải), EM cosine theo công thức `DynamicAnchorModule` (K = 64, τ = 0,1, 20 vòng, không quality gate) trên 200 000 pixel lấy mẫu, rồi gán mọi pixel vào cụm. Với ba nhóm đồng thuận–đúng / đồng thuận–sai / xung đột: độ thuần trung bình của cụm, tỉ lệ ở cụm lẫn (độ thuần < 0,6), tỉ lệ trong dải 3 pixel quanh biên GT; riêng đồng thuận–sai: tỉ lệ nằm trong cụm thuần của lớp bị dự đoán sai. Tổng và theo lớp, trung bình ± độ lệch chuẩn qua 3 seed EM → `diagnose_b<n>.json`, `diagnose_b<n>_per_class.csv`.
- `tools/collect_results_dapn.py --roots <exp/oem> <exp/oem_dapn> [...]`: một `results_summary.csv` cho cả origin và DAPN (cột `code_version`, `method`, `agree_wrong_ratio` lấy từ `final_eval.json` hoặc `eval_ckpt.json`, độ thuần từ `diagnose_b*.json`), in trung bình ± độ lệch chuẩn qua các seed.
- `--dryrun` của preset DAPN: `trial_report.json` thêm `dapn_vs_baseline` (s/iter, peak bộ nhớ/GPU, số giờ dự kiến cho 80 epoch, số phiên Kaggle, % tăng). Baseline lấy theo thứ tự: `--baseline-trial <file>`; dry run origin cùng split trong `exp/oem` của notebook; dry run `ccvc_origin`; dry run origin trong output phiên trước dưới `/kaggle/input`; dry run `ccvc_dapn_off`. Không có thì tự chạy thêm một dry run `ccvc_dapn_off` (tắt bằng `--no-baseline-trial`).

## 6. Công cụ và kiểm thử

| File | Vai trò |
|---|---|
| `tools/verify_origin.py` | xác nhận origin (mục 2) |
| `tools/check_dapn_diff.py` | mọi dòng khác giữa `CCVC_aug.py` và `CCVC_aug_dapn.py` phải nằm trong khối `# [DAPN]` |
| `tools/golden_capture.py` | dấu vân tay hành vi CPU (golden, khởi tạo) |
| `tools/eval_ckpt.py` | đánh giá checkpoint + số đo đồng thuận |
| `tools/diagnose_entanglement.py` | độ thuần cụm đặc trưng của các nhóm pixel (RQ1) |
| `tools/collect_results_dapn.py` | gom kết quả origin + DAPN |
| `tools/trial_compare_dapn.py` | so dry run DAPN với baseline (gọi từ `run_dapn.sh`) |
| `tools/restore_output_dapn.py` | khôi phục `exp/oem_dapn` của phiên trước, không đụng `exp/oem` |
| `tools/make_kaggle_package_dapn.sh` | `dist/ccvc_dapn_oem.zip` từ commit hiện tại |
| `tools/tests_dapn/run_cpu_tests_dapn.sh <dir>` | toàn bộ kiểm thử CPU |

## 7. Cấu trúc file mới

```
CCVC_aug_dapn.py  valid_dapn.py  run_dapn.sh
model/dapn/{__init__, dynamic_anchor, dapg_loss, ccvc_dapg}.py
model/semseg/deeplabv3plus_dapn.py
util/run_utils_dapn.py
tools/{verify_origin, check_dapn_diff, golden_capture, eval_ckpt, restore_output_dapn,
       diagnose_entanglement, collect_results_dapn, trial_compare_dapn}.py
tools/make_kaggle_package_dapn.sh   tools/tests_dapn/
tests/golden/origin_cpu_5iter.json
ORIGIN_MANIFEST.sha256  ORIGIN_SOURCE.json  DAPN_SOURCE.json  README_DAPN.md  HUONG_DAN_CHAY_DAPN.md
```
