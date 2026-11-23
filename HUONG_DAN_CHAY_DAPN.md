# Hướng dẫn chạy — CCVC-DAPN trên Kaggle (2×T4)

Mọi lần chạy dùng **một file**: `bash run_dapn.sh split <x> [--method ...] [--dryrun]`, mỗi lần một split. Chi tiết kỹ thuật: [README_DAPN.md](README_DAPN.md).

Bản DAPN tách biệt hoàn toàn với bản origin:

| | Origin | DAPN |
|---|---|---|
| Thư mục code | `/kaggle/working/CCVC-Origin` | `/kaggle/working/CCVC-DAPN` |
| Lệnh | `bash run_baseline.sh ...` | `bash run_dapn.sh ...` |
| Kết quả | `/kaggle/working/exp/oem` | `/kaggle/working/exp/oem_dapn` |
| Phân vùng | `/kaggle/working/partitions/oem` | `/kaggle/working/partitions/oem_dapn` |

---

## Bước 0 — Chuẩn bị (một lần)

**0.1 Gói code (máy local)**
```bash
cd D:/Claude/Semi-CCVC/CCVC-DAPN
bash tools/make_kaggle_package_dapn.sh        # -> dist/ccvc_dapn_oem.zip
```
Upload `ccvc_dapn_oem.zip` thành một Kaggle Dataset (ví dụ `ccvc-dapn-oem`). Mỗi khi code đổi: commit, đóng gói lại, tạo version mới của dataset.

Cách khác: tạo repo GitHub riêng (ví dụ `HuanLuongMinh/CCVC_DAPN`, **không** dùng CCVC_Demo), `git push`, rồi clone ở Cell 1.

**0.2 Notebook Kaggle**
- Settings → **Accelerator: GPU T4 ×2**, **Internet: On**.
- Add Data: `aletbm/global-land-cover-mapping-openearthmap`, `model_lab`, dataset `ccvc-dapn-oem`.
- (Tùy chọn) output của lần chạy origin, để lấy `final.pth` origin (bước 0a/0b) và đối chiếu phân vùng.

---

## Bước 1 — Notebook 2 cell

**Cell 1 — lấy code**
```python
!rm -rf /kaggle/working/CCVC-DAPN
!Z=$(find /kaggle/input -name ccvc_dapn_oem.zip | head -n 1); \
 if [ -n "$Z" ]; then unzip -q -o "$Z" -d /kaggle/working; \
 else cp -r "$(find /kaggle/input -maxdepth 5 -type d -name CCVC-DAPN | head -n 1)" /kaggle/working/; fi
%cd /kaggle/working/CCVC-DAPN
!python tools/verify_origin.py --steps 1
```
(Kaggle thường tự giải nén zip trong dataset; lệnh trên xử lý cả hai trường hợp. Kết quả nằm ngoài thư mục code nên `rm -rf` không mất gì.)

Dùng GitHub thì thay bằng:
```python
!test -d /kaggle/working/CCVC-DAPN || git clone -q https://github.com/<user>/CCVC_DAPN.git /kaggle/working/CCVC-DAPN
!git -C /kaggle/working/CCVC-DAPN pull -q
%cd /kaggle/working/CCVC-DAPN
```

**Cell 2 — chạy** (mỗi lần chỉ sửa dòng này)
```python
!bash run_dapn.sh split 1/4 --dryrun
```

---

## Bước 2 — Một lệnh: train rồi đo luôn

```python
!bash run_dapn.sh split 1/4
```
Lệnh này làm trọn: kiểm tra origin → train 80 epoch (resume qua nhiều phiên) → **ngay sau khi train xong, tự chạy các phép đo giai đoạn 2**:

1. `diagnose_entanglement.py` nhánh 1 và nhánh 2 trên `final.pth` vừa train.
2. Với **baseline cùng split, cùng seed** (checkpoint CCVC origin): `eval_ckpt.py` (lấy `agree_wrong_ratio`; ba mIoU phải khớp `final_eval.json` của baseline) và `diagnose_entanglement.py` nhánh 1, 2.
   Baseline được tìm theo thứ tự: `--baseline-run <dir>`; `run_baseline.sh` đã chạy trong cùng notebook (`/kaggle/working/exp/oem/...`); `--method ccvc_origin`; **output notebook origin được add làm input** (`/kaggle/input/**/exp/oem/<split>/ccvc_aug/seed<n>`). Không tìm thấy → chỉ đo run này.
3. `results_summary.csv` (gom origin + DAPN) trong `/kaggle/working/exp/oem_dapn/`, và in **ANALYSIS SUMMARY**: mIoU b1/b2/ave, `agree_wrong_ratio`, `agree_right_ratio`, `conflict_ratio`, độ thuần cụm của nhóm đồng thuận–đúng / đồng thuận–sai — baseline và run này đặt cạnh nhau.

**Thời gian phiên.** Mỗi phép đo được ước lượng theo thời gian eval của run (diagnose ≈ 2,5 lần một lượt eval). Phép đo nào không kịp xong trước giới hạn 12 giờ (chừa 20 phút) thì **không bắt đầu**; log báo `analysis INCOMPLETE`. Phiên sau chạy **đúng lệnh cũ**: train đã xong nên lệnh chỉ làm nốt phần đo còn thiếu (phần đã có được giữ, không tính lại). Cũng vậy khi split đã chạy xong từ trước: chạy lại lệnh là bổ sung phép đo.

Tùy chọn: `--no-analysis` (chỉ train), `--no-analysis-baseline` (không đo baseline), `--baseline-run <thư mục run origin>`.
Mã thoát: 0 xong; 3 hết giờ khi đang train (chạy lại lệnh cũ); **4 = train xong nhưng phép đo báo lỗi** (ví dụ `eval_ckpt` MISMATCH) → dừng và gửi log.

**Chuẩn bị để có baseline:** add output của notebook origin (có `exp/oem/1_4_s0/ccvc_aug/seed0/final.pth`) làm input của notebook DAPN.

### Thứ tự chạy đề xuất (anh quyết định từng bước)

| Bước | Lệnh (Cell 2) | Mục đích | Đi tiếp khi |
|---|---|---|---|
| 1 | `!bash run_dapn.sh split 1/4 --dryrun` | bộ nhớ, thời gian, so với baseline | không OOM |
| 2 | `!bash run_dapn.sh split 1/4` | kết quả chính + đo RQ0/RQ1 cho DAPN và baseline | – |
| 3 | `!bash run_dapn.sh split 1/8` | split thứ hai (+ đo) | – |
| 4 | `--method ccvc_dapn_l`, `--method ccvc_dapn_b1` ở 1/4 | ablation (+ đo) | – |
| 5 | `--seed 1`, `--seed 2` cho `--method ccvc_origin` và `ccvc_dapn` | độ tin cậy | – |

**Đo baseline trước khi train (tùy chọn, RQ0/RQ1 riêng):** vẫn chạy riêng được, kết quả được lệnh bước 2 dùng lại (không tính lại):
```python
!python tools/eval_ckpt.py --ckpt <ORIGIN>/exp/oem/1_4_s0/ccvc_aug/seed0/final.pth --part-root <ORIGIN>/partitions/oem --out /kaggle/working/exp/oem_dapn/eval_ckpt/eval_ckpt_1_4_ccvc_aug_seed0.json
!python tools/diagnose_entanglement.py --ckpt <ORIGIN>/exp/oem/1_4_s0/ccvc_aug/seed0/final.pth --part-root <ORIGIN>/partitions/oem --branch 1
```
`<ORIGIN>` = đường dẫn output của notebook origin dưới `/kaggle/input/...`. Đi tiếp khi nhóm đồng thuận–sai có độ thuần thấp hơn rõ nhóm đồng thuận–đúng.

### Chạy chẩn đoán origin (0a/0b)

Một notebook riêng, một lần Save Version, cho checkpoint **origin** split 1/4 và 1/8: `eval_ckpt.py` (0a: đồng thuận–sai chính xác, theo lớp, ma trận nhầm lẫn) rồi `diagnose_entanglement.py` nhánh 1 và 2 (0b). Hai script chạy nguyên bản, ngưỡng mặc định (`--tol 0.01`, `--mixed 0.6`, biên 3 px, K 64, τ 0.1). Đây là mốc so sánh cho `ccvc_dapn`.

**Input cần gắn** (Add Data):
- output version Kaggle của **hai lần chạy origin** (có `exp/oem/1_4_s0/ccvc_aug/seed0/final.pth` và `exp/oem/1_8_s0/ccvc_aug/seed0/final.pth`, cùng `config.json`, `final_eval.json`, `partitions/oem/val.txt`);
- dataset OEM `aletbm/global-land-cover-mapping-openearthmap`;
- gói `ccvc_dapn_oem.zip` mới (hoặc clone branch `DAPN_CCVC`, xem dưới).
- Settings: **GPU T4**, Internet On nếu clone từ GitHub.

Script tự tìm mọi đường dẫn dưới `/kaggle/input` và in bảng đường đã chọn. Tìm thấy 0 hoặc nhiều hơn 1 checkpoint khác nhau cho một split → dừng, in danh sách; khi đó truyền tường minh `--run-1-4 <thư mục run> --run-1-8 <thư mục run>` (tương tự `--part-root <thư mục có val.txt>`, `--data-root <OEM>`). Bản sao giống hệt của cùng một `final.pth` (ví dụ run 1/4 được khôi phục vào output 1/8) chỉ tính một lần.

**Cell 1 — lấy code + kiểm đường (phiên interactive, 1–2 phút)**
```python
!rm -rf /kaggle/working/CCVC-DAPN
!Z=$(find /kaggle/input -name ccvc_dapn_oem.zip | head -n 1); \
 if [ -n "$Z" ]; then unzip -q -o "$Z" -d /kaggle/working; \
 else cp -r "$(find /kaggle/input -maxdepth 5 -type d -name CCVC-DAPN | head -n 1)" /kaggle/working/; fi
%cd /kaggle/working/CCVC-DAPN
!bash tools/run_diag_origin.sh --quick
```
Hoặc lấy code từ GitHub (branch DAPN): `!git clone -q -b DAPN_CCVC --single-branch https://github.com/HuanLuongMinh/CCVC_Demo.git /kaggle/working/CCVC-DAPN`.
`--quick` chạy preflight (torch/GPU, đủ `final.pth`/`config.json`/`final_eval.json`, `val.txt` 384 dòng và đủ ảnh/nhãn, `verify_origin` bước 1, đĩa trống > 1 GB) và một `diagnose` nhỏ (1/4, nhánh 1, 2 ảnh) vào `.../diag_origin/_quick/`. Mã thoát 0 → sang Cell 2.

**Cell 2 — chạy thật (Save Version → Save & Run All)**
```python
!bash tools/run_diag_origin.sh
!cat /kaggle/working/exp/oem_dapn/diag_origin/summary.md
```
Thứ tự: eval 1/4, eval 1/8, diagnose 1/4 b1, 1/8 b1, 1/4 b2, 1/8 b2. Một bước lỗi không dừng các bước sau.
Mã thoát: 0 mọi bước OK; 2 preflight lỗi (chưa chạy gì); 4 có bước lỗi (xem `run_log.txt`; `eval_ckpt` MISMATCH với `final_eval.json` → dừng và gửi log).
Tùy chọn: `--only eval|diagnose`, `--branches "1"`, `--splits 1/4`.

**Đầu ra** — `/kaggle/working/exp/oem_dapn/diag_origin/` (được lưu theo version):
```
run_log.txt  summary.md  summary.csv  paths.env
1_4/eval_ckpt/eval_ckpt.json, eval_ckpt_confusion_{b1,agree_wrong}.csv
1_4/diagnose_b1/diagnose_b1.json, diagnose_b1_per_class.csv      (và diagnose_b2/)
1_8/...                                                            (giống 1_4)
```
`summary.md`: bảng A (mIoU, đối chiếu `final_eval.json`, agree_right/agree_wrong/conflict, err_b1, cận dưới cũ `err_b1 − conflict`), bảng B (agree_wrong theo lớp), bảng C (mỗi split × nhánh: tỉ lệ pixel, độ thuần, cụm lẫn, dải biên, cụm thuần của lớp bị đoán; mean ± std trên 3 seed EM). Chỉ số liệu, không diễn giải.

**Thời gian:** ~20–30 phút trên T4 (mỗi lượt val 384 ảnh ≈ 85–92 s, tổng 10 lượt). Kiểm tra tài khoản có mở được phiên GPU song song với phiên DAPN đang chạy không (giới hạn số phiên GPU đồng thời của Kaggle).

**Lấy về máy:** trang notebook → Output → tải version; chép thư mục `diag_origin/` vào `D:\Papers\Semi_SITC\STIC_semi\CCVC_Search\CCVC-Origin\`.

**Đối chiếu phân vùng với baseline** (khuyến nghị cho bước 2–3):
```python
!bash run_dapn.sh split 1/4 --baseline-part-root <ORIGIN>/partitions/oem
```
Khác dù một dòng → dừng (mã thoát 2).

**Gom lại mọi run** (bất cứ lúc nào, ví dụ sau nhiều seed):
```python
!python tools/collect_results_dapn.py --roots /kaggle/working/exp/oem_dapn <ORIGIN>/exp/oem
```

**Tiêu chí thành công (chốt trước khi chạy):** so với `ccvc_origin` cùng split, `ccvc_dapn` giảm `agree_wrong_ratio` và tăng mIoU nhánh 1 vượt độ lệch chuẩn giữa các seed. Một seed chỉ là tín hiệu ban đầu.

---

## Bước 3 — Chạy thử (`--dryrun`): cần xem gì

300 iteration + 1 lần eval, không để lại checkpoint. Trong log:

| Dòng | Kiểm tra |
|---|---|
| `verify_origin: PASS`, `PASS check_dapn_diff` | code origin nguyên vẹn |
| `DAPN: use_dapg True ... BOTH branches (da1 + da2)` | DAPN gắn trên **cả hai** mạng con |
| `[iter 0..9] L_DAPG ... L^l ... L^u ... b1 ... b2 ...` | L^l, L^u, b1, b2 đều > 0 |
| `GPU0/GPU1 train peak ... GB` | dưới 16 GB; `CUDA out of memory` → **dừng và báo**, không tự giảm crop/batch/C′ |
| `speed ... s/iter`, `full ... -> N session(s)` | thời gian, số phiên |
| `DAPN vs baseline dry run` | bảng s/iter, peak GB, số giờ 80 epoch của DAPN so với baseline và % tăng; ghi vào `trial_report.json` (`dapn_vs_baseline`) |

Baseline cho bảng so sánh: dry run origin cùng split (cùng notebook, hoặc trong output phiên trước được add làm input), hoặc `--baseline-trial <trial_report.json>`. Không có thì lệnh tự chạy thêm một dry run `ccvc_dapn_off` (300 iteration nữa); tắt bằng `--no-baseline-trial`.

## Bước 4 — Chạy thật và resume

- Bỏ `--dryrun`, Save Version → **Save & Run All (Commit)**.
- **Mã thoát 0:** xong cả train và phép đo; kết quả ở khối RESULTS, ANALYSIS SUMMARY và trong `/kaggle/working/exp/oem_dapn/` (`results_summary.csv`).
- **Mã thoát 4:** train xong, phép đo báo lỗi (xem dòng `[analysis]`).
- **Mã thoát 3:** hết giờ phiên, đã lưu `last.pth` (gồm cả `da1`, `da2`). Phiên sau: add output version trước làm input, chạy **đúng lệnh cũ**. Chỉ cây `exp/oem_dapn` + `partitions/oem_dapn` được khôi phục; nếu input có cả `exp/oem` (origin) thì bị bỏ qua kèm cảnh báo.
- Đổi tham số DAPN (`--w-dapg`, `--method`, ...) giữa chừng → resume bị chặn. Muốn chạy lại từ đầu: `--force` (thư mục cũ được đổi tên, không xóa).

## Bước 5 — Khi nào dừng và gửi log

- `verify_origin` FAIL.
- OOM khi train.
- `dapg_nan_skips` tăng > 10 lần/epoch, hoặc `n_valid_proto_b1/b2` tụt về 1 kéo dài (xem `train_log.csv`).
- `eval_ckpt.py` báo MISMATCH (mã thoát 4).

Gửi: `run_*.log`, `config.json`, `train_log.csv`, `eval_log.csv` của thư mục chạy.
