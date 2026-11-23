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

## Bước 2 — Thứ tự chạy đề xuất (anh quyết định từng bước)

| Bước | Lệnh (Cell 2) | Mục đích | Đi tiếp khi |
|---|---|---|---|
| 0a | `!python tools/eval_ckpt.py --ckpt <ORIGIN>/exp/oem/1_4_s0/ccvc_aug/seed0/final.pth --part-root <ORIGIN>/partitions/oem` (và `1_8_s0`) | số lỗi "đồng thuận mà sai" của baseline | – |
| 0b | `!python tools/diagnose_entanglement.py --ckpt <ORIGIN>/exp/oem/1_4_s0/ccvc_aug/seed0/final.pth --part-root <ORIGIN>/partitions/oem --branch 1` (rồi `--branch 2`) | lỗi có nằm ở cụm lẫn không | nhóm đồng thuận–sai có độ thuần thấp hơn rõ nhóm đồng thuận–đúng |
| 1 | `!bash run_dapn.sh split 1/4 --dryrun` | bộ nhớ, thời gian | không OOM |
| 2 | `!bash run_dapn.sh split 1/4` | kết quả chính | – |
| 3 | `!bash run_dapn.sh split 1/8` | split thứ hai | – |
| 4 | `--method ccvc_dapn_l`, `--method ccvc_dapn_b1` ở 1/4 | ablation | – |
| 5 | `--seed 1`, `--seed 2` cho `--method ccvc_origin` và `ccvc_dapn` | độ tin cậy | – |

`<ORIGIN>` = đường dẫn output của notebook origin dưới `/kaggle/input/...`. `eval_ckpt.py` ghi `eval_ckpt_<split>_<method>_seed<n>.json` vào `/kaggle/working/exp/oem_dapn/eval_ckpt/` khi thư mục checkpoint chỉ đọc; ba mIoU phải khớp `final_eval.json` của origin (±0,01), lệch thì lệnh báo lỗi.

**Đối chiếu phân vùng với baseline** (khuyến nghị cho bước 2–3):
```python
!bash run_dapn.sh split 1/4 --baseline-part-root <ORIGIN>/partitions/oem
```
Khác dù một dòng → dừng (mã thoát 2).

**Bước 0b** in bảng `group / share / mean purity / in mixed / near boundary` và ghi `diagnose_b<n>.json`, `diagnose_b<n>_per_class.csv` vào `/kaggle/working/exp/oem_dapn/diagnose/<split>_<method>_seed<n>/` (thư mục checkpoint trong `/kaggle/input` chỉ đọc). Mất khoảng hai lượt eval trên val. Dùng được cả cho checkpoint DAPN sau khi chạy xong.

**Gom kết quả** (bất cứ lúc nào):
```python
!python tools/collect_results_dapn.py --roots /kaggle/working/exp/oem_dapn <ORIGIN>/exp/oem
```
→ `/kaggle/working/exp/oem_dapn/results_summary.csv` (hoặc `--out <file>`). Kết quả `eval_ckpt` / `diagnose` của baseline (ghi dưới `exp/oem_dapn/`) được ghép vào đúng dòng origin.

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
- **Mã thoát 0:** xong; kết quả ở khối RESULTS và trong `/kaggle/working/exp/oem_dapn/`.
- **Mã thoát 3:** hết giờ phiên, đã lưu `last.pth` (gồm cả `da1`, `da2`). Phiên sau: add output version trước làm input, chạy **đúng lệnh cũ**. Chỉ cây `exp/oem_dapn` + `partitions/oem_dapn` được khôi phục; nếu input có cả `exp/oem` (origin) thì bị bỏ qua kèm cảnh báo.
- Đổi tham số DAPN (`--w-dapg`, `--method`, ...) giữa chừng → resume bị chặn. Muốn chạy lại từ đầu: `--force` (thư mục cũ được đổi tên, không xóa).

## Bước 5 — Khi nào dừng và gửi log

- `verify_origin` FAIL.
- OOM khi train.
- `dapg_nan_skips` tăng > 10 lần/epoch, hoặc `n_valid_proto_b1/b2` tụt về 1 kéo dài (xem `train_log.csv`).
- `eval_ckpt.py` báo MISMATCH.

Gửi: `run_*.log`, `config.json`, `train_log.csv`, `eval_log.csv` của thư mục chạy.
