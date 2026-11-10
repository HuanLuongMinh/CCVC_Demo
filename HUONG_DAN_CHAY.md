# Hướng dẫn chạy — CCVC trên OpenEarthMap (Kaggle 2×T4)

Mọi thứ chạy bằng **một file bash ở thư mục gốc repo**, mỗi lần **một split**:

```bash
bash run_baseline.sh split 1/4 --dryrun     # chạy thử
bash run_baseline.sh split 1/4              # chạy thật
bash run_baseline.sh split 1/8              # lần sau: split khác
```

Lệnh này tự làm hết: kiểm tra môi trường → khôi phục kết quả phiên trước → kiểm tra trọng số → kiểm tra dữ liệu và chia split → train và eval định kỳ (lưu checkpoint, panel ảnh, biểu đồ) → in kết quả. Chi tiết kỹ thuật nằm trong [README_OEM.md](README_OEM.md).

---

## Bước 0 — Chuẩn bị (làm một lần)

**0.1 Đẩy code lên GitHub (máy local)**
```bash
cd D:/Claude/Semi-CCVC/CCVC-Origin
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```

**0.2 Notebook Kaggle**
- Settings → **Accelerator: GPU T4 ×2**, **Internet: On**.
- Add Data:
  - `aletbm/global-land-cover-mapping-openearthmap` (dữ liệu OEM)
  - `model_lab` (chứa `resnet101.pth`)

---

## Bước 1 — Notebook chỉ có 2 cell

**Cell 1 — lấy code** (clone lần đầu, pull nếu đã có)
```python
!test -d /kaggle/working/CCVC-Origin || git clone -q https://github.com/<user>/<repo>.git /kaggle/working/CCVC-Origin
!git -C /kaggle/working/CCVC-Origin pull -q
%cd /kaggle/working/CCVC-Origin
```

**Cell 2 — chạy** (mỗi lần chỉ sửa dòng này)
```python
!bash run_baseline.sh split 1/4 --dryrun
```

Muốn gọi từ thư mục khác, dùng đường dẫn đầy đủ: `!bash /kaggle/working/CCVC-Origin/run_baseline.sh split 1/4`.

---

## Bước 2 — Chạy thử (`--dryrun`): ước tính thời gian và bắt lỗi trước khi chạy thật

Mục đích: trước khi tốn một phiên 12 giờ bằng Save Version, chạy ngắn **trong editor** (bấm Run cell, không Save Version) để:
- bắt lỗi môi trường, dữ liệu, trọng số, DDP 2 GPU, AMP, bộ nhớ T4;
- đo thời gian thật, từ đó biết split cần bao nhiêu phiên.

```python
!bash run_baseline.sh split 1/4 --dryrun
```
Lệnh chạy 300 iteration rồi eval 1 lần trên toàn bộ val, mất khoảng 15–30 phút, không để lại checkpoint. Cuối log có **TRIAL REPORT** và khối **RESULTS**; bạn cần xem:

| Dòng | Cần kiểm tra |
|---|---|
| `GPUs ... x 2 (nccl)` | Đủ 2 GPU. Nếu là `gloo` thì vẫn chạy được, chỉ chậm hơn |
| `GPU0/GPU1 train peak ... GB` | Dưới 16 GB. Nếu `CUDA out of memory` thì **dừng lại và báo** |
| `speed ... s/iter` | Tốc độ thực tế |
| `full ... -> X h -> N session(s)` | Số phiên Kaggle cần cho split này |
| `mIoU b1` | Chỉ để chắc mô hình có học; không phải kết quả |

Nếu chạy thử kết thúc bằng `finished` và không có lỗi, sang bước 3. Nếu có lỗi, xem bước 7 trước khi chạy thật. Chạy thử dùng đúng cấu hình của lần chạy thật, chỉ ít iteration hơn, nên kết quả đo được áp dụng trực tiếp.

---

## Bước 3 — Chạy thật

Bỏ `--dryrun`:
```python
!bash run_baseline.sh split 1/4
```

Có hai cách chạy:
- **Khuyến nghị (cách dùng chuẩn sau khi chạy thử ổn): Save Version → Save & Run All (Commit).** Kaggle chạy nền tối đa 12 giờ, rồi toàn bộ `/kaggle/working` (gồm `exp/` và `partitions/`) thành **output** của version. Trước khi commit, kiểm tra Cell 2 không còn `--dryrun`.
- **Chạy tương tác:** bạn phải tự lưu `/kaggle/working` trước khi phiên kết thúc.

Khi bắt đầu, log in số ảnh, iter/epoch, tổng iteration, thư mục kết quả và ước tính số phiên.

Cuối log, khối **RESULTS** in mIoU (nếu xong) và bảng tổng hợp mọi split đã xong. Dòng `[run_baseline]` cuối cùng cho biết kết quả:

| Dòng cuối | Ý nghĩa | Việc cần làm |
|---|---|---|
| `finished: ...` | Xong | Chạy split tiếp theo (bước 5) |
| `split này đã chạy xong` | Đã xong từ trước, bỏ qua (vẫn in kết quả) | Đổi split |
| `stopped by MAX_SESSION_HOURS ...` | Hết giờ phiên, đã lưu `last.pth` | Bước 4 |
| `ERROR` / `training failed` | Lỗi | Bước 7 |

---

## Bước 4 — Chạy tiếp ở phiên sau (resume)

Script tự dừng sau khoảng 11.25 giờ, sau khi chạy xong epoch đang dở và lưu `last.pth`.

1. Đợi version trước chạy xong.
2. Add Data → **Your Work → Notebooks → notebook này → version mới nhất** (output).
3. Chạy lại **đúng lệnh cũ** (`!bash run_baseline.sh split 1/4`), không đổi tham số nào.
4. Script tự chép `exp/` và `partitions/` từ output cũ, rồi in `RESUME from epoch N`.

Lặp lại cho tới khi log báo `finished`. Nếu đổi tham số khi resume, script **chặn** và liệt kê khác biệt; hãy dùng lại đúng tham số cũ.

---

## Bước 5 — Split tiếp theo

Đổi giá trị sau `split`. Các split nằm ở thư mục riêng, không ghi đè nhau. Vẫn add output phiên trước để giữ các split đã xong.

```python
!bash run_baseline.sh split 1/8
```

| Lệnh | Có nhãn / không nhãn | Tổng iteration |
|---|---|---|
| `split 1/2` | 1151 / 1152 | 23 040 |
| `split 1/4` | 575 / 1728 | 34 560 |
| `split 1/8` | 287 / 2016 | 40 320 |
| `split 1/16` | 143 / 2160 | 43 200 |
| `split full` | 2303 / 2303 | 46 080 |

Baseline giám sát để so sánh (cùng số iteration): `!bash run_baseline.sh split 1/4 --method sup_pure`.

Không đổi `--split-seed`. Nhờ vậy split nhỏ luôn nằm trong split lớn (1/16 ⊂ 1/8 ⊂ 1/4 ⊂ 1/2).

---

## Bước 6 — Kết quả

Thư mục một lần chạy: `/kaggle/working/exp/oem/<split>_s0/<method>/seed0/`. Ví dụ `split 1/4` → `1_4_s0/ccvc_aug/seed0/`; chạy thử nằm thêm trong `trial/`.

| File | Xem gì |
|---|---|
| `final_eval.json` | **Kết quả chính: `miou_b1`** (nhánh 1, epoch cuối) và IoU từng lớp |
| `plots/*.png` | `miou_curve`, `loss_curve`, `per_class_iou`, `conflict_ratio`, `pseudo_label_class_dist`, `confidence_reliability` |
| `vis/epoch_XXX_iter_YYYYYY.png` | 4 ảnh val cố định: Image, Pred nhánh 1, Pred nhánh 2, Conflict map, Ground truth |
| `eval_log.csv`, `train_log.csv` | Số liệu gốc |
| `exp/oem/results_summary.csv` | Tổng hợp mọi split đã xong |

Ảnh xem trong khung **Output** của Kaggle (bên phải notebook) hoặc tải về. Các lệnh xem lại không cần train:
```bash
python tools/show_results.py --run_dir /kaggle/working/exp/oem/1_4_s0/ccvc_aug/seed0
python tools/collect_results.py --out_root /kaggle/working/exp/oem
python tools/plot_run.py /kaggle/working/exp/oem/1_4_s0/ccvc_aug/seed0
```

---

## Bước 7 — Khi có lỗi

| Thông báo | Nguyên nhân | Cách xử lý |
|---|---|---|
| `missing split` | Thiếu `split <x>` | `bash run_baseline.sh split 1/4` |
| `only ONE split per call` | Truyền nhiều split | Mỗi lần một split |
| `unknown argument` | Sai tên tùy chọn | `bash run_baseline.sh -h` |
| `resnet101.pth not found` | Chưa add `model_lab` | Add Data `model_lab` |
| `pretrained weights rejected` + danh sách key | File không phải ResNet deep-stem của UniMatch/CCVC | Kiểm tra file trong `model_lab` |
| `need exactly one directory with images/... and label/...` | Thiếu dữ liệu OEM, hoặc add nhiều bản | Add đúng một dataset OEM, hoặc `--data-root <dir>` |
| `images and labels are not paired` / `values outside 0..8` | Dữ liệu lỗi | Dừng, gửi log |
| `WARNING: image counts changed` | Dataset Kaggle đổi phiên bản | Báo lại; kết quả cũ và mới không so trực tiếp được |
| `resume blocked: configuration differs` | Đổi tham số giữa các phiên | Dùng lại tham số cũ, hoặc `--force` để chạy lại từ đầu (thư mục cũ được đổi tên) |
| `already exists with a different content` | Split cũ khác split mới | Dừng, gửi log |
| `CUDA out of memory` khi train | Thiếu bộ nhớ GPU | **Dừng, gửi log** |
| `CUDA OOM in original mode -> switching to sliding_window` | Hết bộ nhớ khi eval toàn ảnh | Tự xử lý, không cần làm gì |
| `NCCL init failed ... falling back to gloo` | NCCL lỗi | Vẫn chạy được (chậm hơn) |
| `WARNING: one epoch + eval takes ... min` | Epoch dài hơn khoảng đệm trước mốc 12 giờ | Thêm `--max-session-hours 10.5` |
| `not enough disk` | Output quá 20 GB | Xóa `final.pth` của split cũ không cần nữa |

**Khi gửi lỗi, kèm theo:**
- Khoảng 50 dòng cuối của log.
- File `run_*.log` trong thư mục chạy.
- Lệnh đã dùng.

---

## Phụ lục — Tùy chọn (`bash run_baseline.sh -h`)

| Tùy chọn | Mặc định | Ý nghĩa |
|---|---|---|
| `split <x>` | (bắt buộc) | `1/16`, `1/8`, `1/4`, `1/2`, `full`, `n<số>` (viết `1_4` cũng được) |
| `--dryrun` | tắt | Chạy thử 300 iteration + 1 eval |
| `--trial-iters <n>` | 300 | Số iteration khi chạy thử |
| `--method <m>` | `ccvc_aug` | `sup_pure`, `sup`, `ccr`, `cvc`, `ccvc` |
| `--full-unlabeled <m>` | `same` | `none` (chỉ giám sát, tự dùng `sup_pure`) — chỉ với `split full` |
| `--lambda <p>` | `blended` | `original`, `readme_original` |
| `--seed <n>` / `--split-seed <n>` | 0 / 0 | Seed huấn luyện / seed chia dữ liệu |
| `--epochs <n>` / `--eval-every <n>` | 80 / 5 | |
| `--max-session-hours <h>` | 11.25 | Giới hạn mỗi phiên |
| `--force` | tắt | Chạy lại split đã xong |
| `--data-root`, `--pretrained-dir`, `--restore-dir`, `--out-root`, `--part-root` | tự tìm | Đường dẫn |
| `--extra "<args>"` | — | Tham số thêm cho script train, ví dụ `--extra "--cc_mask_mixed False"` |

Ví dụ: `!bash run_baseline.sh split full --full-unlabeled none`
