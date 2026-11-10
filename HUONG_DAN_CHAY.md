# Hướng dẫn chạy — CCVC trên OpenEarthMap (Kaggle 2×T4)

Quy tắc chung: **mỗi lần chạy một split**, luôn bằng `bash tools/run_baseline.sh`. Chi tiết kỹ thuật nằm trong [README_OEM.md](README_OEM.md).

---

## Bước 0 — Chuẩn bị (làm một lần)

**0.1 Đẩy code lên GitHub (máy local)**
```bash
cd D:/Claude/Semi-CCVC/CCVC-Origin
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```

**0.2 Tạo notebook Kaggle**
- Settings → **Accelerator: GPU T4 ×2**, **Internet: On**.
- Add Data:
  - `aletbm/global-land-cover-mapping-openearthmap` (dữ liệu OEM)
  - `model_lab` (chứa `resnet101.pth`)

---

## Bước 1 — Các cell của notebook (dùng cho mọi phiên)

**Cell 1 — giờ bắt đầu phiên + lấy code**
```python
import os, time
os.environ['SESSION_START'] = str(int(time.time()))
REPO = 'https://github.com/<user>/<repo>.git'
!test -d /kaggle/working/CCVC-Origin || git clone -q $REPO /kaggle/working/CCVC-Origin
!git -C /kaggle/working/CCVC-Origin pull -q
%cd /kaggle/working/CCVC-Origin
!git rev-parse --short HEAD
```

**Cell 2 — chạy (chỉ sửa dòng `SPLIT`)**
```python
SPLIT = '1_4'
!SPLIT=$SPLIT bash tools/run_baseline.sh
```

**Cell 3 — xem kết quả**
```python
!python tools/collect_results.py --out_root /kaggle/working/exp/oem
from IPython.display import Image, display
import glob
run = f'/kaggle/working/exp/oem/{SPLIT}_s0/ccvc_aug/seed0'   # SPLIT=full: .../full_same_s0/... hoặc full_none_s0/sup_pure
for p in sorted(glob.glob(run + '/plots/*.png')): display(Image(p))
vis = sorted(glob.glob(run + '/vis/*.png'))
if vis: display(Image(vis[-1]))
```

---

## Bước 2 — Chạy thử trước split đầu tiên (khuyến nghị)

Ở Cell 2, tạm thay lệnh bằng:
```python
!SPLIT=1_4 TRIAL=1 bash tools/run_baseline.sh
```
Lệnh này chạy 300 iteration rồi eval 1 lần, mất khoảng 15–30 phút. Cuối log có bảng **TRIAL REPORT**; bạn cần xem:

| Dòng | Cần kiểm tra |
|---|---|
| `GPUs ... x 2 (nccl)` | Đủ 2 GPU. Nếu là `gloo` thì vẫn chạy được, chỉ chậm hơn |
| `GPU0/GPU1 train peak ... GB` | Dưới 16 GB. Nếu OOM thì **dừng lại và báo** (không tự giảm batch hay crop) |
| `speed ... s/iter` | Tốc độ thực tế |
| `full ... -> X h -> N session(s)` | Số phiên Kaggle cần cho split này |
| `mIoU b1` | Chỉ để chắc mô hình có học (> 0 rõ ràng); không phải kết quả |

Chạy thử không để lại checkpoint và không ảnh hưởng lần chạy thật. Xong thì đổi Cell 2 về lệnh chạy thật.

---

## Bước 3 — Chạy thật một split

Có hai cách chạy:

**Cách A (khuyến nghị): Save Version → Save & Run All (Commit)**
- Kaggle chạy toàn bộ notebook ở chế độ nền (tối đa 12 giờ, không cần mở trình duyệt).
- Xong, toàn bộ `/kaggle/working` (gồm `exp/` và `partitions/`) thành **output** của version đó.
- Trước khi commit, nhớ đổi Cell 2 về lệnh chạy thật (không để `TRIAL=1`).

**Cách B: chạy tương tác** (bấm Run từng cell). Khi đó bạn phải tự lưu `/kaggle/working` trước khi phiên kết thúc.

Khi bắt đầu, log in số ảnh có nhãn / không nhãn / val, iter/epoch, tổng iteration, thư mục kết quả và ước tính số phiên.

Dòng cuối của log cho biết kết quả:

| Dòng cuối của log | Ý nghĩa | Việc cần làm |
|---|---|---|
| `finished: ...` | Split chạy xong | Chạy split tiếp theo (bước 5) |
| `split này đã chạy xong` | Split đã xong từ trước, script bỏ qua | Đổi `SPLIT` |
| `stopped by MAX_SESSION_HOURS` / `STOPPED FOR RESUME` | Hết thời gian phiên, đã lưu `last.pth` | Làm bước 4 |
| `ERROR` / `training failed` | Lỗi | Xem bước 7 |

---

## Bước 4 — Chạy tiếp ở phiên sau (resume)

Một split thường cần hơn một phiên 12 giờ. Script tự dừng sau khoảng 11.25 giờ, khi đã chạy xong epoch đang dở và lưu `last.pth`.

1. Đợi version (commit) trước chạy xong.
2. Mở notebook → **Add Data → Your Work → Notebooks → chọn notebook này → version mới nhất**.
3. Chạy lại **đúng** các cell như cũ, **không đổi `SPLIT` hay tham số nào**. Bằng Save & Run All là tiện nhất.
4. Script tự chép `exp/` và `partitions/` từ output cũ về, rồi in `RESUME from epoch N`.

Lặp lại cho tới khi log báo `finished`.

> Nếu đổi tham số (epochs, batch, λ…) khi resume, script sẽ **chặn** và liệt kê khác biệt. Đó là để bảo vệ kết quả; hãy dùng lại đúng tham số cũ.

---

## Bước 5 — Chạy split tiếp theo

Chỉ cần đổi `SPLIT` ở Cell 2. Kết quả các split nằm ở các thư mục riêng, không ghi đè nhau. Vẫn add output phiên trước để giữ các split đã xong.

Thứ tự gợi ý (theo số giờ ước tính, nhanh trước):

| SPLIT | Có nhãn / không nhãn | Tổng iteration |
|---|---|---|
| `1_2` | 1151 / 1152 | 23 040 |
| `1_4` | 575 / 1728 | 34 560 |
| `1_8` | 287 / 2016 | 40 320 |
| `1_16` | 143 / 2160 | 43 200 |
| `full` | 2303 / 2303 | 46 080 |

Baseline giám sát để so sánh (cùng số iteration):
```python
!SPLIT=1_4 METHOD=sup_pure bash tools/run_baseline.sh
```
Khi xem kết quả của phương pháp khác, sửa `ccvc_aug` trong Cell 3 thành tên method tương ứng.

**Không đổi `SPLIT_SEED`** giữa các lần chạy. Nhờ vậy split nhỏ luôn nằm trong split lớn (1/16 ⊂ 1/8 ⊂ 1/4 ⊂ 1/2).

---

## Bước 6 — Đọc kết quả

Thư mục một lần chạy: `/kaggle/working/exp/oem/<split>_s0/<method>/seed0/`

| File | Xem gì |
|---|---|
| `final_eval.json` | **Kết quả chính: `miou_b1`** (nhánh 1, epoch cuối) và IoU từng lớp |
| `plots/miou_curve.png` | mIoU theo iteration (nhánh 1 nét đậm) |
| `plots/loss_curve.png` | L_sup, L_con, L_dis |
| `plots/per_class_iou.png`, `conflict_ratio.png`, `pseudo_label_class_dist.png`, `confidence_reliability.png` | Phân tích thêm |
| `vis/epoch_XXX_iter_YYYYYY.png` | 4 ảnh val cố định: Image, Pred nhánh 1, Pred nhánh 2, Conflict map, Ground truth |
| `eval_log.csv`, `train_log.csv` | Số liệu gốc |
| `results_summary.csv` (trong `exp/oem/`) | Bảng tổng hợp mọi split đã xong (`collect_results.py`) |

Eval diễn ra mỗi 5 epoch và ở epoch cuối, tức 16 lần cho 80 epoch. Mỗi lần eval sinh thêm một panel trong `vis/` và vẽ lại các biểu đồ.

---

## Bước 7 — Khi có lỗi

| Thông báo | Nguyên nhân | Cách xử lý |
|---|---|---|
| `resnet101.pth not found` | Chưa add `model_lab` | Add Data `model_lab` |
| `pretrained weights rejected` + danh sách key | File không phải ResNet deep-stem của UniMatch/CCVC | Kiểm tra lại file trong `model_lab` |
| `need exactly one directory with images/... and label/...` | Chưa add dữ liệu OEM, hoặc add nhiều bản | Add đúng một dataset OEM, hoặc đặt `DATA_ROOT=...` |
| `images and labels are not paired` / `values outside 0..8` | Dữ liệu lỗi | Dừng, gửi log |
| `WARNING: image counts changed` | Dataset Kaggle đổi phiên bản | Kết quả cũ và mới không so trực tiếp được; báo lại |
| `resume blocked: configuration differs` | Đổi tham số giữa các phiên | Dùng lại tham số cũ, hoặc `FORCE=1` để chạy lại từ đầu (thư mục cũ được đổi tên, không xóa) |
| `already exists with a different content` (partition) | Split cũ khác split mới | Dừng, gửi log |
| `CUDA out of memory` khi train | Thiếu bộ nhớ GPU | **Dừng, gửi log** (không tự giảm cấu hình) |
| `CUDA OOM in original mode -> switching to sliding_window` | Hết bộ nhớ khi eval toàn ảnh | Tự xử lý, không cần làm gì |
| `NCCL init failed ... falling back to gloo` | NCCL lỗi | Vẫn chạy được (chậm hơn) |
| `WARNING: one epoch + eval takes ... min` | Một epoch lâu hơn khoảng đệm 45 phút trước mốc 12 giờ | Thêm `MAX_SESSION_HOURS=10.5` vào lệnh ở Cell 2 |
| `not enough disk` | Output quá 20 GB | Xóa bớt `final.pth` của split cũ không cần nữa |

**Khi gửi lỗi, kèm theo:**
- Khoảng 50 dòng cuối của log trong notebook.
- File `run_*.log` trong thư mục chạy.
- Lệnh đã dùng (giá trị `SPLIT`, `METHOD`, các biến đã đổi).

---

## Phụ lục — Biến hay dùng (đặt trước `bash` trong lệnh)

| Biến | Mặc định | Khi nào đổi |
|---|---|---|
| `SPLIT` | `1_4` | Mỗi lần chạy: `1_16`, `1_8`, `1_4`, `1_2`, `full`, `n<số>` |
| `TRIAL=1` | `0` | Chạy thử |
| `METHOD` | `ccvc_aug` | `sup_pure`, `sup`, `ccr`, `cvc`, `ccvc` |
| `FULL_UNLABELED` | `same` | `none` khi `SPLIT=full` để chỉ giám sát |
| `LAMBDA_PRESET` | `blended` | `original` để thử λ của VOC original |
| `FORCE=1` | `0` | Chạy lại một split đã xong |
| `MAX_SESSION_HOURS` | `11.25` | Giảm nếu có cảnh báo về khoảng đệm |
| `EXTRA="--cc_mask_mixed False"` | — | Chỉ khi muốn chạy đúng hành vi upstream của mặt nạ CC (dùng `SEED` khác để không lẫn thư mục) |

Ví dụ: `!SPLIT=full FULL_UNLABELED=none bash tools/run_baseline.sh`
