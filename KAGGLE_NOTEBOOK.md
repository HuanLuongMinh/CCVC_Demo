# Chạy trên Kaggle (2×T4) — các cell mẫu

Mỗi phiên chạy **một split**. Code lấy bằng `git clone` từ repo public của bạn. Kết quả nằm dưới `/kaggle/working`; để dùng tiếp ở phiên sau, add output đó làm input.

## 0. Thiết lập notebook (một lần)
- **Accelerator:** GPU T4 ×2. **Internet:** On (cần để `git clone`).
- **Add Data:**
  1. `aletbm/global-land-cover-mapping-openearthmap` (OEM)
  2. `model_lab` (dataset của bạn, chứa `resnet101.pth`)
  3. *(từ phiên thứ 2 trở đi)* output của version notebook trước: Add Data → Your Work → Notebooks → chọn notebook này → version gần nhất. Output được mount dưới `/kaggle/input/<tên>/` và `run_baseline.sh` sẽ tự khôi phục từ đó.

## 1. Ghi giờ bắt đầu phiên + lấy code
```python
import os, time
os.environ['SESSION_START'] = str(int(time.time()))   # để MAX_SESSION_HOURS tính cả thời gian chuẩn bị
```
```python
REPO = 'https://github.com/<user>/<repo>.git'          # repo public của bạn
!test -d /kaggle/working/CCVC-Origin || git clone -q $REPO /kaggle/working/CCVC-Origin
!git -C /kaggle/working/CCVC-Origin pull -q
%cd /kaggle/working/CCVC-Origin
!git rev-parse HEAD
```

## 2. Kiểm tra môi trường (tùy chọn)
```python
!nvidia-smi --query-gpu=name,memory.total --format=csv
!python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
!find /kaggle/input -maxdepth 4 -type d -name label          # thư mục dữ liệu OEM
!find /kaggle/input -maxdepth 4 -name resnet101.pth         # trọng số
```
Kiểm tra loader trên dữ liệu thật. Đổi đường dẫn thành thư mục cha của `label/` tìm được ở trên:
```python
!python tools/tests/check_loader_real.py --data_root /kaggle/input/<thư mục OEM>
```

## 3. Chạy thử (tùy chọn, khuyến nghị cho split đầu tiên)
```python
!SPLIT=1_4 TRIAL=1 bash tools/run_baseline.sh
```
Lệnh in bảng tóm tắt và ghi `/kaggle/working/exp/oem/1_4_s0/ccvc_aug/seed0/trial/trial_report.json`. Bảng gồm s/iter, bộ nhớ đỉnh mỗi GPU, thời gian eval, số giờ và số phiên dự kiến. Chạy thử không để lại checkpoint.

## 4. Chạy đầy đủ một split
```python
!SPLIT=1_4 bash tools/run_baseline.sh
```
- Dùng `!` để chạy trong **tiến trình riêng**; không gọi `python CCVC_aug.py` trực tiếp trong cell.
- Khi bắt đầu, script in iter/epoch, tổng iteration và ước tính số phiên Kaggle.
- Khi quá `MAX_SESSION_HOURS` (11.25 giờ), script chạy xong epoch hiện tại, lưu `last.pth` rồi dừng ("STOPPED FOR RESUME"). Phiên sau chạy **lại đúng lệnh này**.
- Split khác: đổi `SPLIT` (ví dụ `SPLIT=1_8`). Không đổi `SPLIT_SEED`.
- Split đã xong thì script báo "split này đã chạy xong" và bỏ qua; `FORCE=1` để chạy lại (thư mục cũ được đổi tên, không xóa).

## 5. Xem kết quả
```python
!python tools/collect_results.py --out_root /kaggle/working/exp/oem
```
```python
from IPython.display import Image, display
import glob
run = '/kaggle/working/exp/oem/1_4_s0/ccvc_aug/seed0'
for p in sorted(glob.glob(run + '/plots/*.png')): display(Image(p))
display(Image(sorted(glob.glob(run + '/vis/*.png'))[-1]))      # panel của lần eval mới nhất
```
Vẽ lại biểu đồ bất cứ lúc nào: `!python tools/plot_run.py <run_dir>`.

## 6. Lưu kết quả để dùng ở phiên sau
- **Khuyến nghị:** **Save Version → Save & Run All (Commit)**. Kaggle chạy lại toàn bộ notebook ở chế độ nền (tối đa 12 giờ), rồi lưu toàn bộ `/kaggle/working` (gồm `exp/` và `partitions/`) thành output của version đó.
- Phiên sau: add output version vừa rồi làm input (mục 0.3), rồi chạy lại cùng các cell. `run_baseline.sh` tự chép `exp/` và `partitions/` về `/kaggle/working` (`RESTORE_DIR=auto`) và resume.
- Nếu chạy ở chế độ interactive, bạn phải tự lưu `/kaggle/working` trước khi phiên kết thúc. Hãy kiểm tra lại cách Kaggle hiện hành lưu output ở chế độ này.
- **Dung lượng:** mỗi split chạy xong có `final.pth` khoảng 0.5 GB; split đang dở có thêm `last.pth` khoảng 1 GB. Giới hạn output khoảng 20 GB; xóa bớt split cũ nếu cần.
