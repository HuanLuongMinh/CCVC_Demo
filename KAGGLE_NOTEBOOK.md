# Chạy trên Kaggle — đủ các cell, bắt đầu từ `git clone`

Có sẵn notebook cùng nội dung: **`kaggle_run.ipynb`** (Kaggle → File → Import Notebook → chọn file này, hoặc tạo notebook mới rồi dán lần lượt các cell dưới đây). Hướng dẫn chi tiết từng bước và xử lý lỗi: [HUONG_DAN_CHAY.md](HUONG_DAN_CHAY.md).

# CCVC trên OpenEarthMap — chạy một split

**Thiết lập notebook (một lần):**
- Settings → **Accelerator: GPU T4 ×2**, **Internet: On**.
- Add Data: `aletbm/global-land-cover-mapping-openearthmap` (OEM) và `model_lab` (chứa `resnet101.pth`).
- Từ phiên thứ 2 của cùng một split: Add Data → Your Work → Notebooks → **output version trước của notebook này** (để resume).

**Mỗi lần chạy:** chỉ sửa Cell 1, rồi Run All (hoặc Save Version → Save & Run All).

## Cell 1 — Cấu hình (chỉ sửa ở đây)

```python
import os, time
os.environ['SESSION_START'] = str(int(time.time()))     # tính giới hạn 11.25 h từ đầu phiên

REPO_URL  = 'https://github.com/<user>/<repo>.git'     # repo public của bạn
REPO_DIR  = '/kaggle/working/CCVC-Origin'

SPLIT          = '1_4'        # 1_16 | 1_8 | 1_4 | 1_2 | full | n<số>   — MỘT split mỗi lần
TRIAL          = 0            # 1 = chạy thử ngắn (300 iter + 1 eval); 0 = chạy đầy đủ
METHOD         = 'ccvc_aug'   # ccvc_aug | sup_pure | sup | ccr | cvc | ccvc
FULL_UNLABELED = 'same'       # chỉ khi SPLIT='full': same | none (none -> sup_pure)
LAMBDA_PRESET  = 'blended'    # blended (2/2/1) | original (5/1/2) | readme_original (5/2/1)
SEED           = 0
SPLIT_SEED     = 0            # KHÔNG đổi giữa các lần chạy (split lồng nhau)
EXTRA_ENV      = {}           # biến khác của run_baseline.sh, ví dụ {'MAX_SESSION_HOURS': '10.5', 'FORCE': '1'}
```

## Cell 2 — Lấy code (clone lần đầu, pull các lần sau)

```python
!test -d {REPO_DIR} || git clone -q {REPO_URL} {REPO_DIR}
!git -C {REPO_DIR} pull -q
%cd {REPO_DIR}
!git log --oneline -1
```

## Cell 3 — Kiểm tra môi trường

```python
!nvidia-smi --query-gpu=index,name,memory.total --format=csv
!python -c "import torch; print('torch', torch.__version__, '| GPUs:', torch.cuda.device_count())"
print('--- dữ liệu OEM (thư mục label/):')
!find /kaggle/input -maxdepth 5 -type d -name label 2>/dev/null
print('--- trọng số:')
!find /kaggle/input -maxdepth 5 -name resnet101.pth 2>/dev/null
print('--- output phiên trước (nếu có):')
!find /kaggle/input -maxdepth 4 -type d -path '*exp/oem' 2>/dev/null
```

## Cell 4 — Chạy (một split)

```python
import subprocess
env = dict(os.environ, SPLIT=str(SPLIT), TRIAL=str(TRIAL), METHOD=METHOD, FULL_UNLABELED=FULL_UNLABELED,
           LAMBDA_PRESET=LAMBDA_PRESET, SEED=str(SEED), SPLIT_SEED=str(SPLIT_SEED),
           **{k: str(v) for k, v in EXTRA_ENV.items()})
proc = subprocess.Popen(['bash', 'tools/run_baseline.sh'], cwd=REPO_DIR, env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    if 'Warning' not in line and 'warn(' not in line:
        print(line, end='')
RC = proc.wait()
print('\n' + {0: 'EXIT 0: xong (hoặc split đã xong trước đó / trial xong)',
              3: 'EXIT 3: hết giờ phiên, đã lưu last.pth -> Save Version, phiên sau add output này và chạy lại y hệt',
              }.get(RC, 'EXIT %d: LỖI - xem log phía trên và run_*.log trong thư mục chạy' % RC))
```

## Cell 5 — Kết quả

```python
import glob, json
from IPython.display import Image, display
!python tools/collect_results.py --out_root /kaggle/working/exp/oem

method = 'sup_pure' if (SPLIT == 'full' and FULL_UNLABELED == 'none') else METHOD
tag = f'full_{FULL_UNLABELED}_s{SPLIT_SEED}' if SPLIT == 'full' else f'{SPLIT}_s{SPLIT_SEED}'
mdir = method if LAMBDA_PRESET == 'blended' else f'{method}_{LAMBDA_PRESET}'
run = f'/kaggle/working/exp/oem/{tag}/{mdir}/seed{SEED}' + ('/trial' if TRIAL else '')
print('run dir:', run)
for f in ('final_eval.json', 'trial/trial_report.json' if not TRIAL else 'trial_report.json', 'progress.json'):
    p = os.path.join(run, f)
    if os.path.exists(p):
        d = json.load(open(p))
        print(f, '->', {k: d[k] for k in d if k in ('miou_b1', 'miou_b2', 'miou_ave', 'epochs_done', 'epochs',
                                                   's_per_iter_median', 'projection')})
for p in sorted(glob.glob(run + '/plots/*.png')):
    display(Image(p))
vis = sorted(glob.glob(run + '/vis/*.png'))
if vis:
    print('panel mới nhất:', os.path.basename(vis[-1]))
    display(Image(vis[-1]))
```

## Sau khi chạy
- **EXIT 3 (chưa xong):** Save Version (nếu đang chạy tương tác) → phiên sau: Add Data → output version vừa rồi → **không sửa Cell 1** → Run All. Script tự chép `exp/` + `partitions/` về và resume.
- **EXIT 0 (xong):** đổi `SPLIT` ở Cell 1 cho split tiếp theo (vẫn add output cũ để giữ các split đã xong).
- **Chạy nền 12 h:** Save Version → **Save & Run All (Commit)**; `/kaggle/working` thành output của version. Nhớ đặt `TRIAL = 0` trước khi commit.
- Xử lý lỗi: xem `HUONG_DAN_CHAY.md`, bước 7.
