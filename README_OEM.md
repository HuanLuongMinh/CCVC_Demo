# CCVC (CVPR 2023) trên OpenEarthMap — tái lập theo protocol Pascal VOC

Fork của code chính thức [xiaoyao3302/CCVC](https://github.com/xiaoyao3302/CCVC) @ `274672cca69e9aec809b3f0ecd86ed9885d2a0a5` (xem `UPSTREAM_COMMIT`), chạy trên Kaggle 2×T4. Mỗi lần gọi `bash run_baseline.sh split <x>` chạy **đúng một split**; chạy split khác thì đổi giá trị sau `split`, kết quả không ghi đè lên nhau.

Hướng dẫn chạy từng bước (kèm xử lý lỗi): [HUONG_DAN_CHAY.md](HUONG_DAN_CHAY.md). Các cell Kaggle chi tiết: [KAGGLE_NOTEBOOK.md](KAGGLE_NOTEBOOK.md).

---

## 1. Chạy một split

Mọi thứ chạy bằng **một file bash ở thư mục gốc repo**:
```bash
bash run_baseline.sh split 1/4 --dryrun         # chạy thử: 300 iteration + 1 eval + trial_report.json
bash run_baseline.sh split 1/4                  # chạy thật (80 epoch)
bash run_baseline.sh split 1/8                  # lần sau: split khác
bash run_baseline.sh split 1/4 --method sup_pure
bash run_baseline.sh split full --full-unlabeled none
bash run_baseline.sh -h                         # mọi tùy chọn
```

Script tự làm theo thứ tự:
1. Đọc tham số. Thiếu split, nhiều split (`split 1/4 1/8`), split sai dạng hay tùy chọn lạ thì báo lỗi.
2. In thông tin môi trường (`tools/check_env.py`): GPU, dữ liệu, trọng số, output phiên trước.
3. Khôi phục `exp/` và `partitions/` từ output phiên trước (mục 6).
4. Tìm `resnet101.pth` dưới `/kaggle/input/model_lab*/`, link vào `pretrained/` và kiểm tra key (mục 5.4).
5. Kiểm tra dữ liệu, tạo `data_report.json`, sinh split (`tools/prepare_oem.py`).
6. Train trong một tiến trình riêng. Khi có 2 GPU thì dùng `mp.spawn`.
7. In kết quả của lần chạy và bảng tổng hợp các split đã xong (`tools/show_results.py`).

| Mã thoát | Ý nghĩa |
|---|---|
| 0 | Xong (hoặc split đã xong từ trước, hoặc chạy thử xong) |
| 3 | Dừng vì `--max-session-hours`. Lưu output, phiên sau chạy **lại đúng lệnh cũ** để resume |
| 2 | Lỗi tham số, dữ liệu hoặc trọng số (thông báo ngay phía trên) |
| khác | Lỗi khi train (xem `run_*.log` trong thư mục chạy) |

**Chạy split tiếp theo:** đổi giá trị sau `split`. Giữ nguyên `--split-seed` (mặc định 0) để các split lồng nhau (1/16 ⊂ 1/8 ⊂ 1/4 ⊂ 1/2). Split đã có `final_eval.json` sẽ được bỏ qua (vẫn in kết quả). `--force` chạy lại; khi đó thư mục cũ được đổi tên thành `..._old_<thời gian>` chứ không bị xóa.

## 2. Tùy chọn của `run_baseline.sh`

| Tùy chọn | Biến môi trường tương đương | Mặc định | Ý nghĩa |
|---|---|---|---|
| `split <x>` | `SPLIT` | (bắt buộc) | `1/16`, `1/8`, `1/4`, `1/2`, `full` hoặc `n<số>` (số ảnh có nhãn). Viết `1_4` cũng được. Mỗi lần một giá trị |
| `--dryrun` | `TRIAL=1` | tắt | Chạy `--trial-iters` iteration, eval 1 lần, ghi `trial_report.json` vào `.../trial/`; không để lại checkpoint |
| `--trial-iters <n>` | `TRIAL_ITERS` | 300 | |
| `--method <m>` | `METHOD` | `ccvc_aug` | `sup`, `sup_pure`, `ccr`, `cvc`, `ccvc`, `ccvc_aug` (mục 4) |
| `--seed <n>` | `SEED` | 0 | Seed huấn luyện |
| `--split-seed <n>` | `SPLIT_SEED` | 0 | Seed chia dữ liệu. **Giữ nguyên** giữa các lần chạy |
| `--full-unlabeled <m>` | `FULL_UNLABELED` | `same` | Chỉ với `full`. `same`: tập không nhãn là chính ảnh train (bỏ nhãn). `none`: chỉ giám sát, tự dùng `sup_pure` |
| `--import-labeled <file>` | `IMPORT_LABELED` | — | Danh sách ảnh có nhãn có sẵn (thư mục `<split>_imp<hash>_s<seed>`) |
| `--epochs <n>` | `EPOCHS` | 80 | Theo paper (VOC). 1 epoch = 1 lượt qua tập không nhãn |
| `--eval-every <n>` | `EVAL_EVERY` | 5 | Eval mỗi N epoch và ở epoch cuối |
| `--batch-size <n>` | `BATCH_SIZE` | 2 | Ảnh mỗi GPU, mỗi loader. 2×T4 cho batch hiệu dụng 4 có nhãn + 4 không nhãn |
| `--lambda <p>` | `LAMBDA_PRESET` | `blended` | `blended` = 2/2/1, `original` = 5/1/2, `readme_original` = 5/2/1 (w_CE/w_con/w_dis, mục 3.1) |
| `--max-session-hours <h>` | `MAX_SESSION_HOURS` | 11.25 | Quá mốc: chạy xong epoch hiện tại, lưu `last.pth`, thoát mã 3 |
| `--force` | `FORCE=1` | tắt | Chạy lại split đã xong (thư mục cũ được đổi tên) |
| `--data-root <dir>` | `DATA_ROOT` | tự tìm | Thư mục có `images/{train,val}` và `label/{train,val}` dưới `/kaggle/input` |
| `--pretrained-dir <dir>` | `PRETRAINED_DIR` | tự tìm | Thư mục chứa `resnet101.pth` (mặc định `/kaggle/input/model_lab*`) |
| `--restore-dir <d>` | `RESTORE_DIR` | `auto` | `auto`: tự tìm output phiên trước dưới `/kaggle/input`; hoặc một đường dẫn; hoặc `none` |
| `--out-root <dir>` | `OUT_ROOT` | `/kaggle/working/exp/oem` | Thư mục kết quả |
| `--part-root <dir>` | `PART_ROOT` | `/kaggle/working/partitions/oem` | Thư mục chứa các split |
| `--extra "<args>"` | `EXTRA` | — | Tham số thêm cho script train, ví dụ `--extra "--log_every 100"` |
| — | `NUM_WORKERS` | 2 | Số worker DataLoader cho mỗi GPU |
| — | `SESSION_START` | lúc gọi script | Thời điểm bắt đầu phiên (unix time) cho `--max-session-hours` |

Tham số dòng lệnh được ưu tiên hơn biến môi trường. Thư mục một lần chạy: `$OUT_ROOT/<split>_s<split-seed>/<method>[_<lambda nếu khác blended>]/seed<seed>/` (ví dụ `split 1/4` → `1_4_s0/ccvc_aug/seed0/`). Với `full` là `full_same_s0/` hoặc `full_none_s0/`. Chạy thử nằm thêm trong `trial/`.

Xem lại kết quả mà không cần train: `python tools/show_results.py --run_dir <thư mục chạy>`, `python tools/collect_results.py --out_root /kaggle/working/exp/oem`, `python tools/plot_run.py <thư mục chạy>`.

## 3. Cấu hình

### 3.1 Giá trị lấy từ paper (arXiv 2303.01276, mục 3–4)
| Thành phần | Giá trị |
|---|---|
| Mô hình | 2 × DeepLabV3+ ResNet-101 pretrain ImageNet (deep stem của UniMatch/CCVC), OS16, dilations 6/12/18 |
| Mapping Ψ_map | conv1×1 (304→304) + BN + ReLU + Dropout2d 0.5, chỉ ở nhánh 2 (`--mode_mapping else`) |
| Optimizer | SGD, lr khởi tạo **0.001**, poly scheduler |
| Số epoch | **80** (VOC blended; paper chỉ dùng 40 cho VOC original) |
| Crop | **512** |
| Batch | Số ảnh có nhãn = không nhãn trong mỗi mini-batch |
| Weak aug | Resize ngẫu nhiên 0.5–2.0, crop, lật ngang (code upstream) |
| Strong aug (`ccvc_aug`) | ColorJitter, Grayscale, blur, CutMix (code upstream) |
| λ1/λ2/λ3 | Paper Eq. 5: **L = λ1·L_sup + λ2·L_con + λ3·L_dis**. VOC blended 2/2/1 → `w_CE 2, w_con 2, w_dis 1` (mặc định). VOC original 5/1/2 → `w_CE 5, w_con 1, w_dis 2` |
| γ | **0.9**, so với **xác suất softmax** (`--conf_prob True`) |
| ω_c | **2.0** (`--w_confident 2.0`, truyền tường minh vì `CCVC_aug.py` upstream để mặc định 1.5) |
| Metric / suy luận | mIoU. Suy luận một nhánh: **số chính là mIoU nhánh 1 ở epoch cuối** |

Lý do chọn λ blended: tỉ lệ không nhãn : có nhãn của OEM (15.1 / 7.0 / 3.0 : 1 ở 1/16, 1/8, 1/4) gần với VOC blended hơn VOC original (114/57/28 : 1).

**λ và README upstream không khớp nhau.** README upstream chạy VOC original với `--w_CE 5.0 --w_con 2.0 --w_dis 1.0`, tức tráo λ2/λ3 so với paper (5/1/2). Repo này theo paper. Preset `--lambda readme_original` (5/2/1) chỉ để đối chiếu, không dùng làm mặc định.

### 3.2 Giá trị paper không nói, lấy theo code upstream
| Thành phần | Giá trị |
|---|---|
| Hệ số lr | Backbone ×1, phần còn lại ×10 (README Pascal: `--lr_backbone 1.0 --lr_network 10.0`) |
| Momentum, weight decay | 0.9, 1e-4 |
| Poly power | 0.9. lr được cập nhật sau mỗi `optimizer.step()` theo `base·(1−it/T)^0.9` |
| BatchNorm | SyncBatchNorm khi DDP (2 GPU); BN thường khi 1 GPU |
| L_dis | Đối xứng có `detach`: ½[(1+cos(f1.detach, f̄2)) + (1+cos(f̄2.detach, f1))], trên đặc trưng 304 kênh trước khối `fuse`, cho cả ảnh có nhãn và không nhãn |
| CC | `vote_threshold_label_selection`: pixel xung đột mà nhánh tạo nhãn giả tự tin > γ thì nhân ω_c. Ở lượt CutMix, mặt nạ tính trên dự đoán đã CutMix (sửa lỗi upstream, mục 3.4.7) |
| `ccvc_aug` | Ảnh không nhãn đi qua 3 lượt forward: weak (eval, no_grad) sinh nhãn giả → strong + CutMix → weak (train) |
| Tập có nhãn | Nhân bản cho bằng số ảnh không nhãn (`nsample`) |

### 3.3 Sai khác bắt buộc so với paper (cũng ghi trong `config.json`)
| Sai khác | Lý do |
|---|---|
| **Mixed precision (AMP)**: chỉ forward chạy fp16, đầu ra ép về fp32, mọi loss tính bằng fp32; dùng `GradScaler` | FP32 cần khoảng 17–21 GB/GPU (đo 17.6 GB trên server cũ), T4 chỉ có 16 GB |
| **Batch hiệu dụng nhỏ hơn**: 2+2 ảnh/GPU, tức 4+4 trên 2×T4 (paper 24). **Không** scale lr | Giới hạn bộ nhớ T4 |
| Eval mỗi `EVAL_EVERY` epoch thay vì mỗi epoch | Tiết kiệm thời gian, không ảnh hưởng huấn luyện |
| Không dump dự đoán toàn tập val (`evaluate_save`); chỉ lưu panel 4 ảnh cố định | Dung lượng `/kaggle/working` |
| Không lưu checkpoint best-on-val | Chỉ giữ `last.pth` (khi đang chạy) và `final.pth` |

Không tự giảm crop, backbone hay batch. Nếu OOM khi train với AMP và batch 2/GPU thì dừng và báo lại.

### 3.4 Sửa lỗi upstream (mỗi mục một commit)
1. `str2bool` trả `Trueepochs` (NameError) → `True`.
2. `valid.py` hard-code 19 lớp trong sliding window → `cfg['nclass']`.
3. γ so với logit thô → thêm `--conf_prob` (mặc định `True` = xác suất như paper; `False` = hành vi upstream).
4. Thêm dataset `oem` (đọc `.tif`, ánh xạ 0→255 và 1..8→0..7, **không đọc nhãn ảnh không nhãn**).
5. Eval OEM: chế độ `original` (toàn ảnh) có AMP và `empty_cache()`; OOM thì chuyển hẳn sang `sliding_window` (cửa sổ 512, bước 2/3). Chế độ thực tế được ghi trong `eval_log.csv`.
6. Đã rà chéo `CCVC_aug.py` và `CCVC_no_aug.py`: các sửa đổi áp dụng như nhau.
7. **Mặt nạ CC của lượt strong (CutMix) trong `CCVC_aug.py`** — cờ `--cc_mask_mixed` (mặc định `True`).
   - **Lỗi upstream:** mặt nạ pixel "xung đột và tự tin" (nhân ω_c) được tính trên dự đoán của ảnh **chưa CutMix** (`unlabeled_pred1/2`), nhưng lại áp lên loss của nhãn giả **đã CutMix** (`aug_unlabeled_pseudo_label*_for_mix`). Bên trong hộp CutMix (2–40% diện tích ảnh, ở mọi batch), ω_c vì thế rơi vào sai pixel. Mode `normal` của chính upstream thì dùng đúng bản đã CutMix.
   - **Sửa:** mặt nạ CC của lượt strong được tính trên `aug_unlabeled_pred*_for_mix`, cùng nguồn với nhãn giả đã CutMix. Confidence là xác suất softmax, khớp `--conf_prob`. Bên trong hộp, mặt nạ bây giờ theo dự đoán của ảnh được dán vào. Lượt weak (`raw`) giữ nguyên vì không có CutMix. Thống kê `conflict_ratio`/`cc_ratio` trong `train_log.csv` vẫn tính trên nhãn giả của lượt 1 (chưa trộn).
   - **Quay về hành vi upstream:** `bash run_baseline.sh split 1/4 --extra "--cc_mask_mixed False" --seed 1` (thư mục chạy vẫn là `ccvc_aug`, nên dùng `--seed` hoặc `--out-root` khác để không resume nhầm; giá trị cờ được ghi trong `config.json` và resume bị chặn nếu khác).
   - Kiểm thử: `tools/tests/check_cc_mask.py` đọc tensor do script dump ra ở iteration đầu (`--debug_dump_cc`) và xác nhận mặt nạ bên trong hộp khớp với ảnh được dán vào (mục 15 của `run_cpu_tests.sh`).
8. (thêm) Chạy được với 1 GPU. Upstream gọi `dist.barrier()` hay `model.module` khi không có DDP, và khi `sampler=None` thì 1 GPU không xáo trộn dữ liệu.
9. (thêm) `len(trainloader_u) // 8` bằng 0 khi tập nhỏ.
10. (thêm) PyTorch 2.x: tensorboardX thành tùy chọn; `requirements.txt` đổi `sklearn` → `scikit-learn`.

## 4. Phương pháp (`METHOD`)

| METHOD | Script | Loss không nhãn | mode_confident | L_dis | Mapping |
|---|---|---|---|---|---|
| `sup` | no_aug | không (w_con = w_dis = 0) | no | không | không |
| `sup_pure` | no_aug `--sup_pure` | không có ảnh không nhãn | no | không | không |
| `ccr` | no_aug | cross pseudo-label | no | không | không |
| `cvc` | no_aug | cross pseudo-label | no | có | có |
| `ccvc` | no_aug | cross pseudo-label | vote_threshold | có | có |
| `ccvc_aug` | aug | weak→strong + cross | vote_threshold | có | có |

Ở 10 iteration đầu, log in từng thành phần loss (giá trị thô, giá trị đã nhân hệ số, và có được cộng vào tổng hay không) để kiểm tra preset.

**`sup` và `sup_pure`.**
- **`sup` giữ nguyên như upstream.** `CCVC_no_aug.py` vẫn forward ảnh không nhãn ở chế độ **train, có gradient**. L_con và L_dis vẫn được tính nhưng nhân 0 và không cộng vào tổng.
  - Gradient không bị ảnh hưởng.
  - **BatchNorm thì có.** Thống kê theo batch của lượt có nhãn không bị trộn, nhưng **running mean/var (dùng khi eval) được cập nhật bởi cả ảnh không nhãn**. Với SyncBN, các thống kê này gom trên cả 2 GPU. Vì vậy `sup` không phải baseline "chỉ dùng ảnh có nhãn" thuần.
  - Lượt forward thêm làm `sup` chậm và tốn bộ nhớ hơn.
- **`sup_pure`** chỉ dùng ảnh có nhãn: không có loader ảnh không nhãn, không forward ảnh không nhãn. Tập có nhãn vẫn được nhân bản lên bằng số ảnh trong `unlabeled.txt` (chỉ đếm dòng, không mở ảnh), như upstream. Nhờ vậy iter/epoch, tổng iteration và lịch lr giống hệt `sup` và `ccvc_aug`.

## 5. Dữ liệu, split, kiểm tra

### 5.1 OEM
Kaggle `aletbm/global-land-cover-mapping-openearthmap`: `images/{train,val}/*.tif` và `label/{train,val}/*.tif`. Bản kiểm tra (10/11/2026) có 2303 train và 384 val. Kích thước ảnh không đồng nhất (650, 900, 1000, 1024). Số ảnh luôn được đếm từ thư mục, không hard-code. Nếu số ảnh khác `data_report.json` cũ, script cảnh báo: các split chạy trước sẽ không còn so sánh trực tiếp được.

Nhãn: 0 'unknown' → 255 (ignore); 1..8 → 0..7 = Bareland, Rangeland, Developed space, Road, Tree, Water, Agriculture land, Building.

### 5.2 Split (`tools/prepare_oem.py`)
k = floor(N / mẫu số). Ảnh có nhãn là k ảnh đầu của một hoán vị cố định sinh từ `--split-seed`, nên các split lồng nhau dù chạy ở các lần khác nhau. Phần còn lại là ảnh không nhãn. Với N = 2303:

| Split | Có nhãn | Không nhãn | Iter/epoch (2×T4) | Eval mỗi … iter | Tổng iter (80 epoch) |
|---|---|---|---|---|---|
| 1_16 | 143 | 2160 | 540 | 2700 | 43 200 |
| 1_8 | 287 | 2016 | 504 | 2520 | 40 320 |
| 1_4 | 575 | 1728 | 432 | 2160 | 34 560 |
| 1_2 | 1151 | 1152 | 288 | 1440 | 23 040 |
| full | 2303 | 2303 (same) / 0 (none) | 576 | 2880 | 46 080 |

Iter/epoch = `len(trainloader_u)` = floor(ceil(U/2)/2) (DistributedSampler + drop_last). Với 1 GPU thì gấp đôi. 16 lần eval cho 80 epoch.

### 5.3 Lưu ý về `full`
- **Khác dòng "Full" của paper:** trên VOC, paper vẫn còn 9.118 ảnh SBD làm dữ liệu không nhãn. Ở OEM, `same` dùng lại chính ảnh train (bỏ nhãn) làm không nhãn; `none` chỉ giám sát.
- Với `full`, N_labeled = N_train. Vì vậy 1 epoch (theo tập không nhãn ở `same`, hoặc theo tập có nhãn nhân bản ở `none`/`sup_pure`) **trùng với "1 lượt qua tập có nhãn"**. `full/same` và `full/none` có cùng iter/epoch và cùng tổng iteration. Script in hai con số này khi bắt đầu, và chúng cũng có trong `config.json`.
- `split full --full-unlabeled none` tự dùng `sup_pure`.

### 5.4 Trọng số pretrain
`resnet101.pth` của UniMatch/CCVC (deep stem), **không phải** torchvision. Script dừng và liệt kê đúng từng key lỗi khi:
- thiếu key backbone hoặc key sai shape (bỏ qua `num_batches_tracked`);
- thiếu key deep-stem (`conv1.0/3/6`, `bn1` 128 kênh);
- có key thừa ngoài `fc.weight` và `fc.bias`.

Riêng ngưỡng 95% của spec thì không đủ: state_dict torchvision khớp 96.8% key theo tên.

## 6. Nhiều phiên Kaggle, resume
- `last.pth` được ghi đè cuối mỗi epoch (model, optimizer, GradScaler, epoch, iteration, RNG của mọi GPU). Cách ghi: ghi `last.pth.tmp`, fsync, rồi đổi tên. Kèm theo là `progress.json`.
- Thứ tự trong một epoch là train → eval (nếu đến lượt) → checkpoint, nên resume không bỏ sót lần eval nào.
- Resume tự động khi có `last.pth`. Resume bị **chặn** nếu một tham số ảnh hưởng huấn luyện khác với `config.json` (split, method, λ, epochs, batch, crop, lr, số ảnh, số GPU…); thông báo liệt kê từng khác biệt. `--max-session-hours`, `--eval-every` và `NUM_WORKERS` được phép đổi.
- `--max-session-hours` (mặc định 11.25): khi vượt mốc, chạy xong epoch hiện tại, lưu, thoát mã 3. Nếu một epoch cộng eval mất lâu hơn 45 phút còn lại trước mốc 12 giờ của Kaggle, script in cảnh báo.
- `--restore-dir auto` (mặc định): tìm output phiên trước dưới `/kaggle/input`. Mỗi lần chạy lấy bản đi xa nhất; không bao giờ đè lên bản local đi xa hơn. Split khác nội dung thì dừng.

## 7. Đầu ra mỗi lần chạy
| File | Nội dung |
|---|---|
| `config.json` | Mọi tham số, commit upstream và commit code, số ảnh, batch hiệu dụng, GPU, PyTorch, backend DDP, sai khác so với paper, lịch sử resume |
| `train_log.csv` | Mỗi 50 iter: epoch, iteration, lr backbone/head, L_sup, L_con, L_dis (đã nhân λ), L_total, tỉ lệ xung đột, tỉ lệ CC, s/iter |
| `eval_log.csv` | Mỗi lần eval: chế độ, thời gian, mIoU nhánh 1/2/trung bình, xung đột trên val, IoU 8 lớp × 3 |
| `pseudo_label_dist.csv` | % lớp của nhãn giả nhánh 1/2 giữa hai lần eval (lượt weak no_grad ở `aug`, lượt forward duy nhất ở `no_aug`; bỏ pixel 254/255), cộng một dòng mốc `labeled_gt` |
| `confidence_reliability.csv` | Nhánh 1 trên val: % pixel có max-softmax > γ, pixel accuracy của nhóm > γ, ≤ γ và toàn bộ |
| `final_eval.json` | Kết quả epoch cuối. **Số chính = nhánh 1**; best-on-val chỉ để tham khảo |
| `final.pth` | Trọng số cuối (khoảng 0.5 GB) |
| `vis/epoch_XXX_iter_YYYYYY.png` | 4 ảnh val cố định × 5 cột: Image, Pred nhánh 1 (chính), Pred nhánh 2, Conflict map, Ground truth |
| `plots/*.png` | `miou_curve`, `loss_curve`, `per_class_iou`, `conflict_ratio`, `pseudo_label_class_dist`, `confidence_reliability`. Vẽ lại bằng `python tools/plot_run.py <run_dir>` |
| `log.txt`, `run_*.log` | Log |

Bảng màu OEM chính thức (open-earth-map.org): Bareland #800000, Rangeland #00FF24, Developed space #949494, Road #FFFFFF, Tree #226126, Water #0045FF, Agriculture land #4BB549, Building #DE1F07.

## 8. Kiểm thử
- `bash tools/tests/run_cpu_tests.sh <thư mục tạm>` chạy trên CPU, dữ liệu giả lập, khoảng 15–20 phút. Kết quả xem ở [TEST_REPORT.md](TEST_REPORT.md).
- `python tools/tests/check_loader_real.py --data_root <OEM>` kiểm tra loader trên dữ liệu OEM thật.
