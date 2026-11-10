# Kết quả kiểm thử (CPU, trước bàn giao) — 10/11/2026

Máy: Windows 11, Python 3.12.9, PyTorch 2.13.0+cpu. Lệnh: `bash tools/tests/run_cpu_tests.sh <thư mục tạm>`
Dữ liệu giả lập: 16 ảnh train và 4 ảnh val `.tif` (cạnh 192, một số ảnh 144), nhãn 0..8, file `.pth` giả đúng key deep-stem. Thông số chạy: ResNet-101 thật, crop 64, `EPOCHS=1–3`, `EVAL_EVERY=1`, `--log_every 2`.

**Kết quả: 66 đạt / 0 lỗi** (lần chạy cuối, có mục 15; lệnh: `REAL_PRETRAINED_DIR=D:/Claude/models bash tools/tests/run_cpu_tests.sh <thư mục tạm>`). Lần chạy đầu có 8 lỗi, tất cả cùng một nguyên nhân: `run_baseline.sh` truyền `--mode_criterion` cho `CCVC_no_aug.py`, script này không có cờ đó. Đã sửa (chỉ truyền cho `ccvc_aug`) rồi chạy lại toàn bộ.

| Mục spec 8.1 | Kiểm tra | Kết quả |
|---|---|---|
| 1 | Sinh dữ liệu giả và trọng số giả (1 hợp lệ, 5 lỗi) | ✅ |
| 2 | `SPLIT=1_8` rồi `1_4`: hai thư mục riêng; labeled(1_8) ⊂ labeled(1_4); có nhãn ∩ không nhãn = ∅; hợp = tập train | ✅ |
| 3 | `full` với `same` và `none` đều chạy được; `none` tự chuyển `sup_pure`; `full_same_s0` / `full_none_s0` tách riêng; cùng tổng iteration | ✅ |
| 4 | `SPLIT="1_4 1_8"`, `"1_4,1_8"`, `""`, `1_3` đều bị từ chối (mã 2) | ✅ |
| 5 | Chạy lại split đã xong → "split này đã chạy xong"; `FORCE=1` → thư mục cũ đổi tên `seed0_old_*`, vẫn còn `final_eval.json` | ✅ |
| 6 | `ccvc_aug`: L_con, L_dis được cộng (w 2/2/1). `sup`: L_con = L_dis = 0 `[not added]`, L_total = L_sup | ✅ |
| 7 | `TRIAL=1 TRIAL_ITERS=5`: `trial_report.json` đủ trường; iter/epoch = ceil(U/batch); tổng iter = iter/epoch × 80; 16 lần eval; công thức giờ và số phiên đúng; không còn `.pth` | ✅ |
| 8 | Resume: `MAX_SESSION_HOURS` rất nhỏ → thoát mã 3 sau epoch 1, có `last.pth`, không có `.tmp`. Resume (kèm một `.tmp` dở dang giả lập file bị kill khi đang ghi) → tiếp từ epoch 1. Đổi `EPOCHS` → bị chặn, liệt kê `epochs: saved=3 now=4`. Resume lần 2 → xong. Iteration liên tục, lr khớp lịch poly qua các lần resume | ✅ |
| 9 | Trọng số giả đúng key → khớp 530/530 (100%). Các file torchvision, thiếu key, sai shape, thiếu stem, thừa key đều bị từ chối và liệt kê đúng key. `run_baseline.sh` dừng (mã 2) trước khi tạo thư mục chạy | ✅ |
| 10 | `collect_results.py`: 7 dòng = 7 lần chạy đã xong (bỏ `trial/` và `_old_`) | ✅ |
| thêm | Panel 5 cột: mỗi lần eval một file, cùng 4 ảnh val; `plot_run.py` vẽ lại đủ 6 biểu đồ; `pseudo_label_dist.csv` có dòng mốc + 2 dòng mỗi lần eval, mỗi dòng cộng lại = 100%; `confidence_reliability.csv` có 1 dòng mỗi lần eval | ✅ |
| thêm | `sup_pure`: không có ảnh không nhãn, tổng iteration bằng `sup` | ✅ |
| 15 (thêm) | Mặt nạ CC của lượt CutMix (`--cc_mask_mixed`): đọc tensor do chính `CCVC_aug.py` dump ra ở iteration 10 (trọng số pretrain thật, γ = 0). Kiểm: dự đoán đã trộn = ảnh được dán vào trong hộp; nhãn giả đã trộn = argmax của dự đoán đã trộn. `True`: mặt nạ trong hộp = mặt nạ của **ảnh được dán vào**, ngoài hộp = của chính ảnh. `False`: = mặt nạ của ảnh chưa trộn ở mọi pixel (hành vi upstream). Mặt nạ của ảnh dán vào và của chính ảnh khác nhau ở 4988 / 522 pixel trong hộp, nên phép kiểm phân biệt được hai hành vi | ✅ |
| thêm | Trial với `TRIAL_ITERS` = 1 hoặc lớn hơn tổng số iteration: vẫn kết thúc bằng eval và ghi report (trước đây lỗi) | ✅ |
| thêm | Khôi phục output phiên trước: chép run đã xong và run đang dở, bỏ `trial/` và `_old_`; giữ bản local đi xa hơn; split khác nội dung → dừng (mã 2); `RESTORE_DIR=none` không làm gì | ✅ |

Ghi chú:
- Mục 15 dump ở iteration 10 chứ không phải iteration 0. Ở iteration 0, các head phân loại vừa khởi tạo ngẫu nhiên, mỗi nhánh gần như dự đoán một lớp hằng nên toàn ảnh đều "xung đột"; mặt nạ hai ảnh trùng nhau và phép kiểm không phân biệt được hai hành vi. Không có trọng số thật (`REAL_PRETRAINED_DIR`), phần kiểm độ nhạy chỉ in cảnh báo.
- Mục 8 của spec đề xuất "đổi `LAMBDA_PRESET` → bị chặn". Nhưng mỗi preset λ khác `blended` có thư mục riêng (`<method>_<lambda>`), nên đổi λ là một lần chạy mới chứ không phải resume. Phép thử chặn dùng `EPOCHS` thay thế. λ (`w_CE/w_con/w_dis`) vẫn nằm trong danh sách khóa so sánh khi resume.
- Panel và biểu đồ được kiểm bằng mắt. Mô hình mới train vài chục iteration trên dữ liệu ngẫu nhiên nên dự đoán còn đơn điệu và mIoU gần 0, điều này đúng như mong đợi.

## Kiểm tra loader trên dữ liệu OEM thật (máy local, cùng bản với Kaggle: 2303 train / 384 val)
`python tools/tests/check_loader_real.py --label_root D:/Claude/PixelAnalys/label --image_root D:/Claude/PixelAnalys/OpenEarthMap_wo_xBD`
- 2303/2303 và 384/384 ảnh có nhãn ghép cặp được với ảnh; PIL đọc được (ảnh RGB uint8, nhãn L uint8; kích thước 650 / 900 / 1000 / 1024).
- Giá trị nhãn thô 0..8. Ánh xạ 0→255, c→c−1 đúng trên từng pixel.
- `train_l`: (3,512,512), mask chỉ có giá trị trong {0..7, 255}. `train_u`: không mở file nhãn nào. Padding 254→255 đúng tỉ lệ lý thuyết (0.5971 với ảnh 650 thu nhỏ 0.5).
- Kết luận: **LOADER CHECK: OK**.

## Chưa kiểm được trên CPU (sẽ được xác minh bằng `TRIAL=1` trên Kaggle)
- DDP 2 GPU (`mp.spawn`, NCCL / fallback gloo, SyncBN, gom RNG và histogram giữa các rank).
- AMP fp16 và `GradScaler`; bộ nhớ thực tế trên T4 (train và eval toàn ảnh 1024).
- Tốc độ s/iter thật, và thời gian một epoch so với khoảng đệm 45 phút của `MAX_SESSION_HOURS`.
