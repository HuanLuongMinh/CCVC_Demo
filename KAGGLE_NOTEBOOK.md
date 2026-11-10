# Chạy trên Kaggle — 2 cell

Notebook cùng nội dung: `kaggle_run.ipynb` (Kaggle → File → Import Notebook). Hướng dẫn đầy đủ và xử lý lỗi: [HUONG_DAN_CHAY.md](HUONG_DAN_CHAY.md).

**Thiết lập một lần:**
- Settings → **GPU T4 ×2**, **Internet On**.
- Add Data: `aletbm/global-land-cover-mapping-openearthmap` và `model_lab`.
- Từ phiên thứ 2 của cùng split: add thêm output version trước của notebook này (để resume).

## Cell 1 — lấy code
```python
!test -d /kaggle/working/CCVC-Origin || git clone -q https://github.com/<user>/<repo>.git /kaggle/working/CCVC-Origin
!git -C /kaggle/working/CCVC-Origin pull -q
%cd /kaggle/working/CCVC-Origin
```

## Cell 2 — chạy một split (chỉ sửa dòng này)
```python
!bash run_baseline.sh split 1/4 --dryrun     # chạy thử; chạy thật: bỏ --dryrun; lần sau: split 1/8 ...
```

- **Mã thoát 0:** xong. Kết quả in ở khối RESULTS cuối log; file nằm trong `/kaggle/working/exp/oem/`.
- **Mã thoát 3:** hết giờ phiên, đã lưu `last.pth`. Lưu version, phiên sau add output đó làm input rồi chạy **đúng lệnh cũ**.
- Chạy nền 12 giờ: Save Version → **Save & Run All (Commit)**. Nhớ bỏ `--dryrun` trước khi commit.
