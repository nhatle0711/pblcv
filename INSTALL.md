# Cài đặt & chạy Web Demo (Windows)

Yêu cầu: **Python 3.12**, **Git**, khoảng 6 GB đĩa trống, có mạng ở lần cài đầu.

Mở **PowerShell** và chạy lần lượt:

```powershell
# 1. Clone
git clone https://github.com/NguyenChien510/pblcv.git
cd pblcv

# 2. Tạo môi trường ảo
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Cài thư viện (bắt buộc cài torch TRƯỚC)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 4. Tải checkpoint (~659 MB)
mkdir checkpoints
curl.exe -L -o checkpoints\best_weights.pth https://github.com/NguyenChien510/pblcv/releases/download/v1.0/best_weights.pth

# 5. Chạy
python web_demo.py
```

Trình duyệt sẽ tự mở **http://127.0.0.1:8000**. Lần chạy đầu sẽ tải thêm BERT và YOLO nên mất thêm 1–2 phút. Dừng server bằng **Ctrl + C**.

**Các lần sau** chỉ cần mở PowerShell và chạy:

```powershell
cd pblcv
.venv\Scripts\Activate.ps1
python web_demo.py
```

### Lỗi thường gặp
- **PowerShell chặn `Activate.ps1`:** chạy `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` một lần, sau đó kích hoạt lại môi trường.
- **Có GPU NVIDIA:** ở bước 3, thay `whl/cpu` bằng `whl/cu126`.
- **Cổng 8000 đang bị chiếm:** chạy `python web_demo.py --port 8080`.
- **Lỗi `No usable checkpoint` / `Architecture mismatch`:** file checkpoint chưa tải đủ 659 MB. Xóa file rồi tải lại ở bước 4.
