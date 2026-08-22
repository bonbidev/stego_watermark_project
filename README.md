# 🔐 Hệ thống Giấu tin & Thủy vân số (Stego & Watermark)

Ứng dụng minh họa và thực nghiệm các kỹ thuật **giấu tin mật (steganography)** và **thủy vân số (digital watermarking)** trên ảnh, kết hợp mã hóa **AES**, đánh giá chất lượng ảnh và kiểm tra độ bền trước các phép tấn công phổ biến.

Xây dựng bằng Python + [Streamlit](https://streamlit.io/), giao diện web tương tác, không cần biết lập trình vẫn có thể sử dụng.

---

## ✨ Tính năng chính

| Nhóm | Thuật toán / Chức năng |
|---|---|
| **Giấu tin (Steganography)** | LSB (Least Significant Bit), PVD (Pixel Value Differencing), DCT (Discrete Cosine Transform) |
| **Mã hóa** | AES (mật khẩu do người dùng cung cấp, qua `pycryptodome`) |
| **Thủy vân số (Watermarking)** | DWT (Discrete Wavelet Transform), DWT-SVD (kết hợp Singular Value Decomposition) |
| **Đánh giá chất lượng ảnh** | MSE, PSNR, SSIM (giữa ảnh gốc và ảnh đã xử lý), NC (Normalized Correlation cho watermark trích xuất) |
| **Kiểm tra độ bền (Robustness)** | JPEG compression, Gaussian noise, Salt & Pepper noise, Gaussian blur, Median blur, Resize, Crop, Rotate, Sharpen |
| **Steganalysis** | Phân tích LSB Ratio, LSB Entropy để phát hiện dấu hiệu giấu tin |

---

## 🖥️ Demo giao diện (Streamlit)

Ứng dụng gồm 5 tab chính:

1. **🔐 Giấu tin** — mã hóa nội dung bí mật bằng AES rồi nhúng vào ảnh bằng LSB / PVD / DCT
2. **🔓 Trích xuất** — trích xuất và giải mã nội dung từ ảnh đã giấu tin
3. **©️ Thủy vân** — nhúng watermark (ảnh logo) vào ảnh gốc bằng DWT hoặc DWT-SVD, xem chỉ số PSNR/SSIM/NC
4. **🧪 Robustness** — mô phỏng các phép tấn công lên ảnh để kiểm tra độ bền
5. **🔍 Steganalysis** — phân tích một ảnh để phát hiện khả năng có chứa tin giấu

---

## 📁 Cấu trúc thư mục

```
stego_watermark_project/
├── core/
│   ├── aes_cipher.py          # Mã hóa/giải mã AES
│   ├── lsb_stego.py           # Giấu tin bằng LSB
│   ├── pvd_stego.py           # Giấu tin bằng PVD
│   ├── dct_stego.py           # Giấu tin bằng DCT
│   ├── dwt_watermark.py       # Thủy vân số bằng DWT
│   └── dwt_svd_watermark.py   # Thủy vân số bằng DWT-SVD
├── evaluation/
│   ├── metrics.py             # Tính MSE, PSNR, SSIM, NC
│   ├── attacks.py             # Mô phỏng các phép tấn công
│   └── steganalysis.py        # Phân tích phát hiện giấu tin
├── app.py                     # Giao diện Streamlit chính
├── requirements.txt
└── README.md
```

---

## ⚙️ Cài đặt

### Yêu cầu
- Python 3.9+
- pip

### Các bước

```bash
# 1. Clone repository
git clone https://github.com/bonbidev/stego_watermark_project.git
cd stego_watermark_project

# 2. Tạo virtual environment (khuyến nghị)
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

# 3. Cài đặt thư viện
pip install -r requirements.txt
```

### Thư viện sử dụng

```
numpy
opencv-python
pywavelets
pycryptodome
scikit-image
matplotlib
streamlit
```

---

## 🚀 Chạy ứng dụng

```bash
streamlit run app.py
```

Sau khi chạy, trình duyệt sẽ tự mở tại `http://localhost:8501`.

---

## 📖 Hướng dẫn sử dụng nhanh

### Giấu tin mật vào ảnh
1. Vào tab **🔐 Giấu tin**
2. Tải lên ảnh gốc (PNG/JPG/JPEG/BMP)
3. Chọn thuật toán (LSB / PVD / DCT)
4. Nhập nội dung bí mật và mật khẩu AES
5. Nhấn **Giấu tin** → xem kết quả, chỉ số chất lượng (MSE/PSNR/SSIM) và tải ảnh về

### Trích xuất tin đã giấu
1. Vào tab **🔓 Trích xuất**
2. Tải lên ảnh đã giấu tin, chọn đúng thuật toán và nhập đúng mật khẩu đã dùng khi giấu tin
3. Nhấn **Trích xuất** để xem nội dung gốc

### Nhúng thủy vân số
1. Vào tab **©️ Thủy vân**
2. Tải ảnh gốc và ảnh watermark (logo)
3. Chọn thuật toán (DWT / DWT-SVD) và hệ số **Alpha** (cường độ nhúng, 0.01–0.20)
4. Nhấn **Nhúng watermark** → xem ảnh kết quả, watermark trích xuất lại và chỉ số NC

### Kiểm tra độ bền / Steganalysis
- Tab **🧪 Robustness**: chọn ảnh và loại tấn công (nén JPEG, nhiễu, làm mờ, resize, crop, xoay, làm nét...) để mô phỏng
- Tab **🔍 Steganalysis**: tải ảnh để phân tích chỉ số LSB Ratio và LSB Entropy, đánh giá khả năng chứa dữ liệu ẩn

---

## 🧮 Cơ sở lý thuyết (tóm tắt)

- **LSB**: thay đổi bit có trọng số thấp nhất của từng pixel — đơn giản, sức chứa lớn, nhưng dễ bị phát hiện/phá hủy khi nén ảnh
- **PVD**: dựa trên độ chênh lệch giá trị giữa các cặp pixel liền kề để quyết định số bit giấu — cân bằng giữa sức chứa và khả năng ẩn giấu
- **DCT**: biến đổi ảnh sang miền tần số (giống nguyên lý nén JPEG) rồi giấu tin vào các hệ số — bền hơn trước nén ảnh
- **DWT**: biến đổi wavelet rời rạc, nhúng watermark vào các hệ số tần số thấp/cao để tăng độ bền trước các phép biến đổi hình học và nhiễu
- **DWT-SVD**: kết hợp DWT với phân tích giá trị kỳ dị (SVD) — thường cho độ bền cao hơn DWT thuần

### Chỉ số đánh giá
- **MSE / PSNR**: đo mức độ sai khác giữa ảnh gốc và ảnh sau xử lý (PSNR càng cao → ảnh càng ít bị biến dạng)
- **SSIM**: đo độ tương đồng cấu trúc, gần với cảm nhận thị giác con người hơn PSNR
- **NC (Normalized Correlation)**: đo mức độ giống nhau giữa watermark gốc và watermark trích xuất được (giá trị càng gần 1 càng tốt)

---

## 🗺️ Lộ trình phát triển (Roadmap)

Dự án đang triển khai theo các hạng mục sau (xem chi tiết tại tab **Issues** của repo):

- [x] Cài đặt module mã hóa AES
- [x] Cài đặt module giấu tin LSB
- [x] Cài đặt module thủy vân DWT
- [x] Cài đặt module tính chỉ số đánh giá
- [x] Xây dựng pipeline thống nhất và giao diện demo (Streamlit)
- [x] Xây dựng module mô phỏng tấn công
- [ ] Thu thập bộ dataset chuẩn (ảnh vỏ + watermark logo)
- [ ] Hoàn thiện tài liệu toán học cho LSB, DWT và các công thức đánh giá
- [ ] Thiết kế sơ đồ kiến trúc luồng dữ liệu (pipeline diagram)
- [ ] Chạy thực nghiệm hàng loạt, xuất kết quả `.xlsx`
- [ ] Biên soạn báo cáo khoa học và slide thuyết trình

---

## ⚠️ Lưu ý

- Đây là dự án mang tính **học thuật/nghiên cứu**, phục vụ mục đích tìm hiểu kỹ thuật giấu tin và thủy vân số.
- Mật khẩu AES do người dùng nhập không được lưu trữ hay truyền đi đâu khác ngoài phiên làm việc hiện tại.
- Sức chứa và độ bền của từng thuật toán phụ thuộc nhiều vào kích thước ảnh, định dạng và mức độ nén.

---

## 📄 Giấy phép

Chưa xác định (đề xuất bổ sung file `LICENSE`, ví dụ MIT License, nếu dự định public/chia sẻ mã nguồn).

## 👤 Tác giả

[bonbidev](https://github.com/bonbidev)