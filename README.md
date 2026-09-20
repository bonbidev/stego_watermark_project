# Hệ thống Giấu tin & Thủy vân số

Ứng dụng thực nghiệm các kỹ thuật **giấu tin mật (steganography)** và **thủy vân số (digital watermarking)** trên ảnh, kết hợp mã hóa **AES-256-GCM**, đánh giá chất lượng ảnh (PSNR/SSIM/NC), kiểm tra độ bền trước 9 kiểu tấn công phổ biến, và so sánh trực tiếp các thuật toán trên cùng dữ liệu đầu vào.

Xây dựng bằng Python + [Streamlit](https://streamlit.io/) — giao diện web tương tác, có biểu đồ Plotly, chế độ tối, xuất báo cáo Excel.

**Sinh viên:** Tuấn Kiệt, Thanh Ngôn — Đại học Tôn Đức Thắng, Khoa CNTT

---

## Tính năng chính

| Nhóm | Thuật toán / Chức năng |
|---|---|
| **Giấu tin (Steganography)** | LSB, PVD (Pixel Value Differencing), DCT (Discrete Cosine Transform) — cả 3 đều **giữ nguyên màu ảnh gốc** (chỉ ghi vào kênh Blue) |
| **Mã hóa** | AES-256-GCM (mã hóa có xác thực) với khóa dẫn xuất bằng PBKDF2-HMAC-SHA256, 600.000 vòng lặp |
| **Thủy vân số (Watermarking)** | DWT (Discrete Wavelet Transform), DWT-SVD (kết hợp Singular Value Decomposition) |
| **Đánh giá chất lượng ảnh** | MSE, PSNR, SSIM, NC (Normalized Correlation cho watermark trích xuất) |
| **Kiểm tra độ bền (Robustness)** | JPEG, Gaussian noise, Salt & Pepper, Gaussian blur, Median blur, Resize, Crop, Rotate, Sharpen (9 kiểu) |
| **So sánh thuật toán** | Chạy đồng thời nhiều ảnh, ma trận độ bền đầy đủ (thuật toán × tấn công), biểu đồ radar đa tiêu chí, xuất Excel |
| **Steganalysis** | LSB ratio/entropy, Chi-square test, histogram phân bố pixel/bit |
| **Giao diện** | Chế độ tối, thanh trượt so sánh ảnh trước/sau, nhận xét chi tiết tự động, hướng dẫn thao tác tích hợp từng tab |

---

## Giao diện — 7 tab

1. **Giấu tin** — mã hóa nội dung bằng AES rồi nhúng vào ảnh bằng LSB / PVD / DCT
2. **Trích xuất** — trích xuất và giải mã nội dung từ ảnh đã giấu tin
3. **Thủy vân** — nhúng watermark (ảnh logo) bằng DWT hoặc DWT-SVD, xem PSNR/SSIM/NC
4. **So sánh** — chạy tất cả thuật toán trên cùng ảnh đầu vào (nhiều ảnh cùng lúc), so sánh chất lượng/tốc độ/độ bền, xuất Excel
5. **Robustness** — mô phỏng 1 trong 9 kiểu tấn công lên ảnh để xem mức độ biến dạng
6. **Steganalysis** — phân tích một ảnh để tìm dấu hiệu có chứa tin giấu
7. **Hướng dẫn** — FAQ và giải thích từng tab ngay trong app

---

## Cấu trúc thư mục

```
stego_watermark_project/
├── core/
│   ├── aes_cipher.py           # AES-256-GCM + PBKDF2-HMAC-SHA256
│   ├── lsb_stego.py            # Giấu tin LSB (kênh Blue)
│   ├── pvd_stego.py            # Giấu tin PVD (giữ màu ảnh)
│   ├── dct_stego.py            # Giấu tin DCT — 1 bit/khối 8x8 (giữ màu ảnh)
│   ├── dwt_watermark.py        # Thủy vân DWT
│   ├── dwt_svd_watermark.py    # Thủy vân DWT-SVD
│   └── pipeline.py             # Pipeline AES→LSB→DWT gộp (không dùng bởi app.py — xem Hạn chế)
├── evaluation/
│   ├── metrics.py              # MSE, PSNR, SSIM, NC
│   ├── attacks.py              # 9 kiểu tấn công mô phỏng
│   ├── steganalysis.py         # LSB ratio/entropy, Chi-square test
│   └── benchmark.py            # Chạy & so sánh hàng loạt thuật toán
├── app.py                      # Giao diện Streamlit (7 tab)
├── config.py                   # Cấu hình tập trung (mật khẩu, alpha, tham số tấn công...)
├── example_usage.py            # 8 ví dụ chạy trực tiếp bằng Python (không cần UI)
├── requirements.txt
├── cleanup_repo.sh             # Script gỡ .venv/ khỏi git tracking
├── .gitignore
└── README.md
```

---

## Cài đặt

### Yêu cầu
- Python 3.9+
- pip

### Các bước

```bash
git clone https://github.com/bonbidev/stego_watermark_project.git
cd stego_watermark_project

python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Thư viện sử dụng

```
numpy            opencv-python      pywavelets
cryptography     scipy              pandas
scikit-image     streamlit          plotly
openpyxl
```

> `requirements.txt` đã được rà soát khớp đúng với import thật trong code (không còn gói thừa `pycryptodome`/`matplotlib`, không còn thiếu `cryptography`/`scipy`/`pandas` như bản cũ).

---

## Chạy ứng dụng

```bash
streamlit run app.py
```

Trình duyệt tự mở tại `http://localhost:8501`.

Kiểm tra nhanh không cần mở giao diện web:

```bash
python example_usage.py
```

Nếu thấy dòng cuối **`ALL EXAMPLES COMPLETED SUCCESSFULLY`** là môi trường đã cài đặt đúng.

---

## Hướng dẫn sử dụng nhanh

### Giấu tin mật vào ảnh
1. Tab **Giấu tin** → upload ảnh gốc (PNG khuyến nghị)
2. Chọn thuật toán (LSB / PVD / DCT) — xem thông tin tốc độ/sức chứa/độ bền ngay bên cạnh
3. Nhập nội dung bí mật và mật khẩu AES (nên ≥ 12 ký tự)
4. Nhấn **Giấu tin** → xem kết quả, nhận xét chi tiết, bản đồ sai khác, tải ảnh PNG về

### Trích xuất tin đã giấu
1. Tab **Trích xuất** → upload đúng ảnh đã giấu tin, chọn đúng thuật toán, nhập đúng mật khẩu
2. Nhấn **Trích xuất** để xem lại nội dung gốc

### Nhúng thủy vân số
1. Tab **Thủy vân** → upload ảnh gốc + ảnh watermark (logo nhỏ)
2. Chọn thuật toán (DWT / DWT-SVD) và **Alpha** (0.01–0.20, càng lớn càng bền nhưng ảnh càng biến dạng)
3. Nhấn **Nhúng watermark** → xem watermark trích xuất lại và chỉ số NC

### So sánh & kiểm tra độ bền
- Tab **So sánh**: upload nhiều ảnh cùng lúc, chọn mức kiểm tra độ bền (Nhanh/Đầy đủ), xem bảng + biểu đồ radar + xuất Excel
- Tab **Robustness**: mô phỏng 1 kiểu tấn công cụ thể để xem mức biến dạng chi tiết
- Tab **Steganalysis**: phân tích 1 ảnh bất kỳ để tìm dấu hiệu bất thường trong mặt phẳng LSB

---

## Cơ sở lý thuyết (tóm tắt)

| Thuật toán | Nguyên lý | Sức chứa | Độ bền thực đo |
|---|---|---|---|
| **LSB** | Ghi vào bit thấp nhất kênh Blue | Rất lớn (~1 bit/byte ảnh) | Rất thấp — vỡ ngay cả khi nén nhẹ |
| **PVD** | Số bit nhúng theo độ tương phản cục bộ (cạnh nhúng nhiều hơn vùng phẳng) | Lớn | Thấp — tương tự LSB |
| **DCT** | Biến đổi từng khối 8×8 sang miền tần số, mã hoá 1 bit/khối bằng 2 hệ số | Thấp (1 bit/khối) | Thấp hơn kỳ vọng lý thuyết — xem mục Hạn chế |
| **DWT** | Nhúng vào hệ số subband LH của biến đổi wavelet | Trung bình | **Có bền thực đo**: NC ≈ 0.83 ở JPEG q95, ≈ 0.65 ở q75 |
| **DWT-SVD** | DWT + phân tích giá trị kỳ dị (SVD) | Trung bình | Thấp trong bản cài đặt hiện tại — xem Hạn chế |

### Chỉ số đánh giá
- **PSNR** (dB): sai khác giữa ảnh gốc và ảnh sau xử lý — ≥ 40 dB thường coi là mắt thường khó phân biệt
- **SSIM** (0–1): độ tương đồng cấu trúc, gần cảm nhận thị giác hơn PSNR
- **NC** (0–1): độ giống giữa watermark gốc và watermark trích xuất — ≥ 0.9 là rất tốt

---

## Hạn chế đã biết (dựa trên thực nghiệm, không phải suy đoán)

Các mục dưới đây đã được đo đạc trực tiếp trong quá trình phát triển, không phải nhận định chủ quan:

1. **LSB, PVD, DCT không sống sót qua nén JPEG**, kể cả ở chất lượng rất cao (q=95), kể cả sau khi tăng độ mạnh mã hoá DCT lên gấp 10 lần. Đây là giới hạn của kiểu mã hoá 1 bit đơn lẻ không có mã sửa lỗi, không phải lỗi cài đặt. Ba thuật toán này chỉ đáng tin khi **file ảnh được giữ nguyên y hệt** (gửi trực tiếp, không qua nền tảng tự nén lại).
2. **DWT-SVD cho NC thấp hơn DWT** trong bản cài đặt hiện tại, vì bước trích xuất chỉ khôi phục đường chéo ma trận giá trị kỳ dị (S), bỏ qua U và V. Muốn cải thiện cần nhúng/trích đầy đủ cả 3 thành phần SVD.
3. **`core/pipeline.py`** (gộp AES→LSB→DWT) hiện **không được `app.py` sử dụng**, vì thứ tự bước sẽ khiến DWT phá huỷ dữ liệu LSB đã nhúng trước đó (DWT/IDWT là biến đổi số thực, thay đổi toàn bộ giá trị pixel chứ không chỉ bit thấp nhất). Nếu cần dùng module này, phải sửa lại thứ tự hoặc tách 2 bước riêng.

---

## Bảo mật

- Mã hoá bằng AES-256-GCM (mã hoá **có xác thực** — nếu ảnh bị sửa đổi hoặc sai mật khẩu, hệ thống báo lỗi rõ ràng thay vì trả về dữ liệu rác).
- Khóa dẫn xuất từ mật khẩu bằng PBKDF2-HMAC-SHA256 với 600.000 vòng lặp.
- Mật khẩu không được lưu trữ hay gửi đi đâu ngoài phiên làm việc hiện tại của trình duyệt.
- Nên dùng mật khẩu ≥ 12 ký tự (app có cảnh báo nếu ngắn hơn).

---

## Giấy phép

Dự án học thuật — phục vụ mục đích nghiên cứu và giảng dạy tại Đại học Tôn Đức Thắng.

## Tác giả

Tuấn Kiệt, Thanh Ngôn — [github.com/bonbidev](https://github.com/bonbidev)