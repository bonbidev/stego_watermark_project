import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2

class MetricsCalculator:
    """
    Tính toán các chỉ số đánh giá:
    - PSNR (Peak Signal-to-Noise Ratio): Đánh giá chất lượng ảnh chứa tin so với ảnh gốc.
    - SSIM (Structural Similarity Index): Đánh giá độ tương đồng cấu trúc.
    - BER (Bit Error Rate): Tỷ lệ lỗi bit của dữ liệu trích xuất.
    - NC (Normalized Correlation): Độ tương quan chuẩn hóa của logo trích xuất.
    """

    @staticmethod
    def calculate_psnr(original: np.ndarray, stego: np.ndarray) -> float:
        """Tính chỉ số PSNR (Đơn vị: dB). Giá trị càng cao ảnh càng giống gốc."""
        mse = np.mean((original.astype(np.float64) - stego.astype(np.float64)) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return float(psnr)

    @staticmethod
    def calculate_ssim(original: np.ndarray, stego: np.ndarray) -> float:
        """Tính chỉ số SSIM (Giá trị từ 0 đến 1, càng gần 1 càng tốt)."""
        # Chuyển về ảnh xám để tính SSIM chuẩn xác
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        stego_gray = cv2.cvtColor(stego, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(orig_gray, stego_gray, full=True)
        return float(score)

    @staticmethod
    def calculate_ber(original_bits: bytes, extracted_bits: bytes) -> float:
        """Tính tỷ lệ lỗi bit (Bit Error Rate)."""
        orig_arr = np.unpackbits(np.frombuffer(original_bits, dtype=np.uint8))
        ext_arr = np.unpackbits(np.frombuffer(extracted_bits, dtype=np.uint8))
        
        # Nếu chiều dài khác nhau, cắt bớt phần thừa để so sánh
        min_len = min(len(orig_arr), len(ext_arr))
        if min_len == 0:
            return 1.0
            
        errors = np.sum(orig_arr[:min_len] != ext_arr[:min_len])
        # Cộng thêm phần chiều dài chênh lệch vào số lỗi
        length_diff = abs(len(orig_arr) - len(ext_arr))
        total_errors = errors + (length_diff * 8)
        
        return float(total_errors / (len(orig_arr) * 8))

    @staticmethod
    def calculate_nc(orig_logo: np.ndarray, ext_logo: np.ndarray) -> float:
        """Tính độ tương quan chuẩn hóa (Normalized Correlation) cho logo."""
        orig_f = orig_logo.astype(np.float32).flatten()
        ext_f = ext_logo.astype(np.float32).flatten()
        
        # Chuẩn hóa về [0, 1]
        orig_f = orig_f / 255.0
        ext_f = ext_f / 255.0
        
        numerator = np.sum(orig_f * ext_f)
        denominator = np.sqrt(np.sum(orig_f ** 2)) * np.sqrt(np.sum(ext_f ** 2))
        
        if denominator == 0:
            return 0.0
        return float(numerator / denominator)

# --- KIỂM THỬ ---
if __name__ == "__main__":
    img1 = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    img2 = img1 + np.random.randint(-2, 3, (512, 512, 3), dtype=np.int8).astype(np.uint8)
    
    print(f"PSNR: {MetricsCalculator.calculate_psnr(img1, img2):.2f} dB")
    print(f"SSIM: {MetricsCalculator.calculate_ssim(img1, img2):.4f}")