import numpy as np
import pywt
import cv2

class DWTWatermark:
    """
    Module Thủy vân số (Watermarking) sử dụng DWT 1-level.
    - Nhúng logo nhị phân vào các hệ số tần số cao (LH subband).
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        :param alpha: Hệ số nhúng (càng lớn càng bền nhưng ảnh càng mờ).
        """
        self.alpha = alpha

    def embed(self, image: np.ndarray, watermark: np.ndarray) -> np.ndarray:
        """
        Nhúng logo (watermark) vào ảnh.
        Yêu cầu: watermark là ảnh nhị phân (đen trắng).
        """
        # Chuyển ảnh sang float32 để tính toán DWT
        img_float = np.float32(image) / 255.0
        
        # Chỉ xử lý trên một kênh màu (mặc định kênh 0 - Blue)
        channel = img_float[:, :, 0]
        
        # Thực hiện biến đổi DWT
        coeffs = pywt.dwt2(channel, 'haar')
        LL, (LH, HL, HH) = coeffs
        
        # Resize watermark cho khớp với kích thước subband LH
        wm_resized = cv2.resize(watermark, (LH.shape[1], LH.shape[0]))
        wm_binary = (wm_resized > 128).astype(np.float32)
        
        # Nhúng watermark vào hệ số LH
        LH_embedded = LH + self.alpha * wm_binary
        
        # Biến đổi ngược (IDWT)
        coeffs_embedded = (LL, (LH_embedded, HL, HH))
        watermarked_channel = pywt.idwt2(coeffs_embedded, 'haar')
        
        # Đưa về dạng uint8
        stego_img = image.copy()
        stego_img[:, :, 0] = np.clip(watermarked_channel * 255, 0, 255).astype(np.uint8)
        
        return stego_img

    def extract(self, stego_image: np.ndarray, original_image: np.ndarray) -> np.ndarray:
        """
        Trích xuất logo từ ảnh đã nhúng bằng cách trừ ảnh gốc.
        """
        stego_float = np.float32(stego_image) / 255.0
        orig_float = np.float32(original_image) / 255.0
        
        # DWT cho cả 2 ảnh
        _, (LH_stego, _, _) = pywt.dwt2(stego_float[:, :, 0], 'haar')
        _, (LH_orig, _, _) = pywt.dwt2(orig_float[:, :, 0], 'haar')
        
        # Trích xuất dựa trên sự khác biệt
        wm_extracted = (LH_stego - LH_orig) / self.alpha
        
        # Chuyển về dạng ảnh nhị phân
        wm_binary = (wm_extracted > 0.5).astype(np.uint8) * 255
        return wm_binary

# --- KIỂM THỬ ---
if __name__ == "__main__":
    # Tạo ảnh dummy 512x512
    dummy_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    dummy_wm = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(dummy_wm, (64, 64), 30, 255, -1) # Vẽ logo hình tròn
    
    dwt = DWTWatermark(alpha=0.2)
    
    # Nhúng
    encoded = dwt.embed(dummy_img, dummy_wm)
    # Trích xuất
    extracted = dwt.extract(encoded, dummy_img)
    
    print(">>> DWT Watermarking chạy thành công!")