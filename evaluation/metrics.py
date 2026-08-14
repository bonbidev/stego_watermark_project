import cv2
import numpy as np

class AttackSimulator:
    """
    Mô phỏng các hình thức tấn công phổ biến lên ảnh chứa thủy vân:
    - Nén JPEG (Lossy compression)
    - Thêm nhiễu Gaussian (Noise addition)
    - Làm mờ (Blurring)
    - Cắt xén ảnh (Cropping)
    """

    @staticmethod
    def jpeg_compression(image: np.ndarray, quality: int = 50) -> np.ndarray:
        """Mô phỏng nén JPEG với mức chất lượng (quality từ 1 đến 100)."""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_img = cv2.imencode('.jpg', image, encode_param)
        decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        return decoded_img

    @staticmethod
    def gaussian_noise(image: np.ndarray, mean: float = 0.0, var: float = 0.001) -> np.ndarray:
        """Thêm nhiễu Gauss vào ảnh."""
        img_float = image.astype(np.float32) / 255.0
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, image.shape)
        noisy = img_float + gauss
        noisy = np.clip(noisy * 255, 0, 255).astype(np.uint8)
        return noisy

    @staticmethod
    def median_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Làm mờ lọc trung vị (Median Blur) để phá hoại nhiễu LSB."""
        # Kernel size bắt buộc phải là số lẻ
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.medianBlur(image, kernel_size)

    @staticmethod
    def crop_image(image: np.ndarray, crop_percent: float = 0.1) -> np.ndarray:
        """Cắt xén một phần viền của ảnh."""
        h, w, _ = image.shape
        ch = int(h * crop_percent)
        cw = int(w * crop_percent)
        
        cropped = image.copy()
        # Đen hóa phần bị cắt hoặc thay thế bằng viền trắng
        cropped[0:ch, :] = 0
        cropped[h-ch:h, :] = 0
        cropped[:, 0:cw] = 0
        cropped[:, w-cw:w] = 0
        return cropped

# --- KIỂM THỬ ---
if __name__ == "__main__":
    img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    
    # Test nén JPEG
    img_compressed = AttackSimulator.jpeg_compression(img, quality=70)
    print("Mô phỏng nén JPEG thành công. Kích thước:", img_compressed.shape)