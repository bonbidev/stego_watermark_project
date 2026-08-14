import numpy as np
import cv2
import logging

# Cấu hình logging để tiện theo dõi lỗi khi làm nhóm
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSBSteg:
    """
    Module LSB Steganography nâng cao:
    - Hỗ trợ chọn kênh màu để nhúng (R, G, hoặc B).
    - Tự động kiểm tra giới hạn dung lượng dữ liệu.
    - Xử lý header 32-bit cho độ dài bản tin.
    """
    
    def __init__(self, channel: int = 0):
        """
        :param channel: Kênh màu sẽ nhúng (0: Blue, 1: Green, 2: Red trong OpenCV)
        """
        self.channel = channel

    def _data_to_bits(self, data: bytes) -> np.ndarray:
        """Chuyển dữ liệu bytes thành mảng bit."""
        # Thêm 4 bytes đầu lưu độ dài (Metadata)
        data_len = np.array([len(data)], dtype=np.uint32).view(np.uint8)
        full_data = np.concatenate([data_len, np.frombuffer(data, dtype=np.uint8)])
        return np.unpackbits(full_data)

    def embed(self, image: np.ndarray, secret_data: bytes) -> np.ndarray:
        """Nhúng dữ liệu vào ảnh."""
        if image is None:
            raise ValueError("Ảnh không hợp lệ.")
            
        bits = self._data_to_bits(secret_data)
        
        # Kiểm tra dung lượng ảnh (mỗi pixel nhúng 1 bit)
        max_bits = image.size // 3 # Số pixel * 1 kênh
        if len(bits) > max_bits:
            raise ValueError(f"Dữ liệu quá lớn! Tối đa: {max_bits//8} bytes.")

        stego_image = image.copy()
        
        # Chỉ nhúng vào kênh màu đã chọn
        # Duyệt qua các pixel và thay đổi bit cuối
        height, width, _ = stego_image.shape
        bit_idx = 0
        
        for y in range(height):
            for x in range(width):
                if bit_idx < len(bits):
                    # LSB nhúng: Lấy bit cuối của giá trị pixel, xóa nó và đặt bit mới vào
                    val = int(stego_image[y, x, self.channel])
                    stego_image[y, x, self.channel] = (val & ~1) | bits[bit_idx]
                    bit_idx += 1
                else:
                    return stego_image
        
        logger.info("Nhúng dữ liệu hoàn tất.")
        return stego_image

    def extract(self, stego_image: np.ndarray) -> bytes:
        """Trích xuất dữ liệu từ ảnh."""
        # 1. Lấy 32 bit đầu tiên để đọc độ dài
        len_bits = np.zeros(32, dtype=np.uint8)
        
        height, width, _ = stego_image.shape
        idx = 0
        for y in range(height):
            for x in range(width):
                if idx < 32:
                    len_bits[idx] = stego_image[y, x, self.channel] & 1
                    idx += 1
                else: break
        
        data_len = np.packbits(len_bits).view(np.uint32)[0]
        
        # 2. Lấy dữ liệu thực tế dựa trên độ dài
        data_bits = np.zeros(data_len * 8, dtype=np.uint8)
        idx = 0
        for y in range(height):
            for x in range(width):
                if idx < (data_len * 8):
                    # Bắt đầu đọc từ vị trí 32
                    if (y * width + x) >= 32:
                        data_bits[idx] = stego_image[y, x, self.channel] & 1
                        idx += 1
                else: break
        
        return np.packbits(data_bits).tobytes()

# --- DEMO KIỂM CHỨNG ---
if __name__ == "__main__":
    # Test với một ảnh thật
    img = np.zeros((512, 512, 3), dtype=np.uint8) # Ảnh đen 512x512
    data = b"Bao mat thong tin - Nhom 16"
    
    lsb = LSBSteg(channel=0) # Nhúng vào kênh Blue
    try:
        encoded = lsb.embed(img, data)
        decoded = lsb.extract(encoded)
        print(f"Dữ liệu trích xuất: {decoded.decode('utf-8')}")
        assert data == decoded
    except Exception as e:
        print(f"Lỗi: {e}")