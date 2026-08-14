import cv2
import numpy as np
from .aes_cipher import AESCipher
from .lsb_stego import LSBSteg
from .dwt_watermark import DWTWatermark

class StegoPipeline:
    """
    Pipeline tích hợp: AES -> LSB -> DWT
    Quy trình: 
    1. Mã hóa văn bản bằng AES.
    2. Nhúng bản mã vào ảnh bằng LSB.
    3. Nhúng logo bản quyền vào ảnh bằng DWT.
    """
    
    def __init__(self, password: str, alpha: float = 0.1):
        self.aes = AESCipher(password=password)
        self.lsb = LSBSteg(channel=0) # Nhúng LSB vào kênh Blue
        self.dwt = DWTWatermark(alpha=alpha)

    def run_embed(self, image_path: str, secret_text: str, watermark_path: str, output_path: str):
        # 1. Đọc ảnh và logo
        img = cv2.imread(image_path)
        wm = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
        
        # 2. Mã hóa văn bản
        encrypted_data = self.aes.encrypt(secret_text)
        
        # 3. Nhúng LSB (Thông tin bí mật)
        stego_lsb = self.lsb.embed(img, encrypted_data)
        
        # 4. Nhúng DWT (Logo bản quyền)
        final_img = self.dwt.embed(stego_lsb, wm)
        
        # 5. Lưu ảnh kết quả
        cv2.imwrite(output_path, final_img)
        print(f"Đã lưu ảnh thành công tại: {output_path}")
        return final_img

    def run_extract(self, stego_image_path: str, original_image_path: str, password: str):
        # 1. Đọc ảnh
        stego_img = cv2.imread(stego_image_path)
        orig_img = cv2.imread(original_image_path)
        
        # 2. Trích xuất LSB (Lấy bản mã)
        encrypted_data = self.lsb.extract(stego_img)
        
        # 3. Giải mã AES
        aes = AESCipher(password=password)
        secret_text = aes.decrypt(encrypted_data)
        
        return secret_text

# --- KIỂM THỬ PIPELINE ---
if __name__ == "__main__":
    print("--- CHẠY PIPELINE TÍCH HỢP ---")
    # Lưu ý: Cần có file ảnh test trong folder assets
    # pipeline = StegoPipeline(password="Nhom16_Secret")
    # pipeline.run_embed("assets/cover_images/test.png", "Bi mat", "assets/watermarks/logo.png", "output.png")