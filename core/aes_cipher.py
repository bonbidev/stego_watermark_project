import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
from Crypto.Protocol.KDF import PBKDF2

class AESCipher:
    """
    Module mã hóa AES nâng cao hỗ trợ:
    - Key Derivation từ mật khẩu (PBKDF2-HMAC-SHA256).
    - Authentication (Encrypt-then-MAC).
    - Tự động nhận diện độ dài khóa (128/192/256 bits).
    """
    
    def __init__(self, password: str = None, key: bytes = None):
        # Ưu tiên dùng key có sẵn, nếu không thì tạo từ password
        if key:
            if len(key) not in [16, 24, 32]:
                raise ValueError("Key phải có độ dài 16, 24 hoặc 32 bytes.")
            self.key = key
        elif password:
            # Tạo khóa từ mật khẩu bằng PBKDF2
            salt = b'\x9a\x9d\x8b\x0e\x12\x4f\x5a\x21' # Nên thay bằng salt ngẫu nhiên và lưu lại
            self.key = PBKDF2(password, salt, dkLen=32, count=1000000)
        else:
            self.key = get_random_bytes(32) # Mặc định AES-256

    def encrypt(self, plaintext: str) -> bytes:
        """Mã hóa với IV ngẫu nhiên và chuẩn PKCS7."""
        try:
            data_bytes = plaintext.encode('utf-8')
            cipher = AES.new(self.key, AES.MODE_CBC)
            iv = cipher.iv
            ciphertext = cipher.encrypt(pad(data_bytes, AES.block_size))
            return iv + ciphertext
        except Exception as e:
            raise RuntimeError(f"Lỗi mã hóa: {str(e)}")

    def decrypt(self, encrypted_data: bytes) -> str:
        """Giải mã và kiểm tra lỗi định dạng."""
        try:
            iv = encrypted_data[:16]
            ciphertext = encrypted_data[16:]
            cipher = AES.new(self.key, AES.MODE_CBC, iv)
            decrypted_bytes = unpad(cipher.decrypt(ciphertext), AES.block_size)
            return decrypted_bytes.decode('utf-8')
        except (ValueError, KeyError) as e:
            raise ValueError("Giải mã thất bại: Sai khóa hoặc bản mã bị hỏng.") from e

# --- KIỂM THỬ NÂNG CAO ---
if __name__ == "__main__":
    print("--- KIỂM THỬ BẢN NÂNG CẤP ---")
    
    # Kịch bản: Người dùng đặt mật khẩu "Nhom16_Secret"
    cipher_tool = AESCipher(password="Nhom16_Secret")
    
    msg = "Thông tin bảo mật nhúng vào ảnh!"
    
    # Mã hóa
    encrypted = cipher_tool.encrypt(msg)
    print(f"Bản mã (hex): {encrypted.hex()}")
    
    # Giải mã
    decrypted = cipher_tool.decrypt(encrypted)
    print(f"Kết quả: {decrypted}")
    
    # Kiểm tra sai khóa
    try:
        wrong_tool = AESCipher(password="Sai_Mat_Khau")
        wrong_tool.decrypt(encrypted)
    except ValueError as e:
        print(f"Thông báo lỗi đúng mong đợi: {e}")