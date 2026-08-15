from __future__ import annotations
import argparse
import base64
import getpass
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA256
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

__all__ = [
    "AESMode",
    "AESConfig",
    "EncryptionResult",
    "DecryptionResult",
    "AESCipherError",
    "InvalidKeyError",
    "DecryptionError",
    "InvalidTokenError",
    "AESCipher",
    "bytes_to_bitstream",
    "bitstream_to_bytes",
]
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


class AESCipherError(Exception):
    pass


class InvalidKeyError(AESCipherError):
    pass


class DecryptionError(AESCipherError):
    pass


class InvalidTokenError(AESCipherError):
    pass


class AESMode(str, Enum):
    GCM = "GCM"
    CBC = "CBC"


class _KeySource(str, Enum):
    PASSWORD = "PASSWORD"
    RAW_KEY = "RAW_KEY"


_MAGIC = b"AC1"
_MODE_BYTE = {AESMode.CBC: 0, AESMode.GCM: 1}
_BYTE_MODE = {v: k for k, v in _MODE_BYTE.items()}
_KEYSRC_BYTE = {_KeySource.PASSWORD: 0, _KeySource.RAW_KEY: 1}
_BYTE_KEYSRC = {v: k for k, v in _KEYSRC_BYTE.items()}
_SALT_SIZE = 16
_CBC_IV_SIZE = 16
_GCM_NONCE_SIZE = 12
_GCM_TAG_SIZE = 16
_AES_256_KEY_SIZE = 32


@dataclass
class AESConfig:
    mode: AESMode = AESMode.GCM
    key_size: int = _AES_256_KEY_SIZE
    pbkdf2_iterations: int = 200000

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            self.mode = AESMode(self.mode.upper())
        if self.key_size not in (16, 24, 32):
            raise InvalidKeyError(
                "key_size phải là 16 (AES-128), 24 (AES-192) hoặc 32 (AES-256) byte."
            )
        if self.pbkdf2_iterations < 100000:
            logger.warning(
                "pbkdf2_iterations=%d thấp hơn khuyến nghị OWASP tối thiểu (100,000 vòng cho PBKDF2-HMAC-SHA256). Cân nhắc tăng lên để đảm bảo an toàn trước tấn công brute-force.",
                self.pbkdf2_iterations,
            )


@dataclass
class EncryptionResult:
    token: bytes
    plaintext_size_bytes: int
    ciphertext_size_bytes: int
    elapsed_ms: float
    mode: AESMode
    key_size_bits: int

    def to_base64(self) -> str:
        return base64.b64encode(self.token).decode("ascii")


@dataclass
class DecryptionResult:
    plaintext: str
    elapsed_ms: float
    mode: AESMode


class AESCipher:

    def __init__(
        self,
        password: Optional[str] = None,
        key: Optional[bytes] = None,
        config: Optional[AESConfig] = None,
    ) -> None:
        if (password is None) == (key is None):
            raise InvalidKeyError(
                "Phải cung cấp CHÍNH XÁC MỘT trong hai tham số: `password` (mã hóa dựa trên mật khẩu, dùng PBKDF2) hoặc `key` (dùng thẳng khóa AES thô). Không được cung cấp cả hai hoặc không cung cấp gì."
            )
        self.config = config or AESConfig()
        self._password = password
        self._raw_key = key
        if self._raw_key is not None and len(self._raw_key) != self.config.key_size:
            raise InvalidKeyError(
                f"Khóa AES thô phải dài đúng {self.config.key_size} byte (config.key_size), nhưng nhận được {len(self._raw_key)} byte."
            )
        logger.debug(
            "Khởi tạo AESCipher: mode=%s, key_size=%d bytes, key_source=%s",
            self.config.mode.value,
            self.config.key_size,
            "RAW_KEY" if self._raw_key is not None else "PASSWORD",
        )

    @staticmethod
    def generate_key(key_size: int = _AES_256_KEY_SIZE) -> bytes:
        if key_size not in (16, 24, 32):
            raise InvalidKeyError("key_size phải là 16, 24 hoặc 32 byte.")
        return get_random_bytes(key_size)

    def _derive_key(self, salt: bytes) -> bytes:
        if self._password is None:
            raise InvalidKeyError(
                "Không thể suy khóa: đối tượng này được khởi tạo bằng raw key."
            )
        return PBKDF2(
            password=self._password,
            salt=salt,
            dkLen=self.config.key_size,
            count=self.config.pbkdf2_iterations,
            hmac_hash_module=SHA256,
        )

    def _resolve_key(self, salt: Optional[bytes]) -> bytes:
        if self._raw_key is not None:
            return self._raw_key
        assert salt is not None, "Chế độ PASSWORD bắt buộc phải có salt."
        return self._derive_key(salt)

    @property
    def _key_source(self) -> _KeySource:
        return _KeySource.RAW_KEY if self._raw_key is not None else _KeySource.PASSWORD

    def encrypt(self, plaintext: Union[str, bytes]) -> EncryptionResult:
        start = time.perf_counter()
        data = plaintext.encode("utf-8") if isinstance(plaintext, str) else plaintext
        plaintext_size = len(data)
        salt = b""
        if self._key_source == _KeySource.PASSWORD:
            salt = get_random_bytes(_SALT_SIZE)
        key = self._resolve_key(salt if salt else None)
        if self.config.mode == AESMode.GCM:
            nonce = get_random_bytes(_GCM_NONCE_SIZE)
            cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
            ciphertext, tag = cipher.encrypt_and_digest(data)
            iv_or_nonce = nonce
        else:
            iv_or_nonce = get_random_bytes(_CBC_IV_SIZE)
            cipher = AES.new(key, AES.MODE_CBC, iv=iv_or_nonce)
            padded = pad(data, AES.block_size)
            ciphertext = cipher.encrypt(padded)
            tag = b""
        header = bytearray()
        header += _MAGIC
        header.append(1)
        header.append(_MODE_BYTE[self.config.mode])
        header.append(_KEYSRC_BYTE[self._key_source])
        header += salt
        header += iv_or_nonce
        header += tag
        token = bytes(header) + ciphertext
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "Đã mã hóa %d byte plaintext -> token %d byte (mode=%s, %.2f ms).",
            plaintext_size,
            len(token),
            self.config.mode.value,
            elapsed_ms,
        )
        return EncryptionResult(
            token=token,
            plaintext_size_bytes=plaintext_size,
            ciphertext_size_bytes=len(ciphertext),
            elapsed_ms=elapsed_ms,
            mode=self.config.mode,
            key_size_bits=self.config.key_size * 8,
        )

    def decrypt(self, token: bytes) -> str:
        result = self._decrypt_full(token)
        return result.plaintext

    def decrypt_verbose(self, token: bytes) -> DecryptionResult:
        return self._decrypt_full(token)

    def _decrypt_full(self, token: bytes) -> DecryptionResult:
        start = time.perf_counter()
        if len(token) < len(_MAGIC) + 3:
            raise InvalidTokenError("Token quá ngắn, không đủ header hợp lệ.")
        cursor = 0
        magic = token[cursor : cursor + len(_MAGIC)]
        cursor += len(_MAGIC)
        if magic != _MAGIC:
            raise InvalidTokenError(
                f"Sai MAGIC header (nhận {magic!r}, kỳ vọng {_MAGIC!r}). Token này có thể không phải do AESCipher tạo ra, hoặc đã bị hỏng."
            )
        version = token[cursor]
        cursor += 1
        if version != 1:
            raise InvalidTokenError(
                f"Phiên bản wire format không được hỗ trợ: {version}."
            )
        mode_byte = token[cursor]
        cursor += 1
        if mode_byte not in _BYTE_MODE:
            raise InvalidTokenError(f"Byte mode không hợp lệ: {mode_byte}.")
        mode = _BYTE_MODE[mode_byte]
        keysrc_byte = token[cursor]
        cursor += 1
        if keysrc_byte not in _BYTE_KEYSRC:
            raise InvalidTokenError(f"Byte key_source không hợp lệ: {keysrc_byte}.")
        key_source = _BYTE_KEYSRC[keysrc_byte]
        if key_source != self._key_source:
            raise InvalidKeyError(
                f"Token được mã hóa bằng key_source={key_source.value}, nhưng đối tượng AESCipher hiện tại được khởi tạo với key_source={self._key_source.value}. Hãy dùng đúng cách khởi tạo tương ứng (password vs raw key)."
            )
        salt = b""
        if key_source == _KeySource.PASSWORD:
            salt = token[cursor : cursor + _SALT_SIZE]
            cursor += _SALT_SIZE
            if len(salt) != _SALT_SIZE:
                raise InvalidTokenError("Token thiếu trường SALT hoặc bị cắt xén.")
        if mode == AESMode.GCM:
            nonce = token[cursor : cursor + _GCM_NONCE_SIZE]
            cursor += _GCM_NONCE_SIZE
            tag = token[cursor : cursor + _GCM_TAG_SIZE]
            cursor += _GCM_TAG_SIZE
            if len(nonce) != _GCM_NONCE_SIZE or len(tag) != _GCM_TAG_SIZE:
                raise InvalidTokenError("Token thiếu trường NONCE/TAG hoặc bị cắt xén.")
        else:
            iv = token[cursor : cursor + _CBC_IV_SIZE]
            cursor += _CBC_IV_SIZE
            if len(iv) != _CBC_IV_SIZE:
                raise InvalidTokenError("Token thiếu trường IV hoặc bị cắt xén.")
        ciphertext = token[cursor:]
        if len(ciphertext) == 0:
            raise InvalidTokenError("Token không chứa ciphertext.")
        key = self._resolve_key(salt if salt else None)
        try:
            if mode == AESMode.GCM:
                cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
                plaintext_bytes = cipher.decrypt_and_verify(ciphertext, tag)
            else:
                cipher = AES.new(key, AES.MODE_CBC, iv=iv)
                padded = cipher.decrypt(ciphertext)
                plaintext_bytes = unpad(padded, AES.block_size)
        except (ValueError, KeyError) as exc:
            raise DecryptionError(
                f"Giải mã thất bại: sai mật khẩu/khóa, hoặc dữ liệu đã bị thay đổi (GCM tag không khớp / PKCS7 padding không hợp lệ). Chi tiết kỹ thuật: {exc}"
            ) from exc
        try:
            plaintext_str = plaintext_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DecryptionError(
                "Giải mã ra dữ liệu nhị phân hợp lệ nhưng không phải chuỗi UTF-8 hợp lệ - có thể sai khóa nhưng 'may mắn' vượt qua kiểm tra toàn vẹn (chỉ xảy ra với CBC không có MAC), hoặc plaintext gốc không phải văn bản thuần túy."
            ) from exc
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "Đã giải mã thành công token %d byte -> plaintext %d ký tự (mode=%s, %.2f ms).",
            len(token),
            len(plaintext_str),
            mode.value,
            elapsed_ms,
        )
        return DecryptionResult(
            plaintext=plaintext_str, elapsed_ms=elapsed_ms, mode=mode
        )

    def encrypt_to_bitstream(self, plaintext: Union[str, bytes]) -> str:
        result = self.encrypt(plaintext)
        return bytes_to_bitstream(result.token)

    def decrypt_from_bitstream(self, bitstream: str) -> str:
        token = bitstream_to_bytes(bitstream)
        return self.decrypt(token)


def bytes_to_bitstream(data: bytes) -> str:
    return "".join((format(byte, "08b") for byte in data))


def bitstream_to_bytes(bitstream: str) -> bytes:
    if len(bitstream) % 8 != 0:
        raise InvalidTokenError(
            f"Độ dài bitstream ({len(bitstream)}) không phải bội số của 8 - dữ liệu trích xuất LSB có thể đã bị cắt xén hoặc sai vị trí bắt đầu."
        )
    if any((ch not in "01" for ch in bitstream)):
        raise InvalidTokenError("Bitstream chứa ký tự khác '0'/'1'.")
    return bytes((int(bitstream[i : i + 8], 2) for i in range(0, len(bitstream), 8)))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mã hóa / Giải mã AES cho pipeline Encrypt-then-Hide (core/aes_cipher.py)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    p_enc = subparsers.add_parser("encrypt", help="Mã hóa một thông điệp văn bản.")
    p_enc.add_argument("--message", required=True, help="Thông điệp cần mã hóa.")
    p_enc.add_argument("--mode", default="GCM", choices=[m.value for m in AESMode])
    p_enc.add_argument("--key-size", type=int, default=32, choices=[16, 24, 32])
    p_enc.add_argument(
        "--output",
        required=False,
        help="Lưu token (Base64) ra file, nếu không in ra màn hình.",
    )
    p_dec = subparsers.add_parser("decrypt", help="Giải mã một token Base64.")
    p_dec.add_argument(
        "--token-base64",
        required=False,
        help="Token dạng Base64 (nếu không, đọc từ --input).",
    )
    p_dec.add_argument("--input", required=False, help="Đọc token Base64 từ file.")
    return parser


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = _build_arg_parser().parse_args()
    password = getpass.getpass("Nhập mật khẩu AES: ")
    if args.command == "encrypt":
        config = AESConfig(mode=AESMode(args.mode), key_size=args.key_size)
        cipher = AESCipher(password=password, config=config)
        result = cipher.encrypt(args.message)
        token_b64 = result.to_base64()
        logger.info(
            "Mã hóa thành công: %d byte plaintext -> %d byte token (%.2f ms).",
            result.plaintext_size_bytes,
            len(result.token),
            result.elapsed_ms,
        )
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(token_b64)
            logger.info("Đã lưu token Base64 tại: %s", args.output)
        else:
            print(token_b64)
    elif args.command == "decrypt":
        if args.token_base64:
            token_b64 = args.token_base64
        elif args.input:
            with open(args.input, "r", encoding="utf-8") as f:
                token_b64 = f.read().strip()
        else:
            raise SystemExit("Cần cung cấp --token-base64 hoặc --input.")
        token = base64.b64decode(token_b64)
        cipher = AESCipher(password=password)
        plaintext = cipher.decrypt(token)
        print(plaintext)


if __name__ == "__main__":
    _main()