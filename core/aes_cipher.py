"""
AES-256-GCM encryption/decryption for secret messages.
"""

import base64
import json
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


KEY_SIZE = 32
SALT_SIZE = 16
NONCE_SIZE = 12
PBKDF2_ITERATIONS = 600_000
PAYLOAD_VERSION = 1


class AESCipherError(Exception):
    """Base exception for AES errors."""


class InvalidPasswordError(AESCipherError):
    """Raised when the password is incorrect or data is corrupted."""


class InvalidPayloadError(AESCipherError):
    """Raised when the encrypted payload is invalid."""


class AESCipher:
    """AES-256-GCM encryption/decryption using a password."""

    def __init__(self, password: str):
        if not isinstance(password, str) or not password:
            raise ValueError("Password cannot be empty.")

        self.password = password

    def _derive_key(self, salt: bytes) -> bytes:
        """Derive a 256-bit AES key using PBKDF2-HMAC-SHA256."""

        if len(salt) != SALT_SIZE:
            raise ValueError("Invalid salt length.")

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=KEY_SIZE,
            salt=salt,
            iterations=PBKDF2_ITERATIONS,
        )

        return kdf.derive(self.password.encode("utf-8"))

    def encrypt(self, plaintext: str) -> str:
        """Encrypt plaintext and return a Base64 payload."""

        if not isinstance(plaintext, str):
            raise ValueError("Plaintext must be a string.")

        # Random salt and nonce ensure different ciphertext each time.
        salt = os.urandom(SALT_SIZE)
        nonce = os.urandom(NONCE_SIZE)

        key = self._derive_key(salt)
        aesgcm = AESGCM(key)

        ciphertext = aesgcm.encrypt(
            nonce,
            plaintext.encode("utf-8"),
            None,
        )

        payload = {
            "version": PAYLOAD_VERSION,
            "algorithm": "AES-256-GCM",
            "kdf": "PBKDF2-HMAC-SHA256",
            "iterations": PBKDF2_ITERATIONS,
            "salt": base64.b64encode(salt).decode("ascii"),
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        }

        payload_json = json.dumps(
            payload,
            separators=(",", ":"),
        )

        return base64.b64encode(
            payload_json.encode("utf-8")
        ).decode("ascii")

    def decrypt(self, encrypted_payload: str) -> str:
        """Decrypt a Base64 payload and return the original plaintext."""

        if not isinstance(encrypted_payload, str):
            raise InvalidPayloadError(
                "Encrypted payload must be a string."
            )

        try:
            payload_json = base64.b64decode(
                encrypted_payload,
                validate=True,
            ).decode("utf-8")

            payload = json.loads(payload_json)

        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidPayloadError(
                "Invalid encrypted payload."
            ) from exc

        required_fields = {
            "version",
            "algorithm",
            "kdf",
            "iterations",
            "salt",
            "nonce",
            "ciphertext",
        }

        if not required_fields.issubset(payload):
            raise InvalidPayloadError(
                "Missing required payload fields."
            )

        if payload["version"] != PAYLOAD_VERSION:
            raise InvalidPayloadError(
                "Unsupported payload version."
            )

        if payload["algorithm"] != "AES-256-GCM":
            raise InvalidPayloadError(
                "Unsupported encryption algorithm."
            )

        if payload["kdf"] != "PBKDF2-HMAC-SHA256":
            raise InvalidPayloadError(
                "Unsupported key derivation function."
            )

        try:
            salt = base64.b64decode(
                payload["salt"],
                validate=True,
            )

            nonce = base64.b64decode(
                payload["nonce"],
                validate=True,
            )

            ciphertext = base64.b64decode(
                payload["ciphertext"],
                validate=True,
            )

        except Exception as exc:
            raise InvalidPayloadError(
                "Invalid payload encoding."
            ) from exc

        if len(salt) != SALT_SIZE:
            raise InvalidPayloadError("Invalid salt length.")

        if len(nonce) != NONCE_SIZE:
            raise InvalidPayloadError("Invalid nonce length.")

        if not ciphertext:
            raise InvalidPayloadError("Ciphertext cannot be empty.")

        key = self._derive_key(salt)
        aesgcm = AESGCM(key)

        try:
            plaintext = aesgcm.decrypt(
                nonce,
                ciphertext,
                None,
            )

        except InvalidTag as exc:
            raise InvalidPasswordError(
                "Incorrect password or corrupted data."
            ) from exc

        try:
            return plaintext.decode("utf-8")

        except UnicodeDecodeError as exc:
            raise InvalidPayloadError(
                "Decrypted data is not valid UTF-8."
            ) from exc


def encrypt_text(plaintext: str, password: str) -> str:
    """Convenience function for encryption."""
    return AESCipher(password).encrypt(plaintext)


def decrypt_text(encrypted_payload: str, password: str) -> str:
    """Convenience function for decryption."""
    return AESCipher(password).decrypt(encrypted_payload)


if __name__ == "__main__":
    password = "TestPassword123!"
    plaintext = "Secret message for steganography."

    cipher = AESCipher(password)

    encrypted = cipher.encrypt(plaintext)
    decrypted = cipher.decrypt(encrypted)

    print("Original :", plaintext)
    print("Encrypted:", encrypted)
    print("Decrypted:", decrypted)

    assert decrypted == plaintext
    print("AES test passed.")