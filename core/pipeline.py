"""
Main pipeline for AES + LSB + DWT.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from core.aes_cipher import (
    AESCipher,
    InvalidPasswordError,
    InvalidPayloadError,
)
from core.lsb_stego import (
    LSBStego,
    CapacityError,
    ExtractionError,
)
from core.dwt_watermark import (
    DWTWatermark,
)


@dataclass
class EmbedResult:
    """Result returned after embedding."""

    image: Image.Image
    encrypted_payload: str
    payload_size: int


@dataclass
class ExtractResult:
    """Result returned after extraction."""

    secret_text: str
    encrypted_payload: str


class StegoPipeline:
    """
    AES → LSB → DWT pipeline.

    Embedding:
        Secret text
            ↓
        AES encryption
            ↓
        LSB embedding
            ↓
        DWT watermark
            ↓
        Final image

    Extraction:
        Watermarked image
            ↓
        DWT watermark extraction
            ↓
        LSB extraction
            ↓
        AES decryption
            ↓
        Secret text
    """

    def __init__(
        self,
        password: str,
        alpha: float = 0.05,
    ):
        if not password:
            raise ValueError(
                "Password cannot be empty."
            )

        self.cipher = AESCipher(password)
        self.lsb = LSBStego()
        self.dwt = DWTWatermark(alpha)

    def encrypt_message(
        self,
        secret_text: str,
    ) -> str:
        """Encrypt secret text."""

        if not secret_text:
            raise ValueError(
                "Secret text cannot be empty."
            )

        return self.cipher.encrypt(
            secret_text
        )

    def decrypt_message(
        self,
        encrypted_payload: str,
    ) -> str:
        """Decrypt encrypted payload."""

        return self.cipher.decrypt(
            encrypted_payload
        )

    def embed_secret(
        self,
        image: Image.Image,
        secret_text: str,
    ) -> tuple[Image.Image, str]:
        """
        Encrypt and embed secret text using LSB.
        """

        encrypted_payload = (
            self.encrypt_message(
                secret_text
            )
        )

        payload = encrypted_payload.encode(
            "utf-8"
        )

        stego_image = self.lsb.embed(
            image,
            payload,
        )

        return (
            stego_image,
            encrypted_payload,
        )

    def extract_secret(
        self,
        image: Image.Image,
    ) -> tuple[str, str]:
        """
        Extract encrypted payload and decrypt it.
        """

        payload = self.lsb.extract(
            image
        )

        if not payload:
            raise ExtractionError(
                "No hidden data found."
            )

        encrypted_payload = payload.decode(
            "utf-8"
        )

        secret_text = self.decrypt_message(
            encrypted_payload
        )

        return (
            secret_text,
            encrypted_payload,
        )

    def embed_watermark(
        self,
        image: Image.Image,
        watermark: Image.Image,
    ) -> Image.Image:
        """Embed watermark using DWT."""

        return self.dwt.embed(
            image,
            watermark,
        )

    def extract_watermark(
        self,
        original: Image.Image,
        watermarked: Image.Image,
        watermark_size: tuple[int, int] | None = None,
    ) -> Image.Image:
        """Extract watermark using DWT."""

        return self.dwt.extract(
            original,
            watermarked,
            watermark_size,
        )

    def run_embed(
        self,
        image: Image.Image,
        secret_text: str,
        watermark: Image.Image | None = None,
    ) -> EmbedResult:
        """
        Run the complete embedding pipeline.

        AES → LSB → optional DWT watermark
        """

        stego_image, encrypted_payload = (
            self.embed_secret(
                image,
                secret_text,
            )
        )

        if watermark is not None:
            stego_image = self.embed_watermark(
                stego_image,
                watermark,
            )

        return EmbedResult(
            image=stego_image,
            encrypted_payload=encrypted_payload,
            payload_size=len(
                encrypted_payload.encode("utf-8")
            ),
        )

    def run_extract(
        self,
        original: Image.Image,
        watermarked_image: Image.Image,
        watermark_size: tuple[int, int] | None = None,
    ) -> ExtractResult:
        """
        Run the complete extraction pipeline.

        DWT watermark extraction is optional.
        Secret extraction uses LSB.
        """

        secret_text, encrypted_payload = (
            self.extract_secret(
                watermarked_image
            )
        )

        return ExtractResult(
            secret_text=secret_text,
            encrypted_payload=encrypted_payload,
        )


def run_embed(
    image: Image.Image,
    secret_text: str,
    password: str,
    watermark: Image.Image | None = None,
    alpha: float = 0.05,
) -> EmbedResult:
    """Convenience function for the embedding pipeline."""

    pipeline = StegoPipeline(
        password=password,
        alpha=alpha,
    )

    return pipeline.run_embed(
        image=image,
        secret_text=secret_text,
        watermark=watermark,
    )


def run_extract(
    original: Image.Image,
    watermarked_image: Image.Image,
    password: str,
    watermark_size: tuple[int, int] | None = None,
    alpha: float = 0.05,
) -> ExtractResult:
    """Convenience function for the extraction pipeline."""

    pipeline = StegoPipeline(
        password=password,
        alpha=alpha,
    )

    return pipeline.run_extract(
        original=original,
        watermarked_image=watermarked_image,
        watermark_size=watermark_size,
    )


if __name__ == "__main__":
    image = Image.open("input.png")
    watermark = Image.open("watermark.png")

    password = "TestPassword123!"
    secret = "This is a secret message."

    pipeline = StegoPipeline(
        password=password,
        alpha=0.05,
    )

    # Embed
    result = pipeline.run_embed(
        image=image,
        secret_text=secret,
        watermark=watermark,
    )

    result.image.save(
        "final_output.png"
    )

    print(
        f"Payload size: "
        f"{result.payload_size} bytes"
    )

    # Extract
    extracted = pipeline.run_extract(
        original=image,
        watermarked_image=result.image,
        watermark_size=watermark.size,
    )

    print(
        "Original :",
        secret,
    )

    print(
        "Extracted:",
        extracted.secret_text,
    )

    assert extracted.secret_text == secret

    print("Pipeline test passed.")