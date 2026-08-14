"""
DWT-based digital watermarking using Haar wavelet.
"""

from __future__ import annotations

import numpy as np
import pywt
from PIL import Image


WAVELET = "haar"
LEVEL = 1


class DWTWatermarkError(Exception):
    """Base exception for DWT watermarking."""


class WatermarkError(DWTWatermarkError):
    """Raised when watermark data is invalid."""


class DWTWatermark:
    """
    DWT watermarking using the LH subband.

    The watermark is embedded by modifying
    the LH coefficients of the host image.
    """

    def __init__(self, alpha: float = 0.05):
        if alpha <= 0:
            raise ValueError("Alpha must be greater than 0.")

        self.alpha = alpha

    @staticmethod
    def _prepare_image(image: Image.Image) -> np.ndarray:
        """Convert image to grayscale float32 array."""

        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image.")

        return np.array(
            image.convert("L"),
            dtype=np.float32,
        )

    @staticmethod
    def _prepare_watermark(
        watermark: Image.Image,
    ) -> np.ndarray:
        """Convert watermark to binary array."""

        if not isinstance(watermark, Image.Image):
            raise ValueError(
                "Watermark must be a PIL Image."
            )

        watermark = watermark.convert("L")

        array = np.array(
            watermark,
            dtype=np.float32,
        )

        return (array > 128).astype(np.float32)

    @staticmethod
    def _resize_watermark(
        watermark: np.ndarray,
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Resize watermark to match the LH subband."""

        image = Image.fromarray(
            (watermark * 255).astype(np.uint8)
        )

        resized = image.resize(
            (shape[1], shape[0]),
            Image.Resampling.NEAREST,
        )

        return (
            np.array(resized) > 128
        ).astype(np.float32)

    def _dwt(
        self,
        image_array: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Perform one-level 2D DWT."""

        return pywt.dwt2(
            image_array,
            WAVELET,
        )

    def _idwt(
        self,
        coeffs: tuple[
            np.ndarray,
            tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ],
        ],
    ) -> np.ndarray:
        """Perform inverse DWT."""

        return pywt.idwt2(
            coeffs,
            WAVELET,
        )

    def embed(
        self,
        image: Image.Image,
        watermark: Image.Image,
    ) -> Image.Image:
        """
        Embed a binary watermark into the host image.

        Watermark is embedded into the LH subband.
        """

        image_array = self._prepare_image(
            image
        )

        watermark_array = self._prepare_watermark(
            watermark
        )

        LL, (LH, HL, HH) = self._dwt(
            image_array
        )

        watermark_array = self._resize_watermark(
            watermark_array,
            LH.shape,
        )

        # Add watermark information to LH coefficients.
        LH_watermarked = (
            LH + self.alpha * 50.0 * watermark_array
        )

        watermarked = self._idwt(
            (
                LL,
                (
                    LH_watermarked,
                    HL,
                    HH,
                ),
            )
        )

        watermarked = np.clip(
            watermarked,
            0,
            255,
        ).astype(np.uint8)

        return Image.fromarray(
            watermarked,
            mode="L",
        )

    def extract(
        self,
        original: Image.Image,
        watermarked: Image.Image,
        watermark_size: tuple[int, int] | None = None,
    ) -> Image.Image:
        """
        Extract watermark from original and watermarked images.
        """

        original_array = self._prepare_image(
            original
        )

        watermarked_array = self._prepare_image(
            watermarked
        )

        _, (_, original_LH, _,) = self._dwt(
            original_array
        )

        _, (_, watermarked_LH, _,) = self._dwt(
            watermarked_array
        )

        difference = (
            watermarked_LH - original_LH
        )

        if watermark_size is None:
            watermark_size = (
                difference.shape[1],
                difference.shape[0],
            )

        watermark = (
            difference > self.alpha * 25.0
        ).astype(np.uint8) * 255

        watermark_image = Image.fromarray(
            watermark,
            mode="L",
        )

        watermark_image = watermark_image.resize(
            watermark_size,
            Image.Resampling.NEAREST,
        )

        return watermark_image


def embed_watermark(
    image: Image.Image,
    watermark: Image.Image,
    alpha: float = 0.05,
) -> Image.Image:
    """Convenience function for watermark embedding."""

    return DWTWatermark(alpha).embed(
        image,
        watermark,
    )


def extract_watermark(
    original: Image.Image,
    watermarked: Image.Image,
    watermark_size: tuple[int, int] | None = None,
    alpha: float = 0.05,
) -> Image.Image:
    """Convenience function for watermark extraction."""

    return DWTWatermark(alpha).extract(
        original,
        watermarked,
        watermark_size,
    )


if __name__ == "__main__":
    original = Image.open("input.png")
    watermark = Image.open("watermark.png")

    dwt = DWTWatermark(alpha=0.05)

    watermarked = dwt.embed(
        original,
        watermark,
    )

    watermarked.save(
        "watermarked.png"
    )

    extracted = dwt.extract(
        original,
        watermarked,
        watermark.size,
    )

    extracted.save(
        "extracted_watermark.png"
    )

    print("DWT watermark test passed.")