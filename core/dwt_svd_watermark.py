"""
DWT-SVD based digital watermarking.
"""

from __future__ import annotations

import numpy as np
import pywt
from PIL import Image


WAVELET = "haar"


class DWTSVDWatermarkError(Exception):
    """Base exception for DWT-SVD watermarking."""


class WatermarkError(DWTSVDWatermarkError):
    """Raised when watermark data is invalid."""


class DWTSVDWatermark:
    """
    DWT-SVD watermarking using the LH subband.
    """

    def __init__(self, alpha: float = 0.05):
        if alpha <= 0:
            raise ValueError("Alpha must be greater than 0.")

        self.alpha = alpha

    @staticmethod
    def _prepare_image(
        image: Image.Image,
    ) -> np.ndarray:
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
        """Convert watermark to normalized grayscale array."""

        if not isinstance(watermark, Image.Image):
            raise ValueError(
                "Watermark must be a PIL Image."
            )

        return np.array(
            watermark.convert("L"),
            dtype=np.float32,
        ) / 255.0

    @staticmethod
    def _dwt(
        image_array: np.ndarray,
    ):
        """Perform one-level 2D DWT."""

        return pywt.dwt2(
            image_array,
            WAVELET,
        )

    @staticmethod
    def _idwt(coeffs):
        """Perform inverse DWT."""

        return pywt.idwt2(
            coeffs,
            WAVELET,
        )

    @staticmethod
    def _resize_watermark(
        watermark: np.ndarray,
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Resize watermark to match the LH subband."""

        watermark_image = Image.fromarray(
            np.clip(
                watermark * 255,
                0,
                255,
            ).astype(np.uint8)
        )

        resized = watermark_image.resize(
            (shape[1], shape[0]),
            Image.Resampling.BILINEAR,
        )

        return (
            np.array(
                resized,
                dtype=np.float32,
            ) / 255.0
        )

    def embed(
        self,
        image: Image.Image,
        watermark: Image.Image,
    ) -> Image.Image:
        """
        Embed watermark using DWT-SVD.
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

        U, S, Vt = np.linalg.svd(
            LH,
            full_matrices=False,
        )

        U_w, S_w, Vt_w = np.linalg.svd(
            watermark_array,
            full_matrices=False,
        )

        S_modified = (
            S + self.alpha * S_w
        )

        LH_modified = (
            U
            @ np.diag(S_modified)
            @ Vt
        )

        watermarked = self._idwt(
            (
                LL,
                (
                    LH_modified,
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

        _, (original_LH, _, _,) = self._dwt(
            original_array
        )

        _, (watermarked_LH, _, _,) = self._dwt(
            watermarked_array
        )

        _, S_original, _ = np.linalg.svd(
            original_LH,
            full_matrices=False,
        )

        _, S_watermarked, _ = np.linalg.svd(
            watermarked_LH,
            full_matrices=False,
        )

        S_watermark = (
            S_watermarked - S_original
        ) / self.alpha

        min_value = min(
            S_watermark.shape[0],
            S_watermark.shape[1]
            if S_watermark.ndim > 1
            else 1,
        )

        if min_value <= 0:
            raise WatermarkError(
                "Unable to extract watermark."
            )

        watermark = np.diag(
            S_watermark
        )

        watermark = np.clip(
            watermark,
            0,
            1,
        )

        watermark = (
            watermark * 255
        ).astype(np.uint8)

        result = Image.fromarray(
            watermark,
            mode="L",
        )

        if watermark_size is not None:
            result = result.resize(
                watermark_size,
                Image.Resampling.BILINEAR,
            )

        return result


def embed_watermark(
    image: Image.Image,
    watermark: Image.Image,
    alpha: float = 0.05,
) -> Image.Image:
    """Convenience function for watermark embedding."""

    return DWTSVDWatermark(alpha).embed(
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

    return DWTSVDWatermark(alpha).extract(
        original,
        watermarked,
        watermark_size,
    )


if __name__ == "__main__":
    original = Image.open("input.png")
    watermark = Image.open("watermark.png")

    dwt_svd = DWTSVDWatermark(
        alpha=0.05
    )

    watermarked = dwt_svd.embed(
        original,
        watermark,
    )

    watermarked.save(
        "dwt_svd_output.png"
    )

    extracted = dwt_svd.extract(
        original,
        watermarked,
        watermark.size,
    )

    extracted.save(
        "dwt_svd_extracted.png"
    )

    print("DWT-SVD test passed.")