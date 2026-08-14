"""
Metrics for steganography and digital watermarking evaluation.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


MAX_PIXEL_VALUE = 255.0


def _to_array(
    image: Image.Image | np.ndarray,
) -> np.ndarray:
    """Convert image to NumPy array."""

    if isinstance(image, Image.Image):
        return np.asarray(image)

    if isinstance(image, np.ndarray):
        return image

    raise TypeError(
        "Input must be a PIL Image or NumPy array."
    )


def _validate_same_shape(
    image1: np.ndarray,
    image2: np.ndarray,
) -> None:
    """Check whether two images have the same shape."""

    if image1.shape != image2.shape:
        raise ValueError(
            "Images must have the same shape."
        )


def mse(
    original: Image.Image | np.ndarray,
    processed: Image.Image | np.ndarray,
) -> float:
    """
    Calculate Mean Squared Error.

    Lower MSE indicates smaller image distortion.
    """

    original_array = _to_array(
        original
    ).astype(np.float64)

    processed_array = _to_array(
        processed
    ).astype(np.float64)

    _validate_same_shape(
        original_array,
        processed_array,
    )

    return float(
        np.mean(
            (original_array - processed_array) ** 2
        )
    )


def psnr(
    original: Image.Image | np.ndarray,
    processed: Image.Image | np.ndarray,
) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio.

    Higher PSNR indicates better image quality.
    """

    error = mse(
        original,
        processed,
    )

    if error == 0:
        return float("inf")

    return float(
        10 * np.log10(
            MAX_PIXEL_VALUE ** 2 / error
        )
    )


def ssim(
    original: Image.Image | np.ndarray,
    processed: Image.Image | np.ndarray,
) -> float:
    """
    Calculate Structural Similarity Index.

    Higher SSIM indicates greater structural similarity.
    """

    original_array = _to_array(
        original
    )

    processed_array = _to_array(
        processed
    )

    _validate_same_shape(
        original_array,
        processed_array,
    )

    if original_array.ndim == 3:
        return float(
            structural_similarity(
                original_array,
                processed_array,
                channel_axis=-1,
                data_range=255,
            )
        )

    return float(
        structural_similarity(
            original_array,
            processed_array,
            data_range=255,
        )
    )


def ber(
    original_bits: np.ndarray | list[int],
    extracted_bits: np.ndarray | list[int],
) -> float:
    """
    Calculate Bit Error Rate.

    BER = number of incorrect bits / total bits.
    """

    original = np.asarray(
        original_bits,
        dtype=np.uint8,
    ).flatten()

    extracted = np.asarray(
        extracted_bits,
        dtype=np.uint8,
    ).flatten()

    if len(original) != len(extracted):
        raise ValueError(
            "Bit sequences must have the same length."
        )

    if len(original) == 0:
        raise ValueError(
            "Bit sequences cannot be empty."
        )

    errors = np.count_nonzero(
        original != extracted
    )

    return float(
        errors / len(original)
    )


def normalized_correlation(
    original: Image.Image | np.ndarray,
    extracted: Image.Image | np.ndarray,
) -> float:
    """
    Calculate Normalized Correlation (NC).

    Higher NC indicates better watermark similarity.
    """

    original_array = _to_array(
        original
    ).astype(np.float64)

    extracted_array = _to_array(
        extracted
    ).astype(np.float64)

    _validate_same_shape(
        original_array,
        extracted_array,
    )

    original_array = (
        original_array
        - np.mean(original_array)
    )

    extracted_array = (
        extracted_array
        - np.mean(extracted_array)
    )

    denominator = (
        np.sqrt(
            np.sum(original_array ** 2)
        )
        *
        np.sqrt(
            np.sum(extracted_array ** 2)
        )
    )

    if denominator == 0:
        return 0.0

    return float(
        np.sum(
            original_array
            * extracted_array
        )
        / denominator
    )


def calculate_metrics(
    original: Image.Image | np.ndarray,
    processed: Image.Image | np.ndarray,
) -> dict[str, float]:
    """
    Calculate image quality metrics.
    """

    return {
        "MSE": mse(
            original,
            processed,
        ),
        "PSNR": psnr(
            original,
            processed,
        ),
        "SSIM": ssim(
            original,
            processed,
        ),
    }


def calculate_watermark_metrics(
    original_watermark: Image.Image | np.ndarray,
    extracted_watermark: Image.Image | np.ndarray,
) -> dict[str, float]:
    """
    Calculate watermark extraction metrics.

    The images must have the same dimensions.
    """

    return {
        "NC": normalized_correlation(
            original_watermark,
            extracted_watermark,
        ),
    }


if __name__ == "__main__":
    original = Image.open(
        "original.png"
    ).convert("L")

    processed = Image.open(
        "processed.png"
    ).convert("L")

    metrics = calculate_metrics(
        original,
        processed,
    )

    print("Image Quality Metrics")

    for name, value in metrics.items():
        print(
            f"{name}: {value:.6f}"
        )