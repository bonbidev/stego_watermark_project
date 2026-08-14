"""
Image attacks for steganography and watermark robustness testing.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image


def _to_array(
    image: Image.Image | np.ndarray,
) -> np.ndarray:
    """Convert image to NumPy array."""

    if isinstance(image, Image.Image):
        return np.array(image)

    if isinstance(image, np.ndarray):
        return image.copy()

    raise TypeError(
        "Input must be a PIL Image or NumPy array."
    )


def _to_image(
    image_array: np.ndarray,
) -> Image.Image:
    """Convert NumPy array to PIL Image."""

    return Image.fromarray(
        np.clip(
            image_array,
            0,
            255,
        ).astype(np.uint8)
    )


def jpeg_compression(
    image: Image.Image,
    quality: int = 75,
) -> Image.Image:
    """
    Apply JPEG compression.

    Parameters
    ----------
    quality : int
        JPEG quality from 1 to 100.
    """

    if not 1 <= quality <= 100:
        raise ValueError(
            "Quality must be between 1 and 100."
        )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=quality,
    )

    buffer.seek(0)

    return Image.open(buffer).convert(
        image.mode
        if image.mode in ("L", "RGB")
        else "RGB"
    )


def gaussian_noise(
    image: Image.Image,
    mean: float = 0.0,
    sigma: float = 10.0,
) -> Image.Image:
    """Add Gaussian noise to an image."""

    if sigma < 0:
        raise ValueError(
            "Sigma must be non-negative."
        )

    array = _to_array(image).astype(
        np.float32
    )

    noise = np.random.normal(
        mean,
        sigma,
        array.shape,
    )

    noisy = array + noise

    return _to_image(noisy)


def salt_and_pepper_noise(
    image: Image.Image,
    amount: float = 0.01,
) -> Image.Image:
    """Add salt-and-pepper noise."""

    if not 0 <= amount <= 1:
        raise ValueError(
            "Amount must be between 0 and 1."
        )

    array = _to_array(image)

    result = array.copy()

    total_pixels = array.shape[0] * array.shape[1]

    noise_pixels = int(
        total_pixels * amount
    )

    if noise_pixels == 0:
        return _to_image(result)

    rows = np.random.randint(
        0,
        array.shape[0],
        noise_pixels,
    )

    cols = np.random.randint(
        0,
        array.shape[1],
        noise_pixels,
    )

    half = noise_pixels // 2

    result[
        rows[:half],
        cols[:half],
    ] = 0

    result[
        rows[half:],
        cols[half:],
    ] = 255

    return _to_image(result)


def gaussian_blur(
    image: Image.Image,
    kernel_size: int = 5,
    sigma: float = 0,
) -> Image.Image:
    """Apply Gaussian blur."""

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            "Kernel size must be a positive odd number."
        )

    array = _to_array(image)

    blurred = cv2.GaussianBlur(
        array,
        (
            kernel_size,
            kernel_size,
        ),
        sigma,
    )

    return _to_image(blurred)


def median_blur(
    image: Image.Image,
    kernel_size: int = 3,
) -> Image.Image:
    """Apply median blur."""

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            "Kernel size must be a positive odd number."
        )

    array = _to_array(image)

    blurred = cv2.medianBlur(
        array,
        kernel_size,
    )

    return _to_image(blurred)


def resize_image(
    image: Image.Image,
    scale: float = 0.5,
) -> Image.Image:
    """Resize an image by a scale factor."""

    if scale <= 0:
        raise ValueError(
            "Scale must be greater than 0."
        )

    width, height = image.size

    new_width = max(
        1,
        int(width * scale),
    )

    new_height = max(
        1,
        int(height * scale),
    )

    return image.resize(
        (
            new_width,
            new_height,
        ),
        Image.Resampling.LANCZOS,
    )


def crop_image(
    image: Image.Image,
    crop_ratio: float = 0.8,
) -> Image.Image:
    """Crop the center of an image."""

    if not 0 < crop_ratio <= 1:
        raise ValueError(
            "Crop ratio must be between 0 and 1."
        )

    width, height = image.size

    new_width = int(
        width * crop_ratio
    )

    new_height = int(
        height * crop_ratio
    )

    left = (
        width - new_width
    ) // 2

    top = (
        height - new_height
    ) // 2

    return image.crop(
        (
            left,
            top,
            left + new_width,
            top + new_height,
        )
    )


def rotate_image(
    image: Image.Image,
    angle: float = 5.0,
) -> Image.Image:
    """Rotate an image."""

    return image.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
    )


def sharpen_image(
    image: Image.Image,
    strength: float = 1.5,
) -> Image.Image:
    """Apply unsharp masking."""

    if strength < 0:
        raise ValueError(
            "Strength must be non-negative."
        )

    array = _to_array(image)

    blurred = cv2.GaussianBlur(
        array,
        (0, 0),
        3,
    )

    sharpened = cv2.addWeighted(
        array,
        1 + strength,
        blurred,
        -strength,
        0,
    )

    return _to_image(sharpened)


def apply_attack(
    image: Image.Image,
    attack: str,
    **kwargs,
) -> Image.Image:
    """
    Apply an attack by name.

    Supported attacks:
        jpeg
        gaussian_noise
        salt_pepper
        gaussian_blur
        median_blur
        resize
        crop
        rotate
        sharpen
    """

    attacks = {
        "jpeg": jpeg_compression,
        "gaussian_noise": gaussian_noise,
        "salt_pepper": salt_and_pepper_noise,
        "gaussian_blur": gaussian_blur,
        "median_blur": median_blur,
        "resize": resize_image,
        "crop": crop_image,
        "rotate": rotate_image,
        "sharpen": sharpen_image,
    }

    if attack not in attacks:
        raise ValueError(
            f"Unsupported attack: {attack}"
        )

    return attacks[attack](
        image,
        **kwargs,
    )


if __name__ == "__main__":
    image = Image.open(
        "input.png"
    )

    attacked = jpeg_compression(
        image,
        quality=50,
    )

    attacked.save(
        "attack_jpeg.jpg"
    )

    print("Attack test passed.")