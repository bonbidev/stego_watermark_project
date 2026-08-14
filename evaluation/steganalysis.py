"""
Basic steganalysis methods for image analysis.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.stats import chisquare


def _to_array(
    image: Image.Image | np.ndarray,
) -> np.ndarray:
    """Convert image to grayscale array."""

    if isinstance(image, Image.Image):
        return np.asarray(
            image.convert("L")
        )

    if isinstance(image, np.ndarray):
        if image.ndim == 3:
            return np.asarray(
                Image.fromarray(
                    image.astype(np.uint8)
                ).convert("L")
            )

        return image

    raise TypeError(
        "Input must be a PIL Image or NumPy array."
    )


def _validate_image(
    image: np.ndarray,
) -> None:
    """Validate grayscale image."""

    if image.size == 0:
        raise ValueError(
            "Image cannot be empty."
        )

    if image.ndim != 2:
        raise ValueError(
            "Image must be grayscale."
        )


def lsb_plane(
    image: Image.Image | np.ndarray,
) -> np.ndarray:
    """Return the LSB plane of an image."""

    array = _to_array(image)

    _validate_image(array)

    return array.astype(
        np.uint8
    ) & 1


def lsb_ratio(
    image: Image.Image | np.ndarray,
) -> float:
    """Calculate the ratio of 1s in the LSB plane."""

    plane = lsb_plane(image)

    return float(
        np.mean(plane)
    )


def lsb_histogram(
    image: Image.Image | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return even and odd pixel-value histograms."""

    array = _to_array(image)

    _validate_image(array)

    histogram = np.bincount(
        array.flatten(),
        minlength=256,
    )

    even = histogram[0::2]
    odd = histogram[1::2]

    return even, odd


def lsb_histogram_difference(
    image: Image.Image | np.ndarray,
) -> float:
    """
    Measure the difference between even and odd
    pixel-value distributions.
    """

    even, odd = lsb_histogram(
        image
    )

    total = np.sum(even) + np.sum(odd)

    if total == 0:
        return 0.0

    difference = np.sum(
        np.abs(
            even.astype(np.float64)
            - odd.astype(np.float64)
        )
    )

    return float(
        difference / total
    )


def chi_square_test(
    image: Image.Image | np.ndarray,
) -> dict[str, float]:
    """
    Perform a chi-square test on paired pixel values.

    Lower p-value indicates stronger evidence
    against a balanced even/odd distribution.
    """

    even, odd = lsb_histogram(
        image
    )

    observed = np.vstack(
        [
            even,
            odd,
        ]
    ).astype(np.float64)

    expected = (
        observed.sum(axis=0)
        / 2
    )

    expected = np.vstack(
        [
            expected,
            expected,
        ]
    )

    valid = expected > 0

    statistic, p_value = chisquare(
        f_obs=observed[:, valid].flatten(),
        f_exp=expected[:, valid].flatten(),
    )

    return {
        "chi_square": float(
            statistic
        ),
        "p_value": float(
            p_value
        ),
    }


def entropy(
    image: Image.Image | np.ndarray,
) -> float:
    """
    Calculate Shannon entropy.

    Higher entropy indicates a more complex
    pixel-value distribution.
    """

    array = _to_array(image)

    _validate_image(array)

    histogram = np.bincount(
        array.flatten(),
        minlength=256,
    ).astype(np.float64)

    probabilities = (
        histogram
        / histogram.sum()
    )

    probabilities = probabilities[
        probabilities > 0
    ]

    return float(
        -np.sum(
            probabilities
            * np.log2(probabilities)
        )
    )


def lsb_entropy(
    image: Image.Image | np.ndarray,
) -> float:
    """Calculate entropy of the LSB plane."""

    plane = lsb_plane(image)

    counts = np.bincount(
        plane.flatten(),
        minlength=2,
    ).astype(np.float64)

    probabilities = (
        counts
        / counts.sum()
    )

    probabilities = probabilities[
        probabilities > 0
    ]

    return float(
        -np.sum(
            probabilities
            * np.log2(probabilities)
        )
    )


def rs_analysis(
    image: Image.Image | np.ndarray,
    group_size: int = 4,
) -> dict[str, float]:
    """
    Perform a simplified RS analysis.

    The result estimates the regular and singular
    group ratio under LSB flipping.
    """

    array = _to_array(image)

    _validate_image(array)

    flat = array.flatten().astype(
        np.int16
    )

    usable_length = (
        len(flat)
        - len(flat) % group_size
    )

    flat = flat[:usable_length]

    groups = flat.reshape(
        -1,
        group_size,
    )

    def variation(group):
        return np.sum(
            np.abs(
                np.diff(group)
            )
        )

    original_variations = np.array(
        [
            variation(group)
            for group in groups
        ],
        dtype=np.float64,
    )

    flipped = groups.copy()

    flipped ^= 1

    flipped_variations = np.array(
        [
            variation(group)
            for group in flipped
        ],
        dtype=np.float64,
    )

    regular = np.sum(
        flipped_variations
        > original_variations
    )

    singular = np.sum(
        flipped_variations
        < original_variations
    )

    total = len(groups)

    if total == 0:
        return {
            "regular_ratio": 0.0,
            "singular_ratio": 0.0,
            "rs_difference": 0.0,
        }

    regular_ratio = (
        regular / total
    )

    singular_ratio = (
        singular / total
    )

    return {
        "regular_ratio": float(
            regular_ratio
        ),
        "singular_ratio": float(
            singular_ratio
        ),
        "rs_difference": float(
            regular_ratio
            - singular_ratio
        ),
    }


def analyze(
    image: Image.Image | np.ndarray,
) -> dict[str, float]:
    """
    Run all basic steganalysis methods.
    """

    chi = chi_square_test(
        image
    )

    rs = rs_analysis(
        image
    )

    return {
        "LSB ratio": lsb_ratio(
            image
        ),
        "LSB histogram difference":
            lsb_histogram_difference(
                image
            ),
        "Chi-square":
            chi["chi_square"],
        "Chi-square p-value":
            chi["p_value"],
        "Entropy":
            entropy(image),
        "LSB entropy":
            lsb_entropy(image),
        "RS regular ratio":
            rs["regular_ratio"],
        "RS singular ratio":
            rs["singular_ratio"],
        "RS difference":
            rs["rs_difference"],
    }


if __name__ == "__main__":
    image = Image.open(
        "stego.png"
    )

    results = analyze(
        image
    )

    for name, value in results.items():
        print(
            f"{name}: {value:.6f}"
        )