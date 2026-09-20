"""
Pixel Value Differencing (PVD) image steganography.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


HEADER_SIZE = 32

# PVD ranges: (lower_bound, upper_bound, bits_per_pixel_pair)
RANGES = [
    (0, 7, 3),
    (8, 15, 3),
    (16, 31, 4),
    (32, 63, 5),
    (64, 127, 6),
    (128, 255, 7),
]


class PVDStegoError(Exception):
    """Base exception for PVD steganography."""


class CapacityError(PVDStegoError):
    """Raised when the image cannot store the payload."""


class ExtractionError(PVDStegoError):
    """Raised when hidden data cannot be extracted."""


class PVDStego:
    """
    PVD steganography using pairs of grayscale pixels.

    Larger pixel differences allow more embedded bits.
    """

    @staticmethod
    def _prepare_image(image: Image.Image) -> np.ndarray:
        """
        Return the single channel that PVD actually reads/writes.

        For color images, only the Blue channel is used (same choice
        as LSB) — this is what embed() modifies below, so extract()
        pulling out the same channel is enough to recover the data.
        Using `.convert("L")` here (the original approach) collapsed
        every stego image to grayscale even though only one channel
        was ever touched, permanently discarding the other two
        channels' color information for no benefit.
        """

        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image.")

        if image.mode in ("RGB", "RGBA"):
            return np.array(image.convert("RGB"), dtype=np.uint8)[:, :, 2].copy()

        return np.array(
            image.convert("L"),
            dtype=np.uint8,
        )

    @staticmethod
    def _bytes_to_bits(data: bytes) -> list[int]:
        """Convert bytes to bits."""

        return [
            (byte >> bit) & 1
            for byte in data
            for bit in range(7, -1, -1)
        ]

    @staticmethod
    def _bits_to_bytes(bits: list[int]) -> bytes:
        """Convert bits to bytes."""

        if len(bits) % 8 != 0:
            raise ExtractionError(
                "Bit length must be divisible by 8."
            )

        result = bytearray()

        for i in range(0, len(bits), 8):
            value = 0

            for bit in bits[i:i + 8]:
                value = (value << 1) | bit

            result.append(value)

        return bytes(result)

    @staticmethod
    def _get_range(difference: int) -> tuple[int, int, int]:
        """Return PVD range for a pixel difference."""

        for lower, upper, bits in RANGES:
            if lower <= difference <= upper:
                return lower, upper, bits

        raise ValueError("Invalid pixel difference.")

    @staticmethod
    def _calculate_capacity(image_array: np.ndarray) -> int:
        """Estimate maximum payload capacity in bytes."""

        height, width = image_array.shape
        capacity_bits = 0

        for row in range(height):
            for col in range(0, width - 1, 2):
                p1 = int(image_array[row, col])
                p2 = int(image_array[row, col + 1])

                difference = abs(p1 - p2)

                _, _, bits = PVDStego._get_range(
                    difference
                )

                capacity_bits += bits

        return max(
            0,
            (capacity_bits - HEADER_SIZE) // 8,
        )

    def get_capacity(self, image: Image.Image) -> int:
        """Return estimated payload capacity in bytes."""

        image_array = self._prepare_image(image)

        return self._calculate_capacity(
            image_array
        )

    @staticmethod
    def _adjust_pixels(
        p1: int,
        p2: int,
        new_difference: int,
    ) -> tuple[int, int]:
        """
        Adjust a pixel pair so that abs(p1' - p2') == new_difference,
        while keeping both values within [0, 255].

        NOTE: simply clamping p1/p2 independently to [0, 255] (the
        original implementation) can silently change the resulting
        difference whenever the naive adjustment pushes a value out
        of range. That desynchronizes embed/extract and corrupts the
        payload. Instead, if a value would overflow/underflow, the
        *pair* is shifted together by the overflow amount, which
        preserves (p1' - p2') exactly because both values move by the
        same amount.
        """

        current_difference = abs(p1 - p2)
        delta = new_difference - current_difference

        if p1 >= p2:
            a = p1 + (delta + 1) // 2
            b = p2 - delta // 2
        else:
            a = p1 - delta // 2
            b = p2 + (delta + 1) // 2

        if a > 255:
            shift = a - 255
            a -= shift
            b -= shift
        elif a < 0:
            shift = -a
            a += shift
            b += shift

        if b > 255:
            shift = b - 255
            a -= shift
            b -= shift
        elif b < 0:
            shift = -b
            a += shift
            b += shift

        return (
            max(0, min(255, a)),
            max(0, min(255, b)),
        )

    def embed(
        self,
        image: Image.Image,
        data: bytes,
    ) -> Image.Image:
        """
        Embed data using Pixel Value Differencing.

        The first 32 bits store the payload length. For a color
        image, only the Blue channel is modified — Red and Green are
        copied through unchanged so the result stays a color image
        instead of collapsing to grayscale.
        """

        if not isinstance(data, bytes):
            raise ValueError("Data must be bytes.")

        original_rgb: np.ndarray | None = None
        if image.mode in ("RGB", "RGBA"):
            original_rgb = np.array(image.convert("RGB"), dtype=np.uint8)
            image_array = original_rgb[:, :, 2].copy()
        else:
            image_array = np.array(image.convert("L"), dtype=np.uint8)

        capacity = self._calculate_capacity(
            image_array
        )

        if len(data) > capacity:
            raise CapacityError(
                f"Payload is too large. "
                f"Maximum capacity: {capacity} bytes."
            )

        length_bits = [
            (len(data) >> bit) & 1
            for bit in range(31, -1, -1)
        ]

        payload_bits = self._bytes_to_bits(data)

        bits = length_bits + payload_bits

        bit_index = 0

        height, width = image_array.shape

        for row in range(height):
            if bit_index >= len(bits):
                break

            for col in range(0, width - 1, 2):

                if bit_index >= len(bits):
                    break

                p1 = int(image_array[row, col])
                p2 = int(image_array[row, col + 1])

                difference = abs(p1 - p2)

                lower, upper, capacity_bits = (
                    self._get_range(difference)
                )

                remaining = len(bits) - bit_index

                if remaining < capacity_bits:
                    value = 0

                    for _ in range(remaining):
                        value = (
                            value << 1
                        ) | bits[bit_index]

                        bit_index += 1

                    value <<= (
                        capacity_bits - remaining
                    )

                else:
                    value = 0

                    for _ in range(capacity_bits):
                        value = (
                            value << 1
                        ) | bits[bit_index]

                        bit_index += 1

                new_difference = lower + value

                if new_difference > upper:
                    new_difference = upper

                new_p1, new_p2 = (
                    self._adjust_pixels(
                        p1,
                        p2,
                        new_difference,
                    )
                )

                image_array[row, col] = new_p1
                image_array[row, col + 1] = new_p2

        if original_rgb is not None:
            original_rgb[:, :, 2] = image_array
            return Image.fromarray(original_rgb, mode="RGB")

        return Image.fromarray(image_array)

    def extract(
        self,
        image: Image.Image,
    ) -> bytes:
        """Extract hidden data from a PVD image."""

        image_array = self._prepare_image(image)

        height, width = image_array.shape

        bits: list[int] = []

        for row in range(height):
            for col in range(0, width - 1, 2):

                p1 = int(image_array[row, col])
                p2 = int(image_array[row, col + 1])

                difference = abs(p1 - p2)

                lower, upper, capacity_bits = (
                    self._get_range(difference)
                )

                value = difference - lower

                for bit in range(
                    capacity_bits - 1,
                    -1,
                    -1,
                ):
                    bits.append(
                        (value >> bit) & 1
                    )

        if len(bits) < HEADER_SIZE:
            raise ExtractionError(
                "Image does not contain enough data."
            )

        length = 0

        for bit in bits[:HEADER_SIZE]:
            length = (length << 1) | bit

        payload_bits = length * 8

        if (
            length == 0
            or HEADER_SIZE + payload_bits > len(bits)
        ):
            raise ExtractionError(
                "Invalid payload length."
            )

        return self._bits_to_bytes(
            bits[
                HEADER_SIZE:
                HEADER_SIZE + payload_bits
            ]
        )


def embed_data(
    image: Image.Image,
    data: bytes,
) -> Image.Image:
    """Convenience function for embedding data."""

    return PVDStego().embed(
        image,
        data,
    )


def extract_data(
    image: Image.Image,
) -> bytes:
    """Convenience function for extracting data."""

    return PVDStego().extract(image)


if __name__ == "__main__":
    image = Image.open("input.png")

    secret = b"Hello PVD Steganography!"

    stego = PVDStego()

    print(
        f"Capacity: {stego.get_capacity(image)} bytes"
    )

    encoded_image = stego.embed(
        image,
        secret,
    )

    encoded_image.save("pvd_output.png")

    decoded = stego.extract(
        Image.open("pvd_output.png")
    )

    print("Original :", secret)
    print("Extracted:", decoded)

    assert decoded == secret

    print("PVD test passed.")