"""
LSB image steganography.

Data is embedded into the least significant bits
of the blue channel.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


HEADER_SIZE = 32
CHANNEL_INDEX = 2


class LSBStegoError(Exception):
    """Base exception for LSB steganography."""


class CapacityError(LSBStegoError):
    """Raised when the image cannot store the payload."""


class ExtractionError(LSBStegoError):
    """Raised when hidden data cannot be extracted."""


class LSBStego:
    """LSB steganography using the blue channel."""

    @staticmethod
    def _bytes_to_bits(data: bytes) -> np.ndarray:
        """Convert bytes to a binary array."""

        return np.unpackbits(
            np.frombuffer(data, dtype=np.uint8)
        )

    @staticmethod
    def _bits_to_bytes(bits: np.ndarray) -> bytes:
        """Convert a binary array back to bytes."""

        if len(bits) % 8 != 0:
            raise ExtractionError(
                "Bit length must be divisible by 8."
            )

        return np.packbits(bits).tobytes()

    @staticmethod
    def _prepare_image(image: Image.Image) -> np.ndarray:
        """Convert image to RGB NumPy array."""

        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image.")

        return np.array(image.convert("RGB"))

    @staticmethod
    def _calculate_capacity(image_array: np.ndarray) -> int:
        """Return maximum payload size in bytes."""

        total_bits = image_array.shape[0] * image_array.shape[1]

        if total_bits <= HEADER_SIZE:
            return 0

        return (total_bits - HEADER_SIZE) // 8

    def get_capacity(self, image: Image.Image) -> int:
        """Return the maximum payload size in bytes."""

        image_array = self._prepare_image(image)
        return self._calculate_capacity(image_array)

    def embed(
        self,
        image: Image.Image,
        data: bytes,
    ) -> Image.Image:
        """
        Embed bytes into the blue channel.

        The first 32 bits store the payload length.
        """

        if not isinstance(data, bytes):
            raise ValueError("Data must be bytes.")

        image_array = self._prepare_image(image)

        capacity = self._calculate_capacity(image_array)

        if len(data) > capacity:
            raise CapacityError(
                f"Payload is too large. "
                f"Maximum capacity: {capacity} bytes."
            )

        # Store payload length in the first 32 bits.
        length_bits = np.unpackbits(
            np.array([len(data)], dtype=">u4").view(np.uint8)
        )

        payload_bits = self._bytes_to_bits(data)

        bits = np.concatenate(
            [length_bits, payload_bits]
        )

        blue_channel = image_array[:, :, CHANNEL_INDEX].flatten()

        blue_channel[:len(bits)] &= 0xFE
        blue_channel[:len(bits)] |= bits

        image_array[:, :, CHANNEL_INDEX] = blue_channel.reshape(
            image_array.shape[:2]
        )

        return Image.fromarray(image_array)

    def extract(self, image: Image.Image) -> bytes:
        """Extract hidden bytes from an image."""

        image_array = self._prepare_image(image)

        blue_channel = image_array[:, :, CHANNEL_INDEX].flatten()

        if len(blue_channel) < HEADER_SIZE:
            raise ExtractionError(
                "Image is too small."
            )

        # Read the 32-bit payload length.
        length_bits = blue_channel[:HEADER_SIZE] & 1

        length_bytes = np.packbits(
            length_bits
        ).tobytes()

        payload_length = int.from_bytes(
            length_bytes,
            byteorder="big",
        )

        capacity = self._calculate_capacity(image_array)

        if payload_length > capacity:
            raise ExtractionError(
                "Invalid payload length."
            )

        if payload_length == 0:
            return b""

        payload_bits_count = payload_length * 8

        start = HEADER_SIZE
        end = start + payload_bits_count

        if end > len(blue_channel):
            raise ExtractionError(
                "Incomplete payload."
            )

        payload_bits = blue_channel[start:end] & 1

        return self._bits_to_bytes(payload_bits)


def embed_data(
    image: Image.Image,
    data: bytes,
) -> Image.Image:
    """Convenience function for embedding data."""

    return LSBStego().embed(image, data)


def extract_data(
    image: Image.Image,
) -> bytes:
    """Convenience function for extracting data."""

    return LSBStego().extract(image)


if __name__ == "__main__":
    image = Image.open("input.png")

    secret = b"Hello LSB Steganography!"

    stego = LSBStego()

    print(
        f"Capacity: {stego.get_capacity(image)} bytes"
    )

    encoded_image = stego.embed(
        image,
        secret,
    )

    encoded_image.save("output.png")

    decoded = stego.extract(
        Image.open("output.png")
    )

    print("Original :", secret)
    print("Extracted:", decoded)

    assert decoded == secret

    print("LSB test passed.")