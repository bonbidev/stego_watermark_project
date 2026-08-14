"""
DCT-based image steganography.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


HEADER_SIZE = 32
BLOCK_SIZE = 8

# Hai hệ số DCT dùng để biểu diễn bit.
COEFF_1 = (3, 4)
COEFF_2 = (4, 3)


class DCTStegoError(Exception):
    """Base exception for DCT steganography."""


class CapacityError(DCTStegoError):
    """Raised when the image cannot store the payload."""


class ExtractionError(DCTStegoError):
    """Raised when hidden data cannot be extracted."""


class DCTStego:
    """DCT steganography using 8x8 image blocks."""

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
    def _dct_2d(block: np.ndarray) -> np.ndarray:
        """Calculate 2D DCT for an 8x8 block."""

        return DCTStego._dct_1d(
            DCTStego._dct_1d(block, axis=0),
            axis=1,
        )

    @staticmethod
    def _dct_1d(
        matrix: np.ndarray,
        axis: int,
    ) -> np.ndarray:
        """Calculate DCT-II along one axis."""

        n = matrix.shape[axis]

        x = np.arange(n)
        k = np.arange(n)

        transform = np.cos(
            np.pi / n
            * (x[:, None] + 0.5)
            * k[None, :]
        )

        result = np.tensordot(
            matrix,
            transform,
            axes=([axis], [0]),
        )

        result *= np.sqrt(2 / n)
        result[..., 0] /= np.sqrt(2)

        return result

    @staticmethod
    def _idct_2d(block: np.ndarray) -> np.ndarray:
        """Calculate inverse 2D DCT."""

        return DCTStego._idct_1d(
            DCTStego._idct_1d(block, axis=0),
            axis=1,
        )

    @staticmethod
    def _idct_1d(
        matrix: np.ndarray,
        axis: int,
    ) -> np.ndarray:
        """Calculate inverse DCT-II."""

        n = matrix.shape[axis]

        k = np.arange(n)
        x = np.arange(n)

        transform = np.cos(
            np.pi / n
            * (x[:, None] + 0.5)
            * k[None, :]
        )

        matrix = matrix.copy()
        matrix[..., 0] /= np.sqrt(2)

        result = np.tensordot(
            matrix,
            transform.T,
            axes=([axis], [0]),
        )

        return result * np.sqrt(2 / n)

    @staticmethod
    def _get_blocks(
        image_array: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Return valid 8x8 block positions."""

        height, width = image_array.shape

        blocks = []

        for row in range(
            0,
            height - BLOCK_SIZE + 1,
            BLOCK_SIZE,
        ):
            for col in range(
                0,
                width - BLOCK_SIZE + 1,
                BLOCK_SIZE,
            ):
                blocks.append((row, col))

        return blocks

    @staticmethod
    def _set_bit(
        dct: np.ndarray,
        bit: int,
    ) -> None:
        """Encode one bit using two DCT coefficients."""

        c1 = dct[COEFF_1]
        c2 = dct[COEFF_2]

        strength = max(
            5.0,
            abs(c1 - c2),
        )

        if bit == 0:
            dct[COEFF_1] = c2 + strength / 2
            dct[COEFF_2] = c2
        else:
            dct[COEFF_1] = c2
            dct[COEFF_2] = c2 + strength / 2

    @staticmethod
    def _get_bit(dct: np.ndarray) -> int:
        """Extract one bit from two DCT coefficients."""

        c1 = dct[COEFF_1]
        c2 = dct[COEFF_2]

        return 0 if c1 > c2 else 1

    @staticmethod
    def _calculate_capacity(
        image_array: np.ndarray,
    ) -> int:
        """Return payload capacity in bytes."""

        block_count = len(
            DCTStego._get_blocks(image_array)
        )

        if block_count * 8 <= HEADER_SIZE:
            return 0

        return (
            block_count * 8 - HEADER_SIZE
        ) // 8

    def get_capacity(
        self,
        image: Image.Image,
    ) -> int:
        """Return payload capacity in bytes."""

        image_array = self._prepare_image(image)

        return self._calculate_capacity(
            image_array
        )

    def embed(
        self,
        image: Image.Image,
        data: bytes,
    ) -> Image.Image:
        """
        Embed data into DCT coefficients.

        One bit is stored in each 8x8 block.
        """

        if not isinstance(data, bytes):
            raise ValueError("Data must be bytes.")

        image_array = self._prepare_image(image)

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

        blocks = self._get_blocks(image_array)

        for index, bit in enumerate(bits):

            row, col = blocks[index]

            block = image_array[
                row:row + BLOCK_SIZE,
                col:col + BLOCK_SIZE,
            ]

            block = block - 128.0

            dct = self._dct_2d(block)

            self._set_bit(dct, bit)

            restored = self._idct_2d(dct)

            image_array[
                row:row + BLOCK_SIZE,
                col:col + BLOCK_SIZE,
            ] = restored + 128.0

        image_array = np.clip(
            image_array,
            0,
            255,
        ).astype(np.uint8)

        return Image.fromarray(
            image_array,
            mode="L",
        )

    def extract(
        self,
        image: Image.Image,
    ) -> bytes:
        """Extract hidden data from DCT coefficients."""

        image_array = self._prepare_image(image)

        blocks = self._get_blocks(image_array)

        if len(blocks) * 8 < HEADER_SIZE:
            raise ExtractionError(
                "Image is too small."
            )

        bits = []

        for row, col in blocks:

            block = image_array[
                row:row + BLOCK_SIZE,
                col:col + BLOCK_SIZE,
            ]

            block = block - 128.0

            dct = self._dct_2d(block)

            bits.append(
                self._get_bit(dct)
            )

        length = 0

        for bit in bits[:HEADER_SIZE]:
            length = (length << 1) | bit

        if length == 0:
            return b""

        required_bits = length * 8

        if (
            HEADER_SIZE + required_bits
            > len(bits)
        ):
            raise ExtractionError(
                "Invalid payload length."
            )

        payload_bits = bits[
            HEADER_SIZE:
            HEADER_SIZE + required_bits
        ]

        return self._bits_to_bytes(
            payload_bits
        )


def embed_data(
    image: Image.Image,
    data: bytes,
) -> Image.Image:
    """Convenience function for embedding data."""

    return DCTStego().embed(
        image,
        data,
    )


def extract_data(
    image: Image.Image,
) -> bytes:
    """Convenience function for extracting data."""

    return DCTStego().extract(image)


if __name__ == "__main__":
    image = Image.open("input.png")

    secret = b"Hello DCT Steganography!"

    stego = DCTStego()

    print(
        f"Capacity: {stego.get_capacity(image)} bytes"
    )

    encoded_image = stego.embed(
        image,
        secret,
    )

    encoded_image.save("dct_output.png")

    decoded = stego.extract(
        Image.open("dct_output.png")
    )

    print("Original :", secret)
    print("Extracted:", decoded)

    assert decoded == secret

    print("DCT test passed.")