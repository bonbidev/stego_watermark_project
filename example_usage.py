"""
Runnable usage examples for every algorithm in this project.

This file was previously committed empty (0 bytes). It now
demonstrates the real API of each module and doubles as a smoke
test: run it directly to confirm your environment is set up
correctly.

    python example_usage.py

Every example uses only synthetically generated images, so no
external test images or network access are required.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from core.aes_cipher import AESCipher, InvalidPasswordError
from core.lsb_stego import LSBStego, CapacityError as LSBCapacityError
from core.pvd_stego import PVDStego, CapacityError as PVDCapacityError
from core.dct_stego import DCTStego, CapacityError as DCTCapacityError
from core.dwt_watermark import DWTWatermark
from core.dwt_svd_watermark import DWTSVDWatermark

from evaluation.metrics import calculate_metrics, calculate_watermark_metrics
from evaluation.attacks import apply_attack
from evaluation.steganalysis import analyze


# ============================================================================
# Helpers
# ============================================================================

def make_sample_image(size: tuple[int, int] = (512, 512)) -> Image.Image:
    """A random-noise RGB image — worst case for compressibility, but
    perfectly fine as a steganography/watermarking carrier."""

    width, height = size
    array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


def make_sample_watermark(size: tuple[int, int] = (64, 64)) -> Image.Image:
    """A simple black-and-white pattern, easy to eyeball after extraction."""

    width, height = size
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[height // 4: 3 * height // 4, width // 4: 3 * width // 4] = 255
    return Image.fromarray(array, mode="RGB")


def print_metrics(label: str, original: Image.Image, processed: Image.Image) -> None:
    m = calculate_metrics(original, processed)
    print(f"  {label}: PSNR={m['PSNR']:.2f} dB, SSIM={m['SSIM']:.4f}, MSE={m['MSE']:.4f}")


# ============================================================================
# Example 1: AES encryption on its own
# ============================================================================

def example_aes_encryption() -> None:
    print("\n" + "=" * 60)
    print("EXAMPLE 1: AES-256-GCM encryption")
    print("=" * 60)

    cipher = AESCipher("a-strong-password-123!")
    plaintext = "This message is authenticated as well as encrypted."

    encrypted = cipher.encrypt(plaintext)  # -> base64 str, safe to store as text
    print(f"Encrypted (base64): {encrypted[:60]}...")

    decrypted = cipher.decrypt(encrypted)
    assert decrypted == plaintext
    print(f"Decrypted matches original: {decrypted == plaintext}")

    # Wrong password -> InvalidPasswordError, not silent corruption,
    # because this is AES-GCM (authenticated encryption).
    try:
        AESCipher("wrong-password").decrypt(encrypted)
    except InvalidPasswordError:
        print("Wrong password correctly rejected (InvalidPasswordError).")


# ============================================================================
# Example 2: LSB steganography
# ============================================================================

def example_lsb_steganography() -> None:
    print("\n" + "=" * 60)
    print("EXAMPLE 2: LSB steganography")
    print("=" * 60)

    cover = make_sample_image((512, 512))
    cipher = AESCipher("lsb-password")
    secret = "Hidden with LSB — fast, high capacity, low robustness."
    payload = cipher.encrypt(secret).encode("utf-8")

    stego_tool = LSBStego()
    print(f"  Capacity: {stego_tool.get_capacity(cover)} bytes (payload is {len(payload)} bytes)")

    stego_image = stego_tool.embed(cover, payload)
    print_metrics("Quality", cover, stego_image)

    extracted = stego_tool.extract(stego_image)
    recovered = cipher.decrypt(extracted.decode("utf-8"))
    assert recovered == secret
    print(f"  Recovered: {recovered!r}")


# ============================================================================
# Example 3: PVD steganography
# ============================================================================

def example_pvd_steganography() -> None:
    print("\n" + "=" * 60)
    print("EXAMPLE 3: PVD (Pixel Value Differencing) steganography")
    print("=" * 60)

    cover = make_sample_image((512, 512))
    cipher = AESCipher("pvd-password")
    secret = "Hidden with PVD — adapts capacity to local pixel contrast."
    payload = cipher.encrypt(secret).encode("utf-8")

    stego_tool = PVDStego()
    print(f"  Capacity: {stego_tool.get_capacity(cover)} bytes (payload is {len(payload)} bytes)")

    stego_image = stego_tool.embed(cover, payload)
    print_metrics("Quality", cover, stego_image)

    extracted = stego_tool.extract(stego_image)
    recovered = cipher.decrypt(extracted.decode("utf-8"))
    assert recovered == secret
    print(f"  Recovered: {recovered!r}")


# ============================================================================
# Example 4: DCT steganography
# ============================================================================

def example_dct_steganography() -> None:
    print("\n" + "=" * 60)
    print("EXAMPLE 4: DCT (frequency-domain) steganography")
    print("=" * 60)

    # DCT stores exactly 1 bit per 8x8 block, so capacity is much
    # lower than LSB/PVD for the same image size — use a larger
    # image or a short secret.
    cover = make_sample_image((768, 768))
    cipher = AESCipher("dct-password")
    secret = "Hidden with DCT."
    payload = cipher.encrypt(secret).encode("utf-8")

    stego_tool = DCTStego()
    print(f"  Capacity: {stego_tool.get_capacity(cover)} bytes (payload is {len(payload)} bytes)")

    stego_image = stego_tool.embed(cover, payload)
    print_metrics("Quality", cover, stego_image)

    extracted = stego_tool.extract(stego_image)
    recovered = cipher.decrypt(extracted.decode("utf-8"))
    assert recovered == secret
    print(f"  Recovered: {recovered!r}")


# ============================================================================
# Example 5: DWT watermarking
# ============================================================================

def example_dwt_watermarking() -> None:
    print("\n" + "=" * 60)
    print("EXAMPLE 5: DWT watermarking")
    print("=" * 60)

    cover = make_sample_image((256, 256))
    watermark = make_sample_watermark((64, 64))

    watermarker = DWTWatermark(alpha=0.05)
    watermarked = watermarker.embed(cover, watermark)
    print_metrics("Quality", cover, watermarked)

    extracted = watermarker.extract(cover, watermarked, watermark.size)
    wm_metrics = calculate_watermark_metrics(watermark, extracted)
    print(f"  Extracted watermark NC (similarity, 1.0 = perfect): {wm_metrics['NC']:.4f}")


# ============================================================================
# Example 6: DWT-SVD watermarking
# ============================================================================

def example_dwt_svd_watermarking() -> None:
    print("\n" + "=" * 60)
    print("EXAMPLE 6: DWT-SVD watermarking")
    print("=" * 60)

    cover = make_sample_image((256, 256))
    watermark = make_sample_watermark((64, 64))

    watermarker = DWTSVDWatermark(alpha=0.05)
    watermarked = watermarker.embed(cover, watermark)
    print_metrics("Quality", cover, watermarked)

    extracted = watermarker.extract(cover, watermarked, watermark.size)
    wm_metrics = calculate_watermark_metrics(watermark, extracted)
    print(f"  Extracted watermark NC (similarity, 1.0 = perfect): {wm_metrics['NC']:.4f}")
    print("  Note: this scheme's NC is typically much lower than DWT's — see")
    print("  01_CLEANUP_REPOSITORY / the code review notes for why.")


# ============================================================================
# Example 7: Robustness against attacks
# ============================================================================

def example_robustness() -> None:
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Robustness against common attacks")
    print("=" * 60)

    cover = make_sample_image((512, 512))
    cipher = AESCipher("robustness-password")
    secret = "Testing robustness"
    payload = cipher.encrypt(secret).encode("utf-8")

    stego_tool = LSBStego()
    stego_image = stego_tool.embed(cover, payload)

    attacks = [
        ("jpeg", {"quality": 75}),
        ("gaussian_noise", {"sigma": 10.0}),
        ("gaussian_blur", {"kernel_size": 3}),
    ]

    for attack_name, kwargs in attacks:
        attacked = apply_attack(stego_image, attack_name, **kwargs)
        try:
            extracted = stego_tool.extract(attacked)
            recovered = cipher.decrypt(extracted.decode("utf-8"))
            status = "recovered intact" if recovered == secret else "recovered but corrupted"
        except Exception:
            status = "could not be recovered"
        print(f"  After {attack_name}: {status}")

    print("  LSB is expected to break under most of these — that's the")
    print("  known trade-off for its speed and capacity, not a bug.")


# ============================================================================
# Example 8: Steganalysis
# ============================================================================

def example_steganalysis() -> None:
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Steganalysis (detecting hidden data)")
    print("=" * 60)

    clean = make_sample_image((256, 256))
    cipher = AESCipher("analysis-password")
    payload = cipher.encrypt("Hidden data for analysis").encode("utf-8")
    stego_image = LSBStego().embed(clean, payload)

    for label, image in [("Clean image", clean), ("Stego image", stego_image)]:
        results = analyze(image)
        print(f"  {label}:")
        for key, value in results.items():
            print(f"    {key}: {value:.4f}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 60)
    print("STEGANOGRAPHY & WATERMARKING — USAGE EXAMPLES")
    print("=" * 60)

    np.random.seed(0)  # reproducible sample images

    example_aes_encryption()
    example_lsb_steganography()
    example_pvd_steganography()
    example_dct_steganography()
    example_dwt_watermarking()
    example_dwt_svd_watermarking()
    example_robustness()
    example_steganalysis()

    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()