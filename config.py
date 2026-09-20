"""
Central configuration for the steganography & watermarking project.

This file was previously committed empty (0 bytes), which silently
broke anything that tried `import config`. It now holds the tunable
defaults used across the project, grouped to match the real classes
in `core/` and `evaluation/`:

    core.aes_cipher.AESCipher(password)
    core.lsb_stego.LSBStego()
    core.pvd_stego.PVDStego()
    core.dct_stego.DCTStego()
    core.dwt_watermark.DWTWatermark(alpha=...)
    core.dwt_svd_watermark.DWTSVDWatermark(alpha=...)
    evaluation.attacks.apply_attack(image, name, **kwargs)

Nothing here is imported automatically by the `core`/`evaluation`
modules (they keep their own internal constants, e.g. HEADER_SIZE,
RANGES, PBKDF2_ITERATIONS) — this module exists so app.py, scripts,
and tests have one place to read/override recommended defaults
instead of duplicating magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass


# ============================================================================
# AES (core.aes_cipher.AESCipher)
# ============================================================================

@dataclass(frozen=True)
class CryptoConfig:
    """Recommended password policy. AESCipher itself hardcodes the
    cryptographic parameters (AES-256-GCM, 600,000 PBKDF2 iterations,
    16-byte salt, 12-byte nonce) — those are security-critical and are
    intentionally not made configurable here."""

    min_password_length: int = 12


# ============================================================================
# Steganography (core.lsb_stego / pvd_stego / dct_stego)
# ============================================================================

@dataclass(frozen=True)
class SteganographyConfig:
    """Which algorithm the UI offers by default and its ordering."""

    algorithms: tuple[str, ...] = ("LSB", "PVD", "DCT")
    default_algorithm: str = "LSB"


# ============================================================================
# Watermarking (core.dwt_watermark.DWTWatermark / dwt_svd_watermark.DWTSVDWatermark)
# ============================================================================

@dataclass(frozen=True)
class WatermarkConfig:
    """
    `alpha` controls embedding strength for both DWTWatermark and
    DWTSVDWatermark. Both classes validate `0 < alpha`; the UI slider
    additionally caps it at 0.20 because larger values visibly
    distort the host image.
    """

    algorithms: tuple[str, ...] = ("DWT", "DWT-SVD")
    default_algorithm: str = "DWT"
    default_alpha: float = 0.05
    min_alpha: float = 0.01
    max_alpha: float = 0.20
    alpha_step: float = 0.01


# ============================================================================
# Attacks (evaluation.attacks.apply_attack)
# ============================================================================

@dataclass(frozen=True)
class AttackConfig:
    """Default parameter values matching each function's own default
    in evaluation/attacks.py, kept here so the UI and scripts share
    one source of truth for "what does a moderate attack look like"."""

    jpeg_quality: int = 75
    gaussian_noise_sigma: float = 10.0
    salt_pepper_amount: float = 0.01
    gaussian_blur_kernel: int = 5
    median_blur_kernel: int = 3
    resize_scale: float = 0.5
    crop_ratio: float = 0.8
    rotate_angle: float = 5.0
    sharpen_strength: float = 1.5

    names: tuple[str, ...] = (
        "jpeg",
        "gaussian_noise",
        "salt_pepper",
        "gaussian_blur",
        "median_blur",
        "resize",
        "crop",
        "rotate",
        "sharpen",
    )

    def kwargs_for(self, attack: str) -> dict:
        """Return the default kwargs dict for `apply_attack(image, attack, **kwargs)`."""

        mapping = {
            "jpeg": {"quality": self.jpeg_quality},
            "gaussian_noise": {"sigma": self.gaussian_noise_sigma},
            "salt_pepper": {"amount": self.salt_pepper_amount},
            "gaussian_blur": {"kernel_size": self.gaussian_blur_kernel},
            "median_blur": {"kernel_size": self.median_blur_kernel},
            "resize": {"scale": self.resize_scale},
            "crop": {"crop_ratio": self.crop_ratio},
            "rotate": {"angle": self.rotate_angle},
            "sharpen": {"strength": self.sharpen_strength},
        }
        if attack not in mapping:
            raise ValueError(f"Unknown attack: {attack}")
        return mapping[attack]


# ============================================================================
# Quality metrics (evaluation.metrics)
# ============================================================================

@dataclass(frozen=True)
class MetricsConfig:
    max_pixel_value: float = 255.0

    # Rough, commonly cited PSNR/SSIM thresholds for "the change is
    # imperceptible to a casual viewer" — used only for the plain-
    # language interpretation shown in the UI, not a formal spec.
    good_psnr_db: float = 40.0
    good_ssim: float = 0.95


# ============================================================================
# Streamlit app
# ============================================================================

@dataclass(frozen=True)
class AppConfig:
    page_title: str = "Stego & Watermark"
    page_icon: str = "🔐"
    layout: str = "wide"
    supported_image_types: tuple[str, ...] = ("png", "jpg", "jpeg", "bmp")


# ============================================================================
# Instances used by the rest of the project
# ============================================================================

crypto_config = CryptoConfig()
stego_config = SteganographyConfig()
watermark_config = WatermarkConfig()
attack_config = AttackConfig()
metrics_config = MetricsConfig()
app_config = AppConfig()


def get_all_configs() -> dict:
    return {
        "crypto": crypto_config,
        "steganography": stego_config,
        "watermark": watermark_config,
        "attack": attack_config,
        "metrics": metrics_config,
        "app": app_config,
    }


def print_config() -> None:
    """Print every config value. Useful as a quick sanity check:
    `python config.py`."""

    for section_name, section in get_all_configs().items():
        print(f"\n[{section_name}]")
        for field_name in section.__dataclass_fields__:
            print(f"  {field_name} = {getattr(section, field_name)!r}")


if __name__ == "__main__":
    print_config()