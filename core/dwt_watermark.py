from __future__ import annotations
import argparse
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np
import pywt

__all__ = [
    "Subband",
    "DWTWatermarkConfig",
    "EmbedResult",
    "ExtractResult",
    "DWTWatermarkError",
    "InvalidSubbandError",
    "ShapeMismatchError",
    "UnsupportedWaveletError",
    "DWTWatermarker",
]
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


class DWTWatermarkError(Exception):
    pass


class InvalidSubbandError(DWTWatermarkError):
    pass


class ShapeMismatchError(DWTWatermarkError):
    pass


class UnsupportedWaveletError(DWTWatermarkError):
    pass


class Subband(str, Enum):
    LL = "LL"
    LH = "LH"
    HL = "HL"
    HH = "HH"


_DETAIL_INDEX = {Subband.LH: 0, Subband.HL: 1, Subband.HH: 2}


@dataclass
class DWTWatermarkConfig:
    wavelet: str = "haar"
    level: int = 1
    subband: Subband = Subband.LL
    alpha: float = 0.05
    mode: str = "periodization"
    embed_on_luminance: bool = True

    def __post_init__(self) -> None:
        if self.wavelet not in pywt.wavelist(kind="discrete"):
            raise UnsupportedWaveletError(
                f"Wavelet '{self.wavelet}' không được PyWavelets hỗ trợ. Xem danh sách hợp lệ bằng pywt.wavelist(kind='discrete')."
            )
        if not isinstance(self.level, int) or self.level < 1:
            raise ValueError("`level` phải là số nguyên >= 1.")
        if isinstance(self.subband, str):
            self.subband = Subband(self.subband.upper())
        if self.alpha <= 0:
            raise ValueError("`alpha` phải là số dương (embedding strength > 0).")


@dataclass
class EmbedResult:
    stego_image: np.ndarray
    watermark_shape: Tuple[int, int]
    subband_shape: Tuple[int, int]
    config: DWTWatermarkConfig
    embedded_channel: str = "GRAY"


@dataclass
class ExtractResult:
    watermark: np.ndarray
    nc_score: Optional[float] = None
    bit_accuracy: Optional[float] = None


def _pad_to_multiple(
    channel: np.ndarray, factor: int
) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = channel.shape[:2]
    pad_h = -h % factor
    pad_w = -w % factor
    if pad_h == 0 and pad_w == 0:
        return (channel, (h, w))
    padded = cv2.copyMakeBorder(
        channel, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REPLICATE
    )
    return (padded, (h, w))


def _unpad(channel: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
    h, w = original_shape
    return channel[:h, :w]


def _get_detail_triplet(
    coeffs: list, level: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_level = len(coeffs) - 1
    if not 1 <= level <= total_level:
        raise InvalidSubbandError(
            f"level={level} không hợp lệ, ảnh chỉ được phân rã {total_level} mức."
        )
    idx = total_level - level + 1
    return coeffs[idx]


def _set_detail_triplet(
    coeffs: list, level: int, new_triplet: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> list:
    total_level = len(coeffs) - 1
    idx = total_level - level + 1
    new_coeffs = list(coeffs)
    new_coeffs[idx] = new_triplet
    return new_coeffs


def load_binary_watermark(
    source: Union[str, Path, np.ndarray],
    target_shape: Optional[Tuple[int, int]] = None,
    threshold: int = 127,
) -> np.ndarray:
    if isinstance(source, (str, Path)):
        img = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Không đọc được ảnh watermark tại: {source}")
    else:
        img = source
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)
    if target_shape is not None and binary.shape != tuple(target_shape):
        resized = cv2.resize(
            binary, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST
        )
        _, binary = cv2.threshold(resized, 0, 1, cv2.THRESH_BINARY)
        binary = binary.astype(np.uint8)
    return binary


def _to_bipolar(watermark_binary: np.ndarray) -> np.ndarray:
    return np.where(watermark_binary > 0, 1.0, -1.0).astype(np.float64)


def compute_nc(
    watermark_original: np.ndarray, watermark_extracted: np.ndarray
) -> float:
    if watermark_original.shape != watermark_extracted.shape:
        raise ShapeMismatchError(
            f"Kích thước không khớp: gốc {watermark_original.shape} vs trích xuất {watermark_extracted.shape}."
        )
    w1 = _to_bipolar(watermark_original)
    w2 = _to_bipolar(watermark_extracted)
    numerator = float(np.sum(w1 * w2))
    denominator = float(np.sqrt(np.sum(w1**2) * np.sum(w2**2)))
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def compute_bit_accuracy(
    watermark_original: np.ndarray, watermark_extracted: np.ndarray
) -> float:
    if watermark_original.shape != watermark_extracted.shape:
        raise ShapeMismatchError("Kích thước watermark gốc và trích xuất không khớp.")
    matches = np.sum(
        watermark_original.astype(np.uint8) == watermark_extracted.astype(np.uint8)
    )
    return 100.0 * matches / watermark_original.size


class DWTWatermarker:

    def __init__(self, config: Optional[DWTWatermarkConfig] = None) -> None:
        self.config = config or self.default_config()
        logger.debug("Khởi tạo DWTWatermarker với cấu hình: %s", self.config)

    @staticmethod
    def default_config() -> DWTWatermarkConfig:
        return DWTWatermarkConfig(
            wavelet="haar",
            level=1,
            subband=Subband.LL,
            alpha=0.05,
            mode="periodization",
            embed_on_luminance=True,
        )

    def embed(self, cover_image: np.ndarray, watermark: np.ndarray) -> EmbedResult:
        if cover_image is None or cover_image.ndim not in (2, 3):
            raise ValueError(
                "cover_image phải là mảng ảnh 2D (xám) hoặc 3D (màu, BGR)."
            )
        watermark_binary = self._normalize_watermark_input(watermark)
        is_color = cover_image.ndim == 3 and cover_image.shape[2] == 3
        if is_color and self.config.embed_on_luminance:
            ycrcb = cv2.cvtColor(cover_image, cv2.COLOR_BGR2YCrCb)
            y_channel, cr, cb = cv2.split(ycrcb)
            watermarked_y, subband_shape = self._embed_channel(
                y_channel, watermark_binary
            )
            stego_ycrcb = cv2.merge([watermarked_y, cr, cb])
            stego_image = cv2.cvtColor(stego_ycrcb, cv2.COLOR_YCrCb2BGR)
            embedded_channel = "Y"
        elif is_color:
            channels = cv2.split(cover_image)
            watermarked_channels = []
            subband_shape = None
            for ch in channels:
                wch, subband_shape = self._embed_channel(ch, watermark_binary)
                watermarked_channels.append(wch)
            stego_image = cv2.merge(watermarked_channels)
            embedded_channel = "BGR-ALL"
        else:
            stego_image, subband_shape = self._embed_channel(
                cover_image, watermark_binary
            )
            embedded_channel = "GRAY"
        logger.info(
            "Đã nhúng watermark %s vào subband %s (level=%d, alpha=%.4f). Kích thước subband: %s",
            watermark_binary.shape,
            self.config.subband.value,
            self.config.level,
            self.config.alpha,
            subband_shape,
        )
        return EmbedResult(
            stego_image=stego_image,
            watermark_shape=watermark_binary.shape,
            subband_shape=subband_shape,
            config=self.config,
            embedded_channel=embedded_channel,
        )

    def extract(
        self,
        received_image: np.ndarray,
        original_cover_image: np.ndarray,
        watermark_shape: Tuple[int, int],
        original_watermark: Optional[np.ndarray] = None,
    ) -> ExtractResult:
        if received_image.shape != original_cover_image.shape:
            raise ShapeMismatchError(
                "Ảnh nhận được và ảnh cover gốc phải cùng kích thước. Nếu ảnh vừa bị tấn công crop/resize, hãy khôi phục kích thước (vd: pad/resize) trước khi trích xuất."
            )
        is_color = received_image.ndim == 3 and received_image.shape[2] == 3
        if is_color and self.config.embed_on_luminance:
            recv_y = cv2.cvtColor(received_image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
            orig_y = cv2.cvtColor(original_cover_image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
            watermark_extracted = self._extract_channel(recv_y, orig_y, watermark_shape)
        elif is_color:
            votes = []
            for c in range(3):
                votes.append(
                    self._extract_channel(
                        received_image[:, :, c],
                        original_cover_image[:, :, c],
                        watermark_shape,
                    ).astype(np.int32)
                )
            stacked = np.stack(votes, axis=0)
            watermark_extracted = (stacked.mean(axis=0) >= 0.5).astype(np.uint8)
        else:
            watermark_extracted = self._extract_channel(
                received_image, original_cover_image, watermark_shape
            )
        result = ExtractResult(watermark=watermark_extracted)
        if original_watermark is not None:
            result.nc_score = compute_nc(original_watermark, watermark_extracted)
            result.bit_accuracy = compute_bit_accuracy(
                original_watermark, watermark_extracted
            )
            logger.info(
                "Trích xuất watermark: NC=%.4f | Bit-accuracy=%.2f%%",
                result.nc_score,
                result.bit_accuracy,
            )
        return result

    def _embed_channel(
        self, channel: np.ndarray, watermark_binary: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        factor = 2**self.config.level
        channel_f = channel.astype(np.float64)
        padded, original_shape = _pad_to_multiple(channel_f, factor)
        coeffs = pywt.wavedec2(
            padded,
            wavelet=self.config.wavelet,
            level=self.config.level,
            mode=self.config.mode,
        )
        target = self._read_subband(coeffs)
        subband_shape = target.shape
        watermark_resized = self._resize_watermark_to(watermark_binary, subband_shape)
        bipolar_wm = _to_bipolar(watermark_resized)
        modified_subband = target + self.config.alpha * bipolar_wm
        new_coeffs = self._write_subband(coeffs, modified_subband)
        reconstructed = pywt.waverec2(
            new_coeffs, wavelet=self.config.wavelet, mode=self.config.mode
        )
        reconstructed = _unpad(reconstructed, original_shape)
        stego_channel = np.clip(np.round(reconstructed), 0, 255).astype(np.uint8)
        return (stego_channel, subband_shape)

    def _extract_channel(
        self,
        received_channel: np.ndarray,
        original_channel: np.ndarray,
        watermark_shape: Tuple[int, int],
    ) -> np.ndarray:
        factor = 2**self.config.level
        recv_f = received_channel.astype(np.float64)
        orig_f = original_channel.astype(np.float64)
        recv_padded, recv_shape = _pad_to_multiple(recv_f, factor)
        orig_padded, orig_shape = _pad_to_multiple(orig_f, factor)
        assert (
            recv_shape == orig_shape
        ), "Ảnh nhận được và ảnh gốc phải cùng kích thước."
        recv_coeffs = pywt.wavedec2(
            recv_padded,
            wavelet=self.config.wavelet,
            level=self.config.level,
            mode=self.config.mode,
        )
        orig_coeffs = pywt.wavedec2(
            orig_padded,
            wavelet=self.config.wavelet,
            level=self.config.level,
            mode=self.config.mode,
        )
        recv_subband = self._read_subband(recv_coeffs)
        orig_subband = self._read_subband(orig_coeffs)
        diff = (recv_subband - orig_subband) / self.config.alpha
        diff_resized = cv2.resize(
            diff.astype(np.float32),
            (watermark_shape[1], watermark_shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        watermark_extracted = (diff_resized >= 0).astype(np.uint8)
        return watermark_extracted

    def _read_subband(self, coeffs: list) -> np.ndarray:
        subband = self.config.subband
        if subband == Subband.LL:
            return coeffs[0]
        cH, cV, cD = _get_detail_triplet(coeffs, self.config.level)
        triplet = {Subband.LH: cH, Subband.HL: cV, Subband.HH: cD}
        return triplet[subband]

    def _write_subband(self, coeffs: list, new_values: np.ndarray) -> list:
        subband = self.config.subband
        new_coeffs = list(coeffs)
        if subband == Subband.LL:
            new_coeffs[0] = new_values
            return new_coeffs
        cH, cV, cD = _get_detail_triplet(coeffs, self.config.level)
        triplet = [cH, cV, cD]
        triplet[_DETAIL_INDEX[subband]] = new_values
        new_coeffs = _set_detail_triplet(new_coeffs, self.config.level, tuple(triplet))
        return new_coeffs

    @staticmethod
    def _normalize_watermark_input(watermark: np.ndarray) -> np.ndarray:
        if watermark.ndim == 3:
            watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2GRAY)
        if watermark.dtype != np.uint8 or watermark.max() > 1:
            _, watermark = cv2.threshold(
                watermark.astype(np.uint8), 127, 1, cv2.THRESH_BINARY
            )
        return watermark.astype(np.uint8)

    @staticmethod
    def _resize_watermark_to(
        watermark_binary: np.ndarray, target_shape: Tuple[int, int]
    ) -> np.ndarray:
        if watermark_binary.shape == target_shape:
            return watermark_binary
        resized = cv2.resize(
            watermark_binary,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        _, resized = cv2.threshold(resized, 0, 1, cv2.THRESH_BINARY)
        return resized.astype(np.uint8)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Nhúng / Trích xuất thủy vân số DWT cho ảnh (module core/dwt_watermark.py)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    p_embed = subparsers.add_parser("embed", help="Nhúng watermark vào ảnh cover.")
    p_embed.add_argument("--cover", required=True, help="Đường dẫn ảnh cover đầu vào.")
    p_embed.add_argument(
        "--watermark", required=True, help="Đường dẫn ảnh logo watermark."
    )
    p_embed.add_argument(
        "--output", required=True, help="Đường dẫn lưu ảnh đã nhúng (I_final)."
    )
    p_embed.add_argument("--alpha", type=float, default=0.05)
    p_embed.add_argument("--wavelet", default="haar")
    p_embed.add_argument("--subband", default="LL", choices=[s.value for s in Subband])
    p_embed.add_argument("--level", type=int, default=1)
    p_extract = subparsers.add_parser(
        "extract", help="Trích xuất watermark từ ảnh đã nhúng."
    )
    p_extract.add_argument(
        "--received", required=True, help="Ảnh đã nhúng (có thể đã bị tấn công)."
    )
    p_extract.add_argument(
        "--cover", required=True, help="Ảnh cover gốc (tham chiếu, chưa nhúng)."
    )
    p_extract.add_argument(
        "--watermark-ref", required=False, help="Watermark gốc để tính NC (tùy chọn)."
    )
    p_extract.add_argument(
        "--width", type=int, required=True, help="Chiều rộng watermark gốc."
    )
    p_extract.add_argument(
        "--height", type=int, required=True, help="Chiều cao watermark gốc."
    )
    p_extract.add_argument(
        "--output", required=True, help="Đường dẫn lưu watermark trích xuất."
    )
    p_extract.add_argument("--alpha", type=float, default=0.05)
    p_extract.add_argument("--wavelet", default="haar")
    p_extract.add_argument(
        "--subband", default="LL", choices=[s.value for s in Subband]
    )
    p_extract.add_argument("--level", type=int, default=1)
    return parser


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = _build_arg_parser().parse_args()
    config = DWTWatermarkConfig(
        wavelet=args.wavelet,
        level=args.level,
        subband=Subband(args.subband),
        alpha=args.alpha,
    )
    watermarker = DWTWatermarker(config)
    if args.command == "embed":
        cover = cv2.imread(args.cover, cv2.IMREAD_UNCHANGED)
        if cover is None:
            raise FileNotFoundError(f"Không đọc được ảnh cover: {args.cover}")
        watermark = load_binary_watermark(args.watermark)
        result = watermarker.embed(cover, watermark)
        cv2.imwrite(args.output, result.stego_image)
        logger.info("Đã lưu ảnh watermarked tại: %s", args.output)
        logger.info(
            "Watermark shape=%s | Subband shape=%s | Kênh nhúng=%s",
            result.watermark_shape,
            result.subband_shape,
            result.embedded_channel,
        )
    elif args.command == "extract":
        received = cv2.imread(args.received, cv2.IMREAD_UNCHANGED)
        cover = cv2.imread(args.cover, cv2.IMREAD_UNCHANGED)
        if received is None or cover is None:
            raise FileNotFoundError("Không đọc được ảnh 'received' hoặc 'cover'.")
        original_watermark = None
        if args.watermark_ref:
            original_watermark = load_binary_watermark(
                args.watermark_ref, target_shape=(args.height, args.width)
            )
        result = watermarker.extract(
            received,
            cover,
            watermark_shape=(args.height, args.width),
            original_watermark=original_watermark,
        )
        cv2.imwrite(args.output, result.watermark * 255)
        logger.info("Đã lưu watermark trích xuất tại: %s", args.output)
        if result.nc_score is not None:
            logger.info(
                "NC = %.4f | Bit-accuracy = %.2f%%",
                result.nc_score,
                result.bit_accuracy,
            )


if __name__ == "__main__":
    _main()