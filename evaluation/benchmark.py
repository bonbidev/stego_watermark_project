"""
Benchmark steganography and watermarking algorithms.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from PIL import Image

from core.lsb_stego import LSBStego
from core.pvd_stego import PVDStego
from core.dct_stego import DCTStego
from core.dwt_watermark import DWTWatermark
from core.dwt_svd_watermark import DWTSVDWatermark

from evaluation.metrics import (
    calculate_metrics,
    calculate_watermark_metrics,
)
from evaluation.attacks import apply_attack


@dataclass
class BenchmarkResult:
    """Store benchmark result."""

    algorithm: str
    operation: str
    success: bool
    execution_time: float
    payload_size: int
    mse: float | None = None
    psnr: float | None = None
    ssim: float | None = None
    nc: float | None = None
    error: str | None = None


class Benchmark:
    """Run benchmark tests."""

    def __init__(
        self,
        image: Image.Image,
        secret: bytes | None = None,
        watermark: Image.Image | None = None,
    ):
        self.image = image
        self.secret = secret
        self.watermark = watermark

        self.results: list[BenchmarkResult] = []

    def _add_result(
        self,
        result: BenchmarkResult,
    ) -> None:
        self.results.append(result)

    def benchmark_lsb(self) -> None:
        """Benchmark LSB steganography."""

        if self.secret is None:
            return

        algorithm = LSBStego()

        start = time.perf_counter()

        try:
            stego = algorithm.embed(
                self.image,
                self.secret,
            )

            execution_time = (
                time.perf_counter() - start
            )

            metrics = calculate_metrics(
                self.image,
                stego,
            )

            extracted = algorithm.extract(
                stego
            )

            success = (
                extracted == self.secret
            )

            self._add_result(
                BenchmarkResult(
                    algorithm="LSB",
                    operation="Embedding",
                    success=success,
                    execution_time=execution_time,
                    payload_size=len(
                        self.secret
                    ),
                    mse=metrics["MSE"],
                    psnr=metrics["PSNR"],
                    ssim=metrics["SSIM"],
                )
            )

        except Exception as exc:
            self._add_result(
                BenchmarkResult(
                    algorithm="LSB",
                    operation="Embedding",
                    success=False,
                    execution_time=0,
                    payload_size=len(
                        self.secret
                    ),
                    error=str(exc),
                )
            )

    def benchmark_pvd(self) -> None:
        """Benchmark PVD steganography."""

        if self.secret is None:
            return

        algorithm = PVDStego()

        start = time.perf_counter()

        try:
            stego = algorithm.embed(
                self.image,
                self.secret,
            )

            execution_time = (
                time.perf_counter() - start
            )

            metrics = calculate_metrics(
                self.image,
                stego,
            )

            extracted = algorithm.extract(
                stego
            )

            success = (
                extracted == self.secret
            )

            self._add_result(
                BenchmarkResult(
                    algorithm="PVD",
                    operation="Embedding",
                    success=success,
                    execution_time=execution_time,
                    payload_size=len(
                        self.secret
                    ),
                    mse=metrics["MSE"],
                    psnr=metrics["PSNR"],
                    ssim=metrics["SSIM"],
                )
            )

        except Exception as exc:
            self._add_result(
                BenchmarkResult(
                    algorithm="PVD",
                    operation="Embedding",
                    success=False,
                    execution_time=0,
                    payload_size=len(
                        self.secret
                    ),
                    error=str(exc),
                )
            )

    def benchmark_dct(self) -> None:
        """Benchmark DCT steganography."""

        if self.secret is None:
            return

        algorithm = DCTStego()

        start = time.perf_counter()

        try:
            stego = algorithm.embed(
                self.image,
                self.secret,
            )

            execution_time = (
                time.perf_counter() - start
            )

            metrics = calculate_metrics(
                self.image,
                stego,
            )

            extracted = algorithm.extract(
                stego
            )

            success = (
                extracted == self.secret
            )

            self._add_result(
                BenchmarkResult(
                    algorithm="DCT",
                    operation="Embedding",
                    success=success,
                    execution_time=execution_time,
                    payload_size=len(
                        self.secret
                    ),
                    mse=metrics["MSE"],
                    psnr=metrics["PSNR"],
                    ssim=metrics["SSIM"],
                )
            )

        except Exception as exc:
            self._add_result(
                BenchmarkResult(
                    algorithm="DCT",
                    operation="Embedding",
                    success=False,
                    execution_time=0,
                    payload_size=len(
                        self.secret
                    ),
                    error=str(exc),
                )
            )

    def benchmark_dwt(self) -> None:
        """Benchmark DWT watermarking."""

        if self.watermark is None:
            return

        algorithm = DWTWatermark()

        start = time.perf_counter()

        try:
            watermarked = algorithm.embed(
                self.image,
                self.watermark,
            )

            execution_time = (
                time.perf_counter() - start
            )

            metrics = calculate_metrics(
                self.image,
                watermarked,
            )

            extracted = algorithm.extract(
                self.image,
                watermarked,
                self.watermark.size,
            )

            watermark_metrics = (
                calculate_watermark_metrics(
                    self.watermark,
                    extracted,
                )
            )

            self._add_result(
                BenchmarkResult(
                    algorithm="DWT",
                    operation="Watermarking",
                    success=True,
                    execution_time=execution_time,
                    payload_size=(
                        self.watermark.width
                        * self.watermark.height
                    ),
                    mse=metrics["MSE"],
                    psnr=metrics["PSNR"],
                    ssim=metrics["SSIM"],
                    nc=watermark_metrics["NC"],
                )
            )

        except Exception as exc:
            self._add_result(
                BenchmarkResult(
                    algorithm="DWT",
                    operation="Watermarking",
                    success=False,
                    execution_time=0,
                    payload_size=0,
                    error=str(exc),
                )
            )

    def benchmark_dwt_svd(self) -> None:
        """Benchmark DWT-SVD watermarking."""

        if self.watermark is None:
            return

        algorithm = DWTSVDWatermark()

        start = time.perf_counter()

        try:
            watermarked = algorithm.embed(
                self.image,
                self.watermark,
            )

            execution_time = (
                time.perf_counter() - start
            )

            metrics = calculate_metrics(
                self.image,
                watermarked,
            )

            extracted = algorithm.extract(
                self.image,
                watermarked,
                self.watermark.size,
            )

            watermark_metrics = (
                calculate_watermark_metrics(
                    self.watermark,
                    extracted,
                )
            )

            self._add_result(
                BenchmarkResult(
                    algorithm="DWT-SVD",
                    operation="Watermarking",
                    success=True,
                    execution_time=execution_time,
                    payload_size=(
                        self.watermark.width
                        * self.watermark.height
                    ),
                    mse=metrics["MSE"],
                    psnr=metrics["PSNR"],
                    ssim=metrics["SSIM"],
                    nc=watermark_metrics["NC"],
                )
            )

        except Exception as exc:
            self._add_result(
                BenchmarkResult(
                    algorithm="DWT-SVD",
                    operation="Watermarking",
                    success=False,
                    execution_time=0,
                    payload_size=0,
                    error=str(exc),
                )
            )

    def run_all(self) -> pd.DataFrame:
        """Run all benchmark tests."""

        self.benchmark_lsb()
        self.benchmark_pvd()
        self.benchmark_dct()
        self.benchmark_dwt()
        self.benchmark_dwt_svd()

        return self.to_dataframe()

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""

        return pd.DataFrame(
            [
                asdict(result)
                for result in self.results
            ]
        )

    def save_csv(
        self,
        path: str | Path,
    ) -> None:
        """Save benchmark results to CSV."""

        dataframe = self.to_dataframe()

        dataframe.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )

    def save_excel(
        self,
        path: str | Path,
    ) -> None:
        """Save benchmark results to Excel."""

        dataframe = self.to_dataframe()

        dataframe.to_excel(
            path,
            index=False,
        )


def run_benchmark(
    image: Image.Image,
    secret: bytes | None = None,
    watermark: Image.Image | None = None,
) -> pd.DataFrame:
    """Run all benchmark tests."""

    benchmark = Benchmark(
        image=image,
        secret=secret,
        watermark=watermark,
    )

    return benchmark.run_all()


if __name__ == "__main__":
    image = Image.open(
        "input.png"
    ).convert("L")

    watermark = Image.open(
        "watermark.png"
    ).convert("L")

    secret = (
        b"Benchmark secret message"
    )

    benchmark = Benchmark(
        image=image,
        secret=secret,
        watermark=watermark,
    )

    results = benchmark.run_all()

    print(
        results.to_string(
            index=False
        )
    )

    benchmark.save_csv(
        "benchmark_results.csv"
    )

    print(
        "\nBenchmark completed."
    )