"""
Streamlit application for steganography and digital watermarking.
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st
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

from evaluation.steganalysis import analyze

from evaluation.attacks import (
    apply_attack,
)


st.set_page_config(
    page_title="Stego & Watermark",
    page_icon="🔐",
    layout="wide",
)


def load_image(
    uploaded_file,
) -> Image.Image | None:
    """Load uploaded image."""

    if uploaded_file is None:
        return None

    try:
        return Image.open(
            uploaded_file
        ).convert("RGB")
    except Exception:
        st.error(
            "Không thể đọc file ảnh."
        )
        return None


def image_download_button(
    image: Image.Image,
    filename: str,
    label: str,
) -> None:
    """Create image download button."""

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    st.download_button(
        label=label,
        data=buffer.getvalue(),
        file_name=filename,
        mime="image/png",
    )


def show_image_comparison(
    original: Image.Image,
    processed: Image.Image,
) -> None:
    """Display original and processed images."""

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            original,
            caption="Ảnh gốc",
            use_container_width=True,
        )

    with col2:
        st.image(
            processed,
            caption="Ảnh sau xử lý",
            use_container_width=True,
        )


def show_metrics(
    original: Image.Image,
    processed: Image.Image,
) -> None:
    """Display image quality metrics."""

    metrics = calculate_metrics(
        original,
        processed,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "MSE",
            f"{metrics['MSE']:.6f}",
        )

    with col2:
        if metrics["PSNR"] == float("inf"):
            value = "∞"
        else:
            value = (
                f"{metrics['PSNR']:.2f} dB"
            )

        st.metric(
            "PSNR",
            value,
        )

    with col3:
        st.metric(
            "SSIM",
            f"{metrics['SSIM']:.6f}",
        )


def steganography_tab() -> None:
    """Steganography interface."""

    st.header(
        "🔐 Giấu tin"
    )

    st.write(
        "Mã hóa dữ liệu bằng AES và "
        "giấu vào ảnh bằng thuật toán "
        "steganography được chọn."
    )

    uploaded_file = st.file_uploader(
        "Chọn ảnh",
        type=[
            "png",
            "jpg",
            "jpeg",
            "bmp",
        ],
        key="stego_image",
    )

    if uploaded_file is None:
        st.info(
            "Vui lòng chọn một ảnh."
        )
        return

    image = load_image(
        uploaded_file
    )

    if image is None:
        return

    algorithm = st.selectbox(
        "Thuật toán",
        [
            "LSB",
            "PVD",
            "DCT",
        ],
        key="stego_algorithm",
    )

    secret = st.text_area(
        "Nội dung bí mật",
        height=150,
    )

    password = st.text_input(
        "Mật khẩu AES",
        type="password",
    )

    if not st.button(
        "🔒 Giấu tin",
        type="primary",
    ):
        return

    if not secret:
        st.warning(
            "Vui lòng nhập nội dung bí mật."
        )
        return

    if not password:
        st.warning(
            "Vui lòng nhập mật khẩu."
        )
        return

    try:
        if algorithm == "LSB":
            stego = LSBStego()

        elif algorithm == "PVD":
            stego = PVDStego()

        else:
            stego = DCTStego()

        from core.aes_cipher import (
            AESCipher,
        )

        cipher = AESCipher(
            password
        )

        encrypted = cipher.encrypt(
            secret
        )

        payload = encrypted.encode(
            "utf-8"
        )

        result = stego.embed(
            image,
            payload,
        )

        st.success(
            "Giấu tin thành công."
        )

        show_image_comparison(
            image,
            result,
        )

        show_metrics(
            image,
            result,
        )

        image_download_button(
            result,
            "stego_image.png",
            "⬇️ Tải ảnh đã giấu tin",
        )

        st.session_state[
            "stego_algorithm"
        ] = algorithm

    except Exception as exc:
        st.error(
            f"Không thể giấu tin: {exc}"
        )


def extraction_tab() -> None:
    """Extraction interface."""

    st.header(
        "🔓 Giải mã & trích xuất"
    )

    uploaded_file = st.file_uploader(
        "Chọn ảnh đã giấu tin",
        type=[
            "png",
            "bmp",
            "jpg",
            "jpeg",
        ],
        key="extract_image",
    )

    if uploaded_file is None:
        st.info(
            "Vui lòng chọn ảnh."
        )
        return

    image = load_image(
        uploaded_file
    )

    if image is None:
        return

    algorithm = st.selectbox(
        "Thuật toán",
        [
            "LSB",
            "PVD",
            "DCT",
        ],
        key="extract_algorithm",
    )

    password = st.text_input(
        "Mật khẩu AES",
        type="password",
        key="extract_password",
    )

    if not st.button(
        "🔓 Trích xuất",
        type="primary",
    ):
        return

    if not password:
        st.warning(
            "Vui lòng nhập mật khẩu."
        )
        return

    try:
        if algorithm == "LSB":
            stego = LSBStego()

        elif algorithm == "PVD":
            stego = PVDStego()

        else:
            stego = DCTStego()

        payload = stego.extract(
            image
        )

        encrypted = payload.decode(
            "utf-8"
        )

        from core.aes_cipher import (
            AESCipher,
        )

        cipher = AESCipher(
            password
        )

        secret = cipher.decrypt(
            encrypted
        )

        st.success(
            "Trích xuất thành công."
        )

        st.text_area(
            "Nội dung đã giải mã",
            secret,
            height=200,
        )

    except Exception as exc:
        st.error(
            f"Không thể trích xuất: {exc}"
        )


def watermark_tab() -> None:
    """Digital watermark interface."""

    st.header(
        "©️ Thủy vân số"
    )

    image_file = st.file_uploader(
        "Chọn ảnh gốc",
        type=[
            "png",
            "jpg",
            "jpeg",
            "bmp",
        ],
        key="watermark_image",
    )

    watermark_file = st.file_uploader(
        "Chọn watermark",
        type=[
            "png",
            "jpg",
            "jpeg",
            "bmp",
        ],
        key="watermark_file",
    )

    algorithm = st.selectbox(
        "Thuật toán thủy vân",
        [
            "DWT",
            "DWT-SVD",
        ],
    )

    alpha = st.slider(
        "Alpha",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01,
    )

    if image_file is None:
        st.info(
            "Vui lòng chọn ảnh gốc."
        )
        return

    if watermark_file is None:
        st.info(
            "Vui lòng chọn watermark."
        )
        return

    image = load_image(
        image_file
    )

    watermark = load_image(
        watermark_file
    )

    if image is None or watermark is None:
        return

    if not st.button(
        "©️ Nhúng watermark",
        type="primary",
    ):
        return

    try:
        if algorithm == "DWT":
            watermarking = (
                DWTWatermark(alpha)
            )
        else:
            watermarking = (
                DWTSVDWatermark(alpha)
            )

        result = watermarking.embed(
            image,
            watermark,
        )

        st.success(
            "Nhúng watermark thành công."
        )

        show_image_comparison(
            image,
            result,
        )

        show_metrics(
            image,
            result,
        )

        image_download_button(
            result,
            "watermarked_image.png",
            "⬇️ Tải ảnh thủy vân",
        )

        extracted = watermarking.extract(
            image,
            result,
            watermark.size,
        )

        st.subheader(
            "Watermark trích xuất"
        )

        st.image(
            extracted,
            caption="Watermark",
            width=300,
        )

        try:
            watermark_metrics = (
                calculate_watermark_metrics(
                    watermark,
                    extracted,
                )
            )

            st.metric(
                "NC",
                f"{watermark_metrics['NC']:.6f}",
            )

        except Exception:
            pass

    except Exception as exc:
        st.error(
            f"Không thể nhúng watermark: {exc}"
        )


def attack_tab() -> None:
    """Robustness testing interface."""

    st.header(
        "🧪 Kiểm tra độ bền"
    )

    uploaded_file = st.file_uploader(
        "Chọn ảnh cần kiểm tra",
        type=[
            "png",
            "jpg",
            "jpeg",
            "bmp",
        ],
        key="attack_image",
    )

    if uploaded_file is None:
        st.info(
            "Vui lòng chọn ảnh."
        )
        return

    image = load_image(
        uploaded_file
    )

    if image is None:
        return

    attack = st.selectbox(
        "Attack",
        [
            "jpeg",
            "gaussian_noise",
            "salt_pepper",
            "gaussian_blur",
            "median_blur",
            "resize",
            "crop",
            "rotate",
            "sharpen",
        ],
    )

    if attack == "jpeg":
        quality = st.slider(
            "JPEG Quality",
            10,
            100,
            75,
        )

        kwargs = {
            "quality": quality,
        }

    elif attack == "gaussian_noise":
        sigma = st.slider(
            "Sigma",
            1.0,
            50.0,
            10.0,
        )

        kwargs = {
            "sigma": sigma,
        }

    elif attack == "salt_pepper":
        amount = st.slider(
            "Amount",
            0.001,
            0.10,
            0.01,
        )

        kwargs = {
            "amount": amount,
        }

    elif attack == "gaussian_blur":
        kernel = st.selectbox(
            "Kernel size",
            [3, 5, 7, 9],
        )

        kwargs = {
            "kernel_size": kernel,
        }

    elif attack == "median_blur":
        kernel = st.selectbox(
            "Kernel size",
            [3, 5, 7],
        )

        kwargs = {
            "kernel_size": kernel,
        }

    elif attack == "resize":
        scale = st.slider(
            "Scale",
            0.1,
            1.0,
            0.5,
        )

        kwargs = {
            "scale": scale,
        }

    elif attack == "crop":
        ratio = st.slider(
            "Crop ratio",
            0.1,
            1.0,
            0.8,
        )

        kwargs = {
            "crop_ratio": ratio,
        }

    elif attack == "rotate":
        angle = st.slider(
            "Angle",
            -45.0,
            45.0,
            5.0,
        )

        kwargs = {
            "angle": angle,
        }

    else:
        strength = st.slider(
            "Strength",
            0.1,
            3.0,
            1.5,
        )

        kwargs = {
            "strength": strength,
        }

    if not st.button(
        "🧪 Thực hiện attack",
        type="primary",
    ):
        return

    try:
        attacked = apply_attack(
            image,
            attack,
            **kwargs,
        )

        st.success(
            "Attack hoàn tất."
        )

        show_image_comparison(
            image,
            attacked,
        )

        if image.size == attacked.size:
            show_metrics(
                image,
                attacked,
            )

        image_download_button(
            attacked,
            "attacked_image.png",
            "⬇️ Tải ảnh sau attack",
        )

    except Exception as exc:
        st.error(
            f"Attack thất bại: {exc}"
        )


def steganalysis_tab() -> None:
    """Steganalysis interface."""

    st.header(
        "🔍 Steganalysis"
    )

    uploaded_file = st.file_uploader(
        "Chọn ảnh để phân tích",
        type=[
            "png",
            "jpg",
            "jpeg",
            "bmp",
        ],
        key="analysis_image",
    )

    if uploaded_file is None:
        st.info(
            "Vui lòng chọn ảnh."
        )
        return

    image = load_image(
        uploaded_file
    )

    if image is None:
        return

    if not st.button(
        "🔍 Phân tích",
        type="primary",
    ):
        return

    try:
        results = analyze(
            image
        )

        dataframe = pd.DataFrame(
            {
                "Metric": results.keys(),
                "Value": results.values(),
            }
        )

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "LSB Ratio",
                f"{results['LSB ratio']:.4f}",
            )

        with col2:
            st.metric(
                "LSB Entropy",
                f"{results['LSB entropy']:.4f}",
            )

    except Exception as exc:
        st.error(
            f"Phân tích thất bại: {exc}"
        )


def main() -> None:
    """Run Streamlit application."""

    st.title(
        "🔐 Hệ thống giấu tin và thủy vân số"
    )

    st.caption(
        "LSB • PVD • DCT • DWT • DWT-SVD • AES"
    )

    tabs = st.tabs(
        [
            "🔐 Giấu tin",
            "🔓 Trích xuất",
            "©️ Thủy vân",
            "🧪 Robustness",
            "🔍 Steganalysis",
        ]
    )

    with tabs[0]:
        steganography_tab()

    with tabs[1]:
        extraction_tab()

    with tabs[2]:
        watermark_tab()

    with tabs[3]:
        attack_tab()

    with tabs[4]:
        steganalysis_tab()


if __name__ == "__main__":
    main()