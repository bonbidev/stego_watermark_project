"""
Streamlit application for steganography and digital watermarking.
"""

from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from core.lsb_stego import LSBStego, LSBStegoError
from core.pvd_stego import PVDStego, PVDStegoError
from core.dct_stego import DCTStego, DCTStegoError
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
    elapsed_seconds: float | None = None,
) -> dict:
    """Display image quality metrics with a plain-language read on
    whether the change would be noticeable, plus timing if given.
    Returns the raw metrics dict so callers can reuse it (e.g. for
    the comparison tab) instead of recomputing."""

    # PVD/DCT/DWT/DWT-SVD all convert their output to grayscale
    # internally, while `original` is loaded as RGB. Comparing an
    # (H, W, 3) array against an (H, W) array makes
    # calculate_metrics() raise "Images must have the same shape.",
    # which previously made every one of those algorithms look
    # broken in the UI even though embedding succeeded. Compare like
    # for like by matching `original`'s mode to `processed`'s mode.
    if original.mode != processed.mode:
        original = original.convert(processed.mode)

    metrics = calculate_metrics(
        original,
        processed,
    )

    cols = st.columns(4 if elapsed_seconds is not None else 3)

    with cols[0]:
        st.metric(
            "MSE",
            f"{metrics['MSE']:.6f}",
        )

    with cols[1]:
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

    with cols[2]:
        st.metric(
            "SSIM",
            f"{metrics['SSIM']:.6f}",
        )

    if elapsed_seconds is not None:
        with cols[3]:
            st.metric(
                "Thời gian",
                f"{elapsed_seconds * 1000:.0f} ms",
            )

    st.caption(interpret_quality(metrics["PSNR"], metrics["SSIM"]))

    return metrics


def interpret_quality(psnr: float, ssim: float) -> str:
    """Plain-language read of PSNR/SSIM, using commonly cited
    thresholds for "a casual viewer would not notice the change"."""

    if psnr == float("inf"):
        return "🟢 Ảnh giống hệt bản gốc (không có sai khác nào về mặt pixel)."

    if psnr >= 40 and ssim >= 0.98:
        return "🟢 Chất lượng rất tốt — mắt thường gần như không thể phân biệt được với ảnh gốc."
    if psnr >= 30 and ssim >= 0.90:
        return "🟡 Chất lượng khá tốt — sai khác rất nhỏ, khó nhận ra nếu không so sánh trực tiếp."
    if psnr >= 20:
        return "🟠 Chất lượng trung bình — có thể nhận thấy khác biệt nếu nhìn kỹ hoặc phóng to."
    return "🔴 Chất lượng thấp — ảnh bị biến dạng rõ rệt, dễ nhận ra bằng mắt thường."


def interpret_nc(nc: float) -> str:
    """Plain-language read of Normalized Correlation for watermark
    extraction quality."""

    if nc >= 0.9:
        return "🟢 Rất tốt — watermark trích xuất gần như giống hệt bản gốc."
    if nc >= 0.7:
        return "🟡 Khá tốt — watermark trích xuất vẫn nhận diện được rõ ràng."
    if nc >= 0.5:
        return "🟠 Trung bình — watermark còn nhận ra được nhưng đã suy giảm nhiều."
    return "🔴 Thấp — watermark trích xuất bị méo nặng, khó/không nhận diện được."


WATERMARK_ALGORITHM_INFO = {
    "DWT": {
        "class": DWTWatermark,
        "desc": "Nhúng watermark trực tiếp vào hệ số subband LH của biến đổi wavelet rời rạc. Đơn giản, NC thường cao (0.8-1.0).",
        "robustness": "🟢 Tốt trước nhiễu/nén nhẹ",
    },
    "DWT-SVD": {
        "class": DWTSVDWatermark,
        "desc": "Kết hợp DWT với phân tích giá trị kỳ dị (SVD). Về lý thuyết bền hơn DWT thuần, nhưng cách trích xuất trong bản này chỉ khôi phục đường chéo ma trận nên NC thực tế thường THẤP hơn DWT (0.01-0.3).",
        "robustness": "🟡 Lý thuyết cao hơn DWT, nhưng cách cài đặt hiện tại cho NC thấp hơn — xem tab So sánh",
    },
}


def show_diff_heatmap(
    original: Image.Image,
    processed: Image.Image,
    amplify: float = 10.0,
) -> None:
    """Show an amplified |original - processed| heatmap so the user
    can *see* where an algorithm changed the image, instead of only
    reading a single aggregate number (PSNR/SSIM/MSE)."""

    if original.mode != processed.mode:
        original = original.convert(processed.mode)

    original_array = np.asarray(original, dtype=np.float32)
    processed_array = np.asarray(processed, dtype=np.float32)

    diff = np.abs(original_array - processed_array)
    if diff.ndim == 3:
        diff = diff.mean(axis=2)

    amplified = np.clip(diff * amplify, 0, 255).astype("uint8")
    heatmap_image = Image.fromarray(amplified, mode="L")

    st.image(
        heatmap_image,
        caption=f"Bản đồ sai khác (đã khuếch đại x{amplify:.0f} — càng sáng càng thay đổi nhiều)",
        use_container_width=True,
    )


def capacity_bar(payload_bytes: int, capacity_bytes: int) -> None:
    """Show how much of the algorithm's capacity the payload uses."""

    if capacity_bytes <= 0:
        st.warning("Ảnh quá nhỏ, không đủ sức chứa cho thuật toán này.")
        return

    usage = min(1.0, payload_bytes / capacity_bytes)
    st.progress(
        usage,
        text=f"Dung lượng dùng: {payload_bytes:,} / {capacity_bytes:,} byte ({usage * 100:.1f}%)",
    )


STEGO_ALGORITHM_INFO = {
    "LSB": {
        "class": LSBStego,
        "desc": "Ghi trực tiếp vào bit thấp nhất của kênh Blue. Nhanh nhất, sức chứa lớn nhất, nhưng dễ vỡ nếu ảnh bị nén JPEG/resize/nhiễu.",
        "speed": "⚡⚡⚡ Rất nhanh",
        "capacity": "🟢🟢🟢 Rất lớn",
        "robustness": "🔴 Thấp",
    },
    "PVD": {
        "class": PVDStego,
        "desc": "Nhúng nhiều bit hơn ở vùng có độ tương phản cao (cạnh, chi tiết), ít bit hơn ở vùng phẳng. Cân bằng giữa sức chứa và độ ẩn giấu.",
        "speed": "⚡⚡ Trung bình",
        "capacity": "🟢🟢 Lớn",
        "robustness": "🟡 Trung bình",
    },
    "DCT": {
        "class": DCTStego,
        "desc": "Biến đổi mỗi khối 8x8 sang miền tần số (như JPEG) và mã hoá 1 bit/khối. Sức chứa thấp hơn nhiều nhưng bền hơn trước nén ảnh.",
        "speed": "⚡ Chậm hơn",
        "capacity": "🟠 Thấp (1 bit / khối 8x8)",
        "robustness": "🟢 Cao nhất trong 3 thuật toán",
    },
}


def _log_history(entry: dict) -> None:
    """Append one row to the session's operation history, shown at
    the bottom of each tab so the user can see what they've tried
    without re-running everything."""

    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].append(entry)


def show_history(filter_kind: str | None = None) -> None:
    rows = st.session_state.get("history", [])
    if filter_kind:
        rows = [r for r in rows if r.get("kind") == filter_kind]
    if not rows:
        return
    with st.expander(f"🕘 Lịch sử thao tác ({len(rows)})"):
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


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

    info = STEGO_ALGORITHM_INFO[algorithm]
    with st.expander(f"ℹ️ Về thuật toán {algorithm}", expanded=False):
        st.write(info["desc"])
        c1, c2, c3 = st.columns(3)
        c1.write(f"**Tốc độ**\n\n{info['speed']}")
        c2.write(f"**Sức chứa**\n\n{info['capacity']}")
        c3.write(f"**Độ bền**\n\n{info['robustness']}")

    stego_tool_preview = info["class"]()
    capacity = stego_tool_preview.get_capacity(image)
    st.caption(f"Sức chứa tối đa của ảnh này với {algorithm}: **{capacity:,} byte**")

    secret = st.text_area(
        "Nội dung bí mật",
        height=150,
    )

    password = st.text_input(
        "Mật khẩu AES",
        type="password",
    )
    if password and len(password) < 12:
        st.caption("⚠️ Mật khẩu ngắn hơn 12 ký tự — nên dùng mật khẩu dài hơn cho an toàn.")

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
        stego = info["class"]()

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

        capacity_bar(len(payload), capacity)
        if len(payload) > capacity:
            st.error(
                f"Nội dung ({len(payload):,} byte sau khi mã hoá) vượt quá sức chứa "
                f"({capacity:,} byte). Hãy dùng ảnh lớn hơn, rút ngắn nội dung, "
                f"hoặc đổi sang thuật toán khác."
            )
            return

        start = time.perf_counter()
        result = stego.embed(
            image,
            payload,
        )
        elapsed = time.perf_counter() - start

        st.success(
            "Giấu tin thành công."
        )

        show_image_comparison(
            image,
            result,
        )

        metrics = show_metrics(
            image,
            result,
            elapsed_seconds=elapsed,
        )

        with st.expander("🔍 Xem bản đồ sai khác (nơi thuật toán đã thay đổi ảnh)"):
            show_diff_heatmap(image, result)

        image_download_button(
            result,
            "stego_image.png",
            "⬇️ Tải ảnh đã giấu tin",
        )

        st.session_state[
            "stego_algorithm"
        ] = algorithm

        _log_history({
            "kind": "stego",
            "Thuật toán": algorithm,
            "Payload (byte)": len(payload),
            "Sức chứa (byte)": capacity,
            "PSNR (dB)": round(metrics["PSNR"], 2) if metrics["PSNR"] != float("inf") else "∞",
            "SSIM": round(metrics["SSIM"], 4),
            "Thời gian (ms)": round(elapsed * 1000, 1),
        })

    except (LSBStegoError, PVDStegoError, DCTStegoError) as exc:
        st.error(f"Không thể giấu tin ({algorithm}): {exc}")
    except Exception as exc:
        st.error(
            f"Không thể giấu tin: {exc}"
        )

    show_history(filter_kind="stego")


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

    wm_info = WATERMARK_ALGORITHM_INFO[algorithm]
    with st.expander(f"ℹ️ Về thuật toán {algorithm}", expanded=False):
        st.write(wm_info["desc"])
        st.write(f"**Độ bền:** {wm_info['robustness']}")

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

        start = time.perf_counter()
        result = watermarking.embed(
            image,
            watermark,
        )
        elapsed = time.perf_counter() - start

        st.success(
            "Nhúng watermark thành công."
        )

        show_image_comparison(
            image,
            result,
        )

        metrics = show_metrics(
            image,
            result,
            elapsed_seconds=elapsed,
        )

        with st.expander("🔍 Xem bản đồ sai khác (nơi thuật toán đã thay đổi ảnh)"):
            show_diff_heatmap(image, result)

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

        col1, col2 = st.columns(2)
        col1.image(
            watermark,
            caption="Watermark gốc",
            use_container_width=True,
        )
        col2.image(
            extracted,
            caption="Watermark trích xuất được",
            use_container_width=True,
        )

        nc_value = None
        try:
            # Same RGB-vs-grayscale mismatch as show_metrics() above:
            # `extracted` is always a grayscale ("L") image.
            watermark_for_nc = (
                watermark
                if watermark.mode == extracted.mode
                else watermark.convert(extracted.mode)
            )

            watermark_metrics = (
                calculate_watermark_metrics(
                    watermark_for_nc,
                    extracted,
                )
            )
            nc_value = watermark_metrics["NC"]

            st.metric(
                "NC (Normalized Correlation)",
                f"{nc_value:.4f}",
            )
            st.caption(interpret_nc(nc_value))

        except Exception:
            st.caption("Không tính được NC cho cặp ảnh này.")

        _log_history({
            "kind": "watermark",
            "Thuật toán": algorithm,
            "Alpha": alpha,
            "PSNR (dB)": round(metrics["PSNR"], 2) if metrics["PSNR"] != float("inf") else "∞",
            "SSIM": round(metrics["SSIM"], 4),
            "NC": round(nc_value, 4) if nc_value is not None else "—",
            "Thời gian (ms)": round(elapsed * 1000, 1),
        })

    except Exception as exc:
        st.error(
            f"Không thể nhúng watermark: {exc}"
        )

    show_history(filter_kind="watermark")


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


def comparison_tab() -> None:
    """Run every algorithm on the same input and compare them side
    by side, so the user can see which one is actually best for
    their image/use case instead of guessing from one run at a
    time."""

    st.header("📊 So sánh thuật toán")
    st.write(
        "Chạy tất cả thuật toán trên cùng một ảnh đầu vào để so sánh "
        "trực tiếp chất lượng, tốc độ và độ bền."
    )

    sub_tab_stego, sub_tab_watermark = st.tabs(
        ["🔐 So sánh giấu tin (LSB / PVD / DCT)", "©️ So sánh thủy vân (DWT / DWT-SVD)"]
    )

    # ------------------------------------------------------------------
    # Steganography comparison
    # ------------------------------------------------------------------
    with sub_tab_stego:
        uploaded_file = st.file_uploader(
            "Chọn ảnh", type=["png", "jpg", "jpeg", "bmp"], key="cmp_stego_image"
        )
        secret = st.text_area("Nội dung bí mật", height=100, key="cmp_stego_secret")
        password = st.text_input("Mật khẩu AES", type="password", key="cmp_stego_password")
        run_robustness_check = st.checkbox(
            "Kèm kiểm tra độ bền nhanh (nén JPEG + nhiễu Gaussian)",
            value=True,
            key="cmp_stego_robustness",
        )

        if st.button("📊 So sánh 3 thuật toán", type="primary", key="cmp_stego_run"):
            if uploaded_file is None or not secret or not password:
                st.warning("Vui lòng chọn ảnh, nhập nội dung bí mật và mật khẩu.")
            else:
                image = load_image(uploaded_file)
                if image is not None:
                    _run_stego_comparison(image, secret, password, run_robustness_check)

    # ------------------------------------------------------------------
    # Watermark comparison
    # ------------------------------------------------------------------
    with sub_tab_watermark:
        image_file = st.file_uploader(
            "Chọn ảnh gốc", type=["png", "jpg", "jpeg", "bmp"], key="cmp_wm_image"
        )
        watermark_file = st.file_uploader(
            "Chọn watermark", type=["png", "jpg", "jpeg", "bmp"], key="cmp_wm_watermark"
        )
        alpha = st.slider(
            "Alpha", min_value=0.01, max_value=0.20, value=0.05, step=0.01, key="cmp_wm_alpha"
        )

        if st.button("📊 So sánh DWT vs DWT-SVD", type="primary", key="cmp_wm_run"):
            if image_file is None or watermark_file is None:
                st.warning("Vui lòng chọn cả ảnh gốc và watermark.")
            else:
                image = load_image(image_file)
                watermark = load_image(watermark_file)
                if image is not None and watermark is not None:
                    _run_watermark_comparison(image, watermark, alpha)


def _run_stego_comparison(
    image: Image.Image,
    secret: str,
    password: str,
    run_robustness_check: bool,
) -> None:
    from core.aes_cipher import AESCipher

    cipher = AESCipher(password)
    encrypted = cipher.encrypt(secret)
    payload = encrypted.encode("utf-8")

    quick_attacks = [
        ("Nén JPEG (q=75)", "jpeg", {"quality": 75}),
        ("Nhiễu Gaussian (σ=10)", "gaussian_noise", {"sigma": 10.0}),
    ]

    rows = []
    stego_images: dict[str, Image.Image] = {}

    with st.spinner("Đang chạy LSB, PVD, DCT..."):
        for name, info in STEGO_ALGORITHM_INFO.items():
            tool = info["class"]()
            capacity = tool.get_capacity(image)
            row = {
                "Thuật toán": name,
                "Sức chứa (byte)": capacity,
                "Đủ chỗ?": "✅" if capacity >= len(payload) else "❌",
                "PSNR (dB)": None,
                "SSIM": None,
                "Thời gian (ms)": None,
            }

            if capacity < len(payload):
                row["Ghi chú"] = "Ảnh không đủ sức chứa cho nội dung này"
                rows.append(row)
                continue

            try:
                start = time.perf_counter()
                stego_image = tool.embed(image, payload)
                elapsed = time.perf_counter() - start

                metrics = calculate_metrics(
                    image if image.mode == stego_image.mode else image.convert(stego_image.mode),
                    stego_image,
                )

                row["PSNR (dB)"] = round(metrics["PSNR"], 2) if metrics["PSNR"] != float("inf") else float("inf")
                row["SSIM"] = round(metrics["SSIM"], 4)
                row["Thời gian (ms)"] = round(elapsed * 1000, 1)
                row["Ghi chú"] = ""
                stego_images[name] = stego_image

                if run_robustness_check:
                    for attack_label, attack_name, kwargs in quick_attacks:
                        try:
                            attacked = apply_attack(stego_image, attack_name, **kwargs)
                            extracted = tool.extract(attacked)
                            recovered = cipher.decrypt(extracted.decode("utf-8"))
                            row[attack_label] = "✅" if recovered == secret else "⚠️"
                        except Exception:
                            row[attack_label] = "❌"

            except Exception as exc:
                row["Ghi chú"] = f"Lỗi: {exc}"

            rows.append(row)

    df = pd.DataFrame(rows)
    st.subheader("Bảng so sánh")
    st.dataframe(df, use_container_width=True, hide_index=True)

    successful = [r for r in rows if r.get("PSNR (dB)") is not None]
    if successful:
        st.subheader("Biểu đồ chất lượng (PSNR)")
        chart_df = pd.DataFrame(
            {r["Thuật toán"]: [r["PSNR (dB)"] if r["PSNR (dB)"] != float("inf") else 100] for r in successful}
        ).T
        chart_df.columns = ["PSNR (dB)"]
        st.bar_chart(chart_df)

        st.subheader("👀 Xem ảnh kết quả")
        cols = st.columns(len(stego_images))
        for col, (name, img) in zip(cols, stego_images.items()):
            col.image(img, caption=name, use_container_width=True)

        # ---- Recommendation ----
        best_capacity = max(successful, key=lambda r: r["Sức chứa (byte)"])
        best_quality = max(
            successful,
            key=lambda r: r["PSNR (dB)"] if r["PSNR (dB)"] != float("inf") else 1e9,
        )
        fastest = min(successful, key=lambda r: r["Thời gian (ms)"])

        st.subheader("💡 Đề xuất")
        st.markdown(
            f"- **Sức chứa lớn nhất:** {best_capacity['Thuật toán']} "
            f"({best_capacity['Sức chứa (byte)']:,} byte)\n"
            f"- **Chất lượng ảnh tốt nhất (PSNR cao nhất):** {best_quality['Thuật toán']}\n"
            f"- **Nhanh nhất:** {fastest['Thuật toán']} ({fastest['Thời gian (ms)']} ms)"
        )
        if run_robustness_check:
            robust_ok = [
                r["Thuật toán"] for r in successful
                if r.get("Nén JPEG (q=75)") == "✅" or r.get("Nhiễu Gaussian (σ=10)") == "✅"
            ]
            if robust_ok:
                st.markdown(f"- **Sống sót qua ít nhất 1 phép tấn công:** {', '.join(robust_ok)}")
            else:
                st.markdown(
                    "- **Độ bền:** không thuật toán nào sống sót qua cả 2 phép tấn công thử "
                    "nghiệm — đây là hạn chế chung của giấu tin trong miền không gian/tần số "
                    "đơn giản; nếu cần độ bền cao trước nén ảnh, DCT thường là lựa chọn tốt nhất "
                    "trong 3 thuật toán này, nhưng vẫn có giới hạn."
                )
        st.caption(
            "Không có thuật toán nào 'tốt nhất' tuyệt đối — LSB thắng về tốc độ/sức chứa, "
            "PVD cân bằng, DCT bền hơn trước nén ảnh nhưng chứa được ít hơn nhiều."
        )


def _run_watermark_comparison(
    image: Image.Image,
    watermark: Image.Image,
    alpha: float,
) -> None:
    rows = []
    result_images: dict[str, Image.Image] = {}

    with st.spinner("Đang chạy DWT và DWT-SVD..."):
        for name, info in WATERMARK_ALGORITHM_INFO.items():
            tool = info["class"](alpha)
            row = {"Thuật toán": name}
            try:
                start = time.perf_counter()
                watermarked = tool.embed(image, watermark)
                elapsed = time.perf_counter() - start

                metrics = calculate_metrics(
                    image if image.mode == watermarked.mode else image.convert(watermarked.mode),
                    watermarked,
                )

                extracted = tool.extract(image, watermarked, watermark.size)
                watermark_for_nc = (
                    watermark if watermark.mode == extracted.mode else watermark.convert(extracted.mode)
                )
                wm_metrics = calculate_watermark_metrics(watermark_for_nc, extracted)

                row["PSNR (dB)"] = round(metrics["PSNR"], 2) if metrics["PSNR"] != float("inf") else float("inf")
                row["SSIM"] = round(metrics["SSIM"], 4)
                row["NC"] = round(wm_metrics["NC"], 4)
                row["Thời gian (ms)"] = round(elapsed * 1000, 1)
                row["Ghi chú"] = ""
                result_images[name] = watermarked

            except Exception as exc:
                row["Ghi chú"] = f"Lỗi: {exc}"

            rows.append(row)

    df = pd.DataFrame(rows)
    st.subheader("Bảng so sánh")
    st.dataframe(df, use_container_width=True, hide_index=True)

    successful = [r for r in rows if "NC" in r]
    if successful:
        st.subheader("Biểu đồ NC (độ giống watermark trích xuất)")
        chart_df = pd.DataFrame({r["Thuật toán"]: [r["NC"]] for r in successful}).T
        chart_df.columns = ["NC"]
        st.bar_chart(chart_df)

        st.subheader("👀 Xem ảnh kết quả")
        cols = st.columns(len(result_images))
        for col, (name, img) in zip(cols, result_images.items()):
            col.image(img, caption=name, use_container_width=True)

        best_nc = max(successful, key=lambda r: r["NC"])
        best_quality = max(successful, key=lambda r: r["PSNR (dB)"] if r["PSNR (dB)"] != float("inf") else 1e9)

        st.subheader("💡 Đề xuất")
        st.markdown(
            f"- **Watermark trích xuất giống bản gốc nhất (NC cao nhất):** {best_nc['Thuật toán']} "
            f"(NC = {best_nc['NC']})\n"
            f"- **Ảnh gốc ít bị biến dạng nhất (PSNR cao nhất):** {best_quality['Thuật toán']}"
        )
        st.caption(
            "Trong bản cài đặt hiện tại, DWT-SVD trích xuất watermark bằng cách chỉ khôi phục "
            "đường chéo của ma trận singular values, nên NC thường thấp hơn DWT dù về mặt lý "
            "thuyết DWT-SVD có thể bền hơn nếu trích xuất đầy đủ hơn."
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


def build_sidebar() -> None:
    """Always-visible cheat sheet so the user doesn't have to open
    the comparison tab just to remember which algorithm does what."""

    with st.sidebar:
        st.header("📋 Tra cứu nhanh")

        st.subheader("Giấu tin")
        stego_df = pd.DataFrame(
            [
                {"Thuật toán": name, "Tốc độ": i["speed"], "Sức chứa": i["capacity"], "Độ bền": i["robustness"]}
                for name, i in STEGO_ALGORITHM_INFO.items()
            ]
        )
        st.dataframe(stego_df, use_container_width=True, hide_index=True)

        st.subheader("Thủy vân")
        wm_df = pd.DataFrame(
            [
                {"Thuật toán": name, "Độ bền": i["robustness"]}
                for name, i in WATERMARK_ALGORITHM_INFO.items()
            ]
        )
        st.dataframe(wm_df, use_container_width=True, hide_index=True)

        st.caption(
            "PSNR ≥ 40 dB và SSIM ≥ 0.98: mắt thường khó nhận ra thay đổi.\n\n"
            "NC ≥ 0.9: watermark trích xuất gần như hoàn hảo."
        )

        if st.session_state.get("history"):
            if st.button("🗑️ Xoá lịch sử"):
                st.session_state["history"] = []
                st.rerun()


def main() -> None:
    """Run Streamlit application."""

    build_sidebar()

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
            "📊 So sánh",
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
        comparison_tab()

    with tabs[4]:
        attack_tab()

    with tabs[5]:
        steganalysis_tab()


if __name__ == "__main__":
    main()