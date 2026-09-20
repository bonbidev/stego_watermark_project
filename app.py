"""
Streamlit application for steganography and digital watermarking.
"""

from __future__ import annotations

import base64
import io
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
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

from config import attack_config


# ============================================================================
# Visual theme — professional font + consistent color palette used by
# every chart so the app reads as one coherent product instead of a
# stack of default-styled widgets. Supports a light/dark toggle
# (see build_sidebar()); charts read colors from _theme_colors() so
# they follow whichever mode is active on each rerun.
# ============================================================================

FONT_FAMILY = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', "
    "Roboto, Arial, sans-serif"
)

ALGO_COLORS = {
    "LSB": "#6366F1",       # indigo
    "PVD": "#0EA5E9",       # sky blue
    "DCT": "#F59E0B",       # amber
    "DWT": "#10B981",       # emerald
    "DWT-SVD": "#EC4899",   # pink
}

BG_COLOR = "#FFFFFF"
GRID_COLOR = "#EEF2F6"
TEXT_COLOR = "#1F2937"


def _theme_colors() -> dict:
    """Chart colors for the currently active light/dark mode."""

    if st.session_state.get("dark_mode"):
        return dict(bg="#1E293B", grid="#334155", text="#E5E7EB", line="#475569", hover_bg="#0F172A")
    return dict(bg=BG_COLOR, grid=GRID_COLOR, text=TEXT_COLOR, line="#D1D5DB", hover_bg="white")


def _inject_theme() -> None:
    """Apply a clean, professional UI font — but ONLY to elements that
    are guaranteed to hold plain text and nothing else.

    ROOT CAUSE OF THE BROKEN ICON IN THE SCREENSHOT: CSS font-family
    is inherited. An earlier version set `font-family` (with
    `!important`) on *containers* that also happen to hold a
    Streamlit-rendered icon as a child — the expander's own built-in
    disclosure arrow (`div[data-testid="stExpander"] details summary`),
    the tab bar (`.stTabs [data-baseweb="tab"]`), and every `h1-h4`
    header (several of which contain a `:material/...:` icon in their
    text). Because those icon children don't set their own
    `font-family`, they inherited the override from their ancestor and
    rendered as literal ligature text ("arrow_right") instead of a
    glyph — exactly what the screenshot shows. The overly broad
    `[class*="css"]` selector made this worse by matching almost any
    Streamlit-generated element, icons included.

    FIX: font-family is now applied only to `.stMarkdown` paragraph/
    list/table text, which never contains a Streamlit-rendered icon
    as a sibling/child in this app (every icon here lives in a
    header/tab/button/expander label or inside hand-written SVG in
    render_commentary — never inside a plain markdown paragraph).
    Nothing here uses `!important` on any container that might hold
    an icon, and no icon-related CSS is touched at all — the safest
    fix is to stop interfering with icon elements entirely rather
    than trying to out-guess Streamlit's internal DOM structure.
    """

    st.markdown(
        f"""
        <style>
        .stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th {{
            font-family: {FONT_FAMILY};
        }}

        [data-testid="stMetric"] {{
            background: #F8FAFC;
            border: 1px solid #E5E7EB;
            border-radius: 12px;
            padding: 14px 18px;
        }}

        /* Explicitly pin both the metric box's background AND its
        text color together. Previously only the background was set,
        so the text fell back to Streamlit's own theme-dependent
        default color — if the person's browser/Streamlit theme is
        set to dark (independent of this app's own dark-mode toggle
        below), that default turns near-white, landing on this
        always-light box and becoming unreadable ("white on white").
        Pinning both colors here makes the light box readable
        regardless of what theme is otherwise active. */
        [data-testid="stMetric"] [data-testid="stMetricValue"],
        [data-testid="stMetric"] [data-testid="stMetricLabel"] {{
            color: #1F2937 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _apply_dark_mode(dark: bool) -> None:
    """Inject dark-mode overrides on top of the base theme. Must run
    after the toggle's value is known, i.e. from build_sidebar()."""

    st.session_state["dark_mode"] = dark
    if not dark:
        return

    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"], [data-testid="stSidebar"], [data-testid="stHeader"] {
            background-color: #0F172A !important;
        }
        [data-testid="stMetric"] {
            background: #1E293B !important;
            border-color: #334155 !important;
        }
        [data-testid="stMetric"] [data-testid="stMetricValue"],
        [data-testid="stMetric"] [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
        h1, h2, h3, h4, p, span, label, li, .stMarkdown, .stCaption {
            color: #E5E7EB !important;
        }
        .stTabs [data-baseweb="tab"] { color: #CBD5E1 !important; }
        [data-testid="stDataFrame"] { filter: invert(0.92) hue-rotate(180deg); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _style_fig(fig: go.Figure, title: str | None = None, height: int = 380) -> go.Figure:
    """Apply the shared visual theme (font, colors, grid) to any
    Plotly figure so every chart in the app looks consistent, in
    either light or dark mode."""

    colors = _theme_colors()

    fig.update_layout(
        title=dict(text=title, font=dict(family=FONT_FAMILY, size=18, color=colors["text"])) if title else None,
        font=dict(family=FONT_FAMILY, size=13, color=colors["text"]),
        plot_bgcolor=colors["bg"],
        paper_bgcolor=colors["bg"],
        height=height,
        margin=dict(l=30, r=30, t=60 if title else 20, b=30),
        legend=dict(font=dict(family=FONT_FAMILY, size=12)),
        hoverlabel=dict(font=dict(family=FONT_FAMILY, size=13), bgcolor=colors["hover_bg"]),
    )
    fig.update_xaxes(gridcolor=colors["grid"], showline=True, linecolor=colors["line"])
    fig.update_yaxes(gridcolor=colors["grid"], showline=True, linecolor=colors["line"])
    return fig


def _image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def image_compare_slider(
    before: Image.Image,
    after: Image.Image,
    before_label: str = "Trước",
    after_label: str = "Sau",
    height: int = 420,
) -> None:
    """Draggable before/after slider so the person can sweep across
    the image instead of eyeballing two static side-by-side images."""

    before_b64 = _image_to_base64(before)
    after_b64 = _image_to_base64(after)

    html = f"""
    <div style="font-family:{FONT_FAMILY}; max-width:100%;">
      <div style="position:relative; width:100%; max-width:720px; margin:0 auto; user-select:none;">
        <img src="data:image/png;base64,{after_b64}"
             style="width:100%; display:block; border-radius:10px;" draggable="false">
        <img id="before-img" src="data:image/png;base64,{before_b64}"
             style="width:100%; display:block; border-radius:10px; position:absolute; top:0; left:0;
                    clip-path: inset(0 50% 0 0);" draggable="false">
        <div id="divider"
             style="position:absolute; top:0; bottom:0; left:50%; width:3px; background:#fff;
                    box-shadow:0 0 6px rgba(0,0,0,.5); pointer-events:none;"></div>
        <div style="position:absolute; top:8px; left:12px; background:rgba(0,0,0,.55); color:#fff;
                    padding:2px 10px; border-radius:6px; font-size:12px;">{before_label}</div>
        <div style="position:absolute; top:8px; right:12px; background:rgba(0,0,0,.55); color:#fff;
                    padding:2px 10px; border-radius:6px; font-size:12px;">{after_label}</div>
      </div>
      <input type="range" min="0" max="100" value="50" id="cmp-slider"
             style="width:100%; max-width:720px; display:block; margin:10px auto 0;">
    </div>
    <script>
      const slider = document.getElementById('cmp-slider');
      const beforeImg = document.getElementById('before-img');
      const divider = document.getElementById('divider');
      slider.addEventListener('input', function() {{
        beforeImg.style.clipPath = `inset(0 ${{100 - this.value}}% 0 0)`;
        divider.style.left = this.value + '%';
      }});
    </script>
    """
    components.html(html, height=height + 60)


def download_excel_button(
    dataframe: pd.DataFrame,
    filename: str,
    label: str,
    sheet_name: str = "Sheet1",
) -> None:
    """Excel export for any comparison/history table, for people who
    need to paste results straight into a report."""

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name=sheet_name)
    st.download_button(
        label=label,
        data=buffer.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


st.set_page_config(
    page_title="Stego & Watermark",
    page_icon=":material/lock:",
    layout="wide",
)

_inject_theme()


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

    # DWT/DWT-SVD watermarking still convert their output to
    # grayscale internally (PVD/DCT now preserve color — see
    # core/pvd_stego.py and core/dct_stego.py), while `original` is
    # loaded as RGB. Comparing an (H, W, 3) array against an (H, W)
    # array makes calculate_metrics() raise "Images must have the
    # same shape.", which previously made watermarking look broken
    # in the UI even though embedding succeeded. Compare like for
    # like by matching `original`'s mode to `processed`'s mode. This
    # is a no-op now for PVD/DCT since both stay RGB already.
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
        return ":green[●] Ảnh giống hệt bản gốc (không có sai khác nào về mặt pixel)."

    if psnr >= 40 and ssim >= 0.98:
        return ":green[●] Chất lượng rất tốt — mắt thường gần như không thể phân biệt được với ảnh gốc."
    if psnr >= 30 and ssim >= 0.90:
        return ":orange[●] Chất lượng khá tốt — sai khác rất nhỏ, khó nhận ra nếu không so sánh trực tiếp."
    if psnr >= 20:
        return ":orange[●] Chất lượng trung bình — có thể nhận thấy khác biệt nếu nhìn kỹ hoặc phóng to."
    return ":red[●] Chất lượng thấp — ảnh bị biến dạng rõ rệt, dễ nhận ra bằng mắt thường."


def interpret_nc(nc: float) -> str:
    """Plain-language read of Normalized Correlation for watermark
    extraction quality."""

    if nc >= 0.9:
        return ":green[●] Rất tốt — watermark trích xuất gần như giống hệt bản gốc."
    if nc >= 0.7:
        return ":orange[●] Khá tốt — watermark trích xuất vẫn nhận diện được rõ ràng."
    if nc >= 0.5:
        return ":orange[●] Trung bình — watermark còn nhận ra được nhưng đã suy giảm nhiều."
    return ":red[●] Thấp — watermark trích xuất bị méo nặng, khó/không nhận diện được."


def render_steps(steps: list[str], title: str = ":material/push_pin: Cách thao tác") -> None:
    """Compact, numbered how-to box shown right on the tab itself —
    complements the full :material/menu_book: Hướng dẫn tab for people who don't want
    to leave the screen they're on."""

    with st.expander(title, expanded=False):
        for i, step in enumerate(steps, start=1):
            st.markdown(f"**{i}.** {step}")


_TONE_ICON_SVG = {
    "success": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" '
        'fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
        'style="vertical-align:-3px;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>'
        '<polyline points="22 4 12 14.01 9 11.01"></polyline></svg>'
    ),
    "warning": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" '
        'fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
        'style="vertical-align:-3px;"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z"></path>'
        '<line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'
    ),
    "info": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" '
        'fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
        'style="vertical-align:-3px;"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>'
    ),
    "error": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" '
        'fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
        'style="vertical-align:-3px;"><circle cx="12" cy="12" r="10"></circle>'
        '<line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>'
    ),
}


def render_commentary(title: str, paragraphs: list[str], tone: str = "info") -> None:
    """A visually distinct 'card' with a detailed, multi-paragraph
    written assessment — used everywhere a bare number (PSNR, NC...)
    isn't enough on its own to explain what it means.

    Uses an inline SVG icon (not a `:material/...:` shortcode) because
    this renders inside a raw HTML block (`unsafe_allow_html=True`);
    an SVG is guaranteed to display correctly there regardless of
    whether Streamlit's icon-shortcode substitution applies inside
    raw HTML, while a shortcode is not."""

    border_color = {
        "success": "#10B981", "warning": "#F59E0B", "info": "#6366F1", "error": "#EF4444",
    }.get(tone, "#6366F1")
    icon_svg = _TONE_ICON_SVG.get(tone, _TONE_ICON_SVG["info"]).format(color=border_color)

    with st.container(border=True):
        st.markdown(
            f"<div style='border-left:4px solid {border_color}; padding-left:12px;'>"
            f"<b style='font-size:1.05em;'>{icon_svg} {title}</b></div>",
            unsafe_allow_html=True,
        )
        for paragraph in paragraphs:
            st.markdown(paragraph)


def _usage_capacity_note(usage_pct: float) -> str:
    if usage_pct < 30:
        return "còn rất nhiều dung lượng dự phòng."
    if usage_pct < 70:
        return "dung lượng còn lại vừa phải, đủ dùng cho vài lần giấu tin nữa nếu cần."
    if usage_pct < 95:
        return "gần hết sức chứa — nội dung dài hơn một chút có thể sẽ không vừa."
    return "gần như dùng hết sức chứa tối đa của ảnh này."


def _speed_note(elapsed_seconds: float) -> str:
    ms = elapsed_seconds * 1000
    if ms < 50:
        return "rất nhanh, phù hợp xử lý hàng loạt nhiều ảnh."
    if ms < 500:
        return "chấp nhận được cho xử lý từng ảnh một, nhưng sẽ chậm nếu chạy hàng loạt số lượng lớn."
    return "khá chậm — cân nhắc nếu cần xử lý nhiều ảnh cùng lúc."


def stego_commentary(
    algorithm: str,
    metrics: dict,
    capacity: int,
    payload_len: int,
    elapsed: float,
) -> None:
    """Detailed written assessment after an embed operation,
    combining quality, capacity usage, speed and robustness into one
    place instead of leaving the user to interpret 3 separate
    numbers on their own."""

    psnr = metrics["PSNR"]
    ssim = metrics["SSIM"]
    usage_pct = 100 * payload_len / capacity if capacity else 0
    info = STEGO_ALGORITHM_INFO[algorithm]

    tone = "success" if (psnr == float("inf") or psnr >= 40) and ssim >= 0.98 else "info"

    robustness_note = (
        "Chỉ nên dùng khi chắc chắn ảnh sẽ **không** bị nén/chỉnh sửa lại trước khi tới tay "
        "người nhận (gửi đúng file PNG gốc) — thuật toán này không được thiết kế để chịu đựng "
        "xử lý ảnh."
        if algorithm in ("LSB", "PVD") else
        "Có khả năng chịu đựng tốt hơn LSB/PVD trước một số xử lý nhẹ, nhưng theo số liệu đo "
        "thực tế vẫn **không** đảm bảo sống sót qua mức nén JPEG phổ biến trên mạng xã hội "
        "(quality 75-85) — xem tab Kiểm tra độ bền hoặc So sánh để có số liệu cụ thể."
    )

    render_commentary(
        f"Nhận xét chi tiết — {algorithm}",
        [
            f"**Chất lượng ảnh:** PSNR = {'∞' if psnr == float('inf') else f'{psnr:.2f} dB'}, "
            f"SSIM = {ssim:.4f}. {interpret_quality(psnr, ssim)}",
            f"**Dung lượng:** nội dung đã mã hoá chiếm **{usage_pct:.1f}%** sức chứa "
            f"({payload_len:,} / {capacity:,} byte) — {_usage_capacity_note(usage_pct)}",
            f"**Tốc độ:** xử lý trong **{elapsed * 1000:.0f} ms** — {_speed_note(elapsed)}",
            f"**Độ bền:** {info['robustness']}. {robustness_note}",
        ],
        tone=tone,
    )


def watermark_commentary(
    algorithm: str,
    metrics: dict,
    nc: float | None,
    alpha: float,
    elapsed: float,
) -> None:
    """Detailed written assessment after a watermark embed."""

    psnr = metrics["PSNR"]
    ssim = metrics["SSIM"]
    info = WATERMARK_ALGORITHM_INFO[algorithm]

    tone = "success" if nc is not None and nc >= 0.7 and ssim >= 0.95 else "info"

    alpha_note = (
        "Alpha đang ở mức thấp — watermark khó bị phát hiện bằng mắt nhưng cũng dễ mất khi ảnh "
        "bị xử lý lại; tăng alpha nếu cần bền hơn, đổi lại ảnh gốc sẽ biến dạng nhiều hơn một chút."
        if alpha <= 0.03 else
        "Alpha đang ở mức cao — watermark bền hơn trước xử lý ảnh, nhưng có thể khiến ảnh gốc "
        "biến dạng rõ hơn; giảm alpha nếu ưu tiên giữ ảnh gốc đẹp."
        if alpha >= 0.12 else
        "Alpha đang ở mức cân bằng thường dùng (0.03-0.12)."
    )

    paragraphs = [
        f"**Chất lượng ảnh gốc sau nhúng:** PSNR = {'∞' if psnr == float('inf') else f'{psnr:.2f} dB'}, "
        f"SSIM = {ssim:.4f}. {interpret_quality(psnr, ssim)}",
    ]
    if nc is not None:
        paragraphs.append(f"**Độ chính xác watermark trích xuất:** NC = {nc:.4f}. {interpret_nc(nc)}")
    paragraphs.append(f"**Alpha = {alpha}:** {alpha_note}")
    paragraphs.append(f"**Tốc độ:** {elapsed * 1000:.0f} ms — {_speed_note(elapsed)}")
    paragraphs.append(f"**Độ bền lý thuyết:** {info['robustness']}")

    render_commentary(f"Nhận xét chi tiết — {algorithm}", paragraphs, tone=tone)


WATERMARK_ALGORITHM_INFO = {
    "DWT": {
        "class": DWTWatermark,
        "desc": "Nhúng watermark trực tiếp vào hệ số subband LH của biến đổi wavelet rời rạc. Đơn giản, NC thường cao (0.8-1.0).",
        "robustness": ":green[●] Tốt trước nhiễu/nén nhẹ",
    },
    "DWT-SVD": {
        "class": DWTSVDWatermark,
        "desc": "Kết hợp DWT với phân tích giá trị kỳ dị (SVD). Về lý thuyết bền hơn DWT thuần, nhưng cách trích xuất trong bản này chỉ khôi phục đường chéo ma trận nên NC thực tế thường THẤP hơn DWT (0.01-0.3).",
        "robustness": ":orange[●] Lý thuyết cao hơn DWT, nhưng cách cài đặt hiện tại cho NC thấp hơn — xem tab So sánh",
    },
}


def show_diff_heatmap(
    original: Image.Image,
    processed: Image.Image,
    amplify: float = 10.0,
) -> None:
    """Show an amplified |original - processed| heatmap so the user
    can *see* where an algorithm changed the image, instead of only
    reading a single aggregate number (PSNR/SSIM/MSE). Uses a Plotly
    heatmap (colorbar + per-pixel hover value) instead of a flat
    grayscale image."""

    if original.mode != processed.mode:
        original = original.convert(processed.mode)

    original_array = np.asarray(original, dtype=np.float32)
    processed_array = np.asarray(processed, dtype=np.float32)

    diff = np.abs(original_array - processed_array)
    if diff.ndim == 3:
        diff = diff.mean(axis=2)

    fig = go.Figure(
        go.Heatmap(
            z=diff[::-1],  # flip so row 0 (top of image) renders at the top
            colorscale="Inferno",
            colorbar=dict(title="|Δ pixel|", tickfont=dict(family=FONT_FAMILY)),
            hovertemplate="hàng %{y}, cột %{x}<br>chênh lệch: %{z:.1f}<extra></extra>",
        )
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")
    st.plotly_chart(
        _style_fig(fig, title="Bản đồ sai khác (càng sáng càng thay đổi nhiều)", height=420),
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
        "speed": ":material/bolt: :material/bolt: :material/bolt: Rất nhanh",
        "capacity": ":green[●] :green[●] :green[●] Rất lớn",
        "robustness": ":red[●] Thấp",
    },
    "PVD": {
        "class": PVDStego,
        "desc": "Nhúng nhiều bit hơn ở vùng có độ tương phản cao (cạnh, chi tiết), ít bit hơn ở vùng phẳng. Cân bằng giữa sức chứa và độ ẩn giấu.",
        "speed": ":material/bolt: :material/bolt: Trung bình",
        "capacity": ":green[●] :green[●] Lớn",
        "robustness": ":orange[●] Trung bình",
    },
    "DCT": {
        "class": DCTStego,
        "desc": "Biến đổi mỗi khối 8x8 sang miền tần số (như JPEG) và mã hoá 1 bit/khối. Sức chứa thấp hơn nhiều nhưng bền hơn trước nén ảnh.",
        "speed": ":material/bolt: Chậm hơn",
        "capacity": ":orange[●] Thấp (1 bit / khối 8x8)",
        "robustness": ":green[●] Cao nhất trong 3 thuật toán",
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
    with st.expander(f":material/history: Lịch sử thao tác ({len(rows)})"):
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def steganography_tab() -> None:
    """Steganography interface."""

    st.header(
        ":material/lock: Giấu tin"
    )

    render_steps([
        "Upload ảnh gốc — nên dùng PNG để không bị mất dữ liệu do nén lại.",
        "Chọn thuật toán giấu tin (xem thông tin từng thuật toán ngay bên cạnh).",
        "Nhập nội dung bí mật và đặt mật khẩu (cần nhớ chính xác để trích xuất lại sau này).",
        "Nhấn **Giấu tin**, xem kết quả và tải ảnh về ở định dạng PNG.",
    ])

    st.write(
        "Mã hóa dữ liệu bằng AES và giấu vào ảnh bằng thuật toán steganography được chọn."
    )

    col_upload, col_algo = st.columns([1, 1], gap="large")

    with col_upload:
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

    with col_algo:
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

    with st.container(border=True):
        preview_col, capacity_col = st.columns([1, 2], gap="large")
        with preview_col:
            st.image(image, caption="Ảnh gốc", use_container_width=True)
        with capacity_col:
            stego_tool_preview = info["class"]()
            capacity = stego_tool_preview.get_capacity(image)
            st.metric("Sức chứa tối đa", f"{capacity:,} byte")
            st.caption(f"Với thuật toán {algorithm} trên ảnh {image.width}×{image.height} điểm ảnh.")

    secret = st.text_area(
        "Nội dung bí mật",
        height=150,
    )

    password = st.text_input(
        "Mật khẩu AES",
        type="password",
    )
    if password and len(password) < 12:
        st.caption(":material/warning: Mật khẩu ngắn hơn 12 ký tự — nên dùng mật khẩu dài hơn cho an toàn.")

    if not st.button(
        ":material/lock: Giấu tin",
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

        with st.expander(":material/compare: So sánh bằng thanh trượt kéo"):
            image_compare_slider(image, result, "Ảnh gốc", "Đã giấu tin")

        metrics = show_metrics(
            image,
            result,
            elapsed_seconds=elapsed,
        )

        stego_commentary(algorithm, metrics, capacity, len(payload), elapsed)

        with st.expander(":material/search: Xem bản đồ sai khác (nơi thuật toán đã thay đổi ảnh)"):
            show_diff_heatmap(image, result)

        image_download_button(
            result,
            "stego_image.png",
            ":material/download: Tải ảnh đã giấu tin",
        )

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
        ":material/lock_open: Giải mã & trích xuất"
    )

    render_steps([
        "Upload đúng ảnh đã được giấu tin trước đó (không phải ảnh gốc).",
        "Chọn **đúng thuật toán** đã dùng lúc giấu tin (LSB/PVD/DCT).",
        "Nhập **đúng mật khẩu** đã dùng lúc giấu tin.",
        "Nhấn **Trích xuất** để xem lại nội dung bí mật.",
    ])

    col_upload, col_options = st.columns([1, 1], gap="large")

    with col_upload:
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

    with col_options:
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

    with st.container(border=True):
        st.image(image, caption="Ảnh đã giấu tin", width=320)

    if not st.button(
        ":material/lock_open: Trích xuất",
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

        with st.container(border=True):
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
        ":material/copyright: Thủy vân số"
    )

    render_steps([
        "Upload ảnh gốc cần bảo vệ, và một ảnh watermark (logo/chữ ký) nhỏ để nhúng vào.",
        "Chọn thuật toán (DWT hoặc DWT-SVD) và mức Alpha (độ mạnh nhúng).",
        "Nhấn **Nhúng watermark**, tải ảnh kết quả về.",
        "Muốn kiểm tra: xem lại NC (độ giống watermark trích xuất) hiển thị ngay bên dưới.",
    ])

    render_steps([
        "Upload ảnh gốc cần bảo vệ, và một ảnh watermark (logo/chữ ký) nhỏ để nhúng vào.",
        "Chọn thuật toán (DWT hoặc DWT-SVD) và mức Alpha (độ mạnh nhúng).",
        "Nhấn **Nhúng watermark**, tải ảnh kết quả về.",
        "Muốn kiểm tra: xem lại NC (độ giống watermark trích xuất) hiển thị ngay bên dưới.",
    ])

    col_upload1, col_upload2 = st.columns(2, gap="large")

    with col_upload1:
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

    with col_upload2:
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

    col_algo, col_alpha = st.columns([1, 1], gap="large")

    with col_algo:
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

    with col_alpha:
        alpha = st.slider(
            "Alpha (độ mạnh nhúng)",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
        )
        st.caption("Alpha lớn hơn → watermark bền hơn nhưng ảnh gốc biến dạng nhiều hơn.")

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

    with st.container(border=True):
        preview_col1, preview_col2 = st.columns(2, gap="large")
        preview_col1.image(image, caption="Ảnh gốc", use_container_width=True)
        preview_col2.image(watermark, caption="Watermark", use_container_width=True)

    if not st.button(
        ":material/copyright: Nhúng watermark",
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

        with st.expander(":material/compare: So sánh bằng thanh trượt kéo"):
            image_compare_slider(image, result, "Ảnh gốc", "Đã nhúng watermark")

        metrics = show_metrics(
            image,
            result,
            elapsed_seconds=elapsed,
        )

        with st.expander(":material/search: Xem bản đồ sai khác (nơi thuật toán đã thay đổi ảnh)"):
            show_diff_heatmap(image, result)

        image_download_button(
            result,
            "watermarked_image.png",
            ":material/download: Tải ảnh thủy vân",
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

        watermark_commentary(algorithm, metrics, nc_value, alpha, elapsed)

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


ATTACK_DISPLAY_NAMES = {
    "jpeg": "Nén JPEG",
    "gaussian_noise": "Nhiễu Gaussian",
    "salt_pepper": "Nhiễu muối tiêu",
    "gaussian_blur": "Làm mờ Gaussian",
    "median_blur": "Làm mờ Median",
    "resize": "Thu phóng",
    "crop": "Cắt ảnh",
    "rotate": "Xoay ảnh",
    "sharpen": "Làm nét",
}


def attack_tab() -> None:
    """Robustness testing interface."""

    st.header(
        ":material/science: Kiểm tra độ bền"
    )

    render_steps([
        "Upload ảnh cần kiểm tra — tốt nhất là ảnh đã giấu tin/watermark ở tab trước.",
        "Chọn kiểu tấn công (nén JPEG, nhiễu, mờ, xoay...) và điều chỉnh mức độ.",
        "Nhấn **Thực hiện attack** để xem ảnh sau xử lý và mức độ biến dạng.",
        "Tải ảnh sau attack về, rồi thử trích xuất lại ở tab Trích xuất/Thủy vân để kiểm chứng.",
    ])

    col_upload, col_params = st.columns([1, 1], gap="large")

    with col_upload:
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

    with col_params:
        attack = st.selectbox(
            "Kiểu tấn công",
            list(ATTACK_DISPLAY_NAMES.keys()),
            format_func=lambda key: ATTACK_DISPLAY_NAMES[key],
        )

        if attack == "jpeg":
            quality = st.slider(
                "Chất lượng JPEG",
                10,
                100,
                75,
            )

            kwargs = {
                "quality": quality,
            }

        elif attack == "gaussian_noise":
            sigma = st.slider(
                "Độ lệch chuẩn nhiễu (sigma)",
                1.0,
                50.0,
                10.0,
            )

            kwargs = {
                "sigma": sigma,
            }

        elif attack == "salt_pepper":
            amount = st.slider(
                "Tỉ lệ nhiễu",
                0.001,
                0.10,
                0.01,
            )

            kwargs = {
                "amount": amount,
            }

        elif attack == "gaussian_blur":
            kernel = st.selectbox(
                "Kích thước kernel",
                [3, 5, 7, 9],
            )

            kwargs = {
                "kernel_size": kernel,
            }

        elif attack == "median_blur":
            kernel = st.selectbox(
                "Kích thước kernel",
                [3, 5, 7],
            )

            kwargs = {
                "kernel_size": kernel,
            }

        elif attack == "resize":
            scale = st.slider(
                "Tỉ lệ thu phóng",
                0.1,
                1.0,
                0.5,
            )

            kwargs = {
                "scale": scale,
            }

        elif attack == "crop":
            ratio = st.slider(
                "Tỉ lệ giữ lại",
                0.1,
                1.0,
                0.8,
            )

            kwargs = {
                "crop_ratio": ratio,
            }

        elif attack == "rotate":
            angle = st.slider(
                "Góc xoay (độ)",
                -45.0,
                45.0,
                5.0,
            )

            kwargs = {
                "angle": angle,
            }

        else:
            strength = st.slider(
                "Cường độ làm nét",
                0.1,
                3.0,
                1.5,
            )

            kwargs = {
                "strength": strength,
            }

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

    with st.container(border=True):
        st.image(image, caption="Ảnh gốc trước khi tấn công", width=320)

    if not st.button(
        ":material/science: Thực hiện attack",
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
            metrics = show_metrics(
                image,
                attacked,
            )
            psnr_str = "∞" if metrics["PSNR"] == float("inf") else f"{metrics['PSNR']:.2f} dB"
            render_commentary(
                f"Nhận xét — mức độ tác động của '{ATTACK_DISPLAY_NAMES[attack]}'",
                [
                    f"PSNR = {psnr_str}, SSIM = {metrics['SSIM']:.4f}. "
                    f"{interpret_quality(metrics['PSNR'], metrics['SSIM'])}",
                    "Nếu ảnh này có giấu tin/watermark, hãy thử trích xuất lại từ ảnh vừa tải về ở "
                    "tab tương ứng để xem dữ liệu có còn sống sót qua mức tấn công này hay không.",
                ],
                tone="success" if metrics["PSNR"] == float("inf") or metrics["PSNR"] >= 40 else "warning",
            )
        else:
            st.info(
                "Kích thước ảnh đã thay đổi sau attack (do crop/resize) nên không so sánh trực "
                "tiếp PSNR/SSIM theo từng điểm ảnh được — hãy quan sát ảnh trực quan ở trên."
            )

        image_download_button(
            attacked,
            "attacked_image.png",
            ":material/download: Tải ảnh sau attack",
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

    st.header(":material/bar_chart: So sánh thuật toán")
    st.write(
        "Chạy tất cả thuật toán trên cùng một ảnh đầu vào để so sánh "
        "trực tiếp chất lượng, tốc độ và độ bền."
    )

    render_steps([
        "Chọn tab con: So sánh giấu tin (LSB/PVD/DCT) hoặc So sánh thủy vân (DWT/DWT-SVD).",
        "Upload ảnh (có thể chọn nhiều ảnh cùng lúc ở phần giấu tin để lấy kết quả trung bình).",
        "Nhập nội dung/watermark cần test, chọn mức kiểm tra độ bền (Nhanh/Đầy đủ) nếu cần.",
        "Nhấn nút So sánh — xem bảng, biểu đồ, nhận xét chi tiết và tải kết quả về Excel.",
    ])

    sub_tab_stego, sub_tab_watermark = st.tabs(
        [":material/lock: So sánh giấu tin (LSB / PVD / DCT)", ":material/copyright: So sánh thủy vân (DWT / DWT-SVD)"]
    )

    # ------------------------------------------------------------------
    # Steganography comparison
    # ------------------------------------------------------------------
    with sub_tab_stego:
        with st.container(border=True):
            uploaded_files = st.file_uploader(
                "Chọn ảnh (có thể chọn nhiều ảnh cùng lúc để lấy kết quả trung bình đáng tin cậy hơn)",
                type=["png", "jpg", "jpeg", "bmp"],
                key="cmp_stego_image",
                accept_multiple_files=True,
            )
            secret = st.text_area("Nội dung bí mật", height=100, key="cmp_stego_secret")

            col_pw, col_mode = st.columns([1, 2])
            with col_pw:
                password = st.text_input("Mật khẩu AES", type="password", key="cmp_stego_password")
            with col_mode:
                robustness_mode = st.radio(
                    "Kiểm tra độ bền",
                    ["Không kiểm tra", "Nhanh (2 kiểu tấn công)", "Đầy đủ (9 kiểu tấn công)"],
                    index=1,
                    horizontal=True,
                    key="cmp_stego_robustness_mode",
                )

        if st.button(":material/bar_chart: So sánh 3 thuật toán", type="primary", key="cmp_stego_run"):
            if not uploaded_files or not secret or not password:
                st.warning("Vui lòng chọn ít nhất 1 ảnh, nhập nội dung bí mật và mật khẩu.")
            else:
                images = [img for f in uploaded_files if (img := load_image(f)) is not None]
                if images:
                    _run_stego_comparison(images, secret, password, robustness_mode)

    # ------------------------------------------------------------------
    # Watermark comparison
    # ------------------------------------------------------------------
    with sub_tab_watermark:
        with st.container(border=True):
            col_wm1, col_wm2 = st.columns(2, gap="large")
            with col_wm1:
                image_file = st.file_uploader(
                    "Chọn ảnh gốc", type=["png", "jpg", "jpeg", "bmp"], key="cmp_wm_image"
                )
            with col_wm2:
                watermark_file = st.file_uploader(
                    "Chọn watermark", type=["png", "jpg", "jpeg", "bmp"], key="cmp_wm_watermark"
                )
            alpha = st.slider(
                "Alpha", min_value=0.01, max_value=0.20, value=0.05, step=0.01, key="cmp_wm_alpha"
            )

        if st.button(":material/bar_chart: So sánh DWT vs DWT-SVD", type="primary", key="cmp_wm_run"):
            if image_file is None or watermark_file is None:
                st.warning("Vui lòng chọn cả ảnh gốc và watermark.")
            else:
                image = load_image(image_file)
                watermark = load_image(watermark_file)
                if image is not None and watermark is not None:
                    _run_watermark_comparison(image, watermark, alpha)


def _bar_chart(
    rows: list[dict],
    value_key: str,
    title: str,
    value_format: str = ".2f",
) -> go.Figure:
    """A styled, color-coded bar chart with value labels on top of
    each bar — replaces the flat default st.bar_chart."""

    names = [r["Thuật toán"] for r in rows]
    values = [r[value_key] if r[value_key] != float("inf") else 0 for r in rows]
    colors = [ALGO_COLORS.get(n, "#94A3B8") for n in names]

    fig = go.Figure(
        go.Bar(
            x=names,
            y=values,
            marker=dict(color=colors, line=dict(width=0)),
            text=[format(v, value_format) for v in values],
            textposition="outside",
            textfont=dict(family=FONT_FAMILY, size=13, color=TEXT_COLOR),
            hovertemplate="%{x}: %{y:" + value_format + "}<extra></extra>",
        )
    )
    fig.update_yaxes(title=value_key)
    return _style_fig(fig, title=title, height=320)


_ROBUSTNESS_TIER_SCORE = {
    ":red[●] Thấp": 20,
    ":orange[●] Trung bình": 55,
    ":green[●] Cao nhất trong 3 thuật toán": 85,
}


def _stego_radar_chart(rows: list[dict], robustness_scores: dict[str, float] | None) -> go.Figure:
    """Multi-criteria radar chart (quality / speed / capacity /
    robustness) so the trade-offs between algorithms are visible at
    a glance instead of scattered across several numbers.

    `robustness_scores` maps algorithm name -> pass rate 0-100 from
    the attack matrix actually run this time; when no attacks were
    run, falls back to the qualitative tier from STEGO_ALGORITHM_INFO
    so the radar always has 4 axes."""

    max_capacity = max(r["Sức chứa (byte)"] for r in rows) or 1
    fastest_ms = min(r["Thời gian (ms)"] for r in rows) or 1

    axes = ["Chất lượng", "Tốc độ", "Sức chứa", "Độ bền"]
    fig = go.Figure()

    for r in rows:
        name = r["Thuật toán"]
        psnr = r["PSNR (dB)"] if r["PSNR (dB)"] != float("inf") else 100
        quality_score = min(100, psnr / 60 * 100)
        speed_score = min(100, 100 * fastest_ms / r["Thời gian (ms)"])
        capacity_score = 100 * r["Sức chứa (byte)"] / max_capacity

        if robustness_scores is not None and name in robustness_scores:
            robustness_score = robustness_scores[name]
        else:
            robustness_score = _ROBUSTNESS_TIER_SCORE.get(
                STEGO_ALGORITHM_INFO[name]["robustness"], 50
            )

        values = [quality_score, speed_score, capacity_score, robustness_score]

        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=axes + [axes[0]],
                name=name,
                fill="toself",
                line=dict(color=ALGO_COLORS.get(name, "#94A3B8"), width=2),
                opacity=0.75,
                hovertemplate="%{theta}: %{r:.0f}/100<extra>" + name + "</extra>",
            )
        )

    colors = _theme_colors()
    fig.update_layout(
        polar=dict(
            bgcolor=colors["bg"],
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=colors["grid"]),
            angularaxis=dict(gridcolor=colors["grid"]),
        )
    )
    return _style_fig(fig, title="So sánh đa tiêu chí (điểm 0-100, càng lớn càng tốt)", height=440)


def _robustness_heatmap(
    robustness_hits: dict[str, dict[str, list[int]]],
    attack_labels: list[str],
) -> go.Figure:
    """Algorithm x attack pass-rate matrix — the "ma trận độ bền đầy
    đủ" view, showing every algorithm against every attack at once
    instead of one attack at a time."""

    algo_names = list(robustness_hits.keys())
    z = []
    text = []
    for name in algo_names:
        row_pct = []
        row_text = []
        for label in attack_labels:
            hit, total = robustness_hits[name][label]
            pct = 100 * hit / total if total else 0
            row_pct.append(pct)
            row_text.append(f"{pct:.0f}%")
        z.append(row_pct)
        text.append(row_text)

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=attack_labels,
            y=algo_names,
            text=text,
            texttemplate="%{text}",
            textfont=dict(family=FONT_FAMILY, size=13),
            colorscale=[[0, "#FEE2E2"], [0.5, "#FEF3C7"], [1, "#A7F3D0"]],
            zmin=0,
            zmax=100,
            colorbar=dict(title="% sống sót", tickfont=dict(family=FONT_FAMILY)),
            hovertemplate="%{y} vs %{x}: %{z:.0f}%<extra></extra>",
        )
    )
    return _style_fig(fig, title="Ma trận độ bền: % lần giải mã đúng sau tấn công", height=320)


def _attacks_for_mode(mode: str) -> list[tuple[str, str, dict]]:
    """Return (label, attack_name, kwargs) triples for apply_attack(),
    matching config.py's attack_config so the UI and the underlying
    attack functions never drift apart."""

    if mode == "Đầy đủ (9 kiểu tấn công)":
        return [
            (ATTACK_DISPLAY_NAMES[name], name, attack_config.kwargs_for(name))
            for name in attack_config.names
        ]
    if mode == "Nhanh (2 kiểu tấn công)":
        return [
            ("Nén JPEG", "jpeg", {"quality": 75}),
            ("Nhiễu Gaussian", "gaussian_noise", {"sigma": 10.0}),
        ]
    return []


def _run_stego_comparison(
    images: list[Image.Image],
    secret: str,
    password: str,
    robustness_mode: str,
) -> None:
    from core.aes_cipher import AESCipher

    cipher = AESCipher(password)
    encrypted = cipher.encrypt(secret)
    payload = encrypted.encode("utf-8")

    attacks = _attacks_for_mode(robustness_mode)
    attack_labels = [label for label, _, _ in attacks]

    detail_rows: list[dict] = []
    # robustness_hits[algo][attack_label] = [success_count, attempted_count]
    robustness_hits: dict[str, dict[str, list[int]]] = {
        name: {label: [0, 0] for label in attack_labels} for name in STEGO_ALGORITHM_INFO
    }
    stego_images_preview: dict[str, Image.Image] = {}

    total_runs = len(images) * len(STEGO_ALGORITHM_INFO)
    progress = st.progress(0.0, text="Đang xử lý...")
    done = 0

    for img_idx, image in enumerate(images):
        for name, info in STEGO_ALGORITHM_INFO.items():
            tool = info["class"]()
            capacity = tool.get_capacity(image)
            row = {
                "Ảnh #": img_idx + 1,
                "Thuật toán": name,
                "Sức chứa (byte)": capacity,
                "Đủ chỗ?": capacity >= len(payload),
                "PSNR (dB)": None,
                "SSIM": None,
                "Thời gian (ms)": None,
                "Ghi chú": "",
            }

            if capacity >= len(payload):
                try:
                    start = time.perf_counter()
                    stego_image = tool.embed(image, payload)
                    elapsed = time.perf_counter() - start

                    metrics = calculate_metrics(
                        image if image.mode == stego_image.mode else image.convert(stego_image.mode),
                        stego_image,
                    )
                    row["PSNR (dB)"] = metrics["PSNR"] if metrics["PSNR"] != float("inf") else 100.0
                    row["SSIM"] = metrics["SSIM"]
                    row["Thời gian (ms)"] = elapsed * 1000

                    if img_idx == 0:
                        stego_images_preview[name] = stego_image

                    for attack_label, attack_name, kwargs in attacks:
                        robustness_hits[name][attack_label][1] += 1
                        try:
                            attacked = apply_attack(stego_image, attack_name, **kwargs)
                            extracted = tool.extract(attacked)
                            recovered = cipher.decrypt(extracted.decode("utf-8"))
                            if recovered == secret:
                                robustness_hits[name][attack_label][0] += 1
                        except Exception:
                            pass

                except Exception as exc:
                    row["Ghi chú"] = f"Lỗi: {exc}"
            else:
                row["Ghi chú"] = "Ảnh không đủ sức chứa"

            detail_rows.append(row)
            done += 1
            progress.progress(done / total_runs, text=f"Đang xử lý... ({done}/{total_runs})")

    progress.empty()

    detail_df = pd.DataFrame(detail_rows)

    # ---- Aggregate one row per algorithm (mean across all images) ----
    summary_rows = []
    for name in STEGO_ALGORITHM_INFO:
        sub = detail_df[detail_df["Thuật toán"] == name]
        ok = sub[sub["PSNR (dB)"].notna()]
        summary_rows.append(
            {
                "Thuật toán": name,
                "Số ảnh test": len(sub),
                "Đủ sức chứa (%)": round(100 * sub["Đủ chỗ?"].mean(), 1) if len(sub) else 0.0,
                "Sức chứa (byte)": round(sub["Sức chứa (byte)"].mean()) if len(sub) else 0,
                "PSNR (dB)": round(ok["PSNR (dB)"].mean(), 2) if len(ok) else None,
                "SSIM": round(ok["SSIM"].mean(), 4) if len(ok) else None,
                "Thời gian (ms)": round(ok["Thời gian (ms)"].mean(), 1) if len(ok) else None,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    st.subheader(f"Bảng tổng hợp (trung bình qua {len(images)} ảnh)" if len(images) > 1 else "Bảng so sánh")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    download_excel_button(summary_df, "so_sanh_giau_tin.xlsx", ":material/download: Tải bảng so sánh (Excel)", "So sanh giau tin")

    if len(images) > 1:
        with st.expander(f"Xem chi tiết từng ảnh ({len(images)} ảnh × 3 thuật toán = {len(detail_rows)} dòng)"):
            st.dataframe(detail_df, use_container_width=True, hide_index=True)

    successful = [r for r in summary_rows if r["PSNR (dB)"] is not None]
    if successful:
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.plotly_chart(
                _bar_chart(successful, "PSNR (dB)", "Chất lượng ảnh (PSNR trung bình)"),
                use_container_width=True,
            )
        with col_chart2:
            st.plotly_chart(
                _bar_chart(successful, "SSIM", "Độ tương đồng cấu trúc (SSIM trung bình)", value_format=".4f"),
                use_container_width=True,
            )

        robustness_scores = None
        if attacks:
            robustness_scores = {
                name: 100 * sum(hit for hit, _ in robustness_hits[name].values())
                / max(1, sum(total for _, total in robustness_hits[name].values()))
                for name in STEGO_ALGORITHM_INFO
            }
            st.plotly_chart(_robustness_heatmap(robustness_hits, attack_labels), use_container_width=True)

        st.plotly_chart(_stego_radar_chart(successful, robustness_scores), use_container_width=True)

        if stego_images_preview:
            st.subheader(":material/visibility: Xem ảnh kết quả (ảnh đầu tiên)")
            cols = st.columns(len(stego_images_preview))
            for col, (name, img) in zip(cols, stego_images_preview.items()):
                col.image(img, caption=name, use_container_width=True)

        # ---- Recommendation ----
        best_capacity = max(successful, key=lambda r: r["Sức chứa (byte)"])
        best_quality = max(successful, key=lambda r: r["PSNR (dB)"])
        fastest = min(successful, key=lambda r: r["Thời gian (ms)"])

        st.subheader(":material/lightbulb: Đề xuất")
        rec1, rec2, rec3 = st.columns(3)
        rec1.metric("Sức chứa lớn nhất", best_capacity["Thuật toán"], f"{best_capacity['Sức chứa (byte)']:,} byte", icon=":material/emoji_events:")
        rec2.metric("Chất lượng tốt nhất", best_quality["Thuật toán"], f"{best_quality['PSNR (dB)']} dB", icon=":material/emoji_events:")
        rec3.metric("Nhanh nhất", fastest["Thuật toán"], f"{fastest['Thời gian (ms)']} ms", icon=":material/emoji_events:")

        if robustness_scores:
            best_robust = max(robustness_scores, key=robustness_scores.get)
            st.markdown(
                f"- **Bền nhất trước tấn công:** {best_robust} "
                f"(sống sót trung bình {robustness_scores[best_robust]:.0f}% các phép tấn công đã test)"
            )
            if max(robustness_scores.values()) < 1:
                st.markdown(
                    "- Không thuật toán nào sống sót qua các phép tấn công đã thử — đây là hạn chế "
                    "chung của giấu tin miền không gian/tần số đơn giản, không phải lỗi cài đặt."
                )
        st.caption(
            "Không có thuật toán nào 'tốt nhất' tuyệt đối — LSB thắng về tốc độ/sức chứa, "
            "PVD cân bằng, DCT bền hơn trước nén ảnh nhưng chứa được ít hơn nhiều."
        )

        worst_quality = min(successful, key=lambda r: r["PSNR (dB)"])
        smallest_capacity = min(successful, key=lambda r: r["Sức chứa (byte)"])
        commentary = [
            f"Trong lần so sánh này (trên **{len(images)} ảnh**), **{best_quality['Thuật toán']}** cho chất "
            f"lượng ảnh tốt nhất ({best_quality['PSNR (dB)']} dB), trong khi **{worst_quality['Thuật toán']}** "
            f"thấp nhất ({worst_quality['PSNR (dB)']} dB) — chênh lệch "
            f"{best_quality['PSNR (dB)'] - worst_quality['PSNR (dB)']:.1f} dB.",
            f"Về sức chứa, **{best_capacity['Thuật toán']}** giấu được nhiều nhất "
            f"({best_capacity['Sức chứa (byte)']:,} byte) — gấp "
            f"{best_capacity['Sức chứa (byte)'] / max(1, smallest_capacity['Sức chứa (byte)']):.1f} lần "
            f"**{smallest_capacity['Thuật toán']}** ({smallest_capacity['Sức chứa (byte)']:,} byte).",
            f"Về tốc độ, **{fastest['Thuật toán']}** xử lý nhanh nhất ({fastest['Thời gian (ms)']} ms).",
        ]
        if robustness_scores:
            ranked = sorted(robustness_scores.items(), key=lambda kv: -kv[1])
            commentary.append(
                "Về độ bền trước tấn công (xếp từ cao xuống thấp): "
                + ", ".join(f"**{name}** ({score:.0f}%)" for name, score in ranked) + "."
            )
        commentary.append(
            "**Kết luận:** không có thuật toán nào vượt trội ở mọi tiêu chí cùng lúc — chọn LSB nếu ưu "
            "tiên tốc độ/sức chứa, PVD nếu cần cân bằng, DCT nếu cần khả năng chịu đựng xử lý ảnh tốt "
            "hơn một chút (dù vẫn không bền trước nén JPEG mạnh)."
        )
        render_commentary(":material/edit_note: Nhận xét chi tiết", commentary, tone="info")


def _nc_gauge(name: str, nc: float) -> go.Figure:
    """Gauge indicator for NC (0-1) — reads at a glance instead of
    requiring the viewer to know what a "good" NC number looks like."""

    color = ALGO_COLORS.get(name, "#94A3B8")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=nc,
            number=dict(valueformat=".4f", font=dict(family=FONT_FAMILY, size=28, color=TEXT_COLOR)),
            gauge=dict(
                axis=dict(range=[0, 1], tickfont=dict(family=FONT_FAMILY, size=11)),
                bar=dict(color=color, thickness=0.35),
                bgcolor="white",
                borderwidth=0,
                steps=[
                    dict(range=[0, 0.5], color="#FEE2E2"),
                    dict(range=[0.5, 0.7], color="#FEF3C7"),
                    dict(range=[0.7, 0.9], color="#D1FAE5"),
                    dict(range=[0.9, 1.0], color="#A7F3D0"),
                ],
            ),
        )
    )
    return _style_fig(fig, title=f"{name} — NC", height=230)


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
    download_excel_button(df, "so_sanh_thuy_van.xlsx", ":material/download: Tải bảng so sánh (Excel)", "So sanh thuy van")

    successful = [r for r in rows if "NC" in r]
    if successful:
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.plotly_chart(
                _bar_chart(successful, "PSNR (dB)", "Chất lượng ảnh gốc (PSNR)"),
                use_container_width=True,
            )
        with col_chart2:
            st.plotly_chart(
                _bar_chart(successful, "SSIM", "Độ tương đồng cấu trúc (SSIM)", value_format=".4f"),
                use_container_width=True,
            )

        st.subheader("Đồng hồ đo NC (độ giống watermark trích xuất)")
        gauge_cols = st.columns(len(successful))
        for col, r in zip(gauge_cols, successful):
            with col:
                st.plotly_chart(_nc_gauge(r["Thuật toán"], r["NC"]), use_container_width=True)

        st.subheader(":material/visibility: Xem ảnh kết quả")
        cols = st.columns(len(result_images))
        for col, (name, img) in zip(cols, result_images.items()):
            col.image(img, caption=name, use_container_width=True)

        best_nc = max(successful, key=lambda r: r["NC"])
        best_quality = max(successful, key=lambda r: r["PSNR (dB)"] if r["PSNR (dB)"] != float("inf") else 1e9)

        st.subheader(":material/lightbulb: Đề xuất")
        rec1, rec2 = st.columns(2)
        rec1.metric("NC cao nhất", best_nc["Thuật toán"], f"NC = {best_nc['NC']}", icon=":material/emoji_events:")
        rec2.metric("PSNR cao nhất", best_quality["Thuật toán"], f"{best_quality['PSNR (dB)']} dB", icon=":material/emoji_events:")
        st.caption(
            "Trong bản cài đặt hiện tại, DWT-SVD trích xuất watermark bằng cách chỉ khôi phục "
            "đường chéo của ma trận singular values, nên NC thường thấp hơn DWT dù về mặt lý "
            "thuyết DWT-SVD có thể bền hơn nếu trích xuất đầy đủ hơn."
        )

        nc_gap = best_nc["NC"] - min(r["NC"] for r in successful)
        commentary = [
            f"**{best_nc['Thuật toán']}** cho NC cao nhất ({best_nc['NC']:.4f}) — "
            f"{interpret_nc(best_nc['NC'])}",
            f"**{best_quality['Thuật toán']}** giữ ảnh gốc ít biến dạng nhất "
            f"({best_quality['PSNR (dB)']} dB).",
            f"Chênh lệch NC giữa 2 thuật toán: **{nc_gap:.4f}** — "
            + ("rất đáng kể, nên ưu tiên thuật toán có NC cao hơn nếu mục tiêu chính là "
               "khả năng trích xuất lại watermark chính xác." if nc_gap > 0.3 else
               "không quá lớn, có thể cân nhắc cả 2 tuỳ nhu cầu về chất lượng ảnh."),
            "**Kết luận:** với bản cài đặt hiện tại, DWT thường là lựa chọn thực tế hơn để xác "
            "thực quyền sở hữu ảnh; DWT-SVD cần cải tiến cách trích xuất (dùng đủ U, S, V thay vì "
            "chỉ đường chéo S) mới phát huy được lợi thế lý thuyết của nó.",
        ]
        render_commentary(":material/edit_note: Nhận xét chi tiết", commentary, tone="info")


def steganalysis_tab() -> None:
    """Steganalysis interface."""

    st.header(
        ":material/search: Steganalysis"
    )

    render_steps([
        "Upload bất kỳ ảnh nào bạn muốn kiểm tra (không cần biết trước có giấu tin hay không).",
        "Nhấn **Phân tích** để xem các chỉ số thống kê (LSB ratio, entropy, chi-square...).",
        "Đọc phần Nhận xét (:green[●]/:orange[●]/:material/warning:) để biết ảnh có dấu hiệu khả nghi hay không.",
        "Xem thêm 2 biểu đồ phân bố bên dưới để hiểu trực quan hơn về dữ liệu ảnh.",
    ])

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

    with st.container(border=True):
        st.image(image, caption="Ảnh cần phân tích", width=320)

    if not st.button(
        ":material/search: Phân tích",
        type="primary",
    ):
        return

    try:
        results = analyze(
            image
        )

        st.subheader("Các chỉ số phân tích")
        dataframe = pd.DataFrame(
            {
                "Chỉ số": results.keys(),
                "Giá trị": [round(v, 4) for v in results.values()],
            }
        )
        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True,
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("LSB Ratio", f"{results['LSB ratio']:.4f}", help="Tỉ lệ bit 1 trong mặt phẳng LSB — ảnh tự nhiên thường lệch khỏi 0.5")
        col2.metric("LSB Entropy", f"{results['LSB entropy']:.4f}", help="Entropy của mặt phẳng LSB — càng gần 1.0 càng giống nhiễu ngẫu nhiên")
        col3.metric("Chi-square p-value", f"{results['Chi-square p-value']:.4f}", help="p-value nhỏ = phân bố chẵn/lẻ mất cân bằng bất thường")

        # ---- Verdict ----
        lsb_ratio = results["LSB ratio"]
        near_half = abs(lsb_ratio - 0.5) < 0.02
        low_p_value = results["Chi-square p-value"] < 0.05

        if near_half and results["LSB entropy"] > 0.99:
            render_commentary(
                ":material/warning: Có dấu hiệu khả nghi",
                [
                    f"Tỉ lệ bit LSB = **{lsb_ratio:.4f}** (rất gần 0.5) và entropy = "
                    f"**{results['LSB entropy']:.4f}** (gần 1.0) — đây là đặc trưng thường thấy khi "
                    "dữ liệu đã mã hoá/nén (trông giống nhiễu ngẫu nhiên) được giấu vào mặt phẳng bit "
                    "thấp nhất của ảnh.",
                    "Đây **không phải bằng chứng chắc chắn 100%** — một số ảnh chụp bằng cảm biến "
                    "nhiễu cao (ISO cao, thiếu sáng) cũng có thể cho kết quả tương tự một cách tự "
                    "nhiên. Nên kết hợp thêm ngữ cảnh (nguồn gốc ảnh, kích thước file bất thường...) "
                    "trước khi kết luận.",
                ],
                tone="warning",
            )
        elif low_p_value:
            render_commentary(
                ":orange[●] Có bất thường nhẹ, chưa đủ kết luận",
                [
                    f"Chi-square p-value = **{results['Chi-square p-value']:.4f}** (< 0.05) cho thấy "
                    "phân bố chẵn/lẻ của mặt phẳng LSB hơi lệch khỏi ảnh tự nhiên thông thường, nhưng "
                    f"tỉ lệ LSB ({lsb_ratio:.4f}) hoặc entropy chưa đủ gần ngưỡng khả nghi để kết luận "
                    "chắc chắn có dữ liệu ẩn.",
                ],
                tone="info",
            )
        else:
            render_commentary(
                ":green[●] Không có dấu hiệu rõ ràng",
                [
                    f"Tỉ lệ LSB ({lsb_ratio:.4f}) và entropy ({results['LSB entropy']:.4f}) đều nằm "
                    "trong phạm vi thường thấy ở ảnh tự nhiên chưa qua giấu tin. Không loại trừ hoàn "
                    "toàn khả năng có dữ liệu ẩn (đặc biệt nếu dùng thuật toán PVD/DCT thay vì LSB), "
                    "nhưng không có bằng chứng thống kê rõ ràng ở đây.",
                ],
                tone="success",
            )

        st.divider()

        # ---- Visual charts ----
        gray = np.asarray(image.convert("L"))

        col_hist1, col_hist2 = st.columns(2)

        with col_hist1:
            fig_hist = px.histogram(
                x=gray.flatten(),
                nbins=64,
                labels={"x": "Giá trị pixel (0-255)"},
                color_discrete_sequence=["#6366F1"],
            )
            fig_hist.update_yaxes(title="Số lượng pixel")
            st.plotly_chart(_style_fig(fig_hist, title="Phân bố giá trị pixel", height=320), use_container_width=True)

        with col_hist2:
            lsb_plane = gray & 1
            zeros = int(np.sum(lsb_plane == 0))
            ones = int(np.sum(lsb_plane == 1))
            fig_lsb = go.Figure(
                go.Bar(
                    x=["Bit = 0", "Bit = 1"],
                    y=[zeros, ones],
                    marker=dict(color=["#0EA5E9", "#F59E0B"]),
                    text=[f"{zeros:,}", f"{ones:,}"],
                    textposition="outside",
                    textfont=dict(family=FONT_FAMILY),
                )
            )
            fig_lsb.update_yaxes(title="Số lượng pixel")
            st.plotly_chart(
                _style_fig(fig_lsb, title="Phân bố bit thấp nhất (LSB) — 50/50 là dấu hiệu khả nghi", height=320),
                use_container_width=True,
            )

    except Exception as exc:
        st.error(
            f"Phân tích thất bại: {exc}"
        )


def guide_tab() -> None:
    """In-app usage guide so people don't have to leave the app and
    read a separate README to understand what each tab does."""

    st.header(":material/menu_book: Hướng dẫn sử dụng")

    with st.expander(":material/lock: Giấu tin / :material/lock_open: Trích xuất", expanded=True):
        st.markdown(
            """
            1. Chọn **mục đích sử dụng** để được gợi ý thuật toán (không bắt buộc).
            2. Upload ảnh gốc (nên dùng PNG để tránh mất dữ liệu do nén JPEG lại).
            3. Nhập **nội dung bí mật** và **mật khẩu** (mật khẩu dùng để mã hoá AES — cần nhớ
               chính xác để trích xuất lại sau này, không có cách khôi phục nếu quên).
            4. Nhấn **Giấu tin** → tải ảnh kết quả về.
            5. Muốn lấy lại nội dung: sang tab **Trích xuất**, upload đúng ảnh đã giấu tin,
               chọn đúng thuật toán và nhập đúng mật khẩu đã dùng lúc giấu.

            :material/warning: **Lưu ý:** ảnh phải được tải về ở định dạng **PNG** (không nén mất dữ liệu).
            Nếu gửi qua Zalo/Messenger, một số nền tảng tự nén ảnh JPEG và có thể phá hỏng
            dữ liệu đã giấu (đặc biệt với LSB/PVD).
            """
        )

    with st.expander(":material/copyright: Thủy vân số"):
        st.markdown(
            """
            Dùng để **khẳng định quyền sở hữu ảnh** thay vì giấu tin nhắn bí mật — nhúng một
            ảnh nhỏ (logo, chữ ký) vào ảnh gốc, và có thể trích xuất lại logo đó sau này để
            chứng minh ảnh là của mình, kể cả khi ảnh đã bị chỉnh sửa nhẹ.

            - **Alpha** càng lớn → watermark càng bền nhưng ảnh gốc càng bị biến dạng nhiều hơn.
              Nên bắt đầu ở mức 0.05 rồi tăng/giảm tuỳ nhu cầu.
            - **NC** (Normalized Correlation) đo độ giống giữa watermark gốc và watermark
              trích xuất được — càng gần 1.0 càng tốt.
            """
        )

    with st.expander(":material/bar_chart: So sánh thuật toán"):
        st.markdown(
            """
            Chạy tất cả thuật toán cùng lúc trên **cùng một ảnh đầu vào** để so sánh công
            bằng. Có thể upload **nhiều ảnh** cùng lúc để lấy kết quả trung bình đáng tin
            cậy hơn (tránh kết luận vội từ 1 ảnh ngẫu nhiên).

            - **Nhanh**: chỉ test nén JPEG + nhiễu Gaussian, chạy nhanh.
            - **Đầy đủ**: test cả 9 kiểu tấn công (nén, nhiễu, mờ, xoay, cắt, thu phóng,
              làm nét...) — chậm hơn nhưng cho bức tranh đầy đủ về độ bền.
            - Kết quả có thể tải về **Excel** để đưa vào báo cáo.
            """
        )

    with st.expander(":material/science: Robustness"):
        st.markdown(
            "Test 1 thuật toán với 1 kiểu tấn công cụ thể, xem ảnh bị biến dạng thế nào. "
            "Dùng khi muốn xem chi tiết ảnh sau tấn công trông ra sao, thay vì chỉ xem số."
        )

    with st.expander(":material/search: Steganalysis"):
        st.markdown(
            "Phân tích 1 ảnh bất kỳ (không cần biết trước có giấu tin hay không) để tìm dấu "
            "hiệu bất thường trong mặt phẳng bit thấp nhất (LSB) — hữu ích để tự kiểm tra ảnh "
            "của mình, hoặc học cách các công cụ phát hiện steganography hoạt động."
        )

    st.divider()
    st.subheader(":material/help: Câu hỏi thường gặp")
    with st.expander("Quên mật khẩu thì có lấy lại nội dung được không?"):
        st.write("Không. Mã hoá dùng AES-256-GCM, không có cách khôi phục nếu không có đúng mật khẩu.")
    with st.expander("Tại sao DCT báo 'không đủ sức chứa' với ảnh nhỏ?"):
        st.write(
            "DCT chỉ giấu được **1 bit mỗi khối 8×8 điểm ảnh**, nên sức chứa nhỏ hơn nhiều so "
            "với LSB/PVD. Dùng ảnh lớn hơn (≥512×512) nếu nội dung dài."
        )
    with st.expander("Tại sao DWT-SVD cho NC thấp hơn DWT?"):
        st.write(
            "Đây là hạn chế của cách cài đặt trích xuất hiện tại (chỉ khôi phục đường chéo "
            "ma trận singular values), không phải lỗi phần mềm. Xem tab So sánh để thấy rõ."
        )


def build_sidebar() -> None:
    """Always-visible cheat sheet so the user doesn't have to open
    the comparison tab just to remember which algorithm does what."""

    with st.sidebar:
        dark = st.toggle(":material/dark_mode: Chế độ tối", value=st.session_state.get("dark_mode", False))
        _apply_dark_mode(dark)

        st.header(":material/checklist: Tra cứu nhanh")

        # NOTE: st.dataframe() does NOT render markdown/icon shortcodes
        # inside cells (its MarkdownColumn only shows them in a
        # click-to-open overlay, not inline) — so these small cheat-sheet
        # tables are built as genuine Markdown tables via st.markdown(),
        # which does render `:material/...:` and `:color[...]` inline.
        st.subheader("Giấu tin")
        stego_table = "| Thuật toán | Tốc độ | Sức chứa | Độ bền |\n|---|---|---|---|\n"
        for name, i in STEGO_ALGORITHM_INFO.items():
            stego_table += f"| **{name}** | {i['speed']} | {i['capacity']} | {i['robustness']} |\n"
        st.markdown(stego_table)

        st.subheader("Thủy vân")
        wm_table = "| Thuật toán | Độ bền |\n|---|---|\n"
        for name, i in WATERMARK_ALGORITHM_INFO.items():
            wm_table += f"| **{name}** | {i['robustness']} |\n"
        st.markdown(wm_table)

        st.caption(
            "PSNR ≥ 40 dB và SSIM ≥ 0.98: mắt thường khó nhận ra thay đổi.\n\n"
            "NC ≥ 0.9: watermark trích xuất gần như hoàn hảo."
        )

        if st.session_state.get("history"):
            st.divider()
            history_df = pd.DataFrame(st.session_state["history"])
            download_excel_button(
                history_df, "lich_su_thao_tac.xlsx", ":material/download: Tải lịch sử (Excel)", sheet_name="Lich su"
            )
            if st.button(":material/delete: Xoá lịch sử"):
                st.session_state["history"] = []
                st.rerun()


def main() -> None:
    """Run Streamlit application."""

    build_sidebar()

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
            border-radius: 16px;
            padding: 28px 32px;
            margin-bottom: 8px;
            color: white;
            font-family: {FONT_FAMILY};
        ">
            <div style="font-size: 1.9em; font-weight: 700; display:flex; align-items:center; gap:12px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 24 24"
                     fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="3" y="11" width="18" height="11" rx="2"></rect>
                    <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
                </svg>
                Hệ thống giấu tin &amp; thủy vân số
            </div>
            <div style="font-size: 1em; opacity: 0.92; margin-top: 6px;">
                Steganography (LSB · PVD · DCT) và Digital Watermarking (DWT · DWT-SVD),
                mã hoá AES-256-GCM
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(
        [
            ":material/lock: Giấu tin",
            ":material/lock_open: Trích xuất",
            ":material/copyright: Thủy vân",
            ":material/bar_chart: So sánh",
            ":material/science: Robustness",
            ":material/search: Steganalysis",
            ":material/menu_book: Hướng dẫn",
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

    with tabs[6]:
        guide_tab()


if __name__ == "__main__":
    main()