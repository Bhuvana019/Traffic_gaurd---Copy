"""
app.py — Traffic Guard AI
--------------------------
Main Streamlit application.
Run with:  streamlit run app.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# STANDARD IMPORTS  (all at module level — never inside render blocks)
# ─────────────────────────────────────────────────────────────────────────────
import os

import streamlit as st
from PIL import Image

from detector import (
    analyze_image,
    get_model_status,
    DEFAULT_CONFIDENCE,
    MODEL_STATUS_CUSTOM,
    MODEL_STATUS_COCO,
    MODEL_STATUS_NO_YOLO,
    MODEL_STATUS_ERROR,
    VIOLATION_LABELS,
    LABEL_COLORS,
)
from utils import pil_to_bytes, resize_for_display, get_confidence_color


def get_label_fixed_color(label: str) -> str:
    """
    Fixed color mapping according to project specification:
      - Green:  Helmet (compliant)
      - Red:    No Helmet (violation)
      - Blue:   Number Plate
      - Yellow: Bike / Motorcycle / Rider
    """
    lbl = (label or "").strip().lower()
    if "no" in lbl and "helmet" in lbl:
        return "#ef4444"  # Red
    elif "helmet" in lbl:
        return "#22c55e"  # Green
    elif "plate" in lbl:
        return "#06b6d4"  # Blue
    elif any(k in lbl for k in ("bike", "motorcycle", "rider", "two wheeler")):
        return "#eab308"  # Yellow
    return "#94a3b8"      # Default Grey



# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be the very first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Traffic Guard AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%);
        color: #e8eaf6;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* Hero */
    .hero-header {
        background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 100%);
        border: 1px solid rgba(59,130,246,0.25);
        border-radius: 20px; padding: 2.5rem 3rem;
        margin-bottom: 2rem; text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
        position: relative; overflow: hidden;
    }
    .hero-header::before {
        content: ""; position: absolute; inset: 0;
        background: radial-gradient(ellipse at 30% 50%, rgba(59,130,246,0.08) 0%, transparent 60%),
                    radial-gradient(ellipse at 70% 50%, rgba(239,68,68,0.06) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.8rem; font-weight: 800; letter-spacing: -0.5px;
        background: linear-gradient(90deg, #60a5fa 0%, #ffffff 50%, #f87171 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin: 0 0 0.4rem 0;
    }
    .hero-subtitle { color: #94a3b8; font-size: 1rem; margin: 0; }
    .hero-badge {
        display: inline-block;
        background: rgba(59,130,246,0.15); border: 1px solid rgba(59,130,246,0.35);
        color: #93c5fd; font-size: 0.72rem; font-weight: 600;
        letter-spacing: 1.5px; text-transform: uppercase;
        padding: 4px 14px; border-radius: 50px; margin-bottom: 1rem;
    }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px; padding: 1.4rem 1.6rem; margin-bottom: 1.2rem;
        backdrop-filter: blur(8px); box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .card-title {
        font-size: 0.72rem; font-weight: 600; letter-spacing: 1.8px;
        text-transform: uppercase; color: #60a5fa; margin: 0 0 1rem 0;
        display: flex; align-items: center; gap: 8px;
    }
    .card-title::after { content: ""; flex: 1; height: 1px; background: rgba(255,255,255,0.07); }

    /* Upload */
    .stFileUploader > div {
        background: rgba(255,255,255,0.03) !important;
        border: 2px dashed rgba(96,165,250,0.4) !important;
        border-radius: 14px !important; transition: border-color 0.3s;
    }
    .stFileUploader > div:hover { border-color: rgba(96,165,250,0.8) !important; }

    /* Buttons */
    .stButton > button {
        width: 100%; padding: 0.75rem 1.5rem; border-radius: 12px;
        font-weight: 700; font-size: 0.95rem; border: none;
        cursor: pointer; transition: all 0.25s ease;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white; box-shadow: 0 4px 16px rgba(37,99,235,0.45);
    }
    .stButton > button:hover {
        transform: translateY(-2px); box-shadow: 0 8px 24px rgba(37,99,235,0.6);
    }

    /* Violation rows */
    .violation-row {
        display: flex; align-items: center; gap: 12px;
        padding: 0.65rem 1rem;
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px; margin-bottom: 0.45rem;
    }
    .violation-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
    .violation-label { flex: 1; font-size: 0.88rem; font-weight: 500; color: #e2e8f0; }
    .violation-count {
        font-size: 0.76rem; font-weight: 700;
        padding: 2px 10px; border-radius: 50px;
    }
    .vc-violation { background:rgba(239,68,68,0.2);color:#fca5a5;border:1px solid rgba(239,68,68,0.3); }
    .vc-normal    { background:rgba(148,163,184,0.1);color:#94a3b8;border:1px solid rgba(148,163,184,0.2); }

    /* Confidence rows */
    .conf-row {
        display: flex; align-items: center; justify-content: space-between;
        padding: 0.55rem 1rem;
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px; margin-bottom: 0.4rem;
    }
    .conf-label { font-size: 0.85rem; color: #cbd5e1; }
    .conf-value { font-size: 0.85rem; font-weight: 700; }

    /* Status banners */
    .status-msg {
        padding: 0.85rem 1.2rem; border-radius: 12px;
        font-size: 0.9rem; font-weight: 500; margin-bottom: 1rem;
    }
    .status-ok             { background:rgba(34,197,94,0.12);  border:1px solid rgba(34,197,94,0.3);  color:#86efac; }
    .status-warn           { background:rgba(234,179,8,0.12);  border:1px solid rgba(234,179,8,0.3);  color:#fde047; }
    .status-error          { background:rgba(239,68,68,0.12);  border:1px solid rgba(239,68,68,0.3);  color:#fca5a5; }
    .status-setup_required { background:rgba(234,179,8,0.10);  border:1px solid rgba(234,179,8,0.4);  color:#fde047; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1020 0%, #0d1b2a 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }
    .model-badge {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 6px 14px; border-radius: 50px;
        font-size: 0.78rem; font-weight: 600; margin-bottom: 0.8rem;
    }
    .model-badge-custom { background:rgba(34,197,94,0.15);border:1px solid rgba(34,197,94,0.4);color:#86efac; }
    .model-badge-coco   { background:rgba(234,179,8,0.15); border:1px solid rgba(234,179,8,0.4); color:#fde047; }
    .model-badge-error  { background:rgba(239,68,68,0.15); border:1px solid rgba(239,68,68,0.4); color:#fca5a5; }
    .cls-pill {
        display: inline-block; padding: 3px 10px; margin: 3px 2px;
        border-radius: 50px; font-size: 0.75rem; font-weight: 500;
        background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); color: #cbd5e1;
    }

    /* Stat boxes */
    .stat-grid { display: flex; gap: 0.7rem; margin-bottom: 1rem; flex-wrap: wrap; }
    .stat-box {
        flex: 1; min-width: 70px;
        background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px; padding: 0.9rem 0.6rem; text-align: center;
    }
    .stat-number { font-size: 1.8rem; font-weight: 800; line-height: 1; }
    .stat-label  { font-size: 0.65rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

    hr { border-color: rgba(255,255,255,0.06); margin: 1.5rem 0; }
    .stImage > div > div { color: #64748b !important; font-size: 0.75rem !important; }

    /* Download */
    .stDownloadButton > button {
        width: 100%;
        background: linear-gradient(135deg, #16a34a 0%, #15803d 100%) !important;
        color: white !important; border: none !important;
        border-radius: 12px !important; padding: 0.75rem 1.5rem !important;
        font-weight: 700 !important; font-size: 0.92rem !important;
        box-shadow: 0 4px 16px rgba(22,163,74,0.4) !important;
        transition: all 0.25s ease !important;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(22,163,74,0.55) !important;
    }

    /* No-Helmet detail rows */
    .nh-row {
        display: flex; align-items: center; gap: 10px;
        padding: 0.6rem 1rem;
        background: rgba(239,68,68,0.07); border: 1px solid rgba(239,68,68,0.2);
        border-radius: 10px; margin-bottom: 0.45rem;
    }
    .nh-idx    { font-size: 0.72rem; color: #94a3b8; min-width: 22px; }
    .nh-label  { flex: 1; font-size: 0.88rem; font-weight: 600; color: #fca5a5; }
    .nh-conf   { font-size: 0.82rem; font-weight: 700; }
    .sev-HIGH   { color: #f87171; }
    .sev-MEDIUM { color: #fbbf24; }
    .sev-LOW    { color: #94a3b8; }

    /* Plate OCR rows */
    .plate-row {
        display: flex; align-items: center; gap: 12px;
        padding: 0.65rem 1rem;
        background: rgba(6,182,212,0.07); border: 1px solid rgba(6,182,212,0.25);
        border-radius: 10px; margin-bottom: 0.45rem;
    }
    .plate-icon { font-size: 1.1rem; }
    .plate-text {
        flex: 1; font-family: 'Courier New', monospace;
        font-size: 0.95rem; font-weight: 700; color: #67e8f9; letter-spacing: 2px;
    }
    .plate-unclear { flex: 1; font-size: 0.88rem; color: #64748b; font-style: italic; }
    .plate-conf    { font-size: 0.78rem; color: #94a3b8; }

    /* Rider estimate */
    .rider-box {
        background: rgba(139,92,246,0.08); border: 1px solid rgba(139,92,246,0.25);
        border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.6rem;
    }
    .rider-triple {
        background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.35);
        color: #fca5a5; border-radius: 8px; padding: 0.55rem 1rem;
        font-size: 0.88rem; font-weight: 600; margin-top: 0.5rem;
    }
    .rider-safe {
        background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.3);
        color: #86efac; border-radius: 8px; padding: 0.55rem 1rem;
        font-size: 0.88rem; font-weight: 600; margin-top: 0.5rem;
    }
    .rider-na { color: #64748b; font-size: 0.85rem; font-style: italic; padding: 0.4rem 0; }
    .rider-assoc-row {
        display: flex; justify-content: space-between;
        padding: 0.35rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);
        font-size: 0.83rem;
    }
    .rider-assoc-key { color: #94a3b8; }
    .rider-assoc-val { color: #e2e8f0; font-weight: 600; }

    /* Limitation notices */
    .limit-note {
        display: flex; gap: 10px; align-items: flex-start;
        padding: 0.6rem 1rem;
        background: rgba(234,179,8,0.07); border: 1px solid rgba(234,179,8,0.2);
        border-radius: 8px; margin-bottom: 0.4rem;
        font-size: 0.82rem; color: #fde68a; line-height: 1.45;
    }
    .limit-icon { flex-shrink: 0; font-size: 0.95rem; margin-top: 1px; }

    /* Legend swatches */
    .legend-row {
        display: flex; align-items: center; gap: 10px;
        padding: 0.35rem 0; font-size: 0.82rem; color: #cbd5e1;
    }
    .legend-swatch {
        width: 14px; height: 14px; border-radius: 3px; flex-shrink: 0;
        border: 1px solid rgba(255,255,255,0.15);
    }

    /* Error alert */
    .error-box {
        background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.35);
        border-radius: 12px; padding: 1.2rem 1.4rem; color: #fca5a5;
        font-size: 0.88rem; line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Model Status + Settings + Legend
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        '<p style="font-size:1.1rem;font-weight:800;color:#e2e8f0;margin:0 0 1.2rem 0;">'
        '⚙️ Settings</p>',
        unsafe_allow_html=True,
    )

    # ── Model status ──────────────────────────────────────────────────────────
    model_status = get_model_status()
    ms = model_status["status"]

    badge_cls = (
        "model-badge-custom" if ms == MODEL_STATUS_CUSTOM else
        "model-badge-coco"   if ms == MODEL_STATUS_COCO   else
        "model-badge-error"
    )
    badge_icon = "✅" if ms == MODEL_STATUS_CUSTOM else "⚠️" if ms == MODEL_STATUS_COCO else "❌"
    badge_text = (
        "Custom Weights" if ms == MODEL_STATUS_CUSTOM else
        "COCO Fallback"  if ms == MODEL_STATUS_COCO   else
        "Not Available"
    )

    st.markdown(
        f'<div class="card">'
        f'<p class="card-title">🤖 Model Status</p>'
        f'<div class="model-badge {badge_cls}">{badge_icon} {badge_text}</div>'
        f'<p style="font-size:0.78rem;color:#64748b;margin:0;line-height:1.5;">{model_status["message"]}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Setup instructions ────────────────────────────────────────────────────
    if ms != MODEL_STATUS_CUSTOM:
        with st.expander("📥 How to add custom weights", expanded=False):
            st.markdown(
                """
**Step 1 — Get weights**
Train or download a YOLO helmet-detection model.
Good source: [Roboflow Universe](https://universe.roboflow.com) → search *helmet detection*

**Step 2 — Place the file**
```
Traffic_gaurd/
└── models/
    └── best.pt   ← put it here
```

**Step 3 — Restart the app**
```powershell
streamlit run app.py
```
The model loads automatically.
                """
            )

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:1rem 0'>", unsafe_allow_html=True)

    # ── Confidence threshold ──────────────────────────────────────────────────
    st.markdown(
        '<p class="card-title" style="margin-bottom:0.3rem;">🎯 Confidence Threshold</p>',
        unsafe_allow_html=True,
    )
    confidence_threshold = st.slider(
        label="confidence_threshold",
        min_value=0.10,
        max_value=0.95,
        value=DEFAULT_CONFIDENCE,
        step=0.05,
        format="%.2f",
        label_visibility="collapsed",
        help="Detections below this score are ignored. Lower = more detections, higher = more precise.",
    )
    st.caption(
        f"Current: **{confidence_threshold:.0%}** — "
        + ("stricter" if confidence_threshold > 0.5 else "lenient")
    )

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:1rem 0'>", unsafe_allow_html=True)

    # ── Bounding-box colour legend ────────────────────────────────────────────
    st.markdown(
        '<p class="card-title">🎨 Box Color Legend</p>',
        unsafe_allow_html=True,
    )
    LEGEND_ITEMS = [
        ("#dc2626", "No Helmet  (violation)"),
        ("#22c55e", "Helmet  (compliant)"),
        ("#f59e0b", "Motorcycle / Bike"),
        ("#06b6d4", "Number Plate"),
        ("#fbbf24", "Person / Rider"),
        ("#94a3b8", "Other objects"),
    ]
    legend_html = ""
    for hex_color, label in LEGEND_ITEMS:
        legend_html += (
            f'<div class="legend-row">'
            f'<div class="legend-swatch" style="background:{hex_color};"></div>'
            f'<span>{label}</span></div>'
        )
    st.markdown(legend_html, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:1rem 0'>", unsafe_allow_html=True)

    # ── Model classes (shown after first analysis) ────────────────────────────
    if "results" in st.session_state and st.session_state.results is not None:
        minfo = st.session_state.results.get("model_info", {})
        raw_names = minfo.get("raw_class_names", {})
        disp_names = minfo.get("display_names", {})
        if raw_names:
            st.markdown(
                f'<p class="card-title">📂 Model Classes ({len(raw_names)})</p>',
                unsafe_allow_html=True,
            )
            pills_html = ""
            for cls_id in sorted(raw_names.keys()):
                raw  = raw_names[cls_id]
                disp = disp_names.get(cls_id, raw)
                pills_html += f'<span class="cls-pill" title="ID {cls_id}: {raw}">{disp}</span>'
            st.markdown(pills_html, unsafe_allow_html=True)
            st.caption("Hover a pill to see raw dataset class name.")
    else:
        st.markdown(
            '<p style="color:#334155;font-size:0.78rem;">'
            'Run an analysis to see model class names here.</p>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#1e293b;font-size:0.7rem;text-align:center;">Traffic Guard AI — College Project</p>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div class="hero-header">
        <div class="hero-badge">🚦 AI-Powered Surveillance</div>
        <h1 class="hero-title">Traffic Guard AI</h1>
        <p class="hero-subtitle">
            Upload a traffic image to detect violations automatically.
            Side-by-side original vs. annotated view, confidence scores, OCR plate reading
            and a downloadable report — all in one place.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — Upload + Controls
# ══════════════════════════════════════════════════════════════════════════════

col_left, col_right = st.columns([1.0, 2.0], gap="large")

with col_left:
    # ── Upload card ──
    st.markdown('<div class="card"><p class="card-title">📁 Upload Image</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label="Drag & drop or click to browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Validate & open image ──
    pil_image     = None
    image_error   = None

    if uploaded_file:
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")
            w, h = pil_image.size
            if w < 32 or h < 32:
                image_error = "❌ Image is too small (minimum 32×32 px). Please upload a larger image."
                pil_image = None
        except Exception as exc:
            image_error = f"❌ Could not open image: {exc}"
            pil_image = None

    if image_error:
        st.markdown(f'<div class="error-box">{image_error}</div>', unsafe_allow_html=True)

    # ── Preview ──
    if pil_image is not None:
        st.markdown('<div class="card"><p class="card-title">🖼️ Original Image</p>', unsafe_allow_html=True)
        st.image(
            resize_for_display(pil_image, max_width=500),
            use_container_width=True,
            caption=f"{uploaded_file.name}  •  {pil_image.size[0]}×{pil_image.size[1]} px",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Analyze button ──
    analyze_clicked = st.button(
        "🔍  Analyze Image",
        disabled=(pil_image is None),
        use_container_width=True,
    )
    if pil_image is None and not image_error:
        st.caption("⬆️  Please upload an image to enable analysis.")

    # ── Supported detections card ──
    st.markdown(
        """
        <div class="card" style="margin-top:1rem;">
            <p class="card-title">ℹ️ Supported Detections</p>
            <div style="display:flex;flex-direction:column;gap:6px;">
                <span style="color:#fca5a5;font-size:0.82rem;">🔴 No Helmet (violation)</span>
                <span style="color:#86efac;font-size:0.82rem;">🟢 Helmet (compliant)</span>
                <span style="color:#f59e0b;font-size:0.82rem;">🟡 Motorcycle / Bike</span>
                <span style="color:#67e8f9;font-size:0.82rem;">🔵 Number Plate + OCR</span>
                <span style="color:#fbbf24;font-size:0.82rem;">⚪ Person / Rider</span>
                <span style="color:#fde047;font-size:0.82rem;">⚠️ Triple riding estimate</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — Results
# ══════════════════════════════════════════════════════════════════════════════

with col_right:

    # ── Session state ──
    if "results" not in st.session_state:
        st.session_state.results = None

    # ── Run analysis ──
    if analyze_clicked and pil_image is not None:
        with st.spinner("🔍 Analysing image — please wait…"):
            st.session_state.results = analyze_image(
                pil_image,
                confidence_threshold=confidence_threshold,
            )

    results = st.session_state.results

    # ── Placeholder ──
    if results is None:
        st.markdown(
            """
            <div class="card" style="text-align:center;padding:4rem 2rem;">
                <div style="font-size:4rem;margin-bottom:1rem;">📊</div>
                <p style="color:#64748b;font-size:1rem;margin:0;">
                    Analysis results will appear here.<br>
                    Upload an image and click <strong style="color:#60a5fa;">Analyze Image</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        # ── Guard: error state ────────────────────────────────────────────────
        res_status = results.get("status", "ok")
        msg        = results["message"]
        total      = results["total_violations"]

        if res_status == "error":
            st.markdown(
                f'<div class="error-box"><strong>Analysis Error</strong><br>{msg.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True,
            )
            # Still offer model info if available
            minfo = results.get("model_info", {})
            if minfo.get("model_status") == MODEL_STATUS_NO_YOLO:
                st.info("**Fix:** `pip install ultralytics`")

        else:
            # ── Status banner ─────────────────────────────────────────────────
            css_cls = (
                "status-ok"             if res_status == "ok" and total == 0 else
                "status-warn"           if res_status == "ok" and total > 0  else
                "status-setup_required" if res_status == "setup_required"    else
                "status-error"
            )
            msg_html = msg.replace("\n", "<br>")
            st.markdown(f'<div class="status-msg {css_cls}">{msg_html}</div>', unsafe_allow_html=True)

            # ── Model info strip ──────────────────────────────────────────────
            minfo       = results.get("model_info", {})
            loaded_from = minfo.get("loaded_from", "unknown")
            n_classes   = minfo.get("num_classes", 0)
            threshold   = minfo.get("threshold", DEFAULT_CONFIDENCE)
            weights_name = os.path.basename(loaded_from) if loaded_from else "—"
            st.markdown(
                f'<p style="font-size:0.75rem;color:#334155;margin:0 0 0.8rem 0;">'
                f'Model: <code>{weights_name}</code> &nbsp;|&nbsp; '
                f'Classes: <strong>{n_classes}</strong> &nbsp;|&nbsp; '
                f'Threshold: <strong>{threshold:.0%}</strong> &nbsp;|&nbsp; '
                f'Objects found: <strong>{results.get("total_detections", 0)}</strong>'
                f'</p>',
                unsafe_allow_html=True,
            )

            # ── Stat grid ─────────────────────────────────────────────────────
            vr           = results.get("violation_report", {})
            obj_counts   = vr.get("object_counts", {})
            total_det    = results.get("total_detections", 0)
            moto_count   = obj_counts.get("motorcycles", 0)
            helmet_count = obj_counts.get("helmets", 0)
            no_hmt_count = obj_counts.get("no_helmets", 0)
            plate_count  = obj_counts.get("number_plates", 0)
            rider_count  = vr.get("rider_estimate", {}).get("total_riders", obj_counts.get("riders", helmet_count + no_hmt_count))
            status_color = "#ef4444" if total > 0 else "#34d399"
            status_icon  = "⚠️" if total > 0 else "✅"

            st.markdown(
                f"""
                <div class="stat-grid">
                    <div class="stat-box">
                        <div class="stat-number" style="color:#f87171;">{total}</div>
                        <div class="stat-label">Violations</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number" style="color:#60a5fa;">{total_det}</div>
                        <div class="stat-label">Objects</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number" style="color:#f59e0b;">{moto_count}</div>
                        <div class="stat-label">Motorcycles</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number" style="color:#eab308;">{rider_count}</div>
                        <div class="stat-label">Riders</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number" style="color:#22c55e;">{helmet_count}</div>
                        <div class="stat-label">Helmets</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number" style="color:#ef4444;">{no_hmt_count}</div>
                        <div class="stat-label">No Helmet</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number" style="color:#06b6d4;">{plate_count}</div>
                        <div class="stat-label">Plates</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number" style="color:{status_color};">{status_icon}</div>
                        <div class="stat-label">Status</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Overall Description & Incident Summary Card ────────────────────
            overall_sum = vr.get("overall_summary", {})
            if overall_sum:
                ov_headline  = overall_sum.get("headline", "Traffic Surveillance Summary")
                ov_desc      = overall_sum.get("description", "")
                ov_badge_col = overall_sum.get("badge_color", "#3b82f6")
                is_viol      = overall_sum.get("status_type") == "VIOLATION"
                card_border  = "border-left: 5px solid #ef4444;" if is_viol else "border-left: 5px solid #22c55e;"

                st.markdown(
                    f"""
                    <div class="card" style="{card_border}background:linear-gradient(135deg, rgba(15,23,42,0.95) 0%, rgba(30,41,59,0.85) 100%);padding:1.2rem 1.4rem;margin-bottom:1.2rem;">
                        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;flex-wrap:wrap;gap:8px;">
                            <span style="font-size:1.05rem;font-weight:700;color:#f8fafc;">📋 Incident Description &amp; Overall Summary</span>
                            <span style="background:{ov_badge_col}22;border:1px solid {ov_badge_col}66;color:{ov_badge_col};font-size:0.78rem;font-weight:700;padding:4px 12px;border-radius:50px;">
                                {ov_headline}
                            </span>
                        </div>
                        <p style="font-size:0.92rem;color:#e2e8f0;line-height:1.65;margin:0 0 0.8rem 0;">
                            {ov_desc}
                        </p>
                        <div style="display:flex;gap:0.6rem;flex-wrap:wrap;">
                            <span style="display:inline-flex;align-items:center;gap:6px;font-size:0.75rem;padding:3px 10px;border-radius:50px;background:rgba(245,158,11,0.12);border:1px solid rgba(245,158,11,0.3);color:#fbbf24;">
                                🏍️ Motorcycles: {moto_count}
                            </span>
                            <span style="display:inline-flex;align-items:center;gap:6px;font-size:0.75rem;padding:3px 10px;border-radius:50px;background:rgba(234,179,8,0.12);border:1px solid rgba(234,179,8,0.3);color:#fde047;">
                                👤 Riders: {rider_count}
                            </span>
                            <span style="display:inline-flex;align-items:center;gap:6px;font-size:0.75rem;padding:3px 10px;border-radius:50px;background:rgba(34,197,94,0.12);border:1px solid rgba(34,197,94,0.3);color:#86efac;">
                                🪖 Helmets: {helmet_count}
                            </span>
                            <span style="display:inline-flex;align-items:center;gap:6px;font-size:0.75rem;padding:3px 10px;border-radius:50px;background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.3);color:#fca5a5;">
                                ⚠️ No Helmet: {no_hmt_count}
                            </span>
                            <span style="display:inline-flex;align-items:center;gap:6px;font-size:0.75rem;padding:3px 10px;border-radius:50px;background:rgba(6,182,212,0.12);border:1px solid rgba(6,182,212,0.3);color:#67e8f9;">
                                🔢 Plates: {plate_count}
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ── Side-by-side: Original vs Annotated ──────────────────────────
            st.markdown('<div class="card"><p class="card-title">🖼️ Detection Result</p>', unsafe_allow_html=True)
            img_col1, img_col2 = st.columns(2, gap="small")

            with img_col1:
                st.image(
                    resize_for_display(pil_image, max_width=600),
                    use_container_width=True,
                    caption="Original image",
                )
            with img_col2:
                st.image(
                    resize_for_display(results["annotated_image"], max_width=600),
                    use_container_width=True,
                    caption="Annotated — bounding boxes by detection model",
                )
            st.markdown("</div>", unsafe_allow_html=True)

            # ── Violation Summary + Confidence Scores (side by side) ──────────
            rc1, rc2 = st.columns(2, gap="medium")

            with rc1:
                st.markdown('<div class="card"><p class="card-title">📋 Violation Summary</p>', unsafe_allow_html=True)

                DOT_COLORS = {
                    "No Helmet":    "#ef4444",
                    "Helmet":       "#22c55e",
                    "No Seat Belt": "#ef4444",
                    "Seat Belt":    "#22c55e",
                    "Motorcycle":   "#f59e0b",
                    "Bike":         "#f59e0b",
                    "Number Plate": "#06b6d4",
                    "Person":       "#eab308",
                    "Rider":        "#eab308",
                    "Car":          "#94a3b8",
                    "Pedestrian":   "#94a3b8",
                }

                if results["summary"]:
                    for label, count in results["summary"].items():
                        dot_color = DOT_COLORS.get(label, get_label_fixed_color(label))
                        is_viol   = label in VIOLATION_LABELS
                        vc_cls    = "vc-violation" if is_viol else "vc-normal"
                        st.markdown(
                            f"""
                            <div class="violation-row">
                                <div class="violation-dot" style="background:{dot_color};
                                     box-shadow:0 0 6px {dot_color};"></div>
                                <span class="violation-label">{label}</span>
                                <span class="violation-count {vc_cls}">{count}×</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        '<p style="color:#64748b;font-size:0.87rem;text-align:center;padding:1rem 0;">'
                        "No objects detected.</p>",
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            with rc2:
                st.markdown('<div class="card"><p class="card-title">📈 Per-Detection Confidence</p>', unsafe_allow_html=True)

                if results["violations"]:
                    for v in results["violations"]:
                        conf     = v["confidence"]
                        color    = get_label_fixed_color(v["label"])
                        conf_pct = f"{conf * 100:.1f}%"
                        raw_lbl  = v.get("raw_label", "")
                        label_display = v["label"]
                        if raw_lbl and raw_lbl.lower() != v["label"].lower():
                            label_display = f"{v['label']} <span style='font-size:0.72rem;color:#475569;'>({raw_lbl})</span>"
                        st.markdown(
                            f"""
                            <div class="conf-row">
                                <div style="display:flex;align-items:center;gap:8px;">
                                    <span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:{color};box-shadow:0 0 6px {color};"></span>
                                    <span class="conf-label">{label_display}</span>
                                </div>
                                <span class="conf-value" style="color:{color};font-weight:700;">{conf_pct}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    st.caption("🟢 Green: Helmet (Compliant) · 🔴 Red: No Helmet (Violation) · 🟡 Yellow: Motorcycle/Rider · 🔵 Blue: Number Plate")
                else:
                    st.markdown(
                        '<p style="color:#64748b;font-size:0.87rem;text-align:center;padding:1rem 0;">'
                        "Confidence scores appear here<br>once detections are available.</p>",
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            # ── Download annotated image ──────────────────────────────────────
            img_bytes = pil_to_bytes(results["annotated_image"], fmt="PNG")
            st.download_button(
                label="⬇️  Download Annotated Image (PNG)",
                data=img_bytes,
                file_name="traffic_guard_result.png",
                mime="image/png",
                use_container_width=True,
            )

            # ─────────────────────────────────────────────────────────────────
            # ENRICHED REPORT — No-Helmet / Plate OCR / Rider Estimate
            # ─────────────────────────────────────────────────────────────────

            det_col1, det_col2, det_col3 = st.columns(3, gap="medium")

            # ── No-Helmet Detail Summary ──────────────────────────────────────
            with det_col1:
                nh_summaries = vr.get("no_helmet_summaries", [])
                st.markdown(
                    '<div class="card"><p class="card-title">🪖 No-Helmet Details</p>',
                    unsafe_allow_html=True,
                )
                if nh_summaries:
                    for i, nh in enumerate(nh_summaries, 1):
                        sev = nh["severity"]
                        st.markdown(
                            f"""
                            <div class="nh-row">
                                <span class="nh-idx">#{i}</span>
                                <span class="nh-label">{nh['label']}</span>
                                <span class="nh-conf sev-{sev}">{nh['conf_pct']}</span>
                                <span class="sev-{sev}" style="font-size:0.72rem;font-weight:600;">{sev}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    st.caption("HIGH ≥ 80% · MEDIUM ≥ 55% · LOW < 55%")
                else:
                    st.markdown(
                        '<p style="color:#64748b;font-size:0.85rem;padding:0.5rem 0;">'
                        "No explicit No-Helmet detections.<br>"
                        "<span style='font-size:0.78rem;color:#475569;'>"
                        "⚠️ Absence ≠ compliance.</span></p>",
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            # ── Number Plate OCR ──────────────────────────────────────────────
            with det_col2:
                plate_readings = vr.get("plate_readings", [])
                ocr_available  = vr.get("ocr_available", False)
                st.markdown(
                    '<div class="card"><p class="card-title">🔢 Plate Recognition</p>',
                    unsafe_allow_html=True,
                )
                if not ocr_available:
                    st.markdown(
                        '<p style="color:#64748b;font-size:0.83rem;">'
                        "EasyOCR not installed.<br>"
                        "<code style='font-size:0.78rem;'>pip install easyocr</code></p>",
                        unsafe_allow_html=True,
                    )
                elif not plate_readings:
                    st.markdown(
                        '<p style="color:#64748b;font-size:0.85rem;padding:0.5rem 0;">'
                        "No plate boxes detected by the model.<br>"
                        "<span style='font-size:0.78rem;color:#475569;'>"
                        "OCR runs only on detected plate regions.</span></p>",
                        unsafe_allow_html=True,
                    )
                else:
                    UNCLEAR_TEXTS = {
                        "Plate text unclear",
                        "Plate crop too small for OCR",
                        "OCR initialisation failed",
                    }
                    UNCLEAR_SOURCES = {"unavailable", "error", "skipped"}
                    for pr in plate_readings:
                        is_clear = (
                            pr["display_text"] not in UNCLEAR_TEXTS
                            and pr["source"] not in UNCLEAR_SOURCES
                        )
                        if is_clear:
                            conf_str = f"{pr['confidence']*100:.0f}%" if pr["confidence"] > 0 else ""
                            st.markdown(
                                f'<div class="plate-row">'
                                f'<span class="plate-icon">🔵</span>'
                                f'<span class="plate-text">{pr["display_text"]}</span>'
                                f'<span class="plate-conf">{conf_str}</span></div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f'<div class="plate-row">'
                                f'<span class="plate-icon">❓</span>'
                                f'<span class="plate-unclear">{pr["display_text"]}</span></div>',
                                unsafe_allow_html=True,
                            )
                    st.caption("Min OCR confidence: 40%. Verify with ANPR hardware.")
                st.markdown("</div>", unsafe_allow_html=True)

            # ── Rider Estimate & Triple Riding ────────────────────────────────
            with det_col3:
                rider_est = vr.get("rider_estimate", {})
                st.markdown(
                    '<div class="card"><p class="card-title">🚴 Rider Detection &amp; Estimate</p>',
                    unsafe_allow_html=True,
                )
                if not rider_est:
                    st.markdown(
                        '<p class="rider-na">Rider module did not run.</p>',
                        unsafe_allow_html=True,
                    )
                else:
                    n_moto   = rider_est.get("total_motorcycles", 0)
                    n_riders = rider_est.get("total_riders", rider_est.get("total_persons", 0))
                    triple   = rider_est.get("possible_triple_riding", False)
                    note     = rider_est.get("reliability_note", "")
                    assocs   = rider_est.get("associations", [])

                    st.markdown(
                        f"""
                        <div class="rider-box">
                            <div class="rider-assoc-row">
                                <span class="rider-assoc-key">Motorcycles</span>
                                <span class="rider-assoc-val">{n_moto}</span>
                            </div>
                            <div class="rider-assoc-row">
                                <span class="rider-assoc-key">Riders Detected</span>
                                <span class="rider-assoc-val" style="color:#eab308;font-weight:700;">{n_riders}</span>
                            </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    for j, assoc in enumerate(assocs, 1):
                        rc      = assoc["rider_count"]
                        est_tag = " (est.)" if assoc.get("is_estimate") else ""
                        st.markdown(
                            f"""
                            <div class="rider-assoc-row">
                                <span class="rider-assoc-key">Bike #{j} riders</span>
                                <span class="rider-assoc-val">~{rc}{est_tag}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    st.markdown("</div>", unsafe_allow_html=True)

                    if triple:
                        st.markdown(
                            '<div class="rider-triple">⚠️ Triple Riding Detected<br>'
                            '<span style="font-size:0.75rem;font-weight:400;">'
                            '≥3 riders on one motorcycle (passenger capacity violation).</span></div>',
                            unsafe_allow_html=True,
                        )
                    elif n_moto > 0:
                        st.markdown(
                            '<div class="rider-safe">✅ Safe passenger capacity<br>'
                            '<span style="font-size:0.75rem;font-weight:400;">'
                            'No triple riding detected on motorcycle(s).</span></div>',
                            unsafe_allow_html=True,
                        )
                    elif n_riders > 0:
                        st.markdown(
                            f'<div class="rider-safe">👤 {n_riders} rider(s) detected in scene</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<p class="rider-na">No motorcycles or riders detected.</p>',
                            unsafe_allow_html=True,
                        )
                    if note:
                        st.caption(note)
                st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <p style="text-align:center;color:#334155;font-size:0.78rem;margin:0;">
        Traffic Guard AI — College Project &nbsp;|&nbsp;
        Built with Streamlit &amp; Ultralytics YOLOv8 &nbsp;|&nbsp;
        EasyOCR plate recognition &nbsp;|&nbsp;
        Model: plug in <code>models/best.pt</code>
    </p>
    """,
    unsafe_allow_html=True,
)
