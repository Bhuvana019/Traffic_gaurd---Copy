"""
utils.py — Traffic Guard AI
-----------------------------
Image conversion helpers and OpenCV annotation engine.

ANNOTATION COLOR RULES (BGR format, as used by OpenCV)
────────────────────────────────────────────────────────
  Red    (0,   0,   220)  →  No Helmet, No Seat Belt, Red Light  (violations)
  Green  (0,   200,  70)  →  Helmet, Seat Belt  (compliant)
  Blue   (220,  80,   0)  →  Number Plate, License Plate
  Yellow (0,   210, 230)  →  Bike, Motorcycle, Rider  (vehicle/rider annotations)
  Grey   (160, 160, 160)  →  any label not in the map above

These rules are enforced through ANNOTATION_COLOR_MAP below.
detector.py's LABEL_COLORS is ignored at draw time — the
canonical source of truth for box colour is THIS file.
"""

import io
import math
import cv2
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATION COLOR MAP  (BGR)
#
# Rule-based assignment:
#   Red    → anything containing "no" + "helmet" or "no" + "seat belt"
#   Green  → compliant helmet / seat-belt detections
#   Blue   → number plate / license plate
#   Yellow → bike / motorcycle / rider / vehicle count annotations
#   Grey   → everything else
#
# Keys are the display labels produced by detector._map_class_name().
# Dataset raw names from this project:
#   'bike' → 'Bike',  'helmet' → 'Helmet',
#   'no-helmet' → 'No Helmet',  'number-plate' → 'Number Plate'
# ─────────────────────────────────────────────────────────────────────────────

# BGR tuples
_RED    = (0,   0,   220)   # violations
_GREEN  = (0,   200,  70)   # compliant
_BLUE   = (220,  80,   0)   # number plate
_YELLOW = (0,   210, 230)   # vehicle / rider
_GREY   = (160, 160, 160)   # fallback

ANNOTATION_COLOR_MAP: dict[str, tuple] = {
    # ── Violations → RED ────────────────────────────────────────────────────
    "No Helmet":           _RED,
    "No Seat Belt":        _RED,
    "Red Light":           _RED,
    "No-Helmet":           _RED,   # raw dataset variant with hyphen
    "no-helmet":           _RED,

    # ── Compliant → GREEN ────────────────────────────────────────────────────
    "Helmet":              _GREEN,
    "With Helmet":         _GREEN,
    "Seat Belt":           _GREEN,
    "Green Light":         _GREEN,

    # ── Number / License Plate → BLUE ────────────────────────────────────────
    "Number Plate":        _BLUE,
    "Number-Plate":        _BLUE,   # raw dataset variant
    "number-plate":        _BLUE,
    "License Plate":       _BLUE,
    "Licence Plate":       _BLUE,
    "Numberplate":         _BLUE,

    # ── Vehicle / Rider → YELLOW ─────────────────────────────────────────────
    "Bike":                _YELLOW,
    "bike":                _YELLOW,
    "Bicycle":             _YELLOW,
    "Motorcycle":          _YELLOW,
    "Motorbike":           _YELLOW,
    "Rider":               _YELLOW,
    "Two Wheeler":         _YELLOW,
    "Vehicle":             _YELLOW,

    # ── Other detections → GREY ──────────────────────────────────────────────
    "Person":              _GREY,
    "Pedestrian":          _GREY,
    "Car":                 _GREY,
    "Truck":               _GREY,
    "Bus":                 _GREY,
    "Traffic Light":       _GREY,
}

# Exported for tests
DEFAULT_BOX_COLOR: tuple = _GREY


def get_annotation_color(label: str) -> tuple:
    """
    Return the BGR annotation colour for a given display label.

    Strategy (applied in order):
      1. Exact key lookup in ANNOTATION_COLOR_MAP.
      2. Case-insensitive exact lookup.
      3. Keyword-based rule:
           contains "no" AND ("helmet" OR "seat")  → RED
           contains "helmet" OR "seat belt"          → GREEN
           contains "plate" OR "license"             → BLUE
           contains "bike" OR "motor" OR "rider"     → YELLOW
      4. Fallback: GREY.

    Parameters
    ----------
    label : str — display label from detector (e.g. "No Helmet")

    Returns
    -------
    BGR tuple e.g. (0, 0, 220)
    """
    # 1 — Exact match
    if label in ANNOTATION_COLOR_MAP:
        return ANNOTATION_COLOR_MAP[label]

    # 2 — Case-insensitive
    lower = label.lower()
    for key, color in ANNOTATION_COLOR_MAP.items():
        if key.lower() == lower:
            return color

    # 3 — Keyword rules
    words = lower.replace("-", " ").replace("_", " ").split()
    has_no    = "no" in words or "without" in words or "not" in words
    has_hlmt  = "helmet" in words
    has_belt  = "seat" in words or "belt" in words
    has_plate = "plate" in words or "license" in words or "licence" in words or "number" in words
    has_bike  = any(w in words for w in ("bike", "motor", "motorcycle", "rider",
                                          "bicycle", "two", "wheeler", "vehicle"))

    if has_no and (has_hlmt or has_belt):
        return _RED
    if has_hlmt or has_belt:
        return _GREEN
    if has_plate:
        return _BLUE
    if has_bike:
        return _YELLOW

    return DEFAULT_BOX_COLOR


# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

# Minimum side length (px) of the bounding box to render a label.
# Boxes smaller than this are still drawn but without the text pill.
_MIN_BOX_FOR_LABEL_PX = 20


def draw_violation_boxes(
    cv_image:      np.ndarray,
    detections:    list,
    color_map:     dict | None = None,
    default_color: tuple       = DEFAULT_BOX_COLOR,
) -> np.ndarray:
    """
    Draw bounding boxes with class-name + confidence labels on an OpenCV image.

    Only draws what the model actually detected — no synthetic boxes are added.
    Preserves the original image dimensions throughout.

    Parameters
    ----------
    cv_image      : BGR NumPy array.  Modified in-place; caller should pass a copy.
    detections    : List of detection dicts from detector._run_inference().
                    Each dict must have 'label' (str), 'confidence' (float 0-1),
                    'box' (tuple of 4 ints: x1 y1 x2 y2).
    color_map     : Optional override mapping label→BGR.  If None, uses
                    ANNOTATION_COLOR_MAP (the canonical source).
    default_color : BGR fallback for labels not in the map.

    Returns
    -------
    Annotated BGR NumPy array with the same shape as cv_image.
    """
    if cv_image is None or cv_image.size == 0:
        return cv_image

    h_img, w_img = cv_image.shape[:2]

    for det in detections:
        label      = det.get("label", "Unknown")
        confidence = det.get("confidence", 0.0)
        box        = det.get("box")

        if box is None:
            continue  # skip malformed detections

        # ── Clamp box coordinates to image bounds ─────────────────────────────
        x1, y1, x2, y2 = _clamp_box(box, w_img, h_img)

        # Skip zero-area boxes
        if x2 <= x1 or y2 <= y1:
            continue

        # ── Pick colour ───────────────────────────────────────────────────────
        if color_map is not None and label in color_map:
            color = color_map[label]
        else:
            color = get_annotation_color(label)

        # ── Adaptive stroke width (thicker for larger images) ─────────────────
        stroke = _adaptive_stroke(w_img, h_img)

        # ── Draw bounding rectangle ───────────────────────────────────────────
        cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, thickness=stroke)

        # ── Draw corner accents for a professional look ───────────────────────
        _draw_corner_accents(cv_image, x1, y1, x2, y2, color, stroke)

        # ── Build label string: "NO HELMET 87.4%" ─────────────────────────────
        text = f"{label.upper()}  {confidence * 100:.1f}%"

        # ── Draw text pill only if box is large enough to be readable ─────────
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w >= _MIN_BOX_FOR_LABEL_PX and box_h >= _MIN_BOX_FOR_LABEL_PX:
            _draw_label_pill(cv_image, text, x1, y1, x2, color, w_img, h_img)

    return cv_image


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _draw_label_pill(
    cv_image: np.ndarray,
    text:     str,
    x1:       int,
    y1:       int,
    x2:       int,
    color:    tuple,
    w_img:    int,
    h_img:    int,
) -> None:
    """
    Draw a filled rectangle (pill) above the bounding box with the class label.

    Rules
    ─────
    • If the pill would go above the image top, draw it INSIDE the top of the box.
    • If the pill would go past the right edge, anchor it to the right.
    • Text is always white for maximum contrast against any box colour.
    • A thin dark outline is added to the text for readability on light backgrounds.
    """
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = _adaptive_font_scale(w_img)
    thickness  = max(1, int(font_scale * 1.5))

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = max(3, int(font_scale * 5))

    pill_w = tw + 2 * pad
    pill_h = th + 2 * pad + baseline

    # Horizontal: anchor left, clamp to right edge
    px1 = x1
    px2 = min(px1 + pill_w, w_img)
    px1 = max(0, px2 - pill_w)   # re-anchor if clamped

    # Vertical: prefer above the box; fall back inside the box top
    if y1 - pill_h >= 0:
        py1 = y1 - pill_h
        py2 = y1
        text_y = py2 - pad - baseline
    else:
        py1 = y1
        py2 = y1 + pill_h
        text_y = py2 - pad - baseline

    # Filled background
    cv2.rectangle(cv_image, (px1, py1), (px2, py2), color, cv2.FILLED)

    text_x = px1 + pad

    # Dark outline for contrast (draw text shifted in 4 directions)
    outline_color = (0, 0, 0)
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        cv2.putText(
            cv_image, text,
            (text_x + dx, text_y + dy),
            font, font_scale, outline_color, thickness + 1, cv2.LINE_AA,
        )

    # White foreground text
    cv2.putText(
        cv_image, text,
        (text_x, text_y),
        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )


def _draw_corner_accents(
    cv_image: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple,
    stroke: int,
) -> None:
    """
    Draw L-shaped corner accents at all four corners of the bounding box.
    This makes boxes visually distinct from plain rectangles, especially when
    boxes overlap.

    The accent length is 15 % of the shorter box side, clamped between 6–30 px.
    """
    side   = min(x2 - x1, y2 - y1)
    length = int(max(6, min(30, side * 0.15)))
    sw     = stroke + 1   # slightly thicker than the main box

    corners = [
        # top-left
        [(x1, y1), (x1 + length, y1)],
        [(x1, y1), (x1, y1 + length)],
        # top-right
        [(x2, y1), (x2 - length, y1)],
        [(x2, y1), (x2, y1 + length)],
        # bottom-left
        [(x1, y2), (x1 + length, y2)],
        [(x1, y2), (x1, y2 - length)],
        # bottom-right
        [(x2, y2), (x2 - length, y2)],
        [(x2, y2), (x2, y2 - length)],
    ]
    for pt1, pt2 in corners:
        cv2.line(cv_image, tuple(pt1), tuple(pt2), color, sw, cv2.LINE_AA)


def _clamp_box(box: tuple, w: int, h: int) -> tuple:
    """Clamp (x1, y1, x2, y2) so all coordinates lie within [0, w/h)."""
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(x1, w - 1)))
    y1 = int(max(0, min(y1, h - 1)))
    x2 = int(max(0, min(x2, w - 1)))
    y2 = int(max(0, min(y2, h - 1)))
    # Normalise so x1 < x2, y1 < y2
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _adaptive_stroke(w: int, h: int) -> int:
    """
    Scale bounding-box stroke width with image resolution.
    Returns 1 for tiny images, up to 4 for very large ones.
    """
    area = w * h
    if area < 160_000:    # < ~400×400
        return 1
    elif area < 640_000:  # < ~800×800
        return 2
    elif area < 2_560_000:  # < ~1600×1600
        return 3
    else:
        return 4


def _adaptive_font_scale(w: int) -> float:
    """
    Scale label font size with image width.
    Keeps text readable on both 320 px thumbnails and 1920 px images.
    """
    if w < 400:
        return 0.40
    elif w < 640:
        return 0.50
    elif w < 960:
        return 0.60
    elif w < 1280:
        return 0.65
    else:
        return 0.75


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE FORMAT CONVERTERS
# ─────────────────────────────────────────────────────────────────────────────

def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """
    Convert a PIL (RGB) image to an OpenCV (BGR) NumPy array.
    Dimensions are preserved exactly.
    """
    rgb_array = np.array(pil_image.convert("RGB"))
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return bgr_array


def cv2_to_pil(cv_image: np.ndarray) -> Image.Image:
    """
    Convert an OpenCV (BGR) NumPy array to a PIL (RGB) image.
    Dimensions are preserved exactly.
    """
    rgb_array = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_array)


def pil_to_bytes(pil_image: Image.Image, fmt: str = "PNG") -> bytes:
    """
    Serialize a PIL image to raw bytes.
    Used by Streamlit's download_button — returned bytes can be passed directly
    to st.download_button(data=...).
    """
    buffer = io.BytesIO()
    pil_image.save(buffer, format=fmt)
    return buffer.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_confidence_color(confidence: float) -> str:
    """
    Map a confidence value (0–1) to a CSS hex colour string.
    Used by app.py to colour-code per-detection confidence badges.

    ≥ 0.80  →  green  (#22c55e)
    ≥ 0.55  →  yellow (#eab308)
    < 0.55  →  red    (#ef4444)
    """
    if confidence >= 0.80:
        return "#22c55e"
    elif confidence >= 0.55:
        return "#eab308"
    else:
        return "#ef4444"


def resize_for_display(pil_image: Image.Image, max_width: int = 900) -> Image.Image:
    """
    Downscale an image proportionally if wider than `max_width`.
    The original image is returned unchanged if it already fits.
    Dimensions are preserved proportionally (no cropping, no stretching).
    """
    w, h = pil_image.size
    if w <= max_width:
        return pil_image
    scale = max_width / w
    new_h = int(h * scale)
    return pil_image.resize((max_width, new_h), Image.LANCZOS)


def annotated_pil_from_detections(
    pil_image:  Image.Image,
    detections: list,
) -> Image.Image:
    """
    Convenience wrapper: takes a PIL image + raw detections, draws all boxes
    and returns a new PIL image ready for Streamlit display or download.

    Parameters
    ----------
    pil_image  : Original PIL image (RGB). Not modified.
    detections : List of detection dicts (label, confidence, box).

    Returns
    -------
    PIL.Image in RGB mode with bounding boxes drawn.
    Dimensions are identical to pil_image.
    """
    cv_img   = pil_to_cv2(pil_image)             # make a fresh copy
    annotated = draw_violation_boxes(cv_img, detections)
    return cv2_to_pil(annotated)
