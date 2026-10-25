"""
detector.py — Traffic Guard AI
--------------------------------
Real YOLO-based detection engine using Ultralytics.

┌─────────────────────────────────────────────────────────────────────────────┐
│  WEIGHTS GUIDE                                                              │
│                                                                             │
│  Feature                    Weights file needed                             │
│  ─────────────────────────  ─────────────────────────────────────────────  │
│  Helmet / No-Helmet detect  models/best.pt  (your custom-trained weights)  │
│  Number-plate detection     models/best.pt  (your custom-trained weights)  │
│  Pipeline smoke-test only   yolov8n.pt      (auto-downloaded from COCO)    │
│                                                                             │
│  WHERE TO GET CUSTOM WEIGHTS                                                │
│  1. Train on a helmet dataset (Roboflow, Kaggle, etc.)                     │
│  2. Download pretrained helmet model from:                                  │
│       https://universe.roboflow.com  → search "helmet detection"           │
│  3. Place the .pt file at:   models/best.pt                                │
└─────────────────────────────────────────────────────────────────────────────┘

HOW TO RUN A QUICK TEST (PowerShell)
──────────────────────────────────────
  # Make sure you are in the project folder
  cd "C:\\Users\\Vinoth\\Desktop\\Bhuvana\\Traffic_gaurd"

  # Quick Python test — no Streamlit needed
  python -c "
  from PIL import Image
  from detector import analyze_image
  img = Image.open('test.jpg')        # any JPG you have handy
  r = analyze_image(img)
  print('Status  :', r['status'])
  print('Message :', r['message'])
  print('Model   :', r['model_info']['loaded_from'])
  print('Violations found:', r['total_violations'])
  for v in r['violations']:
      print(f\"  {v['label']}  {v['confidence']*100:.1f}%  box={v['box']}\")
  "

  # Or simply run the full dashboard
  streamlit run app.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import os
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from utils import pil_to_cv2, cv2_to_pil, draw_violation_boxes
from violation_summary import build_violation_report

# Graceful ultralytics import — gives a clear error if not installed
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Path where your custom-trained weights should be placed.
CUSTOM_WEIGHTS_PATH = Path(__file__).parent / "models" / "best.pt"

# Fallback: a tiny COCO-pretrained model used ONLY to verify the pipeline works.
# It will NOT detect helmet violations — only general objects (car, person, etc.)
COCO_FALLBACK_MODEL = "yolov8n.pt"

# Default confidence threshold (0–1). Overridden by the UI slider in app.py.
DEFAULT_CONFIDENCE = 0.40

# ─────────────────────────────────────────────────────────────────────────────
# CLASS-NAME MAPPING
#
# Maps raw class names from your dataset → human-readable display labels.
# This works by string-matching, never by hardcoded class ID numbers.
# If your dataset uses different names, just add them here.
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: "raw_name_from_model_lowercase" -> "Display Label"
CLASS_NAME_MAP: dict[str, str] = {
    # ── No-Helmet Violations (checked with highest priority) ─────────────────
    "no-helmet":                "No Helmet",
    "no_helmet":                "No Helmet",
    "no helmet":                "No Helmet",
    "without-helmet":           "No Helmet",
    "without_helmet":           "No Helmet",
    "without helmet":           "No Helmet",
    "not-wearing-helmet":       "No Helmet",
    "not_wearing_helmet":       "No Helmet",
    "helmetless":               "No Helmet",
    "bare-head":                "No Helmet",
    "bare_head":                "No Helmet",

    # ── Helmet (Compliant) ───────────────────────────────────────────────────
    "helmet":                   "Helmet",
    "with-helmet":              "Helmet",
    "with_helmet":              "Helmet",
    "with helmet":              "Helmet",
    "wearing-helmet":           "Helmet",
    "wearing_helmet":           "Helmet",
    "helmet_on":                "Helmet",

    # ── Rider / Vehicle ──────────────────────────────────────────────────────
    "bike":                     "Bike",
    "bicycle":                  "Bicycle",
    "motorcycle":               "Motorcycle",
    "motorbike":                "Motorcycle",
    "two_wheeler":              "Motorcycle",
    "two-wheeler":              "Motorcycle",
    "scooter":                  "Bike",
    "rider":                    "Rider",
    "person":                   "Person",
    "pedestrian":               "Pedestrian",
    "car":                      "Car",
    "truck":                    "Truck",
    "bus":                      "Bus",
    "vehicle":                  "Vehicle",

    # ── Number / License Plate ───────────────────────────────────────────────
    "number-plate":             "Number Plate",
    "number_plate":             "Number Plate",
    "number plate":             "Number Plate",
    "numberplate":              "Number Plate",
    "license-plate":            "Number Plate",
    "license_plate":            "Number Plate",
    "license plate":            "Number Plate",
    "licence-plate":            "Number Plate",
    "licence_plate":            "Number Plate",
    "licenseplate":             "Number Plate",
    "plate":                    "Number Plate",
    "lp":                       "Number Plate",

    # ── Seat Belt ────────────────────────────────────────────────────────────
    "no-seatbelt":              "No Seat Belt",
    "no_seatbelt":              "No Seat Belt",
    "no-seat-belt":             "No Seat Belt",
    "no_seat_belt":             "No Seat Belt",
    "without-seatbelt":         "No Seat Belt",
    "without_seatbelt":         "No Seat Belt",
    "seatbelt":                 "Seat Belt",
    "seat_belt":                "Seat Belt",

    # ── Traffic signals ──────────────────────────────────────────────────────
    "traffic_light":            "Traffic Light",
    "traffic light":            "Traffic Light",
    "red_light":                "Red Light",
    "green_light":              "Green Light",
}

# Labels that represent active violations (shown in red/warning in UI)
VIOLATION_LABELS: set[str] = {
    "No Helmet",
    "No Seat Belt",
    "Red Light",
}

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR MAP  (BGR format for OpenCV)
# ─────────────────────────────────────────────────────────────────────────────

LABEL_COLORS: dict[str, tuple] = {
    # Violations — red family
    "No Helmet":     (0,   0,   220),   # red
    "No Seat Belt":  (0,   60,  220),   # red-orange
    "Red Light":     (0,   0,   180),   # dark red

    # Safe / compliant — green family
    "Helmet":        (0,   200,  60),   # green
    "Seat Belt":     (0,   180,  80),   # green

    # Vehicles — blue/cyan family
    "Bike":          (200, 100,   0),   # orange-blue
    "Motorcycle":    (210,  90,   0),   # orange
    "Car":           (180, 180, 180),   # grey
    "Truck":         (150, 150, 150),   # grey
    "Bus":           (130, 130, 255),   # lavender
    "Rider":         (255, 180,   0),   # amber
    "Person":        (255, 200,  50),   # amber-yellow
    "Pedestrian":    (255, 220,  80),   # yellow

    # Plates — yellow
    "Number Plate":  (0,   220, 220),   # yellow

    # Signals
    "Traffic Light": (0,   120, 255),   # orange
    "Green Light":   (0,   200,   0),   # green
}

DEFAULT_BOX_COLOR = (180, 180, 180)   # grey for unknown classes


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CACHE  (module-level — survives across Streamlit re-runs)
# ─────────────────────────────────────────────────────────────────────────────

# Dictionary keyed by the absolute path string of the weights file.
# Value: {"model": YOLO instance, "info": dict with metadata}
_MODEL_CACHE: dict[str, dict] = {}


def _load_model_cached(weights_path: str) -> tuple:
    """
    Load a YOLO model and cache it by weights path.
    Returns (model, model_info_dict).
    Raises RuntimeError if loading fails.
    """
    if weights_path in _MODEL_CACHE:
        cached = _MODEL_CACHE[weights_path]
        return cached["model"], cached["info"]

    model = YOLO(weights_path)

    # Read actual class names from the model — never assume IDs
    raw_class_names: dict[int, str] = model.names   # {0: "helmet", 1: "no_helmet", …}

    # Build display-name lookup keyed by class ID
    display_names: dict[int, str] = {}
    for cls_id, raw_name in raw_class_names.items():
        display = _map_class_name(raw_name)
        display_names[cls_id] = display

    info = {
        "loaded_from":      weights_path,
        "raw_class_names":  raw_class_names,
        "display_names":    display_names,
        "num_classes":      len(raw_class_names),
    }

    _MODEL_CACHE[weights_path] = {"model": model, "info": info}
    return model, info


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: DETERMINE WHICH MODEL TO LOAD
# ─────────────────────────────────────────────────────────────────────────────

# Model status constants — used by app.py to display the right UI message
MODEL_STATUS_CUSTOM   = "custom"          # models/best.pt found → full detection
MODEL_STATUS_COCO     = "coco_fallback"   # no custom weights; using COCO (no helmet classes)
MODEL_STATUS_NO_YOLO  = "no_ultralytics"  # ultralytics not installed
MODEL_STATUS_ERROR    = "error"           # unexpected load error


def get_model_status() -> dict:
    """
    Return a dict describing the current model situation without actually
    loading the model (safe to call from the sidebar on every render).

    Returns
    -------
    dict with keys:
        status        : MODEL_STATUS_* constant
        weights_path  : str — path that will be / was loaded
        message       : str — human-readable one-liner for the UI
        can_detect    : bool — True only when custom weights are present
    """
    if not _YOLO_AVAILABLE:
        return {
            "status":       MODEL_STATUS_NO_YOLO,
            "weights_path": "",
            "message":      "❌ ultralytics not installed. Run: pip install ultralytics",
            "can_detect":   False,
        }

    if CUSTOM_WEIGHTS_PATH.exists():
        return {
            "status":       MODEL_STATUS_CUSTOM,
            "weights_path": str(CUSTOM_WEIGHTS_PATH),
            "message":      f"✅ Custom weights loaded — {CUSTOM_WEIGHTS_PATH.name}",
            "can_detect":   True,
        }

    return {
        "status":       MODEL_STATUS_COCO,
        "weights_path": COCO_FALLBACK_MODEL,
        "message":      (
            "⚠️  No custom weights found at models/best.pt. "
            "Using COCO pretrained model for pipeline testing only — "
            "it CANNOT detect helmet violations."
        ),
        "can_detect":   False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def analyze_image(
    pil_image: Image.Image,
    confidence_threshold: float = DEFAULT_CONFIDENCE,
) -> dict:
    """
    Analyse a traffic image and return detection results.

    Parameters
    ----------
    pil_image            : PIL.Image.Image — the image from the user.
    confidence_threshold : float (0–1)     — minimum confidence to keep a box.

    Returns
    -------
    dict with keys:
        annotated_image  : PIL.Image  — image with bounding boxes drawn
        violations       : list[dict] — each dict has 'label', 'confidence', 'box'
        summary          : dict       — counts per display label
        total_violations : int
        status           : "ok" | "error" | "setup_required"
        message          : str        — human-readable status
        model_info       : dict       — what model was used and its class names
    """

    # ── Guard: ultralytics installed? ──────────────────────────────────────
    if not _YOLO_AVAILABLE:
        return _error_result(
            pil_image,
            status="error",
            message=(
                "❌ The 'ultralytics' package is not installed.\n"
                "Fix: pip install ultralytics"
            ),
        )

    # ── Guard: valid image? ─────────────────────────────────────────────────
    if pil_image is None:
        return _error_result(None, status="error", message="❌ No image provided.")

    try:
        pil_image = pil_image.convert("RGB")
    except Exception as exc:
        return _error_result(
            pil_image,
            status="error",
            message=f"❌ Could not read image: {exc}",
        )

    # ── Determine which weights to load ─────────────────────────────────────
    model_status = get_model_status()

    if model_status["status"] == MODEL_STATUS_NO_YOLO:
        return _error_result(pil_image, status="error", message=model_status["message"])

    weights_path = model_status["weights_path"]

    # ── Load (or retrieve cached) model ─────────────────────────────────────
    try:
        model, model_info = _load_model_cached(weights_path)
    except Exception as exc:
        return _error_result(
            pil_image,
            status="error",
            message=(
                f"❌ Failed to load model from '{weights_path}'.\n"
                f"Details: {exc}\n"
                "Make sure the .pt file is not corrupted and ultralytics is installed."
            ),
        )

    # ── Run inference ────────────────────────────────────────────────────────
    try:
        cv_image = pil_to_cv2(pil_image)
        detections = _run_inference(model, model_info, cv_image, confidence_threshold)
    except Exception as exc:
        return _error_result(
            pil_image,
            status="error",
            message=f"❌ Inference failed: {exc}\n{traceback.format_exc()}",
        )

    # ── If COCO fallback: add an honest setup notice ─────────────────────────
    if model_status["status"] == MODEL_STATUS_COCO:
        # Still draw whatever COCO found, but flag it clearly
        setup_note = (
            "⚠️  SETUP REQUIRED: Place your custom helmet-detection weights at "
            "models/best.pt to enable real violation detection.\n"
            "The boxes shown below are generic COCO detections "
            "(cars, people, etc.) — NOT helmet violations."
        )
        status_key = "setup_required"
    else:
        setup_note = None
        status_key = "ok"

    # ── Draw annotations ─────────────────────────────────────────────────────
    color_map = {**LABEL_COLORS}
    cv_image_copy = pil_to_cv2(pil_image)
    annotated_cv  = draw_violation_boxes(cv_image_copy, detections, color_map, DEFAULT_BOX_COLOR)
    annotated_pil = cv2_to_pil(annotated_cv)

    # ── Build summary counts ─────────────────────────────────────────────────
    summary: dict[str, int] = {}
    for d in detections:
        lbl = d["label"]
        summary[lbl] = summary.get(lbl, 0) + 1

    # Count only real violation labels
    violation_count = sum(
        count for lbl, count in summary.items() if lbl in VIOLATION_LABELS
    )

    # ── Build enriched violation report (OCR, rider count, summaries) ────────
    # Keep a BGR copy for OCR crops (cv_image_copy was used for annotation;
    # make a clean one so bounding-box drawings don't corrupt plate crops).
    cv_image_clean = pil_to_cv2(pil_image)
    try:
        violation_report = build_violation_report(detections, cv_image_clean)
    except Exception as exc:
        # If the summary module fails, degrade gracefully rather than crashing
        violation_report = {
            "no_helmet_summaries": [],
            "object_counts":       {},
            "plate_readings":      [],
            "rider_estimate":      {},
            "limitations":         [f"Violation report error: {exc}"],
            "ocr_available":       False,
        }

    # ── Build result dict ────────────────────────────────────────────────────
    result = {
        "annotated_image":   annotated_pil,
        "violations":        detections,
        "summary":           summary,
        "total_violations":  violation_count,
        "total_detections":  len(detections),
        "status":            status_key,
        "message":           setup_note or _build_message(detections, violation_count),
        "model_info":        {
            **model_info,
            "model_status": model_status["status"],
            "threshold":    confidence_threshold,
        },
        "violation_report":  violation_report,
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL: RUN YOLO INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def _run_inference(
    model,
    model_info: dict,
    cv_image: np.ndarray,
    confidence_threshold: float,
) -> list[dict]:
    """
    Run YOLO inference and return a list of detection dicts.

    Each dict has:
        label      : str   — human-readable display label
        raw_label  : str   — original class name from the model
        confidence : float — 0.0 to 1.0
        box        : tuple — (x1, y1, x2, y2) in pixel coordinates
        class_id   : int   — numeric class ID from the model
    """
    results = model(cv_image, conf=confidence_threshold, verbose=False)

    detections = []
    display_names: dict[int, str] = model_info["display_names"]

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls_id     = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Get display label from the mapping (never hardcode IDs)
            raw_label    = result.names[cls_id]
            display_label = display_names.get(cls_id, raw_label)

            detections.append({
                "label":      display_label,
                "raw_label":  raw_label,
                "confidence": confidence,
                "box":        (x1, y1, x2, y2),
                "class_id":   cls_id,
            })

    return detections


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL: CLASS-NAME MAPPER
# ─────────────────────────────────────────────────────────────────────────────

def _map_class_name(raw_name: str) -> str:
    """
    Map a raw model class name to a human-readable display label.

    Strategy
    --------
    1. Exact match in CLASS_NAME_MAP (normalized for underscores, spaces, hyphens).
    2. Explicit violation check: if negation words ('no', 'without', 'bare', 'less')
       are present with 'helmet', ALWAYS return 'No Helmet'.
    3. Longest substring match in CLASS_NAME_MAP.
    4. Fall back to clean title-cased raw name.
    """
    raw_lower = raw_name.strip().lower()
    norm_underscore = raw_lower.replace("-", "_").replace(" ", "_")
    norm_space      = raw_lower.replace("-", " ").replace("_", " ")

    # 1 — Exact match
    for cand in (raw_lower, norm_underscore, norm_space):
        if cand in CLASS_NAME_MAP:
            return CLASS_NAME_MAP[cand]

    # 2 — Explicit check for No-Helmet / violations before generic keywords
    has_negation = any(neg in raw_lower for neg in ("no", "without", "bare", "less", "not"))
    if has_negation and "helmet" in raw_lower:
        return "No Helmet"

    if has_negation and ("seat" in raw_lower or "belt" in raw_lower):
        return "No Seat Belt"

    if "helmet" in raw_lower:
        return "Helmet"

    if any(p in raw_lower for p in ("plate", "licen", "number")):
        return "Number Plate"

    if any(b in raw_lower for b in ("bike", "motor", "scooter", "two_wheeler", "cycle")):
        return "Bike"

    # 3 — Substring match sorted by key length descending (longer keys match first)
    for key in sorted(CLASS_NAME_MAP.keys(), key=len, reverse=True):
        if key in raw_lower or key in norm_underscore or key in norm_space:
            return CLASS_NAME_MAP[key]

    # 4 — Fallback
    return raw_name.replace("_", " ").replace("-", " ").title()



# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL: HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_message(detections: list, violation_count: int) -> str:
    """Build a human-readable summary of what was detected."""
    if not detections:
        return "✅ No objects detected. Try a clearer image or lower the confidence threshold."

    labels = [d["label"] for d in detections]
    unique = list(dict.fromkeys(labels))

    if violation_count > 0:
        violation_labels = [l for l in unique if l in VIOLATION_LABELS]
        return (
            f"⚠️  {violation_count} violation(s) detected: "
            f"{', '.join(violation_labels)}. "
            f"Total objects found: {len(detections)}."
        )

    return (
        f"✅ {len(detections)} object(s) detected — no violations flagged. "
        f"Classes: {', '.join(unique)}."
    )


def _error_result(pil_image, status: str, message: str) -> dict:
    """Return a safe result dict when an error prevents detection."""
    # Return original image (or blank) as the annotated image
    if pil_image is not None:
        try:
            annotated = pil_image.convert("RGB")
        except Exception:
            annotated = Image.new("RGB", (640, 480), (10, 20, 40))
    else:
        annotated = Image.new("RGB", (640, 480), (10, 20, 40))

    return {
        "annotated_image":  annotated,
        "violations":       [],
        "summary":          {},
        "total_violations": 0,
        "total_detections": 0,
        "status":           status,
        "message":          message,
        "model_info":       {
            "loaded_from":     "",
            "raw_class_names": {},
            "display_names":   {},
            "num_classes":     0,
            "model_status":    MODEL_STATUS_ERROR,
            "threshold":       DEFAULT_CONFIDENCE,
        },
        "violation_report": {
            "no_helmet_summaries": [],
            "object_counts":       {},
            "plate_readings":      [],
            "rider_estimate":      {},
            "overall_summary":     {},
            "limitations":         [],
            "ocr_available":       False,
        },
    }
