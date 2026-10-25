"""
violation_summary.py — Traffic Guard AI
-----------------------------------------
Modular post-processing layer that sits on top of raw YOLO detections.

Public API
----------
build_violation_report(detections, cv_image) → dict
    Full report: no-helmet summaries, counts, OCR plates, rider estimate.

Modules inside this file
------------------------
  _summarise_no_helmet  — per-detection No Helmet summaries with confidence
  _count_objects        — motorcycles / helmets / no-helmets totals
  _run_ocr_on_plates    — EasyOCR on cropped plate regions (with fallback)
  _estimate_riders      — person + motorcycle IoU association for triple-riding
  build_violation_report — assembles everything

Design Limitations (documented honestly)
-----------------------------------------
• OCR is only attempted when a "Number Plate" bounding box is detected.
  If the model did not detect a plate, no plate text will appear.

• Rider counting works by spatial proximity / IoU overlap between
  "Person" / "Rider" boxes and "Motorcycle" / "Bike" boxes.
  When boxes are too small, heavily occluded, or the image angle is poor,
  association is unreliable and the count is marked as an estimate.

• Triple riding is ONLY flagged when the association logic assigns ≥ 3
  person-class detections to the same motorcycle bounding box.
  The absence of a triple-riding flag does NOT mean the scene is safe —
  it means the detector cannot confirm it from this frame.

• No helmet violation is NEVER inferred solely from the absence of a
  "Helmet" detection. Only explicit "No Helmet" model detections trigger
  a violation flag.

• Confidence scores are the raw YOLO confidence values (0–1).
  They reflect how certain the model is about its own class assignment —
  not the real-world ground truth.
"""

from __future__ import annotations

import traceback
from typing import Optional

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# EasyOCR — optional dependency, gracefully handled
# ─────────────────────────────────────────────────────────────────────────────

try:
    import easyocr
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False

# Module-level OCR reader cache — expensive to initialise; reused across calls
_OCR_READER: Optional[object] = None   # easyocr.Reader when loaded

# Minimum confidence from EasyOCR to accept a text reading as valid.
# Below this threshold the plate text is reported as "Plate text unclear".
OCR_MIN_CONFIDENCE: float = 0.40

# Minimum IoU (Intersection-over-Union) overlap between a person box
# and a motorcycle box to count that person as a rider on that bike.
RIDER_ASSOC_IOU_THRESHOLD: float = 0.10

# Minimum side length (px) of a cropped plate to attempt OCR.
# Tiny crops produce too much noise to be useful.
OCR_MIN_CROP_PX: int = 20

# Labels the model can emit for each category
_NO_HELMET_LABELS:   frozenset[str] = frozenset({"No Helmet", "no-helmet", "No-Helmet"})
_HELMET_LABELS:      frozenset[str] = frozenset({"Helmet", "helmet", "With Helmet"})
_MOTORCYCLE_LABELS:  frozenset[str] = frozenset({"Motorcycle", "Bike", "bike", "Motorbike",
                                                   "Two Wheeler"})
_PERSON_LABELS:      frozenset[str] = frozenset({"Person", "person", "Rider", "rider", "Pedestrian"})
_RIDER_OR_HEAD_LABELS: frozenset[str] = frozenset({
    "Person", "person", "Rider", "rider", "Pedestrian",
    "Helmet", "helmet", "With Helmet",
    "No Helmet", "no-helmet", "No-Helmet",
})
_PLATE_LABELS:       frozenset[str] = frozenset({"Number Plate", "number-plate", "Number-Plate",
                                                   "License Plate", "Licence Plate", "Numberplate"})


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def build_violation_report(
    detections: list[dict],
    cv_image:   np.ndarray,
) -> dict:
    """
    Build a structured violation report from raw YOLO detections.

    Parameters
    ----------
    detections : list of dicts from detector._run_inference().
                 Each dict has: label (str), confidence (float),
                 box (x1,y1,x2,y2), raw_label (str), class_id (int).
    cv_image   : BGR NumPy array of the original image (for OCR crops).

    Returns
    -------
    dict with keys:
        no_helmet_summaries  : list[dict]  — one entry per No Helmet detection
        object_counts        : dict        — totals per category
        plate_readings       : list[dict]  — OCR result for each plate box
        rider_estimate       : dict        — rider count and triple-riding flag
        overall_summary      : dict        — incident narrative and overall status
        limitations          : list[str]   — plain-English caveats
        ocr_available        : bool
    """
    no_helmet_summaries = _summarise_no_helmet(detections)
    object_counts       = _count_objects(detections)
    plate_readings      = _run_ocr_on_plates(detections, cv_image)
    rider_estimate      = _estimate_riders(detections)
    limitations         = _build_limitations(
        detections, plate_readings, rider_estimate
    )
    overall_summary     = _build_overall_summary(
        no_helmet_summaries, object_counts, plate_readings, rider_estimate, detections
    )

    return {
        "no_helmet_summaries": no_helmet_summaries,
        "object_counts":       object_counts,
        "plate_readings":      plate_readings,
        "rider_estimate":      rider_estimate,
        "overall_summary":     overall_summary,
        "limitations":         limitations,
        "ocr_available":       _OCR_AVAILABLE,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — NO-HELMET SUMMARIES
# ─────────────────────────────────────────────────────────────────────────────

def _summarise_no_helmet(detections: list[dict]) -> list[dict]:
    """
    Return one summary dict per detected No Helmet instance.

    Each dict contains:
        index       : int   — detection index in original list
        label       : str   — "No Helmet"
        confidence  : float — raw YOLO confidence (0–1)
        conf_pct    : str   — formatted e.g. "87.4%"
        box         : tuple — (x1, y1, x2, y2)
        severity    : str   — "HIGH" / "MEDIUM" / "LOW" based on confidence

    NOTE: A helmet violation is ONLY flagged when the model explicitly
    detects a "No Helmet" class.  The absence of a "Helmet" detection
    does NOT constitute a violation — riders may be out of frame, occluded,
    or simply not detected at this confidence threshold.
    """
    summaries = []
    for idx, det in enumerate(detections):
        if det["label"] not in _NO_HELMET_LABELS:
            continue
        conf = det["confidence"]
        summaries.append({
            "index":      idx,
            "label":      det["label"],
            "confidence": conf,
            "conf_pct":   f"{conf * 100:.1f}%",
            "box":        det["box"],
            "severity":   _confidence_severity(conf),
        })
    return summaries


def _confidence_severity(conf: float) -> str:
    """Map confidence to a descriptive severity tier."""
    if conf >= 0.80:
        return "HIGH"
    elif conf >= 0.55:
        return "MEDIUM"
    else:
        return "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — OBJECT COUNTS
# ─────────────────────────────────────────────────────────────────────────────

def _count_objects(detections: list[dict]) -> dict:
    """
    Return structured counts of motorcycles, helmets, and no-helmets.

    Returns
    -------
    dict with keys:
        motorcycles      : int  — total Motorcycle / Bike detections
        helmets          : int  — total Helmet (compliant) detections
        no_helmets       : int  — total No Helmet detections
        persons          : int  — total Person / Rider detections
        number_plates    : int  — total plate detections
        by_label         : dict — raw count for every label seen
    """
    by_label: dict[str, int] = {}
    for det in detections:
        lbl = det["label"]
        by_label[lbl] = by_label.get(lbl, 0) + 1

    motorcycles   = sum(by_label.get(l, 0) for l in _MOTORCYCLE_LABELS)
    helmets       = sum(by_label.get(l, 0) for l in _HELMET_LABELS)
    no_helmets    = sum(by_label.get(l, 0) for l in _NO_HELMET_LABELS)
    persons       = sum(by_label.get(l, 0) for l in _PERSON_LABELS)
    number_plates = sum(by_label.get(l, 0) for l in _PLATE_LABELS)
    # Each helmet or no-helmet is a rider; count persons if separate
    riders        = helmets + no_helmets + persons

    return {
        "motorcycles":   motorcycles,
        "helmets":       helmets,
        "no_helmets":    no_helmets,
        "persons":       persons,
        "riders":        riders,
        "number_plates": number_plates,
        "by_label":      by_label,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — NUMBER PLATE OCR
# ─────────────────────────────────────────────────────────────────────────────

def _get_ocr_reader():
    """Lazily initialise the EasyOCR reader (English only)."""
    global _OCR_READER
    if _OCR_READER is None and _OCR_AVAILABLE:
        try:
            import easyocr
            # gpu=False for maximum compatibility; set True if CUDA available
            _OCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
        except Exception:
            pass   # OCR init failure — caller handles None
    return _OCR_READER


def _run_ocr_on_plates(
    detections: list[dict],
    cv_image:   np.ndarray,
) -> list[dict]:
    """
    For every detected plate bounding box, crop the region and run OCR.

    Each result dict contains:
        box         : tuple — (x1, y1, x2, y2) of the plate box
        raw_text    : str   — raw OCR output (empty string if failed)
        display_text: str   — cleaned text or "Plate text unclear"
        confidence  : float — highest character-level OCR confidence (0–1)
                              or 0.0 when text is unclear / OCR unavailable
        source      : str   — "easyocr" | "unavailable" | "error"

    Limitations
    -----------
    • OCR is only attempted when a plate box is explicitly detected.
    • Text is only accepted when OCR confidence ≥ OCR_MIN_CONFIDENCE.
    • Very small crops (< OCR_MIN_CROP_PX px on either side) are skipped.
    • EasyOCR may misread similar characters (O/0, I/1, S/5).
      Results should be treated as indicative, not authoritative.
    """
    plate_detections = [d for d in detections if d["label"] in _PLATE_LABELS]

    if not plate_detections:
        return []

    if not _OCR_AVAILABLE:
        return [
            {
                "box":          det["box"],
                "raw_text":     "",
                "display_text": "OCR not available (pip install easyocr)",
                "confidence":   0.0,
                "source":       "unavailable",
            }
            for det in plate_detections
        ]

    reader = _get_ocr_reader()
    if reader is None:
        return [
            {
                "box":          det["box"],
                "raw_text":     "",
                "display_text": "OCR initialisation failed",
                "confidence":   0.0,
                "source":       "error",
            }
            for det in plate_detections
        ]

    h_img, w_img = cv_image.shape[:2]
    results = []

    for det in plate_detections:
        x1, y1, x2, y2 = det["box"]
        # Clamp to image bounds
        x1 = max(0, x1);  y1 = max(0, y1)
        x2 = min(w_img - 1, x2);  y2 = min(h_img - 1, y2)

        crop_w = x2 - x1
        crop_h = y2 - y1

        if crop_w < OCR_MIN_CROP_PX or crop_h < OCR_MIN_CROP_PX:
            results.append({
                "box":          det["box"],
                "raw_text":     "",
                "display_text": "Plate crop too small for OCR",
                "confidence":   0.0,
                "source":       "skipped",
            })
            continue

        crop = cv_image[y1:y2, x1:x2]

        # Upscale small crops so OCR has more pixels to work with
        if crop_w < 100 or crop_h < 30:
            scale = max(100 / crop_w, 30 / crop_h, 2.0)
            crop = cv2.resize(
                crop,
                (int(crop_w * scale), int(crop_h * scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        # Convert to greyscale + mild sharpening for plate text
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        grey = cv2.convertScaleAbs(grey, alpha=1.5, beta=10)

        try:
            ocr_results = reader.readtext(grey, detail=1, paragraph=False)
        except Exception as exc:
            results.append({
                "box":          det["box"],
                "raw_text":     "",
                "display_text": "Plate text unclear",
                "confidence":   0.0,
                "source":       "error",
            })
            continue

        # Collect all text fragments above the confidence threshold
        accepted_parts = []
        best_conf = 0.0
        for (_bbox, text, conf) in ocr_results:
            if conf >= OCR_MIN_CONFIDENCE:
                cleaned = _clean_plate_text(text)
                if cleaned:
                    accepted_parts.append(cleaned)
                    best_conf = max(best_conf, conf)

        if accepted_parts:
            raw_text     = " ".join(accepted_parts)
            display_text = raw_text.upper()
        else:
            raw_text     = ""
            display_text = "Plate text unclear"
            best_conf    = 0.0

        results.append({
            "box":          det["box"],
            "raw_text":     raw_text,
            "display_text": display_text,
            "confidence":   best_conf,
            "source":       "easyocr",
        })

    return results


def _clean_plate_text(text: str) -> str:
    """
    Remove characters that are never part of a valid registration number.
    Keeps alphanumerics, hyphens, and spaces.
    """
    import re
    cleaned = re.sub(r"[^A-Za-z0-9\- ]", "", text).strip()
    # Reject single-character fragments — likely noise
    return cleaned if len(cleaned) >= 2 else ""


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 — RIDER COUNTING & TRIPLE RIDING DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_riders(detections: list[dict]) -> dict:
    """
    Estimate how many riders are in the scene and on each detected motorcycle.
    Uses spatial proximity and geometric overlap between rider/head detections
    (Helmet, No Helmet, Person, Rider) and motorcycle bounding boxes.
    """
    moto_dets  = [d for d in detections if d["label"] in _MOTORCYCLE_LABELS]
    rider_dets = [d for d in detections if d["label"] in _RIDER_OR_HEAD_LABELS]

    total_motorcycles = len(moto_dets)
    total_riders = len(rider_dets)

    if not moto_dets:
        return {
            "total_riders":            total_riders,
            "total_persons":           total_riders,
            "total_motorcycles":       0,
            "associations":            [],
            "possible_triple_riding":  False,
            "is_estimate":             True,
            "reliability_note": (
                f"Detected {total_riders} rider(s) in frame." if total_riders > 0
                else "No motorcycles or riders detected."
            ),
        }

    # Associate riders to motorcycles
    associations = []
    triple_riding_detected = False

    assigned_riders = {i: [] for i in range(len(moto_dets))}
    unassigned_riders = []

    for rider in rider_dets:
        r_box = rider["box"]
        best_moto_idx = None
        best_score = 0.0

        for m_idx, moto in enumerate(moto_dets):
            m_box = moto["box"]
            is_assoc, score = _is_rider_associated_with_moto(m_box, r_box)
            if is_assoc and score > best_score:
                best_score = score
                best_moto_idx = m_idx

        if best_moto_idx is not None:
            assigned_riders[best_moto_idx].append({
                "box":        rider["box"],
                "label":      rider["label"],
                "confidence": rider["confidence"],
                "score":      round(best_score, 3),
            })
        else:
            unassigned_riders.append(rider)

    # If there is only 1 motorcycle and unassigned riders exist, assign them to it
    if len(moto_dets) == 1 and unassigned_riders:
        for rider in unassigned_riders:
            assigned_riders[0].append({
                "box":        rider["box"],
                "label":      rider["label"],
                "confidence": rider["confidence"],
                "score":      0.5,
            })
        unassigned_riders = []

    for m_idx, moto in enumerate(moto_dets):
        moto_box = moto["box"]
        associated = assigned_riders[m_idx]
        rider_count = len(associated)
        if rider_count >= 3:
            triple_riding_detected = True

        associations.append({
            "motorcycle_box":  moto_box,
            "moto_confidence": moto["confidence"],
            "rider_count":     rider_count,
            "rider_boxes":     associated,
            "is_estimate":     _association_is_uncertain(associated),
        })

    if total_riders > 0:
        note = f"Detected {total_riders} rider(s) across {total_motorcycles} motorcycle(s)."
    else:
        note = "Motorcycle detected with no active rider detections."

    return {
        "total_riders":            total_riders,
        "total_persons":           total_riders,
        "total_motorcycles":       total_motorcycles,
        "associations":            associations,
        "possible_triple_riding":  triple_riding_detected,
        "is_estimate":             True,
        "reliability_note":        note,
    }


def _is_rider_associated_with_moto(moto_box: tuple, rider_box: tuple) -> tuple[bool, float]:
    """
    Check if a rider or helmet box belongs to a motorcycle using spatial geometry.
    Returns (is_associated, score).
    """
    mx1, my1, mx2, my2 = moto_box
    rx1, ry1, rx2, ry2 = rider_box

    mw = max(1, mx2 - mx1)
    mh = max(1, my2 - my1)
    rw = max(1, rx2 - rx1)
    rh = max(1, ry2 - ry1)

    # Horizontal center of rider/helmet
    rcx = (rx1 + rx2) / 2.0
    h_margin = 0.30 * mw
    h_overlap = (mx1 - h_margin <= rcx <= mx2 + h_margin)

    # Vertical position: rider is above or overlapping the bike
    v_overlap = (ry2 >= my1 - 0.45 * mh) and (ry1 <= my2 + 0.15 * mh)

    # Intersection over rider box area
    inter_x1 = max(mx1, rx1)
    inter_y1 = max(my1, ry1)
    inter_x2 = min(mx2, rx2)
    inter_y2 = min(my2, ry2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    rider_area = rw * rh
    inter_ratio = inter_area / rider_area if rider_area > 0 else 0.0

    iou = _box_iou(moto_box, rider_box)

    if (h_overlap and v_overlap) or inter_ratio >= 0.10 or iou >= 0.03:
        mcx = (mx1 + mx2) / 2.0
        dist_score = max(0.0, 1.0 - (abs(rcx - mcx) / (mw + 1.0)))
        score = max(0.1, round(dist_score * 0.5 + inter_ratio * 0.5 + iou, 3))
        return True, score

    return False, 0.0


def _box_iou(box_a: tuple, box_b: tuple) -> float:
    """
    Compute Intersection-over-Union (IoU) between two (x1,y1,x2,y2) boxes.
    Returns 0.0 if boxes do not intersect.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union  = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def _association_is_uncertain(associated_riders: list[dict]) -> bool:
    """Return True when the association result is considered an estimate."""
    if not associated_riders:
        return True
    return any(r.get("score", r.get("iou", 0)) < 0.25 for r in associated_riders)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5 — OVERALL SUMMARY & INCIDENT NARRATIVE
# ─────────────────────────────────────────────────────────────────────────────

def _build_overall_summary(
    no_helmet_summaries: list[dict],
    object_counts: dict,
    plate_readings: list[dict],
    rider_estimate: dict,
    detections: list[dict],
) -> dict:
    """
    Generate an overall plain-English incident description and status summary.
    """
    n_motos = object_counts.get("motorcycles", 0)
    n_helmets = object_counts.get("helmets", 0)
    n_no_helmets = object_counts.get("no_helmets", 0)
    n_plates = object_counts.get("number_plates", 0)
    total_riders = rider_estimate.get("total_riders", 0)
    triple_riding = rider_estimate.get("possible_triple_riding", False)

    # Valid plates
    valid_plates = [
        p["display_text"] for p in plate_readings
        if p.get("display_text") and not any(
            bad in p["display_text"].lower()
            for bad in ("unclear", "small", "failed", "error", "skipped")
        )
    ]

    sentences = []

    # 1. Vehicle and riders overview
    vehicle_phrase = f"{n_motos} motorcycle{'s' if n_motos > 1 else ''}" if n_motos > 0 else "two-wheeler traffic"
    rider_phrase = f"{total_riders} rider{'s' if total_riders != 1 else ''}" if total_riders > 0 else "traffic scene"

    if total_riders > 0 and n_motos > 0:
        sentences.append(f"AI surveillance detected {vehicle_phrase} carrying {rider_phrase}.")
    elif total_riders > 0:
        sentences.append(f"AI surveillance detected {total_riders} rider{'s' if total_riders > 1 else ''} in the monitored zone.")
    elif n_motos > 0:
        sentences.append(f"AI surveillance detected {vehicle_phrase}.")
    elif detections:
        sentences.append("AI surveillance completed object detection on the image.")
    else:
        sentences.append("No active vehicles or traffic participants were detected at the current threshold.")

    # 2. Helmet & Safety Violations
    violations = []
    if n_no_helmets > 0:
        conf_strs = [nh["conf_pct"] for nh in no_helmet_summaries]
        c_text = f" (confidence: {', '.join(conf_strs)})" if conf_strs else ""
        violations.append(f"{n_no_helmets} rider{'s' if n_no_helmets > 1 else ''} detected without a helmet{c_text}")

    if triple_riding:
        violations.append("Triple-riding detected (passenger capacity violation)")

    if violations:
        sentences.append(f"Safety Violation: {'; '.join(violations)}.")
    elif n_helmets > 0:
        sentences.append(f"All {n_helmets} detected rider(s) are compliant with helmet safety regulations.")
    elif total_riders > 0:
        sentences.append("No helmet safety violations were flagged.")

    # 3. License plate status
    if valid_plates:
        sentences.append(f"License plate identified: {', '.join(valid_plates)}.")
    elif n_plates > 0:
        sentences.append(f"{n_plates} number plate region(s) detected (text unclear at current resolution).")

    # Status and styling
    if n_no_helmets > 0 or triple_riding:
        status_type = "VIOLATION"
        headline = f"🚨 {len(violations)} Safety Violation{'s' if len(violations) > 1 else ''} Detected"
        badge_color = "#ef4444"
    elif n_helmets > 0:
        status_type = "COMPLIANT"
        headline = "✅ Traffic Compliant — Protective Helmet Worn"
        badge_color = "#22c55e"
    else:
        status_type = "NORMAL"
        headline = "ℹ️ Surveillance Analysis Complete"
        badge_color = "#3b82f6"

    return {
        "status_type": status_type,
        "headline": headline,
        "badge_color": badge_color,
        "description": " ".join(sentences),
        "total_riders": total_riders,
        "violations_count": n_no_helmets + (1 if triple_riding else 0),
        "plates_detected": valid_plates,
    }


def _build_limitations(
    detections:    list[dict],
    plate_readings: list[dict],
    rider_estimate: dict,
) -> list[str]:
    """Limitations removed per request."""
    return []

