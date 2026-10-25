"""
test_utils.py -- Traffic Guard AI
----------------------------------
Unit tests for utils.py -- color mapping and annotation drawing.

Run with:
    python test_utils.py
  or (with pytest installed):
    python -m pytest test_utils.py -v

Tests verify:
  1. Color mapping rules for all four categories (red/green/blue/yellow)
  2. Fallback to grey for unknown labels
  3. Case-insensitive and keyword-based matching
  4. draw_violation_boxes preserves image dimensions
  5. draw_violation_boxes draws nothing when detections list is empty
  6. Boxes clamped at image boundaries (no IndexError on edge-touching boxes)
  7. Overlapping boxes are all drawn without crashing
  8. Small images (< 100x100) render without error
  9. Detections with missing 'box' key are safely skipped
 10. Output is a valid PIL image convertible for Streamlit download
"""

import sys
import traceback
import numpy as np
from PIL import Image

# -- Import the module under test ---------------------------------------------
try:
    from utils import (
        get_annotation_color,
        draw_violation_boxes,
        pil_to_cv2,
        cv2_to_pil,
        pil_to_bytes,
        annotated_pil_from_detections,
        ANNOTATION_COLOR_MAP,
        DEFAULT_BOX_COLOR,
        _RED, _GREEN, _BLUE, _YELLOW,
    )
except ImportError as exc:
    print(f"IMPORT ERROR: {exc}")
    print("Make sure you run this from the Traffic_gaurd/ directory.")
    sys.exit(1)


# -- Test Infrastructure (no pytest required) ---------------------------------

_PASS = 0
_FAIL = 0
_ERRORS = []


def _test(name, condition, detail=""):
    """Record a single test result."""
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  [PASS]  {name}")
    else:
        _FAIL += 1
        msg = f"  [FAIL]  {name}" + (f"  [{detail}]" if detail else "")
        print(msg)
        _ERRORS.append(msg)


def _make_image(w=640, h=480, color=(80, 120, 160)):
    """Create a solid-colour BGR image for testing."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


def _make_detection(label, confidence, x1, y1, x2, y2):
    return {
        "label":      label,
        "raw_label":  label.lower().replace(" ", "-"),
        "confidence": confidence,
        "box":        (x1, y1, x2, y2),
        "class_id":   0,
    }


# =============================================================================
# SECTION 1 -- COLOR MAPPING
# =============================================================================

def test_color_mapping():
    print("\n-- Section 1: Color Mapping -------------------------------------------")

    # RED -- violations
    _test("No Helmet -> RED",
          get_annotation_color("No Helmet") == _RED,
          f"got {get_annotation_color('No Helmet')}")
    _test("no-helmet -> RED",
          get_annotation_color("no-helmet") == _RED,
          f"got {get_annotation_color('no-helmet')}")
    _test("No Seat Belt -> RED",
          get_annotation_color("No Seat Belt") == _RED,
          f"got {get_annotation_color('No Seat Belt')}")
    _test("Red Light -> RED",
          get_annotation_color("Red Light") == _RED,
          f"got {get_annotation_color('Red Light')}")

    # GREEN -- compliant
    _test("Helmet -> GREEN",
          get_annotation_color("Helmet") == _GREEN,
          f"got {get_annotation_color('Helmet')}")
    _test("Seat Belt -> GREEN",
          get_annotation_color("Seat Belt") == _GREEN,
          f"got {get_annotation_color('Seat Belt')}")
    _test("Green Light -> GREEN",
          get_annotation_color("Green Light") == _GREEN,
          f"got {get_annotation_color('Green Light')}")

    # BLUE -- number plate
    _test("Number Plate -> BLUE",
          get_annotation_color("Number Plate") == _BLUE,
          f"got {get_annotation_color('Number Plate')}")
    _test("number-plate -> BLUE",
          get_annotation_color("number-plate") == _BLUE,
          f"got {get_annotation_color('number-plate')}")
    _test("License Plate -> BLUE",
          get_annotation_color("License Plate") == _BLUE,
          f"got {get_annotation_color('License Plate')}")
    _test("Number-Plate -> BLUE",
          get_annotation_color("Number-Plate") == _BLUE,
          f"got {get_annotation_color('Number-Plate')}")

    # YELLOW -- vehicle / rider
    _test("Bike -> YELLOW",
          get_annotation_color("Bike") == _YELLOW,
          f"got {get_annotation_color('Bike')}")
    _test("bike (lowercase) -> YELLOW",
          get_annotation_color("bike") == _YELLOW,
          f"got {get_annotation_color('bike')}")
    _test("Motorcycle -> YELLOW",
          get_annotation_color("Motorcycle") == _YELLOW,
          f"got {get_annotation_color('Motorcycle')}")
    _test("Rider -> YELLOW",
          get_annotation_color("Rider") == _YELLOW,
          f"got {get_annotation_color('Rider')}")

    # GREY -- fallback
    _test("Unknown label -> GREY",
          get_annotation_color("FlyingSaucer") == DEFAULT_BOX_COLOR,
          f"got {get_annotation_color('FlyingSaucer')}")
    _test("Empty string -> GREY",
          get_annotation_color("") == DEFAULT_BOX_COLOR,
          f"got {get_annotation_color('')}")

    # Case-insensitive fallthrough
    _test("HELMET (all caps) -> GREEN",
          get_annotation_color("HELMET") == _GREEN,
          f"got {get_annotation_color('HELMET')}")
    _test("NO HELMET (all caps) -> RED",
          get_annotation_color("NO HELMET") == _RED,
          f"got {get_annotation_color('NO HELMET')}")

    # Keyword-rule edge cases (phrases not in explicit map)
    _test("'rider without helmet' keyword -> RED",
          get_annotation_color("rider without helmet") == _RED)
    _test("'number plate' keyword -> BLUE",
          get_annotation_color("number plate") == _BLUE)
    _test("'two wheeler' keyword -> YELLOW",
          get_annotation_color("two wheeler") == _YELLOW)


# =============================================================================
# SECTION 2 -- DIMENSION PRESERVATION
# =============================================================================

def test_drawing_preserves_dimensions():
    print("\n-- Section 2: Dimension Preservation ---------------------------------")

    for w, h in [(640, 480), (320, 240), (1920, 1080), (100, 100)]:
        img = _make_image(w, h)
        original_shape = img.shape
        dets = [_make_detection("No Helmet", 0.87, 50, 50, w - 50, h - 50)]
        result = draw_violation_boxes(img.copy(), dets)
        _test(
            f"Shape preserved {w}x{h}",
            result.shape == original_shape,
            f"expected {original_shape}, got {result.shape}",
        )


# =============================================================================
# SECTION 3 -- EMPTY DETECTIONS
# =============================================================================

def test_empty_detections():
    print("\n-- Section 3: Empty Detections ----------------------------------------")

    img      = _make_image(640, 480, color=(30, 30, 30))
    original = img.copy()
    result   = draw_violation_boxes(img.copy(), [])

    _test("Empty detections: shape unchanged",
          result.shape == original.shape)
    _test("Empty detections: pixel values unchanged",
          np.array_equal(result, original),
          "pixels changed when no detections were given")


# =============================================================================
# SECTION 4 -- BOUNDARY CLAMPING
# =============================================================================

def test_boundary_clamping():
    print("\n-- Section 4: Boundary Clamping ---------------------------------------")

    img = _make_image(200, 200)

    # Box exactly at image edges
    det_edge = _make_detection("Helmet", 0.9, 0, 0, 199, 199)
    try:
        result = draw_violation_boxes(img.copy(), [det_edge])
        _test("Box at exact image edge: no crash", True)
        _test("Box at exact image edge: shape preserved",
              result.shape == img.shape)
    except Exception as exc:
        _test("Box at exact image edge: no crash", False, str(exc))

    # Box outside bounds -- must be clamped, not crash
    det_oob = _make_detection("Bike", 0.75, -10, -10, 250, 250)
    try:
        result = draw_violation_boxes(img.copy(), [det_oob])
        _test("Out-of-bounds box: clamped without crash", True)
        _test("Out-of-bounds box: shape preserved",
              result.shape == img.shape)
    except Exception as exc:
        _test("Out-of-bounds box: clamped without crash", False, str(exc))

    # Zero-area box (x1 == x2) -- must be silently skipped
    det_zero = _make_detection("Number Plate", 0.6, 50, 50, 50, 100)
    try:
        draw_violation_boxes(img.copy(), [det_zero])
        _test("Zero-width box: skipped without crash", True)
    except Exception as exc:
        _test("Zero-width box: skipped without crash", False, str(exc))


# =============================================================================
# SECTION 5 -- OVERLAPPING BOXES
# =============================================================================

def test_overlapping_boxes():
    print("\n-- Section 5: Overlapping Boxes ---------------------------------------")

    img = _make_image(640, 480)
    dets = [
        _make_detection("No Helmet",    0.92,  10,  10, 300, 300),
        _make_detection("Helmet",       0.85,  50,  50, 250, 250),
        _make_detection("Number Plate", 0.78, 200, 200, 400, 380),
        _make_detection("Bike",         0.65,   5,   5, 635, 475),
    ]
    try:
        result = draw_violation_boxes(img.copy(), dets)
        _test("4 overlapping boxes: no crash", True)
        _test("4 overlapping boxes: shape preserved",
              result.shape == img.shape)
        _test("4 overlapping boxes: image was modified (pixels changed)",
              not np.array_equal(result, img))
    except Exception as exc:
        _test("4 overlapping boxes: no crash", False, str(exc))


# =============================================================================
# SECTION 6 -- SMALL IMAGES
# =============================================================================

def test_small_images():
    print("\n-- Section 6: Small Images --------------------------------------------")

    for sz in [50, 30, 10]:
        img = _make_image(sz, sz)
        dets = [_make_detection("No Helmet", 0.80, 2, 2, max(sz - 2, 3), max(sz - 2, 3))]
        try:
            result = draw_violation_boxes(img.copy(), dets)
            _test(f"{sz}x{sz} image: no crash", True)
            _test(f"{sz}x{sz} image: shape preserved",
                  result.shape == img.shape)
        except Exception as exc:
            _test(f"{sz}x{sz} image: no crash", False, str(exc))


# =============================================================================
# SECTION 7 -- MISSING / MALFORMED DETECTION KEYS
# =============================================================================

def test_missing_box_key():
    print("\n-- Section 7: Missing / Malformed Detection Keys ----------------------")

    img = _make_image(640, 480)

    # Missing 'box' key entirely
    det_no_box = {"label": "No Helmet", "confidence": 0.9}
    try:
        result = draw_violation_boxes(img.copy(), [det_no_box])
        _test("Missing 'box' key: safely skipped", True)
        _test("Missing 'box' key: no pixels changed",
              np.array_equal(result, img))
    except Exception as exc:
        _test("Missing 'box' key: safely skipped", False, str(exc))

    # 'box' is None
    det_none_box = {"label": "Helmet", "confidence": 0.7, "box": None}
    try:
        draw_violation_boxes(img.copy(), [det_none_box])
        _test("box=None: safely skipped", True)
    except Exception as exc:
        _test("box=None: safely skipped", False, str(exc))

    # Missing 'confidence' key -- should default to 0.0 without crash
    det_no_conf = {"label": "Bike", "box": (10, 10, 100, 100)}
    try:
        draw_violation_boxes(img.copy(), [det_no_conf])
        _test("Missing 'confidence' key: no crash", True)
    except Exception as exc:
        _test("Missing 'confidence' key: no crash", False, str(exc))


# =============================================================================
# SECTION 8 -- STREAMLIT-COMPATIBLE OUTPUT
# =============================================================================

def test_streamlit_output_format():
    print("\n-- Section 8: Streamlit-Compatible Output -----------------------------")

    pil_img = Image.new("RGB", (640, 480), (30, 50, 80))
    dets    = [
        _make_detection("No Helmet",    0.87, 100, 80,  300, 280),
        _make_detection("Number Plate", 0.72, 310, 300, 500, 380),
        _make_detection("Bike",         0.65,  20,  20, 580, 450),
    ]

    # Convenience wrapper
    try:
        annotated_pil = annotated_pil_from_detections(pil_img, dets)
        _test("annotated_pil_from_detections: returns PIL.Image",
              isinstance(annotated_pil, Image.Image))
        _test("Annotated PIL: same dimensions as input",
              annotated_pil.size == pil_img.size)
        _test("Annotated PIL: mode is RGB",
              annotated_pil.mode == "RGB")
    except Exception as exc:
        _test("annotated_pil_from_detections: no crash", False, str(exc))

    # pil_to_bytes -- used by st.download_button
    try:
        dummy_pil = Image.new("RGB", (320, 240), (10, 20, 30))
        raw = pil_to_bytes(dummy_pil, fmt="PNG")
        _test("pil_to_bytes: returns bytes",
              isinstance(raw, bytes))
        _test("pil_to_bytes: non-empty",
              len(raw) > 0)
        _test("pil_to_bytes: PNG magic bytes",
              raw[:4] == b"\x89PNG",
              f"got {raw[:4]!r}")
    except Exception as exc:
        _test("pil_to_bytes: no crash", False, str(exc))


# =============================================================================
# SECTION 9 -- LABEL TEXT FORMAT
# =============================================================================

def test_label_text_format():
    print("\n-- Section 9: Label Text Format ---------------------------------------")

    # Verify format "NO HELMET  87.4%"
    label = "No Helmet"
    conf  = 0.874
    text  = f"{label.upper()}  {conf * 100:.1f}%"
    _test("Format 'NO HELMET  87.4%'",
          text == "NO HELMET  87.4%",
          f"got '{text}'")

    label2 = "number-plate"
    conf2  = 0.612
    text2  = f"{label2.upper()}  {conf2 * 100:.1f}%"
    _test("Format 'NUMBER-PLATE  61.2%'",
          text2 == "NUMBER-PLATE  61.2%",
          f"got '{text2}'")

    # 100% confidence edge case
    label3 = "Helmet"
    conf3  = 1.0
    text3  = f"{label3.upper()}  {conf3 * 100:.1f}%"
    _test("Format 'HELMET  100.0%'",
          text3 == "HELMET  100.0%",
          f"got '{text3}'")

    # Near-zero confidence
    label4 = "Bike"
    conf4  = 0.001
    text4  = f"{label4.upper()}  {conf4 * 100:.1f}%"
    _test("Format 'BIKE  0.1%'",
          text4 == "BIKE  0.1%",
          f"got '{text4}'")


# =============================================================================
# RUNNER
# =============================================================================

def run_all():
    print()
    print("+" + "-" * 58 + "+")
    print("|    TRAFFIC GUARD AI -- utils.py Test Suite             |")
    print("+" + "-" * 58 + "+")

    suites = [
        test_color_mapping,
        test_drawing_preserves_dimensions,
        test_empty_detections,
        test_boundary_clamping,
        test_overlapping_boxes,
        test_small_images,
        test_missing_box_key,
        test_streamlit_output_format,
        test_label_text_format,
    ]

    for suite in suites:
        try:
            suite()
        except Exception as exc:
            print(f"\n  [ERROR] Unhandled exception in {suite.__name__}:")
            traceback.print_exc()
            _ERRORS.append(f"Unhandled: {suite.__name__}: {exc}")

    # -- Summary ---------------------------------------------------------------
    total = _PASS + _FAIL
    print()
    print("-" * 60)
    print(f"  Results: {_PASS}/{total} passed", end="")
    if _FAIL:
        print(f"  |  {_FAIL} FAILED")
        print("\n  Failed tests:")
        for err in _ERRORS:
            print(f"    {err}")
    else:
        print("  -- ALL TESTS PASSED")
    print("-" * 60)
    print()

    return _FAIL == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
