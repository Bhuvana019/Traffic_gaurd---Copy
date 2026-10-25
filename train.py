"""
train.py -- Traffic Guard AI
-----------------------------
Transfer-learning script: fine-tunes a small pretrained YOLO model
on the helmet + number-plate dataset.

DATASET INSPECTED
-----------------
  Path    : datasets/helmet_plate/
  Classes : 4  ->  bike | helmet | no-helmet | number-plate   (IDs 0-3)
  Train   : 5 921 images  +  5 921 label files
  Valid   : 1 690 images  +  1 690 label files
  Test    :   855 images  +    855 label files
  Format  : YOLOv8 (class_id cx cy w h, all normalised 0-1)

HOW TO RUN (PowerShell)
------------------------
  cd "C:\\Users\\Vinoth\\Desktop\\Bhuvana\\Traffic_gaurd"

  # Trial run (5 epochs, fast sanity check):
  python train.py --epochs 5 --batch 8

  # Full training run:
  python train.py --epochs 50 --batch 8

  # Custom options:
  python train.py --epochs 30 --batch 16 --imgsz 416 --model yolov8s.pt

  # After training, test on the held-out test set:
  python train.py --test-only

WEIGHTS SAVED TO
-----------------
  models/best.pt   <- auto-copied after training completes
"""

import argparse
import shutil
import sys
import os
from pathlib import Path


# -----------------------------------------------------------------------------
# CONSTANTS  (all derived from the real dataset -- no guessing)
# -----------------------------------------------------------------------------

# Absolute path to this script's directory (the project root)
PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset root
DATASET_ROOT = PROJECT_ROOT / "datasets" / "helmet_plate"

# Where to save the best weights after training
DEST_WEIGHTS = PROJECT_ROOT / "models" / "best.pt"

# Expected dataset structure (verified by inspection)
EXPECTED_YAML   = DATASET_ROOT / "data.yaml"
EXPECTED_SPLITS = {
    "train": DATASET_ROOT / "train" / "images",
    "valid": DATASET_ROOT / "valid" / "images",
    "test":  DATASET_ROOT / "test"  / "images",
}
EXPECTED_LABEL_SPLITS = {
    "train": DATASET_ROOT / "train" / "labels",
    "valid": DATASET_ROOT / "valid" / "labels",
    "test":  DATASET_ROOT / "test"  / "labels",
}

# Classes as declared in data.yaml  (read at runtime -- see validate_dataset)
# 0: bike  |  1: helmet  |  2: no-helmet  |  3: number-plate


# -----------------------------------------------------------------------------
# DATASET VALIDATION
# -----------------------------------------------------------------------------

def validate_dataset() -> dict:
    """
    Inspect the dataset and raise clear errors before wasting time training.

    Returns a dict with dataset statistics.
    Exits with a descriptive fix message on any problem.
    """
    print("\n" + "=" * 60)
    print("  DATASET VALIDATION")
    print("=" * 60)

    errors = []

    # -- 1. data.yaml exists --------------------------------------------------
    if not EXPECTED_YAML.exists():
        errors.append(
            f"[MISSING] data.yaml not found at:\n"
            f"  {EXPECTED_YAML}\n"
            f"  Fix: Make sure you extracted the Roboflow ZIP into:\n"
            f"       datasets/helmet_plate/"
        )

    # -- 2. All three splits present ------------------------------------------
    split_counts = {}
    for split, img_dir in EXPECTED_SPLITS.items():
        lbl_dir = EXPECTED_LABEL_SPLITS[split]
        if not img_dir.exists():
            errors.append(f"[MISSING] {split}/images/ not found at: {img_dir}")
            continue
        if not lbl_dir.exists():
            errors.append(f"[MISSING] {split}/labels/ not found at: {lbl_dir}")
            continue

        imgs   = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
        labels = list(lbl_dir.glob("*.txt"))

        if len(imgs) == 0:
            errors.append(f"[EMPTY] {split}/images/ has no image files.")
        if len(labels) == 0:
            errors.append(f"[EMPTY] {split}/labels/ has no label (.txt) files.")

        # Image  label count mismatch warning (not fatal -- some images have no objects)
        mismatch = abs(len(imgs) - len(labels))
        if mismatch > len(imgs) * 0.05:   # allow 5% unlabelled
            print(f"  [WARN]  {split}: {len(imgs)} images vs {len(labels)} labels "
                  f"(mismatch={mismatch}). Some images may have no annotations -- this is OK.")

        split_counts[split] = {"images": len(imgs), "labels": len(labels)}

    # -- 3. Read and verify data.yaml -----------------------------------------
    yaml_data = {}
    if EXPECTED_YAML.exists():
        try:
            import yaml  # bundled with ultralytics
            with open(EXPECTED_YAML, "r") as f:
                yaml_data = yaml.safe_load(f)

            nc    = yaml_data.get("nc", 0)
            names = yaml_data.get("names", [])

            if nc != len(names):
                errors.append(
                    f"[INVALID] data.yaml: nc={nc} but names has {len(names)} entries.\n"
                    f"  Fix: Make sure nc equals the number of class names."
                )
            if nc == 0:
                errors.append("[INVALID] data.yaml: nc=0, no classes defined.")

        except Exception as exc:
            errors.append(f"[PARSE ERROR] Could not read data.yaml: {exc}")

    # -- 4. Sample label format check -----------------------------------------
    sample_label_dir = DATASET_ROOT / "train" / "labels"
    if sample_label_dir.exists():
        sample_files = list(sample_label_dir.glob("*.txt"))[:10]
        nc = yaml_data.get("nc", 4)
        bad_ids = []
        for lf in sample_files:
            try:
                lines = lf.read_text().strip().splitlines()
                for line in lines:
                    parts = line.split()
                    if len(parts) < 5:
                        errors.append(
                            f"[BAD FORMAT] {lf.name}: line '{line}' has <5 fields.\n"
                            f"  Expected: class_id cx cy w h"
                        )
                        break
                    cls_id = int(parts[0])
                    if cls_id < 0 or cls_id >= nc:
                        bad_ids.append((lf.name, cls_id))
            except Exception:
                pass  # non-parseable files will be caught by YOLO itself

        if bad_ids:
            for fname, cid in bad_ids[:3]:
                errors.append(
                    f"[BAD CLASS ID] {fname}: class_id={cid} is out of range 0-{nc-1}.\n"
                    f"  Fix: Re-export the dataset from Roboflow with correct class count."
                )

    # -- Report ---------------------------------------------------------------
    if errors:
        print("\n[ERROR]  Dataset validation FAILED. Fix these issues before training:\n")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}\n")
        sys.exit(1)

    # -- Print summary ---------------------------------------------------------
    names = yaml_data.get("names", [])
    nc    = yaml_data.get("nc", len(names))

    print(f"  [OK] data.yaml found")
    print(f"  [OK] Classes ({nc}): {names}")
    for split, counts in split_counts.items():
        print(f"  [OK] {split:<6}: {counts['images']:>5} images  |  {counts['labels']:>5} labels")

    return {"yaml_data": yaml_data, "split_counts": split_counts}


# -----------------------------------------------------------------------------
# DATA.YAML PATCH  (Roboflow uses relative paths; YOLO needs absolute ones)
# -----------------------------------------------------------------------------

def make_absolute_yaml() -> Path:
    """
    Roboflow exports data.yaml with relative paths like '../train/images'.
    Those paths are designed to be resolved from WITHIN the dataset folder,
    which does NOT match this project layout.
    Instead, we build a patched yaml directly from the EXPECTED_SPLITS
    constants, which already point at the verified absolute locations.

    Returns the path to the patched yaml file.
    """
    import yaml

    with open(EXPECTED_YAML, "r") as f:
        cfg = yaml.safe_load(f)

    # Overwrite split paths using the verified EXPECTED_SPLITS constants.
    # These point at:  datasets/helmet_plate/{train,valid,test}/images
    cfg["train"] = str(EXPECTED_SPLITS["train"])
    cfg["val"]   = str(EXPECTED_SPLITS["valid"])
    cfg["test"]  = str(EXPECTED_SPLITS["test"])

    patched_yaml = PROJECT_ROOT / "datasets" / "helmet_plate_abs.yaml"
    with open(patched_yaml, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return patched_yaml


# -----------------------------------------------------------------------------
# TRAINING
# -----------------------------------------------------------------------------

def train(args):
    """Fine-tune YOLO on the helmet + number-plate dataset."""

    # -- Validate first -------------------------------------------------------
    stats = validate_dataset()

    # -- Patch yaml to absolute paths -----------------------------------------
    abs_yaml = make_absolute_yaml()
    print(f"\n   Using patched yaml: {abs_yaml}")

    # -- Import YOLO ----------------------------------------------------------
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n[ERROR]  ultralytics is not installed.\n"
              "   Fix:  pip install ultralytics\n")
        sys.exit(1)

    # -- Summary before starting -----------------------------------------------
    train_count = stats["split_counts"]["train"]["images"]
    valid_count = stats["split_counts"]["valid"]["images"]

    print("\n" + "-" * 60)
    print("  TRAINING CONFIGURATION")
    print("-" * 60)
    print(f"  Base model  : {args.model}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch}")
    print(f"  Image size  : {args.imgsz}x{args.imgsz} px")
    print(f"  Train images: {train_count}")
    print(f"  Valid images: {valid_count}")
    print(f"  Device      : {'GPU (CUDA)' if _cuda_available() else 'CPU  <- slow, reduce --imgsz if needed'}")
    print(f"  Output dir  : runs/detect/train/")
    print("-" * 60)

    if args.epochs <= 5:
        print("\n  [INFO]  Trial run detected ( 5 epochs).")
        print("     Metrics will not be meaningful -- this is just a pipeline check.\n")

    # -- Load base model ------------------------------------------------------
    print(f"\n  Loading base model: {args.model} ...")
    model = YOLO(args.model)

    # -- Run training ----------------------------------------------------------
    print("\n  [>>] Starting training ...\n")
    results = model.train(
        data        = str(abs_yaml),
        epochs      = args.epochs,
        batch       = args.batch,
        imgsz       = args.imgsz,
        project     = str(PROJECT_ROOT / "runs" / "detect"),
        name        = "train",
        exist_ok    = True,            # overwrite previous run folder
        patience    = max(10, args.epochs // 5),  # early stopping patience
        save        = True,            # save checkpoints
        save_period = -1,              # only save best + last
        plots       = True,            # generate metric plots (requires matplotlib)
        verbose     = True,
        device      = 0 if _cuda_available() else "cpu",
    )

    # -- Print training metrics ------------------------------------------------
    _print_train_metrics(results)

    # -- Copy best weights to models/best.pt ----------------------------------
    best_src = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
    if best_src.exists():
        DEST_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_src, DEST_WEIGHTS)
        print(f"\n  [OK] Best weights saved to: {DEST_WEIGHTS}")
        print(f"     The Streamlit app will use this file automatically.")
    else:
        print(f"\n  [WARN]  Could not find {best_src}")
        print(f"     Training may have been interrupted. Check runs/detect/train/weights/")


# -----------------------------------------------------------------------------
# EVALUATION ON TEST SET
# -----------------------------------------------------------------------------

def test_model(args):
    """Run the saved best.pt on the held-out test set and print real metrics."""

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR]  ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    if not DEST_WEIGHTS.exists():
        print(f"\n[ERROR]  No weights found at: {DEST_WEIGHTS}")
        print(f"   Train the model first:  python train.py --epochs 5 --batch 8\n")
        sys.exit(1)

    abs_yaml = make_absolute_yaml()

    print("\n" + "=" * 60)
    print("  TEST SET EVALUATION")
    print("=" * 60)
    print(f"  Weights : {DEST_WEIGHTS}")
    print(f"  Data    : {abs_yaml}")
    print(f"  Split   : test  ({EXPECTED_SPLITS['test']})")
    print("=" * 60 + "\n")

    model   = YOLO(str(DEST_WEIGHTS))
    metrics = model.val(
        data   = str(abs_yaml),
        split  = "test",
        imgsz  = args.imgsz,
        device = 0 if _cuda_available() else "cpu",
        verbose= True,
    )

    _print_val_metrics(metrics)


# -----------------------------------------------------------------------------
# METRIC PRINTERS
# -----------------------------------------------------------------------------

def _print_train_metrics(results):
    """Print final training metrics from the results object."""
    print("\n" + "=" * 60)
    print("  FINAL TRAINING METRICS")
    print("=" * 60)
    try:
        # results.results_dict contains mAP and other metrics
        rd = results.results_dict
        print(f"  mAP@0.5         : {rd.get('metrics/mAP50(B)',    'N/A')}")
        print(f"  mAP@0.5:0.95    : {rd.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"  Precision       : {rd.get('metrics/precision(B)', 'N/A')}")
        print(f"  Recall          : {rd.get('metrics/recall(B)',    'N/A')}")
        print(f"  Box loss (train): {rd.get('train/box_loss',       'N/A')}")
        print(f"  Cls loss (train): {rd.get('train/cls_loss',       'N/A')}")
        print(f"\n  Full results saved to: runs/detect/train/results.csv")
        print(f"  Plots saved to       : runs/detect/train/*.png")
    except Exception as exc:
        print(f"  Could not parse metrics automatically: {exc}")
        print("  Check runs/detect/train/results.csv for the full metric history.")
    print("=" * 60)


def _print_val_metrics(metrics):
    """Print per-class and aggregate metrics from the val() call."""
    print("\n" + "=" * 60)
    print("  TEST SET METRICS (from models/best.pt)")
    print("=" * 60)
    try:
        print(f"  mAP@0.5        : {metrics.box.map50:.4f}")
        print(f"  mAP@0.5:0.95   : {metrics.box.map:.4f}")
        print(f"  Precision      : {metrics.box.mp:.4f}")
        print(f"  Recall         : {metrics.box.mr:.4f}")
        print()
        # Per-class AP
        names = metrics.names
        for i, (cls_ap50, cls_ap) in enumerate(zip(metrics.box.ap50, metrics.box.ap)):
            cls_name = names.get(i, f"class_{i}")
            print(f"  [{i}] {cls_name:<15} AP@0.5={cls_ap50:.4f}  AP@0.5:0.95={cls_ap:.4f}")
    except Exception as exc:
        print(f"  Could not parse metrics: {exc}")
    print("=" * 60)


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO on the helmet + number-plate dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 5-epoch trial (sanity check -- fast):
  python train.py --epochs 5 --batch 8

  # Full training:
  python train.py --epochs 50 --batch 8

  # Smaller images (faster on CPU):
  python train.py --epochs 50 --batch 8 --imgsz 416

  # Evaluate saved best.pt on test set:
  python train.py --test-only
        """,
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help=(
            "Base pretrained model to fine-tune. Options:\n"
            "  yolov8n.pt  <- nano  (fastest, least accurate -- good for CPU)\n"
            "  yolov8s.pt  <- small (balanced)\n"
            "  yolov8m.pt  <- medium (needs GPU)\n"
            "Default: yolov8n.pt"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs. Start with 5 (trial), use 50+ for real training. Default: 5",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size. Use 8 for CPU or low-VRAM GPU, 16+ for 4 GB+ GPU. Default: 8",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size (square). Use 416 for faster CPU runs. Default: 640",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Skip training. Load models/best.pt and evaluate on the test set.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    print("\n" + "=" * 60)
    print("       TRAFFIC GUARD AI -- YOLO TRAINER")
    print("=" * 60)

    if args.test_only:
        test_model(args)
    else:
        train(args)

    print("\n  Done. [OK]\n")
