# models/ — Custom YOLO Weights Directory

Place your trained YOLO weights file here as `best.pt`.

## File Structure Expected

```
models/
└── best.pt    ← your custom helmet/violation detection weights
```

## How to Get Weights

### Option A — Download from Roboflow (fastest)
1. Go to https://universe.roboflow.com
2. Search: **helmet detection** or **no helmet detection**
3. Choose a dataset → click **Model** tab → download YOLOv8 weights
4. Rename the file to `best.pt` and place it here

### Option B — Train Your Own Model
1. Collect traffic images (Kaggle, Google, your own)
2. Annotate with [Roboflow](https://app.roboflow.com) or [LabelImg](https://github.com/HumanSignal/labelImg)
3. Export as **YOLOv8 format**
4. Train using:
   ```python
   from ultralytics import YOLO
   model = YOLO("yolov8n.pt")           # start from pretrained
   model.train(data="data.yaml", epochs=50, imgsz=640)
   # After training, copy runs/detect/train/weights/best.pt → models/best.pt
   ```

## Recommended Datasets

| Dataset | Classes | Link |
|---------|---------|------|
| Helmet Detection | helmet, no_helmet | [Roboflow](https://universe.roboflow.com/joseph-nelson/hard-hat-workers) |
| Motorcycle Helmet | helmet, no_helmet, motorcycle | [Kaggle](https://www.kaggle.com/datasets/andrewmvd/helmet-detection) |
| Traffic Violation | multiple | Search Roboflow |

## Notes

- The app automatically detects this file on startup — no code changes needed.
- If this file is missing, the app uses a COCO pretrained model for pipeline testing only.
- Weights file is excluded from Git (see .gitignore) to avoid large file uploads.
