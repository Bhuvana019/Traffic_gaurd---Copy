# 🚦 Traffic Guard AI

> An AI-powered traffic surveillance dashboard built with **Streamlit** and **OpenCV**.  
> Upload a traffic image → detect violations → download the annotated result.

---

## 📁 Project Structure

```
Traffic_gaurd/
│
├── app.py            ← Streamlit frontend (UI, layout, session state)
├── detector.py       ← Detection engine (plug your AI model here)
├── utils.py          ← Shared helpers (image conversion, annotation drawing)
├── requirements.txt  ← Python dependencies
└── README.md         ← This file
```

---

## ⚙️ Setup & Run (Windows PowerShell)

### Step 1 — Open PowerShell in the project folder

```powershell
cd "C:\Users\Vinoth\Desktop\Bhuvana\Traffic_gaurd"
```

### Step 2 — (Recommended) Create a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> ⚠️ If you get a script-execution error, run this first (one-time fix):
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### Step 3 — Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 4 — Launch the app

```powershell
streamlit run app.py
```

The browser will open automatically at **http://localhost:8501**

---

## 🖥️ Features

| Feature | Description |
|---------|-------------|
| 📁 Image Upload | JPG, JPEG, PNG supported |
| 🔍 Analyze Image | One-click detection pipeline |
| 🖼️ Annotated Result | Bounding boxes drawn on the image |
| 📋 Violation Summary | Count per violation type |
| 📈 Confidence Scores | Per-detection confidence with colour coding |
| ⬇️ Download | Save annotated PNG to your computer |

---

## 🎨 Theme

- **Background:** Dark navy (`#0a0f1e` → `#0d1b2a`)
- **Accent:** Blue (`#3b82f6`) / White (`#e8eaf6`)
- **Violation indicators:**
  - 🔴 Red — No Helmet, No Seat Belt
  - 🟡 Yellow — Wrong Lane
  - 🔵 Blue — Overspeeding
  - 🟢 Green — Signal Jump
  - 🟠 Orange — Triple Riding

---

## 🤖 Integrating Your AI Model (For Students)

Open **`detector.py`** and find the `_placeholder_detector` function.  
Replace the stub with your real model code:

```python
# Example — YOLO via ultralytics
from ultralytics import YOLO

model = YOLO("weights/best.pt")   # path to your trained weights

def _placeholder_detector(cv_image):
    results = model(cv_image)[0]
    detections = []
    for box in results.boxes:
        cls_id     = int(box.cls[0])
        label      = results.names[cls_id]
        confidence = float(box.conf[0])
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        detections.append({
            "label":      label,
            "confidence": confidence,
            "box":        (x1, y1, x2, y2),
        })
    return detections
```

Then uncomment `ultralytics` in `requirements.txt` and reinstall.

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `opencv-python` | Image processing & annotation |
| `Pillow` | Image loading & format conversion |
| `numpy` | Array operations |

---

## 📝 Notes

- The app uses **no fake AI results** — the placeholder returns empty detections.
- All UI sections (summary, confidence, download) work correctly once your model returns data.
- The pipeline in `analyze_image()` stays the same regardless of the model you plug in.

---

*Traffic Guard AI — College Project*
