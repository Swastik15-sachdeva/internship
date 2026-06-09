# AI-Powered Smart Recruitment and Online Proctoring System
### Using Computer Vision and Deep Learning

> **Note:** This system is a *recruiter assistance tool*, not an automated cheating detector.  
> The AI observes and logs — all final decisions are made by the human recruiter.

---

## Features

| Module | Technology | Detects |
|---|---|---|
| Face recognition | `face_recognition` + dlib | Identity match / mismatch, face count |
| Head pose | MediaPipe Face Mesh | Looking left / right / up / down |
| Mouth movement | MediaPipe Face Mesh | Mouth open, sustained talking |
| Object detection | YOLOv8 (Ultralytics) | Phone, book, laptop, extra person |
| Evidence logging | OpenCV + SQLite | Screenshot + timestamp per violation |
| Dashboard | CustomTkinter | Live alerts, risk score, metric cards |
| Report | ReportLab | PDF with timeline and violation summary |

---

## Project Structure

```
AI_Recruiter/
├── main.py                  ← Entry point (CLI menu)
├── requirements.txt
│
├── database/
│   ├── database.py          ← SQLite helpers
│   └── recruiter.db         ← Auto-created on first run
│
├── modules/
│   ├── camera.py            ← Webcam capture wrapper
│   ├── register.py          ← Candidate registration
│   ├── face_recognition_module.py
│   ├── head_pose.py
│   ├── mouth_detection.py
│   ├── phone_detection.py   ← YOLOv8 object detection
│   ├── logger.py            ← Violation logger + screenshot
│   └── report_generator.py ← ReportLab PDF
│
├── gui/
│   └── dashboard.py         ← CustomTkinter live dashboard
│
├── candidates/              ← (reserved for future image storage)
├── screenshots/             ← Auto-saved violation screenshots
├── reports/                 ← Generated PDF reports
└── models/                  ← Place custom YOLO weights here
```

---

## Installation

### 1. Python version
Requires **Python 3.10 or 3.11**. Python 3.12 is not yet supported by all dependencies.

### 2. Windows — install dlib first
`face_recognition` depends on dlib which requires a C++ compiler.

**Option A (easiest):** Install a pre-built wheel:
```
pip install https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.0-cp311-cp311-win_amd64.whl
```

**Option B:** Install Visual Studio Build Tools and cmake:
```
winget install Microsoft.VisualStudio.2022.BuildTools
pip install cmake dlib
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. PyTorch (GPU optional)
If you have an NVIDIA GPU with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
CPU-only (default, works fine for demos):
```bash
pip install torch torchvision
```

### 5. YOLOv8 model
The first run auto-downloads `yolov8n.pt` (~6 MB). For better accuracy, download a larger model:
```bash
# Place in project root or models/ folder
yolo download model=yolov8s.pt
```
Then update `ObjectDetector(model_path="models/yolov8s.pt")` in `modules/phone_detection.py`.

---

## Usage

```bash
cd AI_Recruiter
python main.py
```

### Menu options

**1. Register new candidate**
- Opens your webcam and captures 15 frames
- Face encodings are averaged and stored in SQLite
- Tip: ensure good lighting and look directly at the camera

**2. Start monitoring session**
- Select a registered candidate
- Opens webcam preview with live annotations
- Optionally opens the recruiter dashboard (recommended)
- Press **Q** in the webcam window to end the session

**3. List candidates**
- Shows all registered candidates with violation counts and risk scores

---

## Risk Scoring

| Event | Points |
|---|---|
| Looking Away (sustained) | 5 |
| Talking (sustained) | 5 |
| No Face (>10 seconds) | 10 |
| Multiple Persons | 20 |
| Phone Detected | 30 |
| Identity Mismatch | 50 |

| Score | Status |
|---|---|
| 0 – 20 | ✅ Safe |
| 21 – 50 | ⚠️ Needs Review |
| 51+ | 🔴 High Risk |

---

## Hardware Requirements

**Minimum:**
- Intel Core i5 / AMD Ryzen 5
- 8 GB RAM
- USB webcam (720p)
- Windows 10/11 or Ubuntu 22.04
- Python 3.11

**Recommended:**
- Intel Core i7 / AMD Ryzen 7
- 16 GB RAM
- NVIDIA GPU (any CUDA-capable)
- 1080p webcam

---

## Tuning Parameters

| File | Variable | Default | Effect |
|---|---|---|---|
| `main.py` | `NO_FACE_ALERT_SECONDS` | 10 | Seconds before no-face alert |
| `main.py` | `LOOK_AWAY_ALERT_SECONDS` | 5 | Seconds before look-away alert |
| `main.py` | `YOLO_EVERY_N_FRAMES` | 5 | Run YOLO every N frames |
| `head_pose.py` | `YAW_THRESHOLD` | 20 | Degrees for left/right |
| `head_pose.py` | `PITCH_THRESHOLD` | 15 | Degrees for up/down |
| `mouth_detection.py` | `MAR_THRESHOLD` | 0.04 | Lip aperture ratio |
| `mouth_detection.py` | `TALKING_FRAMES` | 8 | Frames before talking alert |
| `face_recognition_module.py` | `TOLERANCE` | 0.5 | Face match strictness (lower = stricter) |
| `phone_detection.py` | `CONFIDENCE_THRESHOLD` | 0.45 | YOLO confidence cutoff |

---

## Roadmap

- [ ] **v2:** Eye gaze tracking (MediaPipe Iris)
- [ ] **v2:** Audio analysis (speech detection via pyaudio)
- [ ] **v2:** Browser tab monitoring (browser extension)
- [ ] **v2:** Cloud dashboard (Flask + WebSockets)
- [ ] **v2:** AI-generated session summary (LLM)

---

## Ethical Note

This system is designed as a **decision-support tool**. It surfaces behavioral signals for human review. It does not and should not automatically disqualify candidates. All evidence should be reviewed by a qualified recruiter or invigilator before any action is taken.

---

*AI-Powered Smart Recruitment and Online Proctoring System — Computer Vision + Deep Learning*
