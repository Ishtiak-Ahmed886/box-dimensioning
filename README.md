<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Box%20Dimensioner&fontSize=52&fontColor=fff&animation=twinkling&fontAlignY=36&desc=Measure%20any%20parcel%20with%20your%20phone%20camera%20%F0%9F%93%A6&descAlignY=58&descSize=18" width="100%"/>

<br/>

[![Live App](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit%20Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://ishtiak-ahmed886-box-dimensi-box-dimensioningapplive-app-al7bqt.streamlit.app/)
[![Tests](https://img.shields.io/badge/✅%20Tests-10%2F10%20Passed-22c55e?style=for-the-badge&logo=pytest&logoColor=white)]()
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-f59e0b?style=for-the-badge)](LICENSE)

<br/>

[![YOLOv8](https://img.shields.io/badge/YOLOv8-nano%20ONNX-00BFFF?style=flat-square&logo=pytorch&logoColor=white)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-PWA%20Backend-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Plotly](https://img.shields.io/badge/Plotly-3D%20WebGL-3F4F75?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com/)
[![Objectron](https://img.shields.io/badge/Google%20Objectron-Benchmarked-4285F4?style=flat-square&logo=google&logoColor=white)](https://github.com/google-research-datasets/Objectron)

<br/>

> 📦 **Record a short video of any box → AI measures Length × Width × Height in seconds.**  
> No tape measure. No special hardware. Just your phone.

</div>

---

## ✨ What It Does

```
📱 Phone Video  ──►  🧠 AI Pipeline  ──►  📏 L × W × H  ──►  ⚖️ Volumetric Weight
```

| Step | Technology | Detail |
|:----:|:-----------|:-------|
| 1️⃣ | **Smart Camera Calibration** | EXIF metadata + 200+ phone database + radial undistortion |
| 2️⃣ | **YOLOv8 Box Detection** | ONNX nano model + Hough-line classical fallback |
| 3️⃣ | **3D Reconstruction** | DLT triangulation across multi-view frames |
| 4️⃣ | **Scale Anchor** | A4 paper / Credit card for absolute metric scale |
| 5️⃣ | **Quality Gate** | Auto-reject blurry / truncated / bad-baseline frames |
| 6️⃣ | **3D Wireframe HUD** | Interactive Plotly WebGL digital twin overlay |

---

## 🏆 Accuracy Benchmarks (Google Objectron · 55 sequences)

| Criterion | Target | Achieved | Status |
|:---------:|:------:|:--------:|:------:|
| **A1** Camera RMS Reprojection Error | ≤ 0.60 px | **0.41 px** | ✅ PASS |
| **A2** Mean Absolute Error per dim | ≤ 2.0 cm | **0.65 cm** | ✅ PASS |
| **A3** Volumetric Weight Accuracy | ≥ 75% within ±15% | **82%** | ✅ PASS |
| **A4** 3D IoU (Sutherland-Hodgman) | ≥ 0.50 | **0.89** | ✅ PASS |
| **A5** Method A vs B Comparison | Comparative analysis | Completed | ✅ PASS |
| **A6** Quality Gate (TPR / FPR) | TPR≥90%, FPR≤10% | **100% / 0%** | ✅ PASS |
| **A7** Measurement Latency (CPU) | ≤ 2.0 s | **28 ms** | ✅ PASS |

---

## 📱 Mobile App (PWA)

<div align="center">

```
┌─────────────────────────────┐
│  📦  Box Dimensioner        │
│  ───────────────────────── │
│  🎥  Camera দিয়ে ভিডিও    │
│  📁  Gallery থেকে Upload   │
│  📐  A4 / Credit Card Scale │
│  📊  L × W × H + Vol Weight│
│  📲  Home Screen Install   │
└─────────────────────────────┘
```

</div>

**Install করার নিয়ম:**
1. Chrome-এ Live App URL খোলো
2. `⋮ Menu → "Add to Home Screen"` tap করো
3. Home screen-এ icon পাবে — full-screen app!

> Works on **Android Chrome** and **iOS Safari** — no App Store needed.

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/Ishtiak-Ahmed886/box-dimensioning.git
cd box-dimensioning

# Setup
python -m venv .venv
.\.venv\Scripts\activate        # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
```

### Run Streamlit Dashboard
```bash
streamlit run streamlit_app.py
# → http://localhost:8501
```

### Run Mobile PWA Backend
```bash
uvicorn mobile_app.main:app --reload
# → http://localhost:8000
```

### Run Tests
```bash
pytest box_dimensioning/tests -v
```

```
✅ test_calibration_rms_reprojection_error         PASSED
✅ test_undistortion_preserves_dimensions           PASSED
✅ test_box_creation_and_properties                 PASSED
✅ test_sutherland_hodgman_exact_iou                PASSED
✅ test_box_fit_reconstruction                      PASSED
✅ test_method_a_accuracy_and_latency_on_real_seq   PASSED
✅ test_method_b_pipeline_and_comparison            PASSED
✅ test_quality_gate_blur_rejection                 PASSED
✅ test_quality_gate_baseline_rejection             PASSED
✅ test_quality_gate_truncation_acceptance_a6       PASSED

══════════════ 10 passed in 4.47s ══════════════
```

---

## 🏗️ Architecture

```
Phone Video
    │
    ├── Smart Calibration (EXIF → Phone DB → Heuristic)
    │       └── smart_intrinsics.py
    │
    ├── YOLOv8 Detection (ONNX) ──► Hough fallback
    │       └── deep_detector.py
    │
    ├── Scale Anchor (A4 / Credit Card)
    │       └── scale_anchor.py
    │
    ├── Quality Gate (Blur + Truncation + Baseline)
    │       └── quality_gate.py
    │
    ├── Method A: Multi-View DLT Triangulation
    │       └── method_a_multiview.py
    │
    └── 3D Cuboid → L × W × H → Volumetric Weight
            └── dimensioner.py
```

---

## 📂 Repository Structure

```
box-dimensioning/
├── 📱 mobile_app/
│   ├── main.py                   # FastAPI PWA backend
│   └── static/
│       ├── index.html            # Mobile-first PWA UI
│       └── sw.js                 # Service Worker (offline)
│
├── 📦 box_dimensioning/
│   ├── app/
│   │   ├── live_app.py           # Streamlit dashboard
│   │   └── cli.py                # CLI (calibrate / measure / benchmark)
│   ├── calibration/
│   │   ├── smart_intrinsics.py   # ✨ EXIF + 200-phone DB calibration
│   │   └── camera_calibrator.py  # Chessboard OpenCV calibration
│   ├── segmentation/
│   │   ├── deep_detector.py      # ✨ YOLOv8-nano ONNX detector
│   │   └── classical.py          # Bilateral + Hough fallback
│   ├── pipeline/
│   │   ├── scale_anchor.py       # ✨ A4/Credit card scale resolution
│   │   ├── custom_video_processor.py  # Mobile video pipeline
│   │   ├── method_a_multiview.py # DLT triangulation
│   │   ├── method_b_singleview.py# Ground-plane metrology
│   │   └── quality_gate.py       # Frame rejection filter
│   ├── dataset/                  # Objectron parser + downloader
│   ├── evaluation/               # MAE, IoU, benchmark metrics
│   └── tests/                    # 10/10 pytest suite
│
├── 🗄️ data/
│   ├── objectron/                # 55 Google Objectron sequences
│   ├── calibration/              # Chessboard calibration images
│   └── benchmark/                # Benchmark results JSON
│
├── streamlit_app.py              # Streamlit Cloud entrypoint
├── render.yaml                   # Render.com deploy config
├── Procfile                      # Heroku deploy config
└── requirements.txt
```

---

## 📊 Sample API Output

```json
{
  "sequence_id": "cereal_box/batch-26/22",
  "status": "ACCEPTED",
  "method": "METHOD_A_MULTIVIEW",
  "dimensions_cm": {
    "length": 21.42,
    "width": 15.18,
    "height": 5.22
  },
  "volumetric_weight_kg": 0.3394,
  "quality_flags": {
    "is_blurry": false,
    "blur_score": 119.02,
    "is_truncated": false,
    "baseline_meters": 0.1499,
    "reason_code": "ACCEPT"
  },
  "latency_seconds": 0.028,
  "evaluation": {
    "mae_cm": 0.42,
    "volumetric_weight_error_pct": 7.74,
    "iou_3d": 0.8851
  }
}
```

---

## 🔧 CLI Usage

```bash
# Calibrate camera
python -m box_dimensioning.app.cli calibrate --dir data/calibration

# Measure a parcel (Method A)
python -m box_dimensioning.app.cli measure \
  --seq data/objectron/cereal_box/batch-26/22 --method A

# Full 55-sequence benchmark
python -m box_dimensioning.app.cli benchmark \
  --data-dir data/objectron --max-seq 55
```

---

## 🌐 Deployment

| Platform | Config | Status |
|----------|--------|--------|
| **Streamlit Cloud** | `streamlit_app.py` | 🟢 Live |
| **Render.com** | `render.yaml` | ✅ Ready |
| **Heroku** | `Procfile` | ✅ Ready |

---

## 📐 Volumetric Weight Formula

$$\text{Volumetric Weight (kg)} = \frac{L \text{ (cm)} \times W \text{ (cm)} \times H \text{ (cm)}}{5000}$$

Used by DHL, FedEx, UPS, and all major couriers for dimensional billing.

---

## 🤝 Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![PyTorch](https://img.shields.io/badge/YOLOv8%20ONNX-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-10%2F10-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)

</div>

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

**Made with ❤️ by [Ishtiak Ahmed](https://github.com/Ishtiak-Ahmed886)**

⭐ Star this repo if it helped you!

</div>
