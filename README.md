# 📦 Parcel Dimensioning & Volumetric-Weight Billing from Phone Video

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ishtiak-ahmed886-box-dimensi-box-dimensioningapplive-app-al7bqt.streamlit.app/)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Tests](https://img.shields.io/badge/Tests-10%2F10%20Passed-brightgreen?logo=pytest&logoColor=white)]()
[![Objectron](https://img.shields.io/badge/Dataset-Google%20Objectron-4285F4?logo=google&logoColor=white)](https://github.com/google-research-datasets/Objectron)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Live Cloud Dashboard:** [**https://ishtiak-ahmed886-box-dimensi-box-dimensioningapplive-app-al7bqt.streamlit.app/**](https://ishtiak-ahmed886-box-dimensi-box-dimensioningapplive-app-al7bqt.streamlit.app/)

---

## 🚀 Industry Context & Problem Statement

Couriers and e-commerce logistics companies bill parcels by the greater of **actual weight** and **volumetric weight**:

$$\text{Volumetric Weight (kg)} = \frac{\text{Length (cm)} \times \text{Width (cm)} \times \text{Height (cm)}}{5000}$$

Manual tape-measuring at parcel intake is slow, labor-intensive, and error-prone. This project delivers an automated, computer-vision parcel dimensioning system that measures metric-scale $L \times W \times H$ from smartphone video with known Augmented Reality (AR) camera geometry, filters unreliable frames with a rigorous Quality Gate, and verifies accuracy against ground-truth 3D bounding boxes from **Google Research's Objectron** dataset.

---

## 🏆 Acceptance Criteria Verification (A1 – A7)

Benchmarked across **55 official Objectron test sequences** (`cereal_box` and `book` categories) and the OpenCV chessboard calibration dataset:

| Criterion | Requirement Description | Target Specification | Benchmark Measured Result | Status |
| :---: | :--- | :---: | :---: | :---: |
| **A1** | **Camera Calibration RMS Error** | $\le 0.60\text{ px}$ | **$0.4087\text{ px}$** | **PASSED** ✅ |
| **A2** | **Method A Mean Absolute Error (MAE)** | $\le 2.0\text{ cm}$ per dim | **$0.646\text{ cm}$** | **PASSED** ✅ |
| **A3** | **Volumetric Weight Accuracy** | $\pm 15\%$ for $\ge 75\%$ videos | **$82.0\%$** | **PASSED** ✅ |
| **A4** | **Method A 3D IoU (Sutherland-Hodgman)** | Mean IoU $\ge 0.50$ | **$0.8851$** | **PASSED** ✅ |
| **A5** | **Method B Comparison & Failure Modes** | Comparative evaluation | Full Analysis Completed | **PASSED** ✅ |
| **A6** | **Quality Gate Rejection (Truncation)** | $\text{TPR} \ge 90\%$, $\text{FPR} \le 10\%$ | **$\text{TPR} = 100\%$, $\text{FPR} = 0\%$** | **PASSED** ✅ |
| **A7** | **Method A Measurement Latency** | $\le 2.0\text{ s}$ on CPU | **$0.028\text{ s}$** ($28\text{ ms}$) | **PASSED** ✅ |

---

## 🔬 System Architecture & Methods

```
                                  Phone Video Frame Sequence
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
             Quality Gate (F5)                               Camera Intrinsics
       • Blur Check (Laplacian > 60)                     • Lens Undistortion (F1)
       • Truncation Guard (> 8 px margin)                • OpenGL NDC Projection
       • Baseline Validator (> 5 cm)                                  │
                      │                                               │
             [Frame Accepted]                                         │
                      │                                               │
        ┌─────────────┴──────────────────────────────┐                │
        ▼                                            ▼                │
Method A: Multi-View (F3)                   Method B: Single-View (F4)│
 • DLT Triangulation (>= 2 frames)           • AR Ground Plane Intersect
 • Orthogonal Cuboid Fitting                 • Vertical Ray-Casting   │
        │                                            │                │
        └─────────────────────┬──────────────────────┘                │
                              ▼                                       │
                      Box Dimensions ($L, W, H$)                      │
                              │                                       │
                              ▼                                       │
                 Volumetric Weight ($LWH/5000$)                       │
                              │                                       │
                              ▼                                       │
                 3D HUD Wireframe Overlay ◄───────────────────────────┘
```

### 1. Camera Calibration & Projection (F1, A1)
* **OpenCV Chessboard Calibration:** $9 \times 6$ grid with sub-pixel corner refinement (`cv2.cornerSubPix`).
* **Objectron Projection Formulation:** Implements exact OpenGL view-projection matrix decomposition:
  $$\mathbf{M} = \mathbf{P} \cdot \mathbf{V}, \quad \begin{bmatrix} x^* \mathbf{M}_3 - \mathbf{M}_1 \\ y^* \mathbf{M}_3 - \mathbf{M}_0 \end{bmatrix} \mathbf{X}_w = 0$$

### 2. Method A: Multi-View Geometric Triangulation (F3, A2–A4)
* Selects keyframe pairs with optimal baseline ($10\text{--}20\text{ cm}$) to minimize AR pose drift.
* Solves Direct Linear Transform (DLT) across multi-view ray intersections.
* Fits 9-DoF oriented cuboid with singular value decomposition (SVD).

### 3. Method B: Single-View Ground-Plane Metrology (F4, A5)
* Intersects optical camera rays with the detected AR floor plane.
* Estimates parcel height via vertical edge projection and vanishing line constraints.

### 4. Quality Gate & Honest Frame Rejection (F5, A6)
* **Motion Blur Detection:** Variance of the Laplacian ($\sigma^2 < 60$ rejected).
* **Truncation Rejection:** Evaluates proximity to image margins to discard out-of-frame boxes.
* **Baseline Rejection:** Enforces camera displacement $\ge 5\text{ cm}$ to prevent collinear triangulation breakdown.

---

## 📂 Repository Structure

```
.
├── box_dimensioning/
│   ├── app/
│   │   ├── cli.py                    # Unified CLI (calibrate, measure, benchmark)
│   │   └── live_app.py               # Interactive Streamlit Web App
│   ├── calibration/
│   │   ├── camera_calibrator.py      # Chessboard calibrator & undistorter
│   │   └── intrinsics_checker.py     # Objectron projection sanity checker
│   ├── dataset/
│   │   ├── box_geometry.py           # 3D Box & Sutherland-Hodgman 3D IoU
│   │   ├── downloader.py             # Public Objectron GCS downloader
│   │   ├── objectron_parser.py       # Protobuf metadata & pose parser
│   │   └── proto/                    # Compiled Objectron protobuf schemas
│   ├── evaluation/
│   │   ├── benchmark.py              # Test split automated benchmark suite
│   │   └── metrics.py                # MAE, Volumetric Error & IoU metrics
│   ├── pipeline/
│   │   ├── dimensioner.py            # End-to-end pipeline & wireframe overlay
│   │   ├── method_a_multiview.py     # Method A (multi-view triangulation)
│   │   ├── method_b_singleview.py    # Method B (single-view metrology)
│   │   └── quality_gate.py           # Blur, truncation & baseline filter
│   ├── segmentation/
│   │   ├── classical.py              # Bilateral filtering & edge suppression
│   │   └── comparator.py             # Classical vs learned comparator
│   └── tests/                        # 10/10 automated pytest suite
├── data/
│   ├── benchmark/                    # Benchmark report & raw JSON results
│   ├── calibration/                  # OpenCV sample chessboard images
│   └── objectron/                    # 55 Objectron test sequences
├── requirements.txt                  # Python dependencies
├── streamlit_app.py                  # Streamlit Cloud entrypoint
└── README.md
```

---

## ⚡ Quick Start

### 1. Installation
```bash
git clone https://github.com/Ishtiak-Ahmed886/box-dimensioning.git
cd box-dimensioning
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Interactive Dashboard
```bash
streamlit run streamlit_app.py
```
Open `http://localhost:8501` to test live video sequences.

### 3. Run Automated Tests
```bash
pytest box_dimensioning/tests -v
```
Output:
```text
box_dimensioning/tests/test_calibration.py::test_calibration_rms_reprojection_error PASSED [ 10%]
box_dimensioning/tests/test_calibration.py::test_undistortion_preserves_dimensions PASSED [ 20%]
box_dimensioning/tests/test_geometry.py::test_box_creation_and_properties PASSED [ 30%]
box_dimensioning/tests/test_geometry.py::test_sutherland_hodgman_exact_iou PASSED [ 40%]
box_dimensioning/tests/test_geometry.py::test_box_fit_reconstruction PASSED [ 50%]
box_dimensioning/tests/test_method_a.py::test_method_a_accuracy_and_latency_on_real_sequence PASSED [ 60%]
box_dimensioning/tests/test_method_b.py::test_method_b_pipeline_and_comparison PASSED [ 70%]
box_dimensioning/tests/test_quality_gate.py::test_quality_gate_blur_rejection PASSED [ 80%]
box_dimensioning/tests/test_quality_gate.py::test_quality_gate_baseline_rejection PASSED [ 90%]
box_dimensioning/tests/test_quality_gate.py::test_quality_gate_truncation_acceptance_criterion_a6 PASSED [100%]

============================= 10 passed in 4.47s =============================
```

### 4. CLI Usage Examples

* **Calibrate Camera:**
  ```bash
  python -m box_dimensioning.app.cli calibrate --dir data/calibration
  ```

* **Measure Package (Method A):**
  ```bash
  python -m box_dimensioning.app.cli measure --seq data/objectron/cereal_box/batch-26/22 --method A
  ```

* **Run Full 50-Sequence Benchmark:**
  ```bash
  python -m box_dimensioning.app.cli benchmark --data-dir data/objectron --max-seq 50
  ```

---

## 📊 Sample Output (JSON API)

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
    "is_truncated": false,
    "blur_score": 119.02,
    "is_blurry": false,
    "baseline_meters": 0.1499,
    "fit_residual": 0.02,
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

## 📜 License
This project is licensed under the [MIT License](LICENSE).
