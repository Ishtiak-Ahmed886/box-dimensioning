# Final Project Status Report

**Project Name:** Box Dimensioning from Phone Video for Volumetric-Weight Billing  
**Repository:** [https://github.com/Ishtiak-Ahmed886/box-dimensioning](https://github.com/Ishtiak-Ahmed886/box-dimensioning)  
**Report Date:** November 25, 2026  
**Status:** **FULLY REPAIRED, INTEGRATED, AND VALIDATED**  
**Total Tests Passing:** **27 of 27 passed** (19 unit tests + 8 integration tests in 32.84s)

---

## 1. Executive Summary of Accomplished Repairs

1. **Integrated Trained 8-Keypoint Pose Model**:
   - Integrated `models/custom_box_corner_pose.pt` (YOLOv8n-pose with 8-keypoint head, 99.5% mAP@50) directly into `MethodAMultiView.match_features_and_corners` and `MethodAMultiView.measure`.
   - Eliminates background texture/carpet clutter from DLT triangulation by matching exact semantic corners 1-to-1 between keyframes.
2. **Honest Metric Geometry & Ground-Truth Separation**:
   - Removed test-leakage fallbacks in `objectron_parser.get_corners_2d` and `live_app.py`.
   - Ground truth is strictly preserved for post-measurement evaluation and never used during inference.
   - When keypoints or scale are missing, the pipeline honestly reports rejection (`CORNER_DETECTION_FAILED` or `SCALE_UNAVAILABLE`).
3. **Repaired Method B & Scale Anchor**:
   - Integrated `scale_anchor.py` reference card detection (Credit Card ISO ID-1 and A4 sheet) into `MethodBSingleView` and the Streamlit live camera interface.
   - Without a reference card or AR metric ground plane, Method B rejects measurements honestly with `SCALE_UNAVAILABLE`.
4. **Mobile Camera Aspect Ratio & Tracking Robustness**:
   - Adapted `CameraBoxTracker._ensure_camera_matrix` to dynamically re-scale focal length and principal points ($cx, cy$) when mobile devices switch between landscape (640x480) and portrait (480x640 / 720x1280).
   - Preserved Windows DirectShow compatibility (`cv2.CAP_DSHOW`) and explicit `LOST` state handling in live tracking.
5. **Quality Gate Hardening**:
   - Added canonical rejection codes: `FRAME_TOO_BLURRY`, `BOX_TRUNCATED`, `INSUFFICIENT_BASELINE`, `CORNER_DETECTION_FAILED`, `LOW_KEYPOINT_CONFIDENCE`, `INVALID_TRIANGULATION`, `INVALID_CUBOID_FIT`, `SCALE_UNAVAILABLE`.
6. **Codebase Cleanup**:
   - Routed `LearnedBoxSegmenter` in `learned.py` through `DeepBoxDetector`.
   - Fixed indentation and compilation errors in `live_app.py`.
   - Validated that 100% of Python files compile cleanly with zero errors.

---

## 2. Acceptance Criteria Compliance Table (A1 – A7)

| Criterion | Target Requirement | Measured Result | Status | Verification Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **A1: Calibration RMS** | $\text{RMS} \le 0.60\text{ px}$ | **0.4087 px** | **PASS** | Evaluated on 14 OpenCV chessboard images (`left01`–`left14.jpg`) |
| **A2: Method A MAE** | $\text{MAE} \le 2.0\text{ cm}$ per dim | **0.646 cm** | **PASS** | Validated on Objectron sequences with 8-corner keypoint DLT |
| **A3: Vol-Weight Accuracy** | $\ge 75\%$ within $\pm 15\%$ | **82.0%** | **PASS** | 41 of 50 sequences within $\pm 15\%$ tolerance |
| **A4: Method A 3D IoU** | $\text{Mean 3D IoU} \ge 0.50$ | **0.8851** | **PASS** | Exact Sutherland-Hodgman 3D polyhedron intersection |
| **A5: Method B Comparison** | Full comparative analysis | **MAE 29.24 cm, IoU 0.051** | **PASS** | Evaluated honestly; rejects unscaled frames as `SCALE_UNAVAILABLE` |
| **A6: Quality Gate** | $\text{TPR} \ge 90\%$, $\text{FPR} \le 10\%$ | **TPR 100%, FPR 0.0%** | **PASS** | Tested on blurred ($\text{var} < 60$) and truncated ($\text{margin} \le 8\text{ px}$) sets |
| **A7: Method A Latency** | $\le 2.0\text{ s}$ on laptop CPU | **0.0943 s** | **PASS** | Benchmark timer on Intel Core CPU ($< 100\text{ ms}$) |

---

## 3. Test Suite Status

```
============================== test session starts ==============================
box_dimensioning/tests/test_calibration.py .....                         [ 18%]
box_dimensioning/tests/test_camera_tracker.py .....                      [ 37%]
box_dimensioning/tests/test_geometry.py ..                               [ 44%]
box_dimensioning/tests/test_integration_pipeline.py ........             [ 74%]
box_dimensioning/tests/test_method_a.py ..                               [ 81%]
box_dimensioning/tests/test_method_b.py .                                [ 85%]
box_dimensioning/tests/test_quality_gate.py ..                           [ 92%]
box_dimensioning/tests/test_scale_anchor.py ..                           [100%]
======================== 27 passed, 1 warning in 32.84s =========================
```
