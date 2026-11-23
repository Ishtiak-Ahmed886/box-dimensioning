# Box Dimensioning & Volumetric-Weight Billing: Benchmark Report

**Generated:** 2026-11-23T04:47:44Z

## Acceptance Criteria Compliance (A1 - A7)

| Criterion | Target Requirement | Measured Result | Status |
| :--- | :--- | :--- | :--- |
| **A1: Calibration RMS** | <= 0.60 px | 0.4087 px | PASSED |
| **A2: Method A MAE** | <= 2.0 cm per dim | 0.646 cm | PASSED |
| **A3: Vol-Weight Accuracy** | >= 75% within +-15% | 82.0% | PASSED |
| **A4: Method A 3D IoU** | Mean IoU >= 0.50 | 0.8851 | PASSED |
| **A5: Method B Comparison** | Full comparative eval | MAE 29.237 cm, IoU 0.0513 | PASSED |
| **A6: Quality Gate** | TPR >= 90%, FPR <= 10% | TPR: 100.0%, FPR: 0.0% | PASSED |
| **A7: Method A Latency** | <= 2.0 s per meas | 0.0943 s | PASSED |

## Method Comparison (Method A vs. Method B)

| Metric | Method A (Multi-View Geometric) | Method B (Single-View Metrology) |
| :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | 0.646 cm | 29.237 cm |
| **Vol-Weight Within 15%** | 82.0% | 6.0% |
| **Mean 3D IoU** | 0.8851 | 0.0513 |
| **Mean Latency (CPU)** | 0.0943 s | 0.0228 s |

### Key Failure Modes & Practical Insights
- **Printed Box Artwork**: High-contrast labels (e.g. cereal box logos, nutrition tables) create strong interior edges. Bilateral filtering and convex hull boundary analysis suppress interior false lines.
- **Thin Books**: Small absolute errors (e.g. 1.0 cm) on a 2.5 cm book yield ~40% relative error in volumetric weight; ground-plane metrology with vertical vanishing line calibration stabilizes thin package dimensioning.
- **AR Pose Drift**: Multi-view triangulation is sensitive to AR pose drift over long camera loops; picking keyframe pairs with 10–20 cm baseline minimizes accumulated drift.
- **Honest Frame Rejection**: Rejecting truncated and motion-blurred frames prevents spurious measurements and ensures courier intake compliance.