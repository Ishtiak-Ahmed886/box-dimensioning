# Beginner's Guide: Computer Vision Models, 3D Geometry & Pipeline Architecture

Welcome! If you are new to Computer Vision, Machine Learning, and 3D Geometry, this guide will walk you through the complete scientific and mathematical mechanics behind the **Box Dimensioning from Phone Video** project.

---

## 1. The Core Problem: Why Measuring from a Phone is Hard

When you take a photo with a smartphone, the real, 3D physical world ($X, Y, Z$) is flattened into a 2D grid of colored pixels ($x, y$).

In that flattening process, **all depth ($Z$) is lost**. A small cardboard box held 30 cm from the camera looks identical in size to a massive shipping crate located 5 meters away. This is known as **Monocular Scale Ambiguity**.

To calculate true physical dimensions in centimeters ($L \times W \times H$), a computer vision system must solve two foundational problems:
1. **Geometric Localization**: Where are the 8 corners of the parcel in 3D space?
2. **Metric Scale Recovery**: What is the multiplier that converts pixels into physical centimeters?

---

## 2. Key Terms: Crucial Distinctions You Must Understand

Many beginners confuse the following terms. In computer vision, they are completely different:

| Term | What It Is | Mathematical Representation | Example in this Project |
| :--- | :--- | :--- | :--- |
| **2D Bounding Box** | An axis-aligned rectangle enclosing the outer edges of an object in a photo. | 4 numbers: $[x, y, \text{width}, \text{height}]$ | **Stage 2: Cyan Box** produced by `custom_box_yolo.pt`. |
| **2D Corner Keypoints** | Exact pixel locations of the parcel vertices visible on the 2D image plane. | 8 coordinate pairs: $[(x_1, y_1), \dots, (x_8, y_8)]$ | **Stage 3: Yellow/Cyan dots ($D_1$–$D_8$)** produced by `custom_box_corner_pose.pt`. |
| **3D Cuboid Vertices** | The 8 true physical corners in 3-dimensional real-world space. | 8 3D points: $[(X_1, Y_1, Z_1), \dots, (X_8, Y_8, Z_8)]$ in meters/cm | Reconstructed by **DLT triangulation** or AR sequence geometry. |
| **Projected 3D Wireframe** | 3D cuboid corners mapped back onto a 2D image using camera intrinsics ($P = K[R\|t]$). | 2D pixel coordinates connected by 12 green edges. | **Stage 4: 12 Green Lines & Numbered Badges (1–8)**. |

> [!IMPORTANT]
> A standard YOLO model (like `custom_box_yolo.pt`) detects **only a 2D rectangle**. It cannot tell you how deep the box is or where its 3D corners are. To get the 3D cuboid, you need either **keypoint pose estimation** or **multi-view triangulation**.

---

## 3. Step-by-Step Pipeline Mechanics

```
Input Video Frame
     │
     ▼
[Step 1: Camera Calibration & Intrinsics] ──► Pinhole Camera Matrix K
     │
     ▼
[Step 2: Quality Gate Assessment]          ──► Reject if motion-blurred or truncated
     │
     ▼
[Step 3: Deep Object & Corner Detection]   ──► Bounding Box + 8 2D Corner Keypoints
     │
     ▼
[Step 4: Multi-View Triangulation (DLT)]   ──► 3D Point Cloud (X, Y, Z in meters)
     │
     ▼
[Step 5: 3D Cuboid Fitting & Alignment]    ──► Length, Width, Height (cm)
     │
     ▼
[Step 6: Volumetric-Weight Calculation]    ──► Billing Weight in kg
     │
     ▼
[Step 7: 4-Stage Visual Verification]      ──► UI Overlay & Digital Twin
```

---

### Step 1: Camera Intrinsics and the Pinhole Camera Model

A camera turns 3D real-world points $[X, Y, Z]^T$ into 2D image pixels $[u, v]^T$ using its **Intrinsic Calibration Matrix $K$**:

$$
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \frac{1}{Z} \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X_{cam} \\ Y_{cam} \\ Z_{cam} \end{bmatrix}
$$

- $f_x, f_y$: Focal lengths in pixel units (how much the camera zooms in).
- $c_x, c_y$: The principal point (the optical center of the image sensor, usually near the image center).
- Lens distortion ($k_1, k_2$): Barrel or pincushion warping caused by the physical glass curved lens.

**How our project gets $K$**:
1. **Offline**: Using `CameraCalibrator.calibrate()` with 14 chessboard images to achieve sub-pixel accuracy ($RMS \le 0.41$ px).
2. **On Smartphones**: Using `smart_intrinsics.py`, which reads EXIF camera metadata and queries a built-in database of 200+ phone models (Samsung Galaxy, Apple iPhone, Google Pixel).

---

### Step 2: Quality Gate (Preventing Bad Data)

Before calculating dimensions, the frame is audited by `quality_gate.py`:
- **Blur Assessment**: Computes the **Variance of the Laplacian** operator ($\nabla^2 I$). Sharp edges have high variance ($> 60.0$); blurry frames have low variance and are rejected with code `REJECT_BLURRY`.
- **Truncation Check**: If any corner of the box touches within 8 pixels of the camera border, the box is partially cut off. The frame is rejected with code `REJECT_TRUNCATED`.

---

### Step 3: Deep Neural Network Corner Detection

Our custom-trained model [`custom_box_corner_pose.pt`](file:///c:/Users/Istiak/OneDrive/Documents/content_creation1/models/custom_box_corner_pose.pt) is a YOLOv8-pose neural network:

1. The frame is resized to $256 \times 256$ pixels and passed through convolutional layers.
2. The network's detection head predicts the parcel bounding box $[x, y, w, h]$.
3. The network's keypoint head simultaneously outputs the 2D pixel coordinates of all **8 cuboid corners** ($D_1$ through $D_8$).
4. Because the model was trained on 1,200 packaging and video frames, it ignores high-contrast graphics (e.g. logos, barcodes) and locks onto the physical corners of the box.

---

### Step 4: Multi-View Reconstruction (DLT Triangulation)

When you move your phone around a box, you capture it from two different positions: **Frame 1** (camera center $C_1$) and **Frame 2** (camera center $C_2$). The distance between $C_1$ and $C_2$ is called the **baseline**.

1. For each corner point $u_1$ in Frame 1 and $u_2$ in Frame 2, a line (optical ray) is projected into 3D space.
2. The Direct Linear Transform (**DLT**) sets up a linear system $A X = 0$:

$$
A = \begin{bmatrix}
x_1^* M_1^{(3)} - M_1^{(1)} \\
y_1^* M_1^{(3)} - M_1^{(0)} \\
x_2^* M_2^{(3)} - M_2^{(1)} \\
y_2^* M_2^{(3)} - M_2^{(0)}
\end{bmatrix}
$$

3. Using **Singular Value Decomposition (SVD)** on matrix $A$, the algorithm finds the 3D point $X = (X, Y, Z)$ where the two camera rays intersect.

---

### Step 5: 3D Cuboid Fitting

Once the 3D point cloud $[X_1 \dots X_8]$ is reconstructed:
1. The center point is computed: $C = \frac{1}{N} \sum X_i$.
2. **Principal Component Analysis (PCA)** finds the 3 orthogonal principal axes along the box edges.
3. The points are projected onto the principal axes to calculate the true metric dimensions:
   - **Length ($L$)**: Longest dimension in cm.
   - **Width ($W$)**: Intermediate dimension in cm.
   - **Height ($H$)**: Smallest dimension in cm.

---

### Step 6: Volumetric-Weight Calculation

In commercial logistics (FedEx, DHL, UPS, courier services), packages are billed based on either physical weight or **Volumetric Weight** (the space they occupy in a cargo plane or delivery truck):

$$
\text{Volumetric Weight (kg)} = \frac{\text{Length (cm)} \times \text{Width (cm)} \times \text{Height (cm)}}{5000}
$$

If a package is very light (e.g. pillows or potato chips in a huge box), the courier charges based on this volumetric weight.

---

### Step 7: The 4 Visual Verification Stages

To ensure couriers and customers can trust the measurement, the application displays 4 distinct visual stages:

```
┌─────────────────────────┐  ┌─────────────────────────┐
│ Stage 1: Raw Frame      │  │ Stage 2: YOLO BBox      │
│ (Unprocessed video)     │  │ (2D bounding rectangle) │
└─────────────────────────┘  └─────────────────────────┘
┌─────────────────────────┐  ┌─────────────────────────┐
│ Stage 3: 2D Corners     │  │ Stage 4: 3D Cuboid      │
│ (D1 to D8 yellow dots)  │  │ (12 green edges + 1-8)  │
└─────────────────────────┘  └─────────────────────────┘
```

1. **Stage 1 (Raw Frame)**: Shows the unmodified camera feed.
2. **Stage 2 (YOLO BBox)**: Shows the detected region-of-interest (ROI).
3. **Stage 3 (2D Corners)**: Shows the 8 detected vertex locations ($D_1$–$D_8$) on the package.
4. **Stage 4 (3D Cuboid)**: Renders the 12 green cuboid edges connecting the corners in 3D perspective, with numbered yellow-and-red badges (1 through 8) and a center centroid marker.

---

## 4. Common Failure Modes & Why They Happen

| Failure Mode | Physical Cause | Mathematical Result | How the Pipeline Mitigates It |
| :--- | :--- | :--- | :--- |
| **Textureless Packages** | Smooth surfaces (e.g. plain red book) have no internal texture. SIFT matching defaults to background rugs or floors. | Triangulated points collapse onto the floor plane ($H \approx 0$). | Uses `custom_box_corner_pose.pt` which detects object boundaries rather than texture points. |
| **Insufficient Baseline** | User moves camera less than 4 cm between frames. | Rays are almost parallel; small 1-pixel errors cause massive depth errors ($Z \to \infty$). | `QualityGate` rejects pairs where baseline $< 5$ cm (`REJECT_INSUFFICIENT_BASELINE`). |
| **Motion Blur** | Quick hand shake while capturing video. | Edge gradients are smoothed out; corner detector cannot locate vertices within 10 px. | Laplacian variance drops $< 60$; frame is rejected (`REJECT_BLURRY`). |
| **Printed Artwork** | Cereal boxes with logos, text, nutrition tables. | Classical Canny edges trigger hundreds of lines inside the box face. | Bilateral filtering + deep neural pose model trained to isolate outer bounding vertices. |
