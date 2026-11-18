"""3D Box Visualization & Rendering Module.
Provides:
1. True 3D perspective wireframe & shaded face projection onto camera frames.
2. Interactive 3D WebGL Mesh & Coordinate space rendering via Plotly.
"""

from typing import Tuple, List, Optional
import cv2
import numpy as np
import plotly.graph_objects as go


def render_true_3d_wireframe_hud(
    image: np.ndarray,
    center_3d: np.ndarray,
    rot_3d: np.ndarray,
    length_cm: float,
    width_cm: float,
    height_cm: float,
    K: np.ndarray,
    volumetric_weight_kg: float
) -> np.ndarray:
    """Projects a true 3D oriented cuboid onto the 2D camera image with 12 wireframe edges,
    translucent 3D shaded faces, vertex spheres, and 3D metric dimension annotations.
    """
    annotated = image.copy()
    h_img, w_img = annotated.shape[:2]

    # Convert dimensions from cm to meters in camera frame
    sx = max(0.04, length_cm / 100.0)
    sy = max(0.02, height_cm / 100.0)
    sz = max(0.03, width_cm / 100.0)

    w2 = sx / 2.0
    h2 = sy / 2.0
    d2 = sz / 2.0

    # 8 local 3D corners centered at local box origin
    local_corners = np.array([
        [-w2, -h2, -d2],  # 0: bottom-left-back
        [-w2, -h2, +d2],  # 1: bottom-left-front
        [-w2, +h2, -d2],  # 2: top-left-back
        [-w2, +h2, +d2],  # 3: top-left-front
        [+w2, -h2, -d2],  # 4: bottom-right-back
        [+w2, -h2, +d2],  # 5: bottom-right-front
        [+w2, +h2, -d2],  # 6: top-right-back
        [+w2, +h2, +d2],  # 7: top-right-front
    ], dtype=np.float64)

    # Transform 3D corners into camera coordinate system: X_cam = R * X_local + C
    corners_cam = (rot_3d @ local_corners.T).T + center_3d.reshape(1, 3)

    # Project the 8 3D vertices using pinhole intrinsics K: [u, v] = [fx*X/Z + cx, fy*Y/Z + cy]
    pts_2d = []
    valid_count = 0
    for pt in corners_cam:
        z = pt[2]
        if z > 0.05:
            u = float(K[0, 0] * (pt[0] / z) + K[0, 2])
            v = float(K[1, 1] * (pt[1] / z) + K[1, 2])
            # Clamp within reasonable projection bounds
            u_clamped = int(np.clip(u, -w_img, 2 * w_img))
            v_clamped = int(np.clip(v, -h_img, 2 * h_img))
            pts_2d.append([u_clamped, v_clamped])
            valid_count += 1
        else:
            pts_2d.append([w_img // 2, h_img // 2])

    pts_2d = np.array(pts_2d, dtype=np.int32)

    # 1. Shaded 3D Faces (Translucent AR effect)
    overlay = annotated.copy()
    try:
        # Top Face: [2, 3, 7, 6] (Cyan)
        top_poly = np.array([pts_2d[2], pts_2d[3], pts_2d[7], pts_2d[6]], dtype=np.int32)
        cv2.fillPoly(overlay, [top_poly], (255, 200, 0))

        # Front Face: [1, 3, 7, 5] (Green)
        front_poly = np.array([pts_2d[1], pts_2d[3], pts_2d[7], pts_2d[5]], dtype=np.int32)
        cv2.fillPoly(overlay, [front_poly], (0, 220, 100))

        # Right Side Face: [5, 7, 6, 4] (Magenta / Purple)
        side_poly = np.array([pts_2d[5], pts_2d[7], pts_2d[6], pts_2d[4]], dtype=np.int32)
        cv2.fillPoly(overlay, [side_poly], (200, 50, 200))

        # Blend with original image (35% opacity)
        cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)
    except Exception:
        pass

    # 2. Draw 12 True 3D Edges
    # 4 Bottom floor edges (Yellow/Orange)
    bottom_edges = [(0, 1), (1, 5), (5, 4), (4, 0)]
    for i, j in bottom_edges:
        cv2.line(annotated, tuple(pts_2d[i]), tuple(pts_2d[j]), (0, 215, 255), 2, cv2.LINE_AA)

    # 4 Vertical height posts (Bright Green)
    pillar_edges = [(0, 2), (1, 3), (4, 6), (5, 7)]
    for i, j in pillar_edges:
        cv2.line(annotated, tuple(pts_2d[i]), tuple(pts_2d[j]), (0, 255, 0), 3, cv2.LINE_AA)

    # 4 Top roof edges (Cyan)
    top_edges = [(2, 3), (3, 7), (7, 6), (6, 2)]
    for i, j in top_edges:
        cv2.line(annotated, tuple(pts_2d[i]), tuple(pts_2d[j]), (255, 255, 0), 3, cv2.LINE_AA)

    # 3. Draw 8 3D Corner Vertex Spheres with Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, (ux, vy) in enumerate(pts_2d):
        cv2.circle(annotated, (ux, vy), 7, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(annotated, (ux, vy), 9, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"V{idx}", (ux + 6, vy - 6), font, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    # 4. Draw 3D Dimension Callouts along projected lines
    # Length along bottom front edge (1 -> 5)
    mid_l = ((pts_2d[1] + pts_2d[5]) // 2).astype(int)
    cv2.putText(annotated, f"L: {length_cm:.1f}cm", (mid_l[0] - 30, mid_l[1] + 20), font, 0.60, (0, 255, 255), 2, cv2.LINE_AA)

    # Height along vertical pillar (1 -> 3)
    mid_h = ((pts_2d[1] + pts_2d[3]) // 2).astype(int)
    cv2.putText(annotated, f"H: {height_cm:.1f}cm", (mid_h[0] - 70, mid_h[1]), font, 0.60, (0, 255, 0), 2, cv2.LINE_AA)

    # Width along right side edge (5 -> 4)
    mid_w = ((pts_2d[5] + pts_2d[4]) // 2).astype(int)
    cv2.putText(annotated, f"W: {width_cm:.1f}cm", (mid_w[0] + 10, mid_w[1] + 15), font, 0.60, (255, 255, 0), 2, cv2.LINE_AA)

    # 5. Top-Left Courier HUD Banner
    cv2.rectangle(annotated, (20, 20), (560, 160), (0, 0, 0), -1)
    cv2.rectangle(annotated, (20, 20), (560, 160), (0, 255, 0), 2)
    cv2.putText(annotated, "3D VOLUMETRIC PARCEL INTAKE (AR HUD)", (35, 52), font, 0.70, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(annotated, f"L: {length_cm:.1f} cm   W: {width_cm:.1f} cm   H: {height_cm:.1f} cm", (35, 88), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated, f"Volumetric Weight: {volumetric_weight_kg:.3f} kg (L*W*H/5000)", (35, 125), font, 0.68, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated, f"3D Vertices: 8 corners | Edges: 12 wireframe rails", (35, 150), font, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    return annotated


def create_interactive_3d_plotly_figure(
    length_cm: float,
    width_cm: float,
    height_cm: float,
    volumetric_weight_kg: float,
    title: str = "Interactive 3D Parcel Model"
) -> go.Figure:
    """Creates a fully rotatable, zoomable 3D WebGL cuboid model with metric dimensions,
    3D shaded faces, wireframe edge rails, and ground grid.
    """
    l = float(length_cm)
    w = float(width_cm)
    h = float(height_cm)

    # 8 3D vertices positioned above ground plane (z >= 0)
    # x: length, y: width, z: height
    x = [0, l, l, 0, 0, l, l, 0]
    y = [0, 0, w, w, 0, 0, w, w]
    z = [0, 0, 0, 0, h, h, h, h]

    # Triangular faces (12 triangles for 6 rectangular faces)
    i_indices = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
    j_indices = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    k_indices = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]

    fig = go.Figure()

    # 1. 3D Semi-transparent Shaded Box
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        i=i_indices, j=j_indices, k=k_indices,
        opacity=0.45,
        color='cyan',
        flatshading=True,
        name='3D Parcel Volume',
        hoverinfo='skip'
    ))

    # 2. 12 3D Wireframe Edges
    edges = [
        # Bottom
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Pillars
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    edge_x = []
    edge_y = []
    edge_z = []
    for p1, p2 in edges:
        edge_x.extend([x[p1], x[p2], None])
        edge_y.extend([y[p1], y[p2], None])
        edge_z.extend([z[p1], z[p2], None])

    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='#00ff88', width=5),
        name='3D Wireframe Edges',
        hoverinfo='skip'
    ))

    # 3. 8 3D Corner Spheres with Hover Tooltips
    labels = [f"V{idx}: ({x[idx]:.1f}, {y[idx]:.1f}, {z[idx]:.1f}) cm" for idx in range(8)]
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers+text',
        marker=dict(size=6, color='red'),
        text=[f"V{idx}" for idx in range(8)],
        textposition="top center",
        hovertext=labels,
        hoverinfo="text",
        name='3D Vertices'
    ))

    # Set up isometric camera viewpoint and layout
    max_dim = max(l, w, h, 20.0)
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><sup>L={l:.1f}cm × W={w:.1f}cm × H={h:.1f}cm | Volumetric: {volumetric_weight_kg:.3f} kg</sup>",
            x=0.05, y=0.92
        ),
        scene=dict(
            xaxis=dict(title='Length (cm)', range=[0, max_dim * 1.2], backgroundcolor="#111", gridcolor="#333"),
            yaxis=dict(title='Width (cm)', range=[0, max_dim * 1.2], backgroundcolor="#111", gridcolor="#333"),
            zaxis=dict(title='Height (cm)', range=[0, max_dim * 1.2], backgroundcolor="#111", gridcolor="#333"),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.2),
                up=dict(x=0, y=0, z=1)
            )
        ),
        margin=dict(l=10, r=10, b=10, t=50),
        paper_bgcolor="#0e1117",
        height=450
    )

    return fig
