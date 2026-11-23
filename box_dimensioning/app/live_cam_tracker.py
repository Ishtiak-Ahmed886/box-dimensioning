"""Standalone Live Laptop Camera Tracker Application.

Supports:
1. Physical laptop webcam stream (default device 0).
2. Simulated video / synthetic test loop when no camera is present.
3. Interactive keyboard controls ('q' to quit, 'r' to reset, 's' to snapshot).

Usage:
    python -m box_dimensioning.app.live_cam_tracker [--device 0] [--simulated] [--distance-cm 60]
"""

import argparse
import sys
import os
import time
import cv2
import numpy as np

# Add repo root to sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from box_dimensioning.pipeline.camera_box_tracker import CameraBoxTracker, TrackingState


def generate_synthetic_box_frame(
    width: int = 640,
    height: int = 480,
    box_pos: tuple = (200, 150),
    box_size: tuple = (220, 160),
    bg_color: tuple = (235, 230, 225),
    box_color: tuple = (65, 120, 185),
    noise_level: float = 5.0,
) -> np.ndarray:
    """Generates synthetic office desk environment with a cardboard box for testing."""
    frame = np.full((height, width, 3), bg_color, dtype=np.uint8)

    # Desk edge line
    cv2.line(frame, (0, int(height * 0.75)), (width, int(height * 0.75)), (180, 170, 160), 2)

    # Distractor clutter (cup, keyboard strip)
    # 1. Round coffee cup (distractor that must be filtered out)
    cv2.circle(frame, (80, 360), 28, (210, 210, 210), -1)
    cv2.circle(frame, (80, 360), 28, (120, 120, 120), 2)
    cv2.circle(frame, (80, 360), 18, (170, 170, 170), -1)

    # 2. Thin rectangular pen/stick
    cv2.rectangle(frame, (480, 370), (600, 378), (40, 40, 180), -1)

    # Main Cardboard Box with depth/perspective faces
    bx, by = box_pos
    bw, bh = box_size

    # Front face
    pts_front = np.array([
        [bx, by + 30],
        [bx + bw - 30, by + 30],
        [bx + bw - 30, by + bh],
        [bx, by + bh]
    ], dtype=np.int32)
    cv2.fillPoly(frame, [pts_front], box_color)
    cv2.polylines(frame, [pts_front], True, (40, 75, 120), 2)

    # Top face
    pts_top = np.array([
        [bx, by + 30],
        [bx + 30, by],
        [bx + bw, by],
        [bx + bw - 30, by + 30]
    ], dtype=np.int32)
    cv2.fillPoly(frame, [pts_top], (box_color[0] + 25, box_color[1] + 25, box_color[2] + 25))
    cv2.polylines(frame, [pts_top], True, (40, 75, 120), 2)

    # Right face
    pts_right = np.array([
        [bx + bw - 30, by + 30],
        [bx + bw, by],
        [bx + bw, by + bh - 30],
        [bx + bw - 30, by + bh]
    ], dtype=np.int32)
    cv2.fillPoly(frame, [pts_right], (max(0, box_color[0] - 25), max(0, box_color[1] - 25), max(0, box_color[2] - 25)))
    cv2.polylines(frame, [pts_right], True, (40, 75, 120), 2)

    # Add subtle Gaussian noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, frame.shape).astype(np.float32)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return frame


def run_camera_tracker(
    device_id: int = 0,
    video_path: str = None,
    simulated: bool = False,
    distance_cm: float = 60.0,
    detection_interval: int = 15,
    tracker_type: str = "mil",
    window_name: str = "Live Laptop Camera Box Tracker",
):
    """Runs real-time camera tracking loop."""
    print("=" * 65)
    print(" 📹 Live Laptop Camera Box Tracker & Volumetric Dimensioner")
    print("=" * 65)
    print(f" Target Distance: {distance_cm} cm | Re-detect Interval: {detection_interval} frames")
    print(f" Tracking Engine: {tracker_type.upper()}")
    print(" Keyboard Controls: [q] Quit  [r] Reset Tracker  [s] Save Frame")
    print("=" * 65)

    tracker = CameraBoxTracker(
        default_distance_cm=distance_cm,
        detection_interval=detection_interval,
        tracker_type=tracker_type,
    )

    cap = None
    use_synthetic = simulated

    if not use_synthetic:
        if video_path and os.path.isfile(video_path):
            print(f"[Tracker] Opening video file: {video_path}")
            cap = cv2.VideoCapture(video_path)
        else:
            print(f"[Tracker] Opening webcam device {device_id}...")
            if sys.platform.startswith("win"):
                cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(device_id)
            else:
                cap = cv2.VideoCapture(device_id)

        if not cap or not cap.isOpened():
            print(f"[Tracker] Unable to open camera device {device_id}. Falling back to simulated stream.")
            use_synthetic = True
            if cap:
                cap.release()

    sim_x = 180
    sim_dx = 2
    snapshot_idx = 0

    try:
        while True:
            if use_synthetic:
                # Oscillate box horizontally to simulate live movement and tracking
                sim_x += sim_dx
                if sim_x > 260 or sim_x < 140:
                    sim_dx = -sim_dx
                frame = generate_synthetic_box_frame(box_pos=(sim_x, 140))
                time.sleep(0.033)  # ~30 FPS
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[Tracker] Stream ended or frame read failed. Restarting video...")
                    if video_path:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break

            # Process frame through tracker
            result = tracker.process_frame(frame)
            annotated = result["annotated_frame"]

            # Display frame
            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Esc or 'q'
                print("[Tracker] Exiting.")
                break
            elif key == ord('r'):
                print("[Tracker] Resetting tracking state...")
                tracker.reset()
            elif key == ord('s'):
                snapshot_idx += 1
                fname = f"box_tracker_snapshot_{snapshot_idx}.png"
                cv2.imwrite(fname, annotated)
                print(f"[Tracker] Snapshot saved to {fname}")

    finally:
        if cap is not None and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Live Laptop Camera Box Tracker & Volumetric Dimensioner")
    parser.add_argument("--device", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Optional video file path for offline simulation")
    parser.add_argument("--simulated", action="store_true", help="Force synthetic simulated video stream")
    parser.add_argument("--distance-cm", type=float, default=60.0, help="Camera-to-box distance in cm (default: 60.0)")
    parser.add_argument("--interval", type=int, default=15, help="Re-detection frame interval (default: 15)")
    parser.add_argument("--tracker", type=str, default="mil", choices=["mil", "csrt", "kcf", "optical_flow"],
                        help="Tracking algorithm (default: mil)")

    args = parser.parse_args()
    run_camera_tracker(
        device_id=args.device,
        video_path=args.video,
        simulated=args.simulated,
        distance_cm=args.distance_cm,
        detection_interval=args.interval,
        tracker_type=args.tracker,
    )


if __name__ == "__main__":
    main()
