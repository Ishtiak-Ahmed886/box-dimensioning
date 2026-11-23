"""Live Laptop Webcam Verification Script.
Connects directly to laptop webcam via DirectShow, logs visual detections,
saves live frames and annotations to artifacts, and verifies box detection.
"""

import os
import sys
import time
import json
import cv2
import numpy as np

# Add repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from box_dimensioning.pipeline.camera_box_tracker import CameraBoxTracker, TrackingState

def run_live_verification(output_dir: str, num_frames: int = 15):
    print("=" * 60)
    print("📸 CONNECTING DIRECTLY TO LAPTOP WEBCAM (DEVICE 0)...")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Open camera with DirectShow on Windows
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(0, backend)
    
    if not cap.isOpened():
        print("❌ ERROR: Unable to open physical webcam device 0.")
        sys.exit(1)
        
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Warm up camera sensor (auto-exposure / white balance)
    print("⏳ Warming up camera sensor (auto-exposure / white balance)...")
    for _ in range(5):
        cap.read()
        time.sleep(0.05)
        
    tracker = CameraBoxTracker(
        default_distance_cm=60.0,
        detection_interval=5,
        tracker_type="mil",
    )
    
    print(f"🎬 Capturing and analyzing {num_frames} live camera frames...")
    
    saved_frames = []
    detection_logs = []
    
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"⚠️ Frame {i+1}: Failed to grab frame.")
            continue
            
        h, w, c = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate scene lighting and sharpness
        mean_brightness = float(np.mean(gray))
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        # Run tracker
        result = tracker.process_frame(frame)
        state = result["state"]
        fps = result["fps"]
        bbox = result.get("bbox")
        corners = result.get("corners_2d")
        dims = result["dimensions_cm"]
        vw = result["volumetric_weight_kg"]
        annotated = result["annotated_frame"]
        
        # Contour inspection: what is the detector seeing?
        # Threshold / Edge analysis for scene clutter vs box
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects_seen = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  # notable object
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
                x, y, bw, bh = cv2.boundingRect(cnt)
                solidity = area / (bw * bh) if (bw * bh) > 0 else 0
                circularity = (4 * np.pi * area) / (peri * peri) if peri > 0 else 0
                
                is_box = tracker.is_box_like_contour(cnt, (w, h))
                objects_seen.append({
                    "bbox": [int(x), int(y), int(bw), int(bh)],
                    "area": float(area),
                    "vertices": len(approx),
                    "solidity": round(float(solidity), 3),
                    "circularity": round(float(circularity), 3),
                    "classified_as_box": bool(is_box),
                })
        
        frame_log = {
            "frame_idx": i + 1,
            "resolution": f"{w}x{h}",
            "mean_brightness": round(mean_brightness, 1),
            "blur_score": round(blur_score, 1),
            "state": str(state),
            "fps": round(fps, 1),
            "box_detected": state == TrackingState.TRACKING,
            "bbox_2d": bbox,
            "dimensions_cm": dims,
            "volumetric_weight_kg": round(vw, 4),
            "objects_analyzed": len(objects_seen),
            "box_candidate_count": sum(1 for obj in objects_seen if obj["classified_as_box"]),
        }
        detection_logs.append(frame_log)
        
        print(f"--- [Frame {i+1:02d}/{num_frames}] ---")
        print(f"  Lighting: {mean_brightness:.1f}/255 | Sharpness: {blur_score:.1f} (Threshold > 60.0)")
        print(f"  Objects analyzed in scene: {len(objects_seen)}")
        for idx, obj in enumerate(objects_seen[:3]):
            tag = "📦 BOX CANDIDATE" if obj["classified_as_box"] else "❌ REJECTED CLUTTER"
            print(f"    Object {idx+1}: Area={obj['area']:.0f}px, Vertices={obj['vertices']}, Solidity={obj['solidity']}, Circularity={obj['circularity']} -> {tag}")
            
        print(f"  Tracker State: {state}")
        if state == TrackingState.TRACKING and bbox:
            print(f"  ✅ BOX DETECTED & TRACKED at bbox: {bbox}")
            print(f"  Estimated Dimensions: {dims['length']:.1f} x {dims['width']:.1f} x {dims['height']:.1f} cm")
            print(f"  Volumetric Weight: {vw:.4f} kg")
            if corners is not None and len(corners) == 8:
                print(f"  8 Projected Corners (C1-C8):")
                for c_idx, (cx, cy) in enumerate(corners):
                    print(f"    C{c_idx+1}: ({cx:.1f}, {cy:.1f})")
        else:
            print("  ℹ️ No box-like rectangular package detected in this frame (waiting for box in camera field of view).")
            
        # Save key sample frames (initial, middle, final)
        if i in [0, num_frames // 2, num_frames - 1] or state == TrackingState.TRACKING:
            raw_path = os.path.join(output_dir, f"webcam_raw_frame_{i+1}.jpg")
            annot_path = os.path.join(output_dir, f"webcam_annotated_frame_{i+1}.jpg")
            cv2.imwrite(raw_path, frame)
            cv2.imwrite(annot_path, annotated)
            saved_frames.append({
                "frame_idx": i + 1,
                "raw_image": raw_path,
                "annotated_image": annot_path,
                "state": str(state),
            })
            
        time.sleep(0.06)
        
    cap.release()
    print("=" * 60)
    print("✅ CAMERA FEED CAPTURE & ANALYSIS COMPLETE!")
    print(f"Saved visual frames to: {output_dir}")
    print("=" * 60)
    
    # Save JSON summary log
    summary_path = os.path.join(output_dir, "camera_detection_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "total_frames": num_frames,
            "saved_frames": saved_frames,
            "detection_logs": detection_logs,
        }, f, indent=2)

if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO_ROOT, "camera_evidence")
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    run_live_verification(out_dir, frames)
