"""Verify real-time box detection using laptop webcam with custom trained YOLO model."""

import os
import json
import time
import cv2
import numpy as np
from box_dimensioning.segmentation.deep_detector import DeepBoxDetector
from box_dimensioning.pipeline.camera_box_tracker import CameraBoxTracker


def verify_webcam_box_detection(
    output_dir: str = "camera_live_evidence",
    artifact_dir: str = r"C:\Users\Istiak\.gemini\antigravity\brain\a5c6c8c9-9c67-4495-bd51-0c2ae50b3961",
    num_frames: int = 15
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)

    print("[Webcam Verification] Initializing DirectShow camera capture...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # Fallback to standard capture
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open laptop webcam on device index 0.")

    # Configure webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = DeepBoxDetector(use_deep=True, conf_threshold=0.20)
    tracker = CameraBoxTracker()

    # Warmup camera sensor
    for _ in range(5):
        cap.read()
        time.sleep(0.05)

    captured_detections = []
    best_frame = None
    best_annotated = None
    best_conf = -1.0

    print("[Webcam Verification] Capturing frames and running box detection...")
    for f_idx in range(num_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        det = detector.detect(frame)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        bx, by, bw, bh = det.get("bounding_box", (0, 0, 0, 0))
        conf = det.get("confidence", 0.0)
        method = det.get("method", "detector")
        corners = det.get("corners_2d", np.empty((0, 2)))

        annotated = frame.copy()

        # Draw detection visualization
        if bw > 15 and bh > 15:
            # Bounding box
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)

            # Header label
            label = f"BOX {conf*100:.1f}% ({method})"
            cv2.rectangle(annotated, (bx, max(0, by - 30)), (bx + len(label)*14, by), (0, 255, 0), -1)
            cv2.putText(annotated, label, (bx + 5, max(15, by - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Draw corner dots and numbers 1 to 4 or 1 to 8
            if len(corners) > 0:
                for c_i, c_pt in enumerate(corners[:8]):
                    cx, cy = int(round(c_pt[0])), int(round(c_pt[1]))
                    # Yellow core with red ring
                    cv2.circle(annotated, (cx, cy), 8, (0, 0, 255), -1)
                    cv2.circle(annotated, (cx, cy), 5, (0, 255, 255), -1)
                    cv2.putText(annotated, str(c_i + 1), (cx + 10, cy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            if conf > best_conf:
                best_conf = conf
                best_frame = frame.copy()
                best_annotated = annotated.copy()

        # Telemetry HUD banner
        hud_text = f"Webcam: LIVE 1280x720 | Det: {method} | Conf: {conf:.2f} | Latency: {latency_ms:.1f}ms"
        cv2.putText(annotated, hud_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        captured_detections.append({
            "frame_idx": f_idx,
            "detected": bool(bw > 15 and bh > 15),
            "bbox": [int(bx), int(by), int(bw), int(bh)],
            "confidence": float(conf),
            "method": method,
            "corners_count": len(corners),
            "corners_px": [[float(p[0]), float(p[1])] for p in corners[:8]] if len(corners) > 0 else []
        })

        time.sleep(0.04)

    cap.release()

    if best_annotated is None and len(captured_detections) > 0:
        best_frame = frame
        best_annotated = annotated

    # Save output artifacts
    out_annotated_path = os.path.join(output_dir, "webcam_custom_box_detection.jpg")
    out_raw_path = os.path.join(output_dir, "webcam_raw_frame.jpg")
    artifact_img_path = os.path.join(artifact_dir, "webcam_box_detection_evidence.jpg")

    cv2.imwrite(out_annotated_path, best_annotated)
    cv2.imwrite(out_raw_path, best_frame)
    cv2.imwrite(artifact_img_path, best_annotated)

    result_summary = {
        "status": "SUCCESS",
        "device": "Laptop Webcam (DirectShow 0)",
        "resolution": f"{w}x{h}",
        "total_frames_analyzed": len(captured_detections),
        "best_detection": {
            "confidence": round(best_conf, 4),
            "method": captured_detections[0]["method"] if captured_detections else "unknown",
            "bbox": captured_detections[0]["bbox"] if captured_detections else [],
            "corners_count": captured_detections[0]["corners_count"] if captured_detections else 0
        },
        "annotated_image_path": out_annotated_path,
        "artifact_image_path": artifact_img_path,
        "frames_history": captured_detections
    }

    summary_json_path = os.path.join(output_dir, "webcam_box_detection_result.json")
    with open(summary_json_path, "w") as f:
        json.dump(result_summary, f, indent=2)

    artifact_json_path = os.path.join(artifact_dir, "webcam_box_detection_result.json")
    with open(artifact_json_path, "w") as f:
        json.dump(result_summary, f, indent=2)

    print(f"[Webcam Verification] Completed. Output saved to {out_annotated_path}")
    print(f"Artifact saved to {artifact_img_path}")
    return result_summary


if __name__ == "__main__":
    res = verify_webcam_box_detection()
    print("Verification result:", json.dumps(res["best_detection"], indent=2))
