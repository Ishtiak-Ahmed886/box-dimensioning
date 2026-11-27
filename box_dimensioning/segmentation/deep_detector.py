"""Deep Learning Box Detector (Step 3 Upgrade).

Improvements over classical edge + contour detection:
1. YOLOv8-nano ONNX inference for fast box/parcel detection (no GPU needed).
2. If YOLO unavailable, falls back to MediaPipe Object Detector.
3. Final fallback: improved classical pipeline with Hough line intersection corners.

Returns 8 projected corner points (2D) + confidence score per detection.
"""

import os
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# YOLO ONNX Inference (lightweight, CPU-only, no ultralytics dependency)
# ---------------------------------------------------------------------------

COCO_PARCEL_CLASSES = {
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    28: "suitcase",
    39: "bottle",
    56: "chair",
    60: "dining table",
    73: "book",
    # Generic box-like objects
    0:  "person",   # exclude
}
BOX_LIKE_CLASSES = {28, 73, 60, 56, 25}  # suitcase, book, table, chair, umbrella


def _yolo_preprocess(image: np.ndarray, input_size: int = 640) -> Tuple[np.ndarray, float, int, int]:
    """Letterbox resize to 640×640 for YOLOv8 input."""
    h, w = image.shape[:2]
    scale = min(input_size / h, input_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    # BGR → RGB, HWC → NCHW, normalize
    blob = padded[:, :, ::-1].transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
    return blob, scale, (input_size - new_h) // 2, (input_size - new_w) // 2


def _yolo_postprocess(
    output: np.ndarray,
    orig_h: int, orig_w: int,
    scale: float, pad_h: int, pad_w: int,
    conf_thresh: float = 0.30,
    iou_thresh: float = 0.45,
) -> List[Dict]:
    """Decodes YOLOv8 output [1, 84, 8400] → list of detections."""
    pred = output[0].T  # (8400, 84)
    # Columns: cx, cy, w, h, cls0..cls79
    xy = pred[:, :2]
    wh = pred[:, 2:4]
    scores = pred[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]

    mask = confidences >= conf_thresh
    if not np.any(mask):
        return []

    xy = xy[mask]; wh = wh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    # Convert cx,cy,w,h (640-space) → x1,y1,x2,y2 (original image)
    input_size = 640
    x1 = ((xy[:, 0] - wh[:, 0] / 2) - pad_w) / scale
    y1 = ((xy[:, 1] - wh[:, 1] / 2) - pad_h) / scale
    x2 = ((xy[:, 0] + wh[:, 0] / 2) - pad_w) / scale
    y2 = ((xy[:, 1] + wh[:, 1] / 2) - pad_h) / scale

    x1 = np.clip(x1, 0, orig_w); x2 = np.clip(x2, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h); y2 = np.clip(y2, 0, orig_h)

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    # NMS
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(), confidences.tolist(), conf_thresh, iou_thresh
    )
    if indices is None or len(indices) == 0:
        return []

    detections = []
    for idx in indices.flatten():
        detections.append({
            "bbox": boxes_xyxy[idx].astype(int).tolist(),  # [x1,y1,x2,y2]
            "confidence": float(confidences[idx]),
            "class_id": int(class_ids[idx]),
        })
    return detections


class DeepBoxDetector:
    """Deep learning box/parcel detector with classical fallback.

    Priority:
    1. YOLOv8-nano ONNX (auto-downloaded on first use, ~6MB)
    2. Classical improved pipeline (Hough line intersections)
    """

    YOLO_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx"
    YOLO_MODEL_CACHE = None  # Will be set to user cache path

    def __init__(self, use_deep: bool = True, conf_threshold: float = 0.20):
        self.use_deep = use_deep
        self.conf_threshold = conf_threshold
        self._onnx_session = None
        self._onnx_tried = False
        self._custom_yolo = None
        self._custom_yolo_tried = False
        self._custom_pose = None
        self._custom_pose_tried = False
        self._ultralytics_yolo = None
        self._ultralytics_yolo_tried = False

    def _load_custom_pose(self) -> bool:
        """Loads custom fine-tuned YOLO 8-corner keypoint model if available."""
        if self._custom_pose is not None:
            return True
        if self._custom_pose_tried:
            return False
        self._custom_pose_tried = True
        custom_weights = os.path.join("models", "custom_box_corner_pose.pt")
        if os.path.isfile(custom_weights):
            try:
                cfg_dir = os.path.abspath(".ultralytics")
                os.makedirs(cfg_dir, exist_ok=True)
                os.environ["YOLO_CONFIG_DIR"] = cfg_dir

                from ultralytics import YOLO
                self._custom_pose = YOLO(custom_weights)
                print(f"[DeepBoxDetector] Loaded custom 8-corner pose YOLO detector from {custom_weights}")
                return True
            except Exception as e:
                print(f"[DeepBoxDetector] Failed to load custom pose weights: {e}")
        return False

    def _detect_custom_pose(self, image: np.ndarray) -> Optional[Dict]:
        """Runs custom YOLO 8-corner keypoint model on image."""
        try:
            results = self._custom_pose(image, conf=self.conf_threshold, verbose=False)
            if results and len(results) > 0 and len(results[0].boxes) > 0:
                confs = results[0].boxes.conf.cpu().numpy()
                best_idx = int(np.argmax(confs))
                best_conf = float(confs[best_idx])
                coords = [int(v) for v in results[0].boxes.xyxy[best_idx].cpu().numpy().tolist()]
                x1, y1, x2, y2 = coords
                bbox = (x1, y1, x2 - x1, y2 - y1)

                corners_8 = None
                if results[0].keypoints is not None and len(results[0].keypoints.xy) > best_idx:
                    kpts = results[0].keypoints.xy[best_idx].cpu().numpy()
                    if len(kpts) == 8:
                        corners_8 = self._refine_corners_in_roi(
                            image, (x1, y1, x2, y2), kpts.astype(np.float32)
                        )

                if corners_8 is None:
                    corners_8 = self._extract_roi_corners(image, (x1, y1, x2, y2))

                return {
                    "corners_2d": corners_8,
                    "corners_8_2d": corners_8,
                    "bounding_rect": bbox,
                    "bounding_box": bbox,
                    "confidence": best_conf,
                    "class_id": 0,
                    "method": "custom_trained_yolo_corner_pose",
                    "mask": self._make_mask(image.shape[:2], x1, y1, x2, y2),
                }
        except Exception as e:
            print(f"[DeepBoxDetector] Custom YOLO pose prediction error: {e}")
        return None

    def _load_custom_yolo(self) -> bool:
        """Loads custom fine-tuned YOLO box detector if available."""
        if self._custom_yolo is not None:
            return True
        if self._custom_yolo_tried:
            return False
        self._custom_yolo_tried = True
        custom_weights = os.path.join("models", "custom_box_yolo.pt")
        if os.path.isfile(custom_weights):
            try:
                # Direct Ultralytics config to workspace to avoid sandbox permission errors
                cfg_dir = os.path.abspath(".ultralytics")
                os.makedirs(cfg_dir, exist_ok=True)
                os.environ["YOLO_CONFIG_DIR"] = cfg_dir

                from ultralytics import YOLO
                self._custom_yolo = YOLO(custom_weights)
                print(f"[DeepBoxDetector] Loaded custom box YOLO detector from {custom_weights}")
                return True
            except Exception as e:
                print(f"[DeepBoxDetector] Failed to load custom YOLO weights: {e}")
        return False

    def _detect_custom_yolo(self, image: np.ndarray) -> Optional[Dict]:
        """Runs custom fine-tuned YOLO box detector on image."""
        try:
            results = self._custom_yolo(image, conf=self.conf_threshold, verbose=False)
            if results and len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                best_box = None
                best_conf = -1.0
                for b in boxes:
                    conf = float(b.conf[0])
                    if conf > best_conf:
                        best_conf = conf
                        best_box = b

                if best_box is not None:
                    coords = [int(v) for v in best_box.xyxy[0].tolist()]
                    x1, y1, x2, y2 = coords
                    bbox = (x1, y1, x2 - x1, y2 - y1)
                    corners = self._bbox_to_corners(x1, y1, x2, y2)
                    return {
                        "corners_2d": corners,
                        "bounding_rect": bbox,
                        "bounding_box": bbox,
                        "confidence": best_conf,
                        "class_id": 0,
                        "method": "custom_trained_yolo_box",
                        "mask": self._make_mask(image.shape[:2], x1, y1, x2, y2),
                    }
        except Exception as e:
            print(f"[DeepBoxDetector] Custom YOLO prediction error: {e}")
        return None

    def _load_ultralytics_yolo(self) -> bool:
        """Loads pretrained YOLOv8n detector from Ultralytics (CPU)."""
        if self._ultralytics_yolo is not None:
            return True
        if self._ultralytics_yolo_tried:
            return False
        self._ultralytics_yolo_tried = True
        try:
            cfg_dir = os.path.abspath(".ultralytics")
            os.makedirs(cfg_dir, exist_ok=True)
            os.environ["YOLO_CONFIG_DIR"] = cfg_dir

            from ultralytics import YOLO
            model_path = "yolov8n.pt"
            if not os.path.isfile(model_path):
                alt_path = os.path.join("models", "yolov8n.pt")
                if os.path.isfile(alt_path):
                    model_path = alt_path
            self._ultralytics_yolo = YOLO(model_path)
            return True
        except Exception as e:
            print(f"[DeepBoxDetector] Failed to load Ultralytics YOLO: {e}")
            return False

    def _detect_ultralytics_yolo(self, image: np.ndarray) -> Optional[Dict]:
        """Runs pretrained YOLOv8 to detect parcels, boxes, books, or luggage, rejecting people."""
        try:
            results = self._ultralytics_yolo(image, conf=max(0.18, self.conf_threshold), verbose=False)
            if not results or len(results) == 0 or len(results[0].boxes) == 0:
                return None

            boxes = results[0].boxes
            best_box = None
            best_score = -1.0
            # Target classes: 73: book, 28: suitcase, 24: backpack, 39: bottle, 60: dining table, 56: chair
            PRIORITY_CLASSES = {73: 2.0, 28: 2.0, 24: 1.5, 39: 1.2, 60: 1.0, 56: 0.8}

            for b in boxes:
                cls_id = int(b.cls[0])
                if cls_id == 0:  # STRICTLY REJECT 'person' to avoid author portraits / figures on covers
                    continue
                conf = float(b.conf[0])
                weight = PRIORITY_CLASSES.get(cls_id, 1.0)
                score = conf * weight
                if score > best_score:
                    best_score = score
                    best_box = b

            if best_box is not None:
                coords = [int(v) for v in best_box.xyxy[0].tolist()]
                x1, y1, x2, y2 = coords
                bbox = (x1, y1, x2 - x1, y2 - y1)
                corners = self._bbox_to_corners(x1, y1, x2, y2)
                cls_id = int(best_box.cls[0])
                cls_name = results[0].names.get(cls_id, "object")
                return {
                    "corners_2d": corners,
                    "bounding_rect": bbox,
                    "bounding_box": bbox,
                    "confidence": float(best_box.conf[0]),
                    "class_id": cls_id,
                    "method": f"ultralytics_{cls_name}",
                    "mask": self._make_mask(image.shape[:2], x1, y1, x2, y2),
                }
        except Exception as e:
            print(f"[DeepBoxDetector] Ultralytics YOLO detection error: {e}")
        return None

    def _load_onnx(self) -> bool:
        """Lazily loads YOLOv8-nano ONNX model, returns True if successful."""
        if self._onnx_session is not None:
            return True
        if self._onnx_tried:
            return False
        self._onnx_tried = True
        try:
            import onnxruntime as ort
            import os, urllib.request

            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "box_dimensioning")
            os.makedirs(cache_dir, exist_ok=True)
            model_path = os.path.join(cache_dir, "yolov8n.onnx")

            if not os.path.isfile(model_path) or os.path.getsize(model_path) < 1_000_000:
                print("[DeepBoxDetector] Downloading YOLOv8-nano ONNX model (~6 MB)...")
                urllib.request.urlretrieve(self.YOLO_MODEL_URL, model_path)
                print("[DeepBoxDetector] Model downloaded.")

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            opts.intra_op_num_threads = 2
            self._onnx_session = ort.InferenceSession(
                model_path, sess_options=opts, providers=['CPUExecutionProvider']
            )
            return True
        except Exception as e:
            print(f"[DeepBoxDetector] ONNX load failed: {e}. Using classical fallback.")
            return False

    def detect_corners(self, image: np.ndarray, candidate_bounding_rect: Optional[Tuple[int, int, int, int]] = None) -> Dict:
        """Alias for detect() providing polymorphic compatibility with classical and learned segmenters."""
        return self.detect(image)

    def detect(self, image: np.ndarray) -> Dict:
        """Main detection entry point. Returns corners_2d, bbox, confidence, method."""
        h, w = image.shape[:2]

        if self.use_deep:
            # Priority 1: Custom fine-tuned YOLO 8-corner keypoint model
            if self._load_custom_pose():
                res_pose = self._detect_custom_pose(image)
                if res_pose is not None:
                    return res_pose

            # Priority 2: Custom fine-tuned YOLO box detector
            if self._load_custom_yolo():
                res = self._detect_custom_yolo(image)
                if res is not None:
                    return res

            # Priority 3: Pretrained Ultralytics YOLOv8n (detects books, luggage, boxes; ignores persons)
            if self._load_ultralytics_yolo():
                res_u = self._detect_ultralytics_yolo(image)
                if res_u is not None:
                    return res_u

            # Priority 4: General YOLOv8-nano ONNX
            if self._load_onnx():
                result = self._detect_yolo(image)
                if result is not None:
                    return result

        # Classical fallback with Hough line intersections
        return self._detect_classical_improved(image)

    def _detect_yolo(self, image: np.ndarray) -> Optional[Dict]:
        """Runs YOLOv8 inference, picks largest detected box-like object."""
        try:
            h, w = image.shape[:2]
            blob, scale, pad_h, pad_w = _yolo_preprocess(image)
            input_name = self._onnx_session.get_inputs()[0].name
            outputs = self._onnx_session.run(None, {input_name: blob})
            detections = _yolo_postprocess(
                outputs[0], h, w, scale, pad_h, pad_w,
                conf_thresh=self.conf_threshold
            )

            if not detections:
                return None

            # Pick highest-confidence detection
            best = max(detections, key=lambda d: d['confidence'])
            x1, y1, x2, y2 = best['bbox']
            bbox = (x1, y1, x2 - x1, y2 - y1)
            corners = self._bbox_to_corners(x1, y1, x2, y2)

            return {
                "corners_2d": corners,
                "bounding_rect": bbox,
                "bounding_box": bbox,
                "confidence": best['confidence'],
                "class_id": best['class_id'],
                "method": "yolov8_deep",
                "mask": self._make_mask(image.shape[:2], x1, y1, x2, y2),
            }
        except Exception as e:
            print(f"[DeepBoxDetector] YOLO inference error: {e}")
            return None

    def _detect_classical_improved(self, image: np.ndarray) -> Dict:
        """Improved classical: bilateral + adaptive Canny + Hough line intersections.
        Completely bulletproof with fallback bounding box if any vision step fails.
        """
        h, w = image.shape[:2]
        corners = self._bbox_to_corners(w // 6, h // 6, 5 * w // 6, 5 * h // 6)
        bbox = (w // 6, h // 6, 2 * w // 3, 2 * h // 3)

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

            # Step 1: Bilateral filter (edge-preserving noise removal)
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)

            # Step 2: Adaptive Canny
            v = float(np.median(filtered))
            edges = cv2.Canny(filtered, int(max(0, 0.67 * v)), int(min(255, 1.33 * v)))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

            # Step 3: Find dominant central parcel/box contour
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            img_area = float(w * h)
            valid_cnts = []
            img_center = np.array([w * 0.5, h * 0.5])

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 0.02 * img_area <= area <= 0.85 * img_area:
                    rect = cv2.boundingRect(cnt)
                    bx, by, bw, bh = rect
                    cnt_center = np.array([bx + bw * 0.5, by + bh * 0.5])
                    dist_to_center = float(np.linalg.norm(cnt_center - img_center))
                    valid_cnts.append((area, dist_to_center, cnt, rect))

            # Prioritize large contours near the center
            if valid_cnts:
                # Score = area / (dist_to_center + 100)
                valid_cnts.sort(key=lambda item: item[0] / (item[1] + 100.0), reverse=True)
                best_area, _, best_cnt, best_rect = valid_cnts[0]
                bx, by, bw, bh = best_rect
                bbox = (bx, by, bw, bh)

                # Focus edge mask exclusively within this parcel ROI
                parcel_mask = np.zeros_like(edges)
                cv2.drawContours(parcel_mask, [best_cnt], -1, 255, thickness=-1)
                edges = cv2.bitwise_and(edges, edges, mask=parcel_mask)

                peri = cv2.arcLength(best_cnt, True)
                approx = cv2.approxPolyDP(best_cnt, 0.03 * peri, True)
                if len(approx) in (4, 8):
                    corners = approx.reshape(-1, 2).astype(np.float32)
                else:
                    corners = self._bbox_to_corners(bx, by, bx + bw, by + bh)
            else:
                bx, by, bw, bh = bbox

            # Step 4: Hough line detection inside parcel region
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                     minLineLength=int(min(bw, bh) * 0.15), maxLineGap=15)

            if lines is not None:
                try:
                    segs = np.asarray(lines, dtype=np.float32).reshape(-1, 4)
                    if len(segs) >= 4:
                        intersections = []
                        n_segs = len(segs)
                        for i in range(n_segs):
                            for j in range(i + 1, min(n_segs, i + 12)):
                                pt = self._line_intersection(segs[i], segs[j])
                                if pt is not None:
                                    x_pt, y_pt = pt
                                    # Ensure intersections stay strictly within the detected parcel ROI
                                    if bx - 5 <= x_pt <= bx + bw + 5 and by - 5 <= y_pt <= by + bh + 5:
                                        intersections.append([x_pt, y_pt])

                        if len(intersections) >= 4:
                            pts = np.array(intersections, dtype=np.float32)
                            n_clusters = min(8, len(pts))
                            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
                            _, _, centers = cv2.kmeans(
                                pts, n_clusters, None, criteria, 5, cv2.KMEANS_PP_CENTERS
                            )
                            corners = centers.astype(np.float32)
                except Exception:
                    pass

        except Exception as e:
            print(f"[DeepBoxDetector] Classical fallback caught error: {e}")

        x, y, bw, bh = bbox
        return {
            "corners_2d": corners,
            "bounding_rect": bbox,
            "bounding_box": bbox,
            "confidence": 0.5,
            "method": "classical_hough_improved",
            "mask": self._make_mask(image.shape[:2], x, y, x + bw, y + bh),
        }

    @staticmethod
    def _line_intersection(seg1, seg2) -> Optional[Tuple[float, float]]:
        """Finds intersection of two line segments (x1,y1,x2,y2)."""
        x1, y1, x2, y2 = map(float, seg1)
        x3, y3, x4, y4 = map(float, seg2)
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None
        t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
        t = t_num / denom
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return x, y

    @staticmethod
    def _extract_roi_corners(image: np.ndarray, xyxy: Tuple[int, int, int, int]) -> np.ndarray:
        """Extracts up to 8 salient 2D corners strictly inside the YOLO bounding box ROI."""
        h_img, w_img = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        x1 = max(0, min(x1, w_img - 2))
        y1 = max(0, min(y1, h_img - 2))
        x2 = max(x1 + 2, min(x2, w_img))
        y2 = max(y1 + 2, min(y2, h_img))
        bw, bh = x2 - x1, y2 - y1

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        roi = gray[y1:y2, x1:x2]
        min_dist = max(8.0, min(bw, bh) * 0.18)
        pts = cv2.goodFeaturesToTrack(roi, maxCorners=8, qualityLevel=0.04, minDistance=min_dist)
        if pts is not None and len(pts) >= 4:
            corners = pts.reshape(-1, 2).astype(np.float32)
            corners[:, 0] += x1
            corners[:, 1] += y1
            return corners
        return DeepBoxDetector._bbox_to_corners(x1, y1, x2, y2)

    @classmethod
    def _refine_corners_in_roi(
        cls,
        image: np.ndarray,
        xyxy: Tuple[int, int, int, int],
        kpts: np.ndarray,
    ) -> np.ndarray:
        """Keeps 2D keypoints strictly on the parcel; replaces wildly drifted ones with ROI corners."""
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        tol_x, tol_y = bw * 0.08, bh * 0.08

        out_of_box = (
            (kpts[:, 0] < x1 - tol_x) | (kpts[:, 0] > x2 + tol_x) |
            (kpts[:, 1] < y1 - tol_y) | (kpts[:, 1] > y2 + tol_y)
        )
        if int(np.sum(out_of_box)) >= 2:
            # 256px pose keypoints drifted outside the box — use clean ROI corners instead
            return cls._extract_roi_corners(image, (int(x1), int(y1), int(x2), int(y2)))

        refined = kpts.copy()
        refined[:, 0] = np.clip(refined[:, 0], x1, x2)
        refined[:, 1] = np.clip(refined[:, 1], y1, y2)
        return refined

    @staticmethod
    def _bbox_to_corners(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """Returns 4 corners of a bounding box as (N,2) array."""
        return np.array([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
        ], dtype=np.float32)

    @staticmethod
    def _make_mask(shape, x1, y1, x2, y2) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
        return mask
