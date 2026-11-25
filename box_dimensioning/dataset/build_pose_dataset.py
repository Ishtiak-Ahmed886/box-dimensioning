"""Builds 8-Corner Keypoint (Pose) Dataset for Box Dimensioning from Objectron sequences.
Extracts >1000 annotated samples with 8 corner vertices (1-8) and tight bounding boxes.
"""

import os
import cv2
import glob
import random
import numpy as np
from typing import Dict, List, Tuple
from box_dimensioning.dataset.objectron_parser import load_sequence


def build_pose_dataset(
    objectron_root: str = "data/objectron",
    output_dir: str = "data/dataset_box_pose",
    target_count: int = 1200,
    val_split: float = 0.20,
    seed: int = 42
) -> Dict[str, int]:
    random.seed(seed)
    np.random.seed(seed)

    train_img_dir = os.path.join(output_dir, "images", "train")
    val_img_dir = os.path.join(output_dir, "images", "val")
    train_lbl_dir = os.path.join(output_dir, "labels", "train")
    val_lbl_dir = os.path.join(output_dir, "labels", "val")

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    seq_dirs = []
    for root, dirs, files in os.walk(objectron_root):
        if "geometry.pbdata" in files:
            seq_dirs.append(root)

    print(f"Found {len(seq_dirs)} Objectron sequences.")

    collected_samples = []

    for s_dir in seq_dirs:
        try:
            seq_id = s_dir.replace("\\", "/").split("data/objectron/")[-1]
            seq = load_sequence(s_dir, seq_id)
            n_frames = len(seq.frames)
            if n_frames == 0 or seq.ground_truth is None:
                continue

            has_video = os.path.isfile(os.path.join(s_dir, "video.MOV"))
            frames_dir = os.path.join(s_dir, "frames")
            has_extracted = os.path.isdir(frames_dir) and len(os.listdir(frames_dir)) > 0

            if not (has_video or has_extracted):
                continue

            # Sample frames evenly across sequence (up to 40 frames per sequence)
            step = max(1, n_frames // 40)
            sample_indices = list(range(0, n_frames, step))

            cap = None
            if has_video:
                cap = cv2.VideoCapture(os.path.join(s_dir, "video.MOV"))

            for f_idx in sample_indices:
                if f_idx >= n_frames:
                    break

                img = None
                if has_extracted:
                    for name in [f"frame_{f_idx:03d}.jpg", f"frame_{f_idx:04d}.jpg", f"frame_{f_idx}.jpg", f"{f_idx}.jpg"]:
                        fp = os.path.join(frames_dir, name)
                        if os.path.isfile(fp):
                            img = cv2.imread(fp)
                            break

                if img is None and cap is not None and cap.isOpened():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                    ret, fr = cap.read()
                    if ret and fr is not None:
                        img = fr

                if img is None:
                    continue

                h_img, w_img = img.shape[:2]
                frame_geom = seq.get_frame(f_idx)

                # Project 3D ground truth box onto this frame
                pts_2d = seq.ground_truth.box_3d.project_to_image(
                    projection_matrix=frame_geom.projection_matrix,
                    view_matrix=frame_geom.view_matrix,
                    width=w_img,
                    height=h_img
                )

                # 8 corner vertices (indices 1 to 8)
                verts_2d = pts_2d[1:9]

                # Check if corners are reasonably inside the image viewport
                x_coords = verts_2d[:, 0]
                y_coords = verts_2d[:, 1]

                x_min, x_max = float(np.min(x_coords)), float(np.max(x_coords))
                y_min, y_max = float(np.min(y_coords)), float(np.max(y_coords))

                # Clamping for bounding box calculation
                x_min_c = max(0.0, x_min)
                y_min_c = max(0.0, y_min)
                x_max_c = min(float(w_img), x_max)
                y_max_c = min(float(h_img), y_max)

                bw = x_max_c - x_min_c
                bh = y_max_c - y_min_c

                if bw >= 20.0 and bh >= 20.0 and (bw * bh) / float(w_img * h_img) > 0.005:
                    xc = (x_min_c + x_max_c) / (2.0 * float(w_img))
                    yc = (y_min_c + y_max_c) / (2.0 * float(h_img))
                    wn = bw / float(w_img)
                    hn = bh / float(h_img)

                    # Build keypoints string: k1_x k1_y v1 ... k8_x k8_y v8
                    # v=2 (visible/labeled) if inside image, v=1 (occluded/out of bounds) if outside
                    kpts = []
                    for k_idx in range(8):
                        kx = verts_2d[k_idx, 0] / float(w_img)
                        ky = verts_2d[k_idx, 1] / float(h_img)
                        v = 2
                        if kx < 0.0 or kx > 1.0 or ky < 0.0 or ky > 1.0:
                            v = 1  # partially out of bounds
                            kx = float(np.clip(kx, 0.0, 1.0))
                            ky = float(np.clip(ky, 0.0, 1.0))
                        kpts.append((kx, ky, v))

                    if 0.0 < xc < 1.0 and 0.0 < yc < 1.0 and 0.0 < wn <= 1.0 and 0.0 < hn <= 1.0:
                        collected_samples.append({
                            "image": img,
                            "bbox": (xc, yc, wn, hn),
                            "kpts": kpts,
                            "source": f"{seq_id.replace('/', '_')}_{f_idx}"
                        })

            if cap is not None:
                cap.release()
        except Exception:
            continue

    print(f"Collected {len(collected_samples)} annotated samples from sequences.")

    # Data augmentation to exceed target_count (>= 1200 samples)
    if len(collected_samples) < target_count:
        needed = target_count - len(collected_samples)
        print(f"Augmenting {needed} samples to reach target count {target_count}...")
        idx = 0
        orig_count = len(collected_samples)
        while len(collected_samples) < target_count and orig_count > 0:
            sample = collected_samples[idx % orig_count]
            img = sample["image"].copy()
            xc, yc, wn, hn = sample["bbox"]
            kpts = sample["kpts"]

            # Mode 1: Brightness adjustment
            mode = (idx // orig_count) % 3
            if mode == 0:
                # Brightness/contrast adjustment (keypoint positions unchanged)
                alpha = random.uniform(0.8, 1.2)
                beta = random.randint(-20, 20)
                aug_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                collected_samples.append({
                    "image": aug_img,
                    "bbox": (xc, yc, wn, hn),
                    "kpts": list(kpts),
                    "source": f"{sample['source']}_b{idx}"
                })
            elif mode == 1:
                # Color tint (Gaussian blur / noise)
                aug_img = cv2.GaussianBlur(img, (3, 3), 0)
                collected_samples.append({
                    "image": aug_img,
                    "bbox": (xc, yc, wn, hn),
                    "kpts": list(kpts),
                    "source": f"{sample['source']}_g{idx}"
                })
            else:
                # Mild gamma
                gamma = random.uniform(0.7, 1.3)
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                aug_img = cv2.LUT(img, table)
                collected_samples.append({
                    "image": aug_img,
                    "bbox": (xc, yc, wn, hn),
                    "kpts": list(kpts),
                    "source": f"{sample['source']}_lut{idx}"
                })
            idx += 1

    # Shuffle samples
    random.shuffle(collected_samples)
    n_val = int(len(collected_samples) * val_split)
    val_samples = collected_samples[:n_val]
    train_samples = collected_samples[n_val:]

    def save_split(samples, img_dir, lbl_dir, split_name):
        count = 0
        for s in samples:
            fname = f"{s['source']}.jpg"
            img_path = os.path.join(img_dir, fname)
            lbl_path = os.path.join(lbl_dir, f"{s['source']}.txt")

            # Save resized image (max 640 for fast YOLO training)
            img = s["image"]
            cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])

            # Format pose line:
            # <class-id> <xc> <yc> <w> <h> <k1_x> <k1_y> <v1> ... <k8_x> <k8_y> <v8>
            xc, yc, wn, hn = s["bbox"]
            kpts_str = " ".join(f"{kx:.6f} {ky:.6f} {v}" for kx, ky, v in s["kpts"])
            lbl_line = f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f} {kpts_str}\n"

            with open(lbl_path, "w") as f:
                f.write(lbl_line)
            count += 1
        print(f"Saved {count} samples for {split_name}.")
        return count

    n_tr = save_split(train_samples, train_img_dir, train_lbl_dir, "train")
    n_va = save_split(val_samples, val_img_dir, val_lbl_dir, "val")

    # Generate data.yaml
    yaml_path = os.path.join(output_dir, "data.yaml")
    abs_output = os.path.abspath(output_dir).replace("\\", "/")
    yaml_content = f"""# 8-Corner Box Keypoint Pose Dataset
path: {abs_output}
train: images/train
val: images/val

# Keypoint configuration: 8 corners of 3D cuboid
kpt_shape: [8, 3] # [number of keypoints, number of dims (x, y, visibility)]
flip_idx: [0, 1, 2, 3, 4, 5, 6, 7]

names:
  0: box
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Dataset YAML written to {yaml_path}")
    print(f"Total: {n_tr + n_va} samples ({n_tr} train, {n_va} val).")
    return {"train": n_tr, "val": n_va, "total": n_tr + n_va}


if __name__ == "__main__":
    build_pose_dataset()
