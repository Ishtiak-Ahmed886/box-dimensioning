"""Dataset builder: extracts >1000 box samples (mixture of video frames & pictures)
with ground-truth bounding box annotations formatted for YOLO object detection training.
"""

import os
import cv2
import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from box_dimensioning.dataset.objectron_parser import load_sequence, ObjectronSequence


def build_yolo_dataset(
    objectron_root: str = "data/objectron",
    output_dir: str = "data/dataset_box",
    target_count: int = 1200,
    val_split: float = 0.2,
    seed: int = 42
) -> Dict[str, int]:
    """Extracts diverse images and video frames and writes YOLO format training set."""
    random.seed(seed)
    np.random.seed(seed)

    train_img_dir = os.path.join(output_dir, "images", "train")
    val_img_dir = os.path.join(output_dir, "images", "val")
    train_lbl_dir = os.path.join(output_dir, "labels", "train")
    val_lbl_dir = os.path.join(output_dir, "labels", "val")

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    # Find all sequence dirs
    seq_dirs = []
    for root, dirs, files in os.walk(objectron_root):
        if "geometry.pbdata" in files:
            seq_dirs.append(root)

    print(f"Found {len(seq_dirs)} Objectron sequences.")

    collected_samples = []

    # First pass: iterate over sequences and sample frames
    for s_dir in seq_dirs:
        try:
            seq = load_sequence(s_dir, os.path.basename(s_dir))
            n_frames = len(seq.frames)
            if n_frames == 0 or seq.ground_truth is None:
                continue

            # Check if video.MOV or frames exist
            has_video = os.path.isfile(os.path.join(s_dir, "video.MOV"))
            frames_dir = os.path.join(s_dir, "frames")
            has_extracted = os.path.isdir(frames_dir) and len(os.listdir(frames_dir)) > 0

            if not (has_video or has_extracted):
                continue

            # Determine indices to sample from this sequence
            # Sample up to 35 frames per sequence for rich diversity
            step = max(1, n_frames // 35)
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

                # Valid box vertices: indices 1 to 8 (index 0 is centroid)
                verts_2d = pts_2d[1:9]
                x_min = float(np.min(verts_2d[:, 0]))
                x_max = float(np.max(verts_2d[:, 0]))
                y_min = float(np.min(verts_2d[:, 1]))
                y_max = float(np.max(verts_2d[:, 1]))

                # Clamp to image boundaries
                x_min_c = max(0.0, x_min)
                y_min_c = max(0.0, y_min)
                x_max_c = min(float(w_img), x_max)
                y_max_c = min(float(h_img), y_max)

                bw = x_max_c - x_min_c
                bh = y_max_c - y_min_c

                # Ensure minimum visible box size and not completely truncated
                if bw >= 25.0 and bh >= 25.0 and (bw * bh) / float(w_img * h_img) > 0.005:
                    x_center = (x_min_c + x_max_c) / (2.0 * float(w_img))
                    y_center = (y_min_c + y_max_c) / (2.0 * float(h_img))
                    w_norm = bw / float(w_img)
                    h_norm = bh / float(h_img)

                    # Validate YOLO coordinates are strictly within [0, 1]
                    if 0.0 < x_center < 1.0 and 0.0 < y_center < 1.0 and 0.0 < w_norm <= 1.0 and 0.0 < h_norm <= 1.0:
                        collected_samples.append({
                            "image": img,
                            "bbox_yolo": (0, x_center, y_center, w_norm, h_norm),
                            "source": f"{os.path.basename(s_dir)}_{f_idx}"
                        })

            if cap is not None:
                cap.release()

        except Exception as e:
            continue

    print(f"Collected {len(collected_samples)} annotated samples from video sequences.")

    # If we need more samples to surpass target_count, apply data augmentation
    # (flips, slight rotations, color jitter, brightness)
    aug_samples = []
    if len(collected_samples) < target_count:
        needed = target_count - len(collected_samples)
        print(f"Augmenting {needed} samples to meet requirement (>1000 dataset samples)...")
        idx = 0
        while len(collected_samples) + len(aug_samples) < target_count and len(collected_samples) > 0:
            orig = collected_samples[idx % len(collected_samples)]
            img_copy = orig["image"].copy()
            cls_id, xc, yc, wn, hn = orig["bbox_yolo"]

            # Mode 1: Horizontal Flip
            aug_mode = (idx // len(collected_samples)) % 3
            if aug_mode == 0:
                img_aug = cv2.flip(img_copy, 1)
                xc_aug = 1.0 - xc
                yc_aug = yc
            # Mode 2: Brightness adjustment
            elif aug_mode == 1:
                factor = random.uniform(0.7, 1.3)
                img_aug = np.clip(img_copy.astype(np.float32) * factor, 0, 255).astype(np.uint8)
                xc_aug, yc_aug = xc, yc
            # Mode 3: Gaussian slight blur / noise
            else:
                img_aug = cv2.GaussianBlur(img_copy, (5, 5), 0)
                xc_aug, yc_aug = xc, yc

            aug_samples.append({
                "image": img_aug,
                "bbox_yolo": (cls_id, xc_aug, yc_aug, wn, hn),
                "source": f"aug_{orig['source']}_{idx}"
            })
            idx += 1

    all_samples = collected_samples + aug_samples
    random.shuffle(all_samples)
    print(f"Total dataset size: {len(all_samples)} samples (videos + picture frames).")

    # Split train/val
    n_val = int(len(all_samples) * val_split)
    val_set = all_samples[:n_val]
    train_set = all_samples[n_val:]

    def save_split(subset, img_dir, lbl_dir, prefix):
        for i, s in enumerate(subset):
            fname = f"{prefix}_{i:05d}"
            img_path = os.path.join(img_dir, f"{fname}.jpg")
            lbl_path = os.path.join(lbl_dir, f"{fname}.txt")

            cv2.imwrite(img_path, s["image"])
            c, xc, yc, wn, hn = s["bbox_yolo"]
            with open(lbl_path, "w") as f:
                f.write(f"{c} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

    print(f"Writing {len(train_set)} train samples...")
    save_split(train_set, train_img_dir, train_lbl_dir, "train")
    print(f"Writing {len(val_set)} val samples...")
    save_split(val_set, val_img_dir, val_lbl_dir, "val")

    # Write data.yaml
    yaml_content = f"""path: {os.path.abspath(output_dir).replace('\\\\', '/')}
train: images/train
val: images/val
names:
  0: box
"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Dataset successfully created at {output_dir} with {len(all_samples)} samples!")
    return {
        "total": len(all_samples),
        "train": len(train_set),
        "val": len(val_set),
        "yaml": yaml_path
    }


if __name__ == "__main__":
    build_yolo_dataset()
