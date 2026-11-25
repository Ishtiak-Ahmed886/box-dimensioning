"""Train Custom YOLO 8-Corner Keypoint (Pose) Model for Box Dimensioning.
Detects the exact 8 2D corners of the 3D parcel/box directly from image/video.
Optimized for multi-core CPU execution.
"""

import os
import shutil
import torch
from ultralytics import YOLO

# Allocate multi-threading for CPU acceleration
torch.set_num_threads(8)

# Keep Ultralytics configuration inside workspace to prevent sandbox permission issues
os.environ["YOLO_CONFIG_DIR"] = os.path.abspath(".ultralytics")


def train_box_pose(
    data_yaml: str = "data/dataset_box_pose/data.yaml",
    model_name: str = "yolov8n-pose.pt",
    epochs: int = 5,
    imgsz: int = 256,
    batch: int = 32,
    output_dir: str = "models"
) -> str:
    """Trains custom YOLOv8-pose model for 8 box corners and exports weights."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading base pose architecture: {model_name}")
    model = YOLO(model_name)

    print(f"Starting optimized training on {data_yaml} ({epochs} epochs, imgsz={imgsz}, batch={batch})...")
    results = model.train(
        data=os.path.abspath(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=0,  # 0 avoids Windows multiprocessing overhead/pickling issues
        project="runs/pose",
        name="box_corner_train",
        exist_ok=True,
        verbose=True
    )

    candidate_weights = [
        os.path.join("runs", "pose", "runs", "pose", "box_corner_train", "weights", "best.pt"),
        os.path.join("runs", "pose", "box_corner_train", "weights", "best.pt"),
        os.path.join("runs", "pose", "runs", "pose", "box_corner_train", "weights", "last.pt"),
        os.path.join("runs", "pose", "box_corner_train", "weights", "last.pt"),
    ]
    target_pt = os.path.join(output_dir, "custom_box_corner_pose.pt")

    saved = False
    for pt in candidate_weights:
        if os.path.exists(pt):
            shutil.copyfile(pt, target_pt)
            print(f"Trained corner model copied from {pt} to {target_pt}")
            saved = True
            break

    if not saved:
        print("Saving active model instance...")
        model.save(target_pt)

    val_res = model.val()
    if hasattr(val_res, "pose"):
        print(f"Validation Pose mAP50: {val_res.pose.map50:.4f}, Pose mAP50-95: {val_res.pose.map:.4f}")
    if hasattr(val_res, "box"):
        print(f"Validation Box mAP50: {val_res.box.map50:.4f}")

    return target_pt


if __name__ == "__main__":
    train_box_pose()
