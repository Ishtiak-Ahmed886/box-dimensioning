"""Train custom YOLO box detector on the 1200+ samples dataset."""

import os
import shutil
from ultralytics import YOLO


def train_box_detector(
    data_yaml: str = "data/dataset_box/data.yaml",
    model_name: str = "yolov8n.pt",
    epochs: int = 5,
    imgsz: int = 416,
    batch: int = 16,
    output_dir: str = "models"
) -> str:
    """Trains custom YOLOv8 model for box detection and exports best weights."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading base architecture: {model_name}")
    model = YOLO(model_name)

    print(f"Starting training on {data_yaml} ({epochs} epochs, imgsz={imgsz})...")
    results = model.train(
        data=os.path.abspath(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=2,
        project="runs/detect",
        name="box_train",
        exist_ok=True,
        verbose=True
    )

    # Copy best model to models/
    best_pt = os.path.join("runs/detect/box_train/weights/best.pt")
    target_pt = os.path.join(output_dir, "custom_box_yolo.pt")

    if not os.path.exists(best_pt):
        best_pt = os.path.join("runs/detect/box_train/weights/last.pt")

    if os.path.exists(best_pt):
        shutil.copyfile(best_pt, target_pt)
        print(f"Best model saved to {target_pt}")
    else:
        print("Warning: Trained weights not found at default path, saving active model...")
        model.save(target_pt)

    # Run quick validation
    val_results = model.val()
    print(f"Validation mAP50: {val_results.box.map50:.4f}, mAP50-95: {val_results.box.map:.4f}")

    return target_pt


if __name__ == "__main__":
    train_box_detector()
