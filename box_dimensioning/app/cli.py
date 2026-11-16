"""Command-Line Interface (CLI) for Box Dimensioning from Phone Video."""

import argparse
import json
import os
import sys

from box_dimensioning.calibration.camera_calibrator import CameraCalibrator, download_opencv_chessboards
from box_dimensioning.dataset.downloader import download_index_file, download_dataset_batch
from box_dimensioning.dataset.objectron_parser import load_sequence
from box_dimensioning.pipeline.dimensioner import BoxDimensioner
from box_dimensioning.evaluation.benchmark import run_benchmark


def main():
    parser = argparse.ArgumentParser(
        description="Box Dimensioning from Phone Video for Volumetric-Weight Billing"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: calibrate
    p_calib = subparsers.add_parser("calibrate", help="Calibrate camera using OpenCV chessboard sample set (F1, A1)")
    p_calib.add_argument("--dir", default="data/calibration", help="Directory containing left01-left14 chessboard images")
    p_calib.add_argument("--download", action="store_true", help="Download missing chessboard images from OpenCV repository")

    # Subcommand: measure
    p_meas = subparsers.add_parser("measure", help="Measure package dimensions from an Objectron video sequence (F3, F4, F6)")
    p_meas.add_argument("--seq", required=True, help="Path to sequence directory containing geometry.pbdata and annotation.pbdata")
    p_meas.add_argument("--method", choices=["A", "B"], default="A", help="Method A (multi-view) or Method B (single-view)")
    p_meas.add_argument("--output-img", default=None, help="Optional output path for reprojected 3D wireframe image")

    # Subcommand: benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run automated benchmark suite on test split (A1-A7)")
    p_bench.add_argument("--data-dir", default="data/objectron", help="Base directory of Objectron dataset")
    p_bench.add_argument("--out-dir", default="data/benchmark", help="Directory to save benchmark reports")
    p_bench.add_argument("--max-seq", type=int, default=50, help="Maximum number of sequences to evaluate")

    # Subcommand: download
    p_down = subparsers.add_parser("download", help="Download official Objectron sequences")
    p_down.add_argument("--category", choices=["cereal_box", "book"], default="cereal_box")
    p_down.add_argument("--split", choices=["train", "test"], default="test")
    p_down.add_argument("--count", type=int, default=5, help="Number of sequences to download")
    p_down.add_argument("--out-dir", default="data/objectron", help="Output directory")

    args = parser.parse_args()

    if args.command == "calibrate":
        if args.download or not os.path.exists(args.dir):
            print(f"Downloading OpenCV chessboard images to {args.dir}...")
            download_opencv_chessboards(args.dir)

        import glob
        images = sorted(glob.glob(os.path.join(args.dir, "left*.jpg")))
        calibrator = CameraCalibrator()
        res = calibrator.calibrate(images)
        print("Calibration Results:")
        print(f"  RMS Reprojection Error: {res['rms_reprojection_error_px']:.4f} px")
        print(f"  Acceptance Criterion A1 (<= 0.6 px): {'PASSED' if res['meets_acceptance_criterion_A1'] else 'FAILED'}")
        print(f"  Images used: {res['valid_images_count']}")

    elif args.command == "measure":
        if "data/objectron" in args.seq:
            seq_id = os.path.relpath(args.seq, "data/objectron").replace("\\", "/")
        else:
            seq_id = os.path.basename(os.path.normpath(args.seq))
        seq = load_sequence(args.seq, seq_id)
        dimensioner = BoxDimensioner()
        method_name = "METHOD_A_MULTIVIEW" if args.method == "A" else "METHOD_B_SINGLEVIEW"
        res = dimensioner.dimension_sequence(seq, method=method_name, output_image_path=args.output_img)
        print(json.dumps(res, indent=2))

    elif args.command == "benchmark":
        print("Running benchmark suite across test split...")
        report = run_benchmark(data_dir=args.data_dir, output_dir=args.out_dir, max_sequences=args.max_seq)
        print("\nBenchmark Summary:")
        print(json.dumps(report.get("benchmark_summary", {}), indent=2))
        print(f"\nReport written to: {os.path.join(args.out_dir, 'benchmark_report.md')}")

    elif args.command == "download":
        idx_key = f"{args.category}_{args.split}"
        print(f"Fetching official index for {idx_key}...")
        video_ids = download_index_file(idx_key)
        selected = video_ids[:args.count]
        print(f"Downloading metadata for {len(selected)} sequences into {args.out_dir}...")
        download_dataset_batch(selected, output_base_dir=args.out_dir)
        print("Download complete.")


if __name__ == "__main__":
    main()
