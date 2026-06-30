"""
CareWatch AI — MultiCam Dataset Evaluator

Real nursing-home-reenacted fall scenarios (24 chutes, re-enactments of
actual falls observed in nursing home residents), as opposed to Le2i's
generic coffee-room/office/lecture-room staged falls.

Ground truth format (data_tuple3.csv):
    chute, cam, start_frame, end_frame, label
    label == 1.0 means that frame range is a fall.
    Each chute can have multiple labeled segments (falls interleaved
    with confounding ADL events) — we take the union of all label==1
    segments as a single fall window per (chute, cam) for evaluate_video's
    "is detection inside the fall window + grace" check. Chutes 23-24 have
    no fall segments (false-positive stress test).

Usage:
    python src/evaluate_multicam.py --dataset data/multicam --csv data/multicam/data_tuple3.csv
"""
import argparse
import csv as csv_module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from ultralytics import YOLO

from config import POSE_MODEL
from evaluate import evaluate_video


def load_ground_truth(csv_path: Path) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """
    Returns {(chute, cam): [(start, end), ...]} for label==1 segments only.
    """
    gt: dict[tuple[int, int], list[tuple[int, int]]] = {}
    with open(csv_path) as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            if float(row["label"]) != 1.0:
                continue
            chute = int(float(row["chute"]))
            cam   = int(float(row["cam"]))
            start = int(float(row["start"]))
            end   = int(float(row["end"]))
            gt.setdefault((chute, cam), []).append((start, end))
    return gt


def write_temp_annotation(start: int, end: int, tmp_path: Path) -> None:
    """evaluate_video() expects a Le2i-style 2-line annotation file. Fake one."""
    tmp_path.write_text(f"{start}\n{end}\n")


def run_evaluation(dataset_dir: Path, csv_path: Path, cam: int = 2) -> None:
    model = YOLO(POSE_MODEL)
    gt_map = load_ground_truth(csv_path)

    videos = sorted(dataset_dir.glob(f"chute*_cam{cam}.avi"))
    if not videos:
        print(f"No videos found matching chute*_cam{cam}.avi in {dataset_dir}")
        return

    print(f"\nEvaluating {len(videos)} MultiCam videos (cam{cam})...\n")
    print(f"{'Video':<22} {'Has Fall':<10} {'Detected':<10} {'TP':<4} {'FN':<4} {'FP':<4} {'Latency'}")
    print("-" * 80)

    total_tp = total_fn = total_fp = 0
    latencies = []
    tmp_ann_dir = dataset_dir / "_tmp_annotations"
    tmp_ann_dir.mkdir(exist_ok=True)

    for video_path in videos:
        # chute01_cam2.avi -> chute=1
        chute_str = video_path.stem.split("_")[0].replace("chute", "")
        chute = int(chute_str)

        segments = gt_map.get((chute, cam), [])
        # Use the earliest start / latest end as one fall window per video
        # (handles videos with a single dominant fall event)
        ann_path = tmp_ann_dir / f"{video_path.stem}.txt"
        if segments:
            start = min(s for s, _ in segments)
            end   = max(e for _, e in segments)
            write_temp_annotation(start, end, ann_path)
        else:
            # No-fall chute — write an empty/invalid file so parse_annotation returns None
            ann_path.write_text("")

        r = evaluate_video(video_path, ann_path, model, camera_id=f"chute{chute}")

        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue

        total_tp += r["tp"]
        total_fn += r["fn"]
        total_fp += r["fp"]
        if r["latency_frames"] is not None:
            latencies.append(r["latency_frames"])

        det_str = f"frame {r['detected_frame']}" if r["detected_frame"] else (
            "MISSED" if r["has_fall"] else "—")
        lat_str = f"{r['latency_frames']}f" if r["latency_frames"] is not None else "—"

        status = ""
        if r["has_fall"] and r["tp"]: status = "OK"
        elif r["has_fall"] and r["fn"]: status = "MISS"
        elif not r["has_fall"] and r["fp"] == 0: status = "OK"
        elif r["fp"] > 0: status = f"{r['fp']} FP"

        print(f"{r['video']:<22} {str(r['has_fall']):<10} {det_str:<10} "
              f"{r['tp']:<4} {r['fn']:<4} {r['fp']:<4} {lat_str}  {status}")

    # cleanup temp annotation files
    for f in tmp_ann_dir.glob("*.txt"):
        f.unlink()
    tmp_ann_dir.rmdir()

    precision = total_tp / max(total_tp + total_fp, 1)
    recall    = total_tp / max(total_tp + total_fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    avg_lat   = np.mean(latencies) if latencies else 0

    print("\n" + "=" * 80)
    print(f"RESULTS SUMMARY (MultiCam — real nursing-home-reenacted falls, cam{cam})")
    print(f"  Videos evaluated : {len(videos)}")
    print(f"  True Positives   : {total_tp}")
    print(f"  False Negatives  : {total_fn}  (missed falls)")
    print(f"  False Positives  : {total_fp}  (wrong alerts)")
    print(f"  Precision        : {precision:.1%}")
    print(f"  Recall           : {recall:.1%}")
    print(f"  F1 Score         : {f1:.1%}")
    print(f"  Avg Latency      : {avg_lat:.1f} frames")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Folder containing chuteNN_camX.avi files")
    parser.add_argument("--csv",     required=True, help="Path to data_tuple3.csv ground truth")
    parser.add_argument("--cam",     type=int, default=2, help="Camera index used (default 2)")
    args = parser.parse_args()

    run_evaluation(Path(args.dataset), Path(args.csv), args.cam)
