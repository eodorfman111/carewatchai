"""
CareWatch AI — Le2i Dataset Evaluator

Runs the full pipeline against Le2i ground truth annotations and reports:
- True Positives (falls correctly detected)
- False Positives (alerts fired when no fall)
- False Negatives (falls missed)
- Precision, Recall, F1
- Average detection latency (how many frames after fall start did we alert?)

Usage:
    python src/evaluate.py --dataset "C:/path/to/Le2i/Coffee_room_01"
    python src/evaluate.py --dataset "C:/path/to/Le2i" --all
"""
import argparse
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from ultralytics import YOLO

from config import POSE_MODEL, CONF_THRESHOLD, IOU_THRESHOLD, TRACKER_CONFIG
from pose_utils import body_axis_angle, hip_y_fraction, centroid_from_keypoints, bbox_aspect_ratio
from fall_fsm import FallFSM, FallState
from inactivity_timer import InactivityTimer


def parse_annotation(ann_file: Path) -> tuple[int, int] | None:
    """
    Parse Le2i annotation file.
    Format: first line = fall start frame, second line = fall end frame.
    Returns (start_frame, end_frame) or None if no fall.
    """
    try:
        lines = ann_file.read_text().strip().splitlines()
        if len(lines) < 2:
            return None
        start = int(lines[0].split()[0])
        end   = int(lines[1].split()[0])
        return start, end
    except Exception:
        return None


def evaluate_video(
    video_path: Path,
    ann_path: Path,
    model: YOLO,
    camera_id: str = "eval",
    detector: str = "rule",
) -> dict:
    """
    Run pipeline on one video, compare to ground truth.
    detector: "rule" (hand-coded FallFSM) or "ml" (trained classifier, MLFallFSM).
    Returns result dict.
    """
    gt = parse_annotation(ann_path)
    has_fall = gt is not None
    fall_start, fall_end = gt if gt else (None, None)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": f"Cannot open {video_path}"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if detector == "ml":
        from ml_fall_fsm import MLFallFSM as FSMClass
    else:
        FSMClass = FallFSM

    fall_fsms:   dict[int, FallFSM]         = {}
    inactivity:  dict[int, InactivityTimer] = {}

    detected_frame = None   # frame when our alert fired
    false_positives = 0
    frame_n = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_n += 1

        results = model.track(
            frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            tracker=TRACKER_CONFIG,
            persist=True,
            verbose=False,
        )

        if results[0].boxes.id is None:
            continue

        ids      = results[0].boxes.id.int().cpu().tolist()
        boxes    = results[0].boxes.xyxy.cpu().numpy()
        kps_list = (results[0].keypoints.data.cpu().numpy()
                    if results[0].keypoints else [])

        for i, tid in enumerate(ids):
            kps = kps_list[i] if i < len(kps_list) else None
            if tid not in fall_fsms:
                fall_fsms[tid]  = FSMClass(tid)
                inactivity[tid] = InactivityTimer(tid)

            angle  = body_axis_angle(kps)
            hip_y  = hip_y_fraction(kps, h)
            ar     = bbox_aspect_ratio(tuple(boxes[i])) if i < len(boxes) else None
            _, fired = fall_fsms[tid].update(angle, hip_y, ar)

            if fired:
                if has_fall and fall_start <= frame_n <= fall_end + 150:
                    # Correct detection — within fall window + 5s grace
                    if detected_frame is None:
                        detected_frame = frame_n
                else:
                    # False positive — no fall or outside window
                    false_positives += 1

    cap.release()

    tp = 1 if (has_fall and detected_frame is not None) else 0
    fn = 1 if (has_fall and detected_frame is None)     else 0
    fp = false_positives

    latency = (detected_frame - fall_start) if (tp and fall_start) else None

    return {
        "video":          video_path.name,
        "has_fall":       has_fall,
        "fall_start":     fall_start,
        "detected_frame": detected_frame,
        "tp":             tp,
        "fn":             fn,
        "fp":             fp,
        "latency_frames": latency,
        "total_frames":   total_frames,
    }


def run_evaluation(dataset_root: Path, subset: str | None = None) -> None:
    model = YOLO(POSE_MODEL)

    # Find all video + annotation pairs
    pairs: list[tuple[Path, Path]] = []

    search_roots = []
    if subset:
        search_roots = [dataset_root / subset]
    else:
        # All subsets
        search_roots = [p for p in dataset_root.iterdir() if p.is_dir()]

    for root in search_roots:
        ann_dir   = root / "Annotation_files"
        video_dir = root

        if not ann_dir.exists():
            # Try one level up
            ann_dir = root.parent / "Annotation_files"

        for video_file in sorted(video_dir.glob("*.avi")) + sorted(video_dir.glob("*.mp4")):
            # Match annotation: "video (1).txt" for "video (1).avi"
            ann_name = video_file.stem + ".txt"
            ann_file = ann_dir / ann_name
            if ann_file.exists():
                pairs.append((video_file, ann_file))

    if not pairs:
        print(f"No video+annotation pairs found in {dataset_root}")
        print("Expected structure: subset_folder/video (N).avi + Annotation_files/video (N).txt")
        return

    print(f"\nEvaluating {len(pairs)} videos...\n")
    print(f"{'Video':<35} {'Has Fall':<10} {'Detected':<10} {'TP':<4} {'FN':<4} {'FP':<4} {'Latency'}")
    print("-" * 80)

    total_tp = total_fn = total_fp = 0
    latencies = []
    errors = []

    for video_path, ann_path in pairs:
        r = evaluate_video(video_path, ann_path, model)

        if "error" in r:
            errors.append(r["error"])
            continue

        total_tp += r["tp"]
        total_fn += r["fn"]
        total_fp += r["fp"]
        if r["latency_frames"] is not None:
            latencies.append(r["latency_frames"])

        det_str = f"frame {r['detected_frame']}" if r["detected_frame"] else "MISSED"
        lat_str = f"{r['latency_frames']}f" if r["latency_frames"] is not None else "—"

        status = ""
        if r["has_fall"] and r["tp"]: status = "✅"
        elif r["has_fall"] and r["fn"]: status = "❌ MISS"
        elif not r["has_fall"] and r["fp"] == 0: status = "✅"
        elif r["fp"] > 0: status = f"⚠️  {r['fp']} FP"

        print(f"{r['video']:<35} {str(r['has_fall']):<10} {det_str:<10} "
              f"{r['tp']:<4} {r['fn']:<4} {r['fp']:<4} {lat_str}  {status}")

    # ── Summary ──────────────────────────────────────────────────────────────
    precision = total_tp / max(total_tp + total_fp, 1)
    recall    = total_tp / max(total_tp + total_fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    avg_lat   = np.mean(latencies) if latencies else 0

    print("\n" + "=" * 80)
    print(f"RESULTS SUMMARY")
    print(f"  Videos evaluated : {len(pairs)}")
    print(f"  True Positives   : {total_tp}")
    print(f"  False Negatives  : {total_fn}  (missed falls)")
    print(f"  False Positives  : {total_fp}  (wrong alerts)")
    print(f"  Precision        : {precision:.1%}")
    print(f"  Recall           : {recall:.1%}")
    print(f"  F1 Score         : {f1:.1%}")
    print(f"  Avg Latency      : {avg_lat:.1f} frames ({avg_lat/30:.1f}s at 30fps)")
    print("=" * 80)

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="Path to Le2i dataset root or subset folder")
    parser.add_argument("--subset",  default=None,
                        help="Specific subset e.g. Coffee_room_01")
    args = parser.parse_args()

    run_evaluation(Path(args.dataset), args.subset)
