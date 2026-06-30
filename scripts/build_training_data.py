"""
CareWatch AI — Build a labeled training dataset from Le2i + MultiCam videos.

For every frame of every tracked person, extracts the same features the
hand-coded FSM uses (angle, hip_y, aspect_ratio) PLUS short-window velocity
features (how fast each is changing) — the thing the hand-tuned rules
couldn't reliably use. Labels each frame 1 (fall) if it falls inside that
video's ground-truth fall window, else 0.

Output: data/training/frame_features.csv
    One row per (video, track_id, frame). Columns:
        video, dataset, frame, track_id,
        angle, hip_y, aspect_ratio,
        angle_vel, hip_vel, ar_vel,
        label

Usage:
    python scripts/build_training_data.py
"""
import sys
from pathlib import Path
from collections import deque, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import csv
import cv2
from ultralytics import YOLO

from config import POSE_MODEL, CONF_THRESHOLD, IOU_THRESHOLD, TRACKER_CONFIG
from pose_utils import body_axis_angle, hip_y_fraction, bbox_aspect_ratio
from evaluate import parse_annotation
from evaluate_multicam import load_ground_truth as load_multicam_gt, write_temp_annotation

REPO_ROOT    = Path(__file__).parent.parent
LE2I_DIR     = REPO_ROOT / "data" / "le2i"
MULTICAM_DIR = REPO_ROOT / "data" / "multicam"
MULTICAM_CSV = MULTICAM_DIR / "data_tuple3.csv"
OUT_DIR      = REPO_ROOT / "data" / "training"
OUT_CSV      = OUT_DIR / "frame_features.csv"

VELOCITY_WINDOW = 5   # frames back to compute rate-of-change


def discover_le2i() -> list[dict]:
    items = []
    video_dir = LE2I_DIR / "Videos"
    ann_dir   = LE2I_DIR / "Annotation_files"
    if not video_dir.exists():
        return items
    for video_file in sorted(video_dir.glob("*.avi")):
        ann_file = ann_dir / f"{video_file.stem}.txt"
        if ann_file.exists():
            items.append({"video": video_file, "ann": ann_file, "dataset": "Le2i"})
    return items


def discover_multicam(cam: int = 2) -> list[dict]:
    items = []
    if not MULTICAM_CSV.exists():
        return items
    gt_map = load_multicam_gt(MULTICAM_CSV)
    tmp_ann_dir = MULTICAM_DIR / "_tmp_annotations"
    tmp_ann_dir.mkdir(exist_ok=True)
    for video_file in sorted(MULTICAM_DIR.glob(f"chute*_cam{cam}.avi")):
        chute = int(video_file.stem.split("_")[0].replace("chute", ""))
        segments = gt_map.get((chute, cam), [])
        ann_path = tmp_ann_dir / f"{video_file.stem}.txt"
        if segments:
            start = min(s for s, _ in segments)
            end   = max(e for _, e in segments)
            write_temp_annotation(start, end, ann_path)
        else:
            ann_path.write_text("")
        items.append({"video": video_file, "ann": ann_path, "dataset": "MultiCam"})
    return items


def extract_video_features(video_path: Path, ann_path: Path, dataset: str, model: YOLO) -> list[dict]:
    gt = parse_annotation(ann_path)
    fall_start, fall_end = gt if gt else (None, None)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    angle_hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=VELOCITY_WINDOW))
    hip_hist:   dict[int, deque] = defaultdict(lambda: deque(maxlen=VELOCITY_WINDOW))
    ar_hist:    dict[int, deque] = defaultdict(lambda: deque(maxlen=VELOCITY_WINDOW))

    rows = []
    frame_n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_n += 1

        results = model.track(
            frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
            tracker=TRACKER_CONFIG, persist=True, verbose=False,
        )
        if results[0].boxes.id is None:
            continue

        ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        kps_list = results[0].keypoints.data.cpu().numpy() if results[0].keypoints else []

        for i, tid in enumerate(ids):
            kps = kps_list[i] if i < len(kps_list) else None
            angle = body_axis_angle(kps)
            hip_y = hip_y_fraction(kps, h)
            ar    = bbox_aspect_ratio(tuple(boxes[i])) if i < len(boxes) else None

            if angle is None or hip_y is None or ar is None:
                continue

            angle_hist[tid].append(angle)
            hip_hist[tid].append(hip_y)
            ar_hist[tid].append(ar)

            angle_vel = angle_hist[tid][-1] - angle_hist[tid][0] if len(angle_hist[tid]) >= 2 else 0.0
            hip_vel   = hip_hist[tid][-1]   - hip_hist[tid][0]   if len(hip_hist[tid])   >= 2 else 0.0
            ar_vel    = ar_hist[tid][-1]    - ar_hist[tid][0]    if len(ar_hist[tid])    >= 2 else 0.0

            label = 1 if (fall_start is not None and fall_start <= frame_n <= fall_end) else 0

            rows.append({
                "video": video_path.name, "dataset": dataset, "frame": frame_n, "track_id": tid,
                "angle": round(angle, 2), "hip_y": round(hip_y, 3), "aspect_ratio": round(ar, 3),
                "angle_vel": round(angle_vel, 2), "hip_vel": round(hip_vel, 3), "ar_vel": round(ar_vel, 3),
                "label": label,
            })

    cap.release()
    return rows


def main() -> None:
    model = YOLO(POSE_MODEL)
    items = discover_le2i() + discover_multicam()
    if not items:
        print("[ERROR] No videos found in data/le2i or data/multicam.")
        sys.exit(1)

    print(f"[INFO] Extracting features from {len(items)} videos...\n")
    all_rows = []
    for idx, item in enumerate(items, 1):
        rows = extract_video_features(item["video"], item["ann"], item["dataset"], model)
        all_rows.extend(rows)
        n_fall_frames = sum(1 for r in rows if r["label"] == 1)
        print(f"  [{idx}/{len(items)}] {item['video'].name:25} "
              f"{len(rows)} frames, {n_fall_frames} labeled fall")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    total_fall = sum(1 for r in all_rows if r["label"] == 1)
    print(f"\n[INFO] Saved {len(all_rows)} labeled frames ({total_fall} fall, "
          f"{len(all_rows) - total_fall} non-fall) -> {OUT_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
