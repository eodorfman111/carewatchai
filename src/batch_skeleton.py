"""
Batch-generate skeleton-only (stick figure) renders for every video in a folder.

Usage:
    python src/batch_skeleton.py --input data/le2i/Videos --output data/le2i/Skeletons
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
from ultralytics import YOLO

from config import POSE_MODEL, CONF_THRESHOLD, IOU_THRESHOLD, TRACKER_CONFIG
from overlay import render_skeleton_only


def batch_render(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(POSE_MODEL)

    videos = sorted(input_dir.glob("*.avi")) + sorted(input_dir.glob("*.mp4"))
    if not videos:
        print(f"No videos found in {input_dir}")
        return

    print(f"Found {len(videos)} videos. Rendering skeletons...\n")

    for idx, video_path in enumerate(videos, 1):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  [{idx}/{len(videos)}] SKIP (cannot open): {video_path.name}")
            continue

        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        out_path = output_dir / f"{video_path.stem}_skeleton.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

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
            skeleton_frame = render_skeleton_only(frame, results[0])
            writer.write(skeleton_frame)

        cap.release()
        writer.release()
        print(f"  [{idx}/{len(videos)}] {video_path.name} -> {out_path.name} ({frame_n} frames)")

    print(f"\nDone. {len(videos)} skeleton videos saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Folder containing source videos")
    parser.add_argument("--output", required=True, help="Folder to save skeleton renders")
    args = parser.parse_args()
    batch_render(Path(args.input), Path(args.output))
