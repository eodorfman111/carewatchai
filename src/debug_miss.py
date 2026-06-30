"""
Debug tool — trace exact angle/hip_y values during a known fall window
to understand why the FSM missed it.

Usage:
    python src/debug_miss.py --video "data/le2i/Videos/video (12).avi" --ann "data/le2i/Annotation_files/video (12).txt"
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from ultralytics import YOLO

from config import POSE_MODEL, CONF_THRESHOLD, IOU_THRESHOLD, TRACKER_CONFIG
from pose_utils import body_axis_angle, hip_y_fraction, bbox_aspect_ratio
from fall_fsm import FallFSM
from evaluate import parse_annotation


def debug_video(video_path: Path, ann_path: Path) -> None:
    gt = parse_annotation(ann_path)
    if gt is None:
        print("No fall in this video.")
        return
    fall_start, fall_end = gt
    print(f"Ground truth: fall from frame {fall_start} to {fall_end}\n")

    model = YOLO(POSE_MODEL)
    cap = cv2.VideoCapture(str(video_path))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fall_fsms: dict[int, FallFSM] = {}
    frame_n = 0

    # Print window: 20 frames before fall start to 60 frames after
    print_start = max(1, fall_start - 20)
    print_end = fall_end + 80

    print(f"{'Frame':<8}{'TrackID':<10}{'Angle':<10}{'Hip_Y':<10}{'AspectR':<10}{'FSM State':<12}{'Down?'}")
    print("-" * 80)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_n += 1

        if frame_n > print_end:
            break

        results = model.track(
            frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
            tracker=TRACKER_CONFIG, persist=True, verbose=False,
        )

        if results[0].boxes.id is None:
            if print_start <= frame_n <= print_end:
                print(f"{frame_n:<8}{'—':<10}{'NO DETECTION':<40}")
            continue

        ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        kps_list = results[0].keypoints.data.cpu().numpy() if results[0].keypoints else []

        for i, tid in enumerate(ids):
            kps = kps_list[i] if i < len(kps_list) else None
            if tid not in fall_fsms:
                fall_fsms[tid] = FallFSM(tid)

            angle = body_axis_angle(kps)
            hip_y = hip_y_fraction(kps, h)
            ar    = bbox_aspect_ratio(tuple(boxes[i])) if i < len(boxes) else None
            state, fired = fall_fsms[tid].update(angle, hip_y, ar)

            if print_start <= frame_n <= print_end:
                angle_str = f"{angle:.1f}" if angle is not None else "None"
                hip_str   = f"{hip_y:.2f}" if hip_y is not None else "None"
                ar_str    = f"{ar:.2f}" if ar is not None else "None"
                is_down = "DOWN" if FallFSM._is_down_pose(angle, hip_y, ar) else ""
                fired_str = " 🔴 FIRED!" if fired else ""

                print(f"{frame_n:<8}{tid:<10}{angle_str:<10}{hip_str:<10}"
                      f"{ar_str:<10}{state.name:<12}{is_down}{fired_str}")

    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--ann",   required=True)
    args = parser.parse_args()
    debug_video(Path(args.video), Path(args.ann))
