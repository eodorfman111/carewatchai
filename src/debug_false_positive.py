"""
Debug tool — trace exact angle/aspect-ratio/hip_y values around a false-positive
fall alert to see which signal fired and why.

Usage:
    python src/debug_false_positive.py --video "data/multicam/chute24_cam2.avi" --around-frame 1020
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
from ultralytics import YOLO

from config import POSE_MODEL, CONF_THRESHOLD, IOU_THRESHOLD, TRACKER_CONFIG
from pose_utils import body_axis_angle, hip_y_fraction, bbox_aspect_ratio
from fall_fsm import FallFSM, FALL_ANGLE_THRESHOLD as ANGLE_TH, ASPECT_RATIO_THRESHOLD as AR_TH
from fall_fsm import FLOOR_PROXIMITY_FRACTION as HIP_TH


def debug_video(video_path: Path, around_frame: int, window: int = 40) -> None:
    model = YOLO(POSE_MODEL)
    cap = cv2.VideoCapture(str(video_path))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fall_fsms: dict[int, FallFSM] = {}
    frame_n = 0
    print_start = max(1, around_frame - window)
    print_end = around_frame + window

    print(f"Tracing frames {print_start}-{print_end} (alert fired near {around_frame})")
    print(f"Thresholds: angle>{ANGLE_TH} hip_y>{HIP_TH} OR aspect_ratio>{AR_TH}\n")
    print(f"{'Frame':<8}{'TrackID':<10}{'Angle':<10}{'Hip_Y':<10}{'AspectR':<10}{'FSM State':<12}{'Trigger'}")
    print("-" * 90)

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

                angle_trig = (angle is not None and hip_y is not None
                              and angle > ANGLE_TH and hip_y > HIP_TH)
                ar_trig = (ar is not None and ar > AR_TH)
                trigger = ""
                if angle_trig and ar_trig: trigger = "BOTH"
                elif angle_trig: trigger = "ANGLE"
                elif ar_trig: trigger = "ASPECT_RATIO"

                fired_str = " <-- ALERT FIRED" if fired else ""

                print(f"{frame_n:<8}{tid:<10}{angle_str:<10}{hip_str:<10}"
                      f"{ar_str:<10}{state.name:<12}{trigger}{fired_str}")

    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--around-frame", type=int, required=True)
    parser.add_argument("--window", type=int, default=40)
    args = parser.parse_args()
    debug_video(Path(args.video), args.around_frame, args.window)
