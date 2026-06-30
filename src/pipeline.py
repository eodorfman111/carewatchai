"""
CareWatch AI — main inference pipeline.

All 4 detection types:
  - Fall detection (FSM: UPRIGHT → SUSPECT → FALLEN)
  - Prolonged inactivity (zone-aware timer)
  - Nighttime wandering (zone crossing counter)
  - Restricted zone breach (polygon + time gate)

Usage:
    python src/pipeline.py --source data/sample_videos/test.mp4 --camera cam_01
    python src/pipeline.py --source rtsp://192.168.1.x/stream --camera room_214
    python src/pipeline.py --source 0 --camera cam_01
"""
import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    POSE_MODEL, CONF_THRESHOLD, IOU_THRESHOLD, TRACKER_CONFIG,
    RESTRICTED_HOURS, STORE_SKELETON_CLIP,
)
from pose_utils import body_axis_angle, hip_y_fraction, centroid_from_keypoints, bbox_aspect_ratio
from fall_fsm import FallFSM, FallState
from inactivity_timer import InactivityTimer
from wandering_detector import WanderingDetector
from zone_manager import ZoneManager
from incident_log import Incident, init_db
from gait_tracker import GaitTracker
from snapshot import save_snapshot, save_skeleton_only
from overlay import draw_frame, render_skeleton_only

OUT_DIR = Path(__file__).parent.parent / "data" / "sample_videos"


def _make_writer(path: Path, fps: float, w: int, h: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


def run(source: str | int, camera_id: str, display: bool = False) -> None:
    init_db()
    model = YOLO(POSE_MODEL)
    zones = ZoneManager(camera_id)

    fall_fsms:        dict[int, FallFSM]           = {}
    inactivity:       dict[int, InactivityTimer]   = {}
    wandering:        dict[int, WanderingDetector] = {}
    gait:             dict[int, GaitTracker]       = {}
    active_flags:     dict[int, dict]              = {}   # per-track alert flags
    active_alerts:    list[dict]                   = []   # banner alerts

    cap = cv2.VideoCapture(source if isinstance(source, int) else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stem = Path(str(source)).stem if isinstance(source, str) else "live"
    annotated_writer = _make_writer(OUT_DIR / f"{stem}_pipeline.mp4",    fps, w, h)
    skeleton_writer  = _make_writer(OUT_DIR / f"{stem}_skeleton.mp4",    fps, w, h)

    frame_n = 0
    print(f"Processing {'live stream' if total <= 0 else f'{total} frames'}...")

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
            show=False,
        )

        track_data: dict[int, dict] = {}

        if results[0].boxes.id is not None:
            ids       = results[0].boxes.id.int().cpu().tolist()
            boxes     = results[0].boxes.xyxy.cpu().numpy()
            kps_list  = (results[0].keypoints.data.cpu().numpy()
                         if results[0].keypoints else [])

            for i, tid in enumerate(ids):
                kps = kps_list[i] if i < len(kps_list) else None
                box = boxes[i]

                cx_norm = ((box[0] + box[2]) / 2) / w
                cy_norm = ((box[1] + box[3]) / 2) / h
                zone    = zones.zone_for_point(cx_norm, cy_norm)

                # ── initialise per-track objects ──────────────────────────
                if tid not in fall_fsms:
                    fall_fsms[tid]    = FallFSM(tid)
                    inactivity[tid]   = InactivityTimer(tid)
                    wandering[tid]    = WanderingDetector(tid)
                    gait[tid]         = GaitTracker(tid)
                    active_flags[tid] = {}

                flags = active_flags[tid]

                # ── Fall FSM ──────────────────────────────────────────────
                angle   = body_axis_angle(kps)
                hip_y   = hip_y_fraction(kps, h)
                aspect_ratio = bbox_aspect_ratio(tuple(box))
                fall_state, fall_fired = fall_fsms[tid].update(angle, hip_y, aspect_ratio)

                if fall_fired:
                    flags["fall"] = True
                    alert = {"type": "fall", "track_id": tid,
                             "camera_id": camera_id, "zone": zone}
                    active_alerts.append(alert)
                    clip = save_snapshot(frame, "fall", tid, camera_id)
                    save_skeleton_only(frame, "fall", tid, camera_id)
                    Incident(camera_id, tid, "fall", zone, clip_path=clip).save()
                    print(f"[FALL]        Track {tid} | {camera_id} | zone={zone}")

                # Clear fall flag when person stands up
                if fall_state == FallState.UPRIGHT:
                    flags.pop("fall", None)

                # ── Inactivity timer ──────────────────────────────────────
                centroid = centroid_from_keypoints(kps)
                inact_fired = inactivity[tid].update(centroid, zone)
                if inact_fired:
                    flags["inactivity"] = True
                    alert = {"type": "inactivity", "track_id": tid,
                             "camera_id": camera_id, "zone": zone}
                    active_alerts.append(alert)
                    clip = save_snapshot(frame, "inactivity", tid, camera_id)
                    Incident(camera_id, tid, "inactivity", zone, clip_path=clip).save()
                    print(f"[INACTIVITY]  Track {tid} | {camera_id} | zone={zone}")

                # Reset inactivity flag when moving again
                if centroid and inactivity[tid]._still_frames == 0:
                    flags.pop("inactivity", None)

                # ── Wandering ─────────────────────────────────────────────
                wander_fired = wandering[tid].update(zone)
                if wander_fired:
                    flags["wandering"] = True
                    alert = {"type": "wandering", "track_id": tid,
                             "camera_id": camera_id, "zone": zone}
                    active_alerts.append(alert)
                    clip = save_snapshot(frame, "wandering", tid, camera_id)
                    Incident(camera_id, tid, "wandering", zone, clip_path=clip).save()
                    print(f"[WANDERING]   Track {tid} | {camera_id} | zone={zone}")

                # ── Restricted zone ───────────────────────────────────────
                if zones.is_restricted(cx_norm, cy_norm):
                    hour = time.localtime().tm_hour
                    start_h, end_h = RESTRICTED_HOURS
                    in_hours = (hour >= start_h or hour < end_h
                                if start_h > end_h
                                else start_h <= hour < end_h)
                    if in_hours and not flags.get("restricted_zone"):
                        flags["restricted_zone"] = True
                        alert = {"type": "restricted_zone", "track_id": tid,
                                 "camera_id": camera_id, "zone": zone}
                        active_alerts.append(alert)
                        clip = save_snapshot(frame, "restricted_zone", tid, camera_id)
                        Incident(camera_id, tid, "restricted_zone", zone,
                                 clip_path=clip).save()
                        print(f"[ZONE]        Track {tid} | {camera_id} | zone={zone}")
                else:
                    flags.pop("restricted_zone", None)

                # ── Gait speed ────────────────────────────────────────────
                gait_speed = gait[tid].update(kps) if kps is not None else None
                if gait[tid].is_declining():
                    flags["gait_decline"] = True

                track_data[tid] = {
                    "fall_state":  fall_state,
                    "zone":        zone,
                    "flags":       dict(flags),
                    "gait_speed":  gait_speed,
                }

        # Keep alert banner to last 4, clear old ones every 300 frames
        if frame_n % 300 == 0:
            active_alerts = active_alerts[-4:]

        annotated = draw_frame(frame, results[0], track_data, active_alerts[-4:])
        skeleton  = render_skeleton_only(frame, results[0])

        annotated_writer.write(annotated)
        skeleton_writer.write(skeleton)

        if display:
            cv2.imshow("CareWatch AI", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_n % 30 == 0 and total > 0:
            print(f"  Frame {frame_n}/{total}")

    cap.release()
    annotated_writer.release()
    skeleton_writer.release()
    if display:
        cv2.destroyAllWindows()

    print(f"\nDone.")
    print(f"  Annotated : {OUT_DIR / f'{stem}_pipeline.mp4'}")
    print(f"  Skeleton  : {OUT_DIR / f'{stem}_skeleton.mp4'}")
    print(f"  Snapshots : incidents/snapshots/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",    default="0")
    parser.add_argument("--camera",    default="cam_01")
    parser.add_argument("--display",   action="store_true")
    parser.add_argument("--no-display",action="store_true")
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    run(source, args.camera, display=args.display)
