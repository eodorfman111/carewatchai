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
import json
import logging
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.nn.tasks import BaseModel

# Ultralytics always fuses Conv+BatchNorm layers on model load (a pure
# inference-speed optimization — mathematically identical output, just
# fewer ops) and doesn't expose a way to opt out via YOLO()/.track()
# kwargs. On some Linux CI runners this crashes with a glibc heap
# corruption ("corrupted double-linked list" / "double free or
# corruption (!prev)") inside fuse_conv_and_bn — reproduced repeatedly on
# GitHub Actions ubuntu-latest, never locally on macOS, root cause not
# fully isolated. Since fusing doesn't affect what the model predicts,
# and our fixed 15-video CI benchmark doesn't need the speed win, this
# no-ops it out entirely rather than continuing to chase the crash.
BaseModel.fuse = lambda self, verbose=True: self

from config import (
    POSE_MODEL, CONF_THRESHOLD, IOU_THRESHOLD, TRACKER_CONFIG,
    FALL_ANGLE_THRESHOLD, FLOOR_PROXIMITY_FRACTION, FALL_CONFIRM_FRAMES,
)
from pose_utils import body_axis_angle, hip_y_fraction, centroid_from_keypoints, bbox_aspect_ratio
from fall_fsm import FallFSM, FallState, ASPECT_RATIO_THRESHOLD
from inactivity_timer import InactivityTimer

logger = logging.getLogger(__name__)


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

    # Diagnostics captured only within the annotated fall window, so a miss
    # can be explained after the fact instead of re-running the video.
    fall_window_total_frames   = 0   # frames in [fall_start, fall_end] we processed
    fall_window_tracked_frames = 0   # ...of those, frames YOLO had >=1 track for
    fall_window_diag: list[dict] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_n += 1

        in_fall_window = has_fall and fall_start <= frame_n <= fall_end

        results = model.track(
            frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            tracker=TRACKER_CONFIG,
            persist=True,
            verbose=False,
        )

        if results[0].boxes.id is None:
            if in_fall_window:
                fall_window_total_frames += 1
                logger.debug(
                    "%s frame %d: YOLO lost tracking (no boxes.id) during fall window",
                    video_path.name, frame_n,
                )
            continue

        ids      = results[0].boxes.id.int().cpu().tolist()
        boxes    = results[0].boxes.xyxy.cpu().numpy()
        kps_list = (results[0].keypoints.data.cpu().numpy()
                    if results[0].keypoints else [])

        if in_fall_window:
            fall_window_total_frames += 1
            fall_window_tracked_frames += 1

        for i, tid in enumerate(ids):
            kps = kps_list[i] if i < len(kps_list) else None
            if tid not in fall_fsms:
                fall_fsms[tid]  = FSMClass(tid)
                inactivity[tid] = InactivityTimer(tid)

            angle  = body_axis_angle(kps)
            hip_y  = hip_y_fraction(kps, h)
            ar     = bbox_aspect_ratio(tuple(boxes[i])) if i < len(boxes) else None
            state, fired = fall_fsms[tid].update(angle, hip_y, ar)

            if in_fall_window:
                fall_window_diag.append({
                    "frame":     frame_n,
                    "track_id":  tid,
                    "angle":     angle,
                    "hip_y":     hip_y,
                    "ar":        ar,
                    "fsm_state": state.name,
                })
                logger.debug(
                    "%s frame %d track %s: angle=%s hip_y=%s ar=%s fsm_state=%s",
                    video_path.name, frame_n, tid,
                    f"{angle:.1f}" if angle is not None else None,
                    f"{hip_y:.2f}" if hip_y is not None else None,
                    f"{ar:.2f}" if ar is not None else None,
                    state.name,
                )

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

    miss_reason = None
    if fn:
        miss_reason = _diagnose_miss(
            video_path, fall_start, fall_end,
            fall_window_total_frames, fall_window_tracked_frames,
            fall_window_diag,
        )

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
        "miss_reason":    miss_reason,
    }


def _diagnose_miss(
    video_path: Path,
    fall_start: int,
    fall_end: int,
    window_total_frames: int,
    window_tracked_frames: int,
    diag: list[dict],
) -> dict:
    """
    Explain why a video with an annotated fall got no alert: whether YOLO
    lost tracking during the fall window, what the body angle looked like,
    and how far the fall FSM progressed. Logs the explanation and returns
    it as a structured dict so callers can persist it alongside the run's
    pass/fail results (see scripts/run_eval.py).
    """
    logger.warning(
        "MISS: %s — fall annotated at frames %d-%d, no alert fired",
        video_path.name, fall_start, fall_end,
    )

    reason: dict = {
        "fall_window": [fall_start, fall_end],
        "window_total_frames":   window_total_frames,
        "window_tracked_frames": window_tracked_frames,
    }

    if window_total_frames == 0:
        logger.warning("  No frames were processed in the annotated fall window "
                        "(video shorter than annotation, or read failed early)")
        reason["category"] = "no_frames_processed"
        reason["diagnosis"] = ("No frames were processed in the annotated fall window "
                                "(video shorter than annotation, or read failed early)")
        return reason

    lost_frames = window_total_frames - window_tracked_frames
    lost_pct = 100 * lost_frames / window_total_frames
    reason["tracking_lost_frames"] = lost_frames
    reason["tracking_lost_pct"]    = round(lost_pct, 1)
    logger.warning(
        "  YOLO tracking: %d/%d frames in fall window had a tracked person "
        "(%d frames / %.0f%% lost tracking entirely)",
        window_tracked_frames, window_total_frames, lost_frames, lost_pct,
    )

    if not diag:
        logger.warning("  No pose data available during fall window — "
                        "YOLO never produced a track, so angle/FSM cannot be evaluated")
        reason["category"] = "no_track_data"
        reason["diagnosis"] = ("YOLO never produced a track during the fall window — "
                                "angle/FSM state cannot be evaluated")
        return reason

    angle_at_start = next((d["angle"] for d in diag if d["frame"] == fall_start), None)
    reason["angle_at_fall_start"] = round(angle_at_start, 1) if angle_at_start is not None else None
    reason["angle_threshold"]     = FALL_ANGLE_THRESHOLD
    if angle_at_start is not None:
        logger.warning(
            "  Body angle at fall_start frame (%d): %.1f° (threshold=%.1f°)",
            fall_start, angle_at_start, FALL_ANGLE_THRESHOLD,
        )
    else:
        logger.warning(
            "  Body angle at fall_start frame (%d): unavailable "
            "(no track/keypoints on that exact frame)", fall_start,
        )

    angles = [d["angle"] for d in diag if d["angle"] is not None]
    reason["angle_min"] = round(min(angles), 1) if angles else None
    reason["angle_max"] = round(max(angles), 1) if angles else None
    if angles:
        logger.warning(
            "  Body angle range across fall window: min=%.1f° max=%.1f° (threshold=%.1f°)",
            min(angles), max(angles), FALL_ANGLE_THRESHOLD,
        )
    else:
        logger.warning("  Body angle: never computed during fall window "
                        "(keypoints missing/low confidence every frame)")

    hip_ys = [d["hip_y"] for d in diag if d["hip_y"] is not None]
    reason["hip_y_min"] = round(min(hip_ys), 2) if hip_ys else None
    reason["hip_y_max"] = round(max(hip_ys), 2) if hip_ys else None
    reason["hip_y_threshold"] = FLOOR_PROXIMITY_FRACTION

    ars = [d["ar"] for d in diag if d["ar"] is not None]
    reason["aspect_ratio_min"] = round(min(ars), 2) if ars else None
    reason["aspect_ratio_max"] = round(max(ars), 2) if ars else None
    reason["aspect_ratio_threshold"] = ASPECT_RATIO_THRESHOLD

    states_reached = sorted({d["fsm_state"] for d in diag},
                             key=lambda s: ["UPRIGHT", "SUSPECT", "FALLEN", "STANDING"].index(s))
    reason["fsm_states_reached"] = states_reached
    logger.warning("  Fall FSM states reached during window: %s", ", ".join(states_reached))

    # Longest run of consecutive frames the FSM held SUSPECT — how close it
    # got to FALL_CONFIRM_FRAMES before something (noise or a track swap)
    # knocked it back to UPRIGHT.
    longest_streak = current_streak = 0
    for d in diag:
        if d["fsm_state"] == "SUSPECT":
            current_streak += 1
            longest_streak = max(longest_streak, current_streak)
        else:
            current_streak = 0
    reason["longest_suspect_streak_frames"] = longest_streak
    reason["fall_confirm_frames_required"]  = FALL_CONFIRM_FRAMES

    track_ids = sorted({d["track_id"] for d in diag})
    reason["track_ids_in_window"] = track_ids

    if "FALLEN" not in states_reached and "SUSPECT" in states_reached and len(track_ids) > 1:
        reason["category"] = "track_id_churn"
        reason["diagnosis"] = (
            f"FSM entered SUSPECT (longest streak {longest_streak}/{FALL_CONFIRM_FRAMES} "
            f"frames) but YOLO/ByteTrack reassigned the person to a new track ID mid-fall "
            f"(track IDs seen: {track_ids}) — each new track starts a fresh FSM at UPRIGHT, "
            f"resetting confirmation progress"
        )
        logger.warning(
            "  Track ID changed mid-window (%s) — new tracks reset the FSM to UPRIGHT, "
            "discarding SUSPECT progress (longest streak was %d/%d frames)",
            track_ids, longest_streak, FALL_CONFIRM_FRAMES,
        )
    elif "FALLEN" not in states_reached and "SUSPECT" in states_reached:
        reason["category"] = "suspect_not_sustained"
        reason["diagnosis"] = (
            f"FSM entered SUSPECT but longest consecutive down-pose streak was only "
            f"{longest_streak}/{FALL_CONFIRM_FRAMES} frames — angle/hip_y likely oscillated "
            f"around the threshold instead of holding steady"
        )
        logger.warning(
            "  FSM entered SUSPECT but never confirmed FALLEN — longest streak was only "
            "%d/%d frames (angle/hip_y dipped back above threshold before confirmation)",
            longest_streak, FALL_CONFIRM_FRAMES,
        )
    elif "SUSPECT" not in states_reached:
        reason["category"] = "never_triggered"

        # Which of the two independent signals (angle+hip_y vs. bbox aspect
        # ratio) got closer to tripping — tells us which one is worth fixing.
        angle_frac = (reason["angle_max"] / FALL_ANGLE_THRESHOLD) if reason["angle_max"] is not None else 0
        ar_frac    = (reason["aspect_ratio_max"] / ASPECT_RATIO_THRESHOLD) if reason["aspect_ratio_max"] is not None else 0
        if angle_frac >= ar_frac:
            reason["closer_signal"] = "angle"
            closer_desc = (f"angle got closer ({reason['angle_max']}°/{FALL_ANGLE_THRESHOLD}° "
                            f"vs. bbox {reason['aspect_ratio_max']}/{ASPECT_RATIO_THRESHOLD})")
        else:
            reason["closer_signal"] = "aspect_ratio"
            closer_desc = (f"bbox aspect ratio got closer ({reason['aspect_ratio_max']}/{ASPECT_RATIO_THRESHOLD} "
                            f"vs. angle {reason['angle_max']}°/{FALL_ANGLE_THRESHOLD}°)")

        reason["diagnosis"] = (
            f"is_down_pose never triggered — angle stayed at/below {FALL_ANGLE_THRESHOLD}° "
            f"(max observed {reason['angle_max']}°, hip_y max {reason['hip_y_max']} vs. "
            f"threshold {FLOOR_PROXIMITY_FRACTION}) and bbox aspect ratio never crossed its "
            f"threshold either (max observed {reason['aspect_ratio_max']} vs. "
            f"{ASPECT_RATIO_THRESHOLD}); {closer_desc}. This fall is likely foreshortened "
            f"toward/away from the camera, which the current angle+bbox signals can't catch"
        )
        logger.warning(
            "  FSM never left UPRIGHT — is_down_pose never triggered "
            "(angle > %.1f° and hip_y > %.2f, or bbox aspect ratio > %.1f, never held). "
            "%s",
            FALL_ANGLE_THRESHOLD, FLOOR_PROXIMITY_FRACTION, ASPECT_RATIO_THRESHOLD, closer_desc,
        )
    else:
        reason["category"] = "unknown"
        reason["diagnosis"] = "FSM never reached SUSPECT or FALLEN; no further signal available"

    return reason


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
    all_results: list[dict] = []

    for video_path, ann_path in pairs:
        r = evaluate_video(video_path, ann_path, model)
        all_results.append(r)

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

    # Machine-readable per-video results (including miss diagnostics) for
    # scripts/run_eval.py to persist into data/eval_results/ — avoids
    # re-deriving this from the printed table above.
    print("RESULTS_JSON:" + json.dumps(all_results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="Path to Le2i dataset root or subset folder")
    parser.add_argument("--subset",  default=None,
                        help="Specific subset e.g. Coffee_room_01")
    parser.add_argument("--debug", action="store_true",
                        help="Log per-frame angle/hip_y/FSM state during fall windows")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    run_evaluation(Path(args.dataset), args.subset)
