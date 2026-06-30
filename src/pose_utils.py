"""Keypoint indices and geometric helpers for YOLO-pose (COCO 17-point)."""
import numpy as np

# COCO keypoint indices
NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

MIN_CONF = 0.3   # keypoint confidence gate


def _kp(keypoints: np.ndarray, idx: int) -> tuple[float, float, float]:
    """Return (x, y, conf) for keypoint idx; zeros if shape mismatch."""
    if keypoints is None or len(keypoints) <= idx:
        return 0.0, 0.0, 0.0
    kp = keypoints[idx]
    return float(kp[0]), float(kp[1]), float(kp[2])


def midpoint(kp1: tuple, kp2: tuple) -> tuple[float, float]:
    return ((kp1[0] + kp2[0]) / 2, (kp1[1] + kp2[1]) / 2)


def body_axis_angle(keypoints: np.ndarray) -> float | None:
    """
    Angle (degrees) of the torso vector from vertical.
    0° = fully upright, 90° = lying horizontal.
    Returns None if keypoints are below confidence threshold.
    """
    ls = _kp(keypoints, L_SHOULDER)
    rs = _kp(keypoints, R_SHOULDER)
    lh = _kp(keypoints, L_HIP)
    rh = _kp(keypoints, R_HIP)

    shoulder_conf = min(ls[2], rs[2])
    hip_conf = min(lh[2], rh[2])
    if shoulder_conf < MIN_CONF or hip_conf < MIN_CONF:
        return None

    mid_shoulder = midpoint(ls, rs)
    mid_hip = midpoint(lh, rh)

    dx = mid_shoulder[0] - mid_hip[0]
    dy = mid_shoulder[1] - mid_hip[1]   # positive = shoulder below hip (inverted)

    # angle from vertical (y-axis pointing down in image coords)
    angle = abs(np.degrees(np.arctan2(abs(dx), abs(dy))))
    return float(angle)


def hip_y_fraction(keypoints: np.ndarray, frame_height: int) -> float | None:
    """
    Normalised Y position of mid-hip (0=top, 1=bottom of frame).
    Higher values mean hips are lower — closer to floor.
    """
    lh = _kp(keypoints, L_HIP)
    rh = _kp(keypoints, R_HIP)
    if min(lh[2], rh[2]) < MIN_CONF or frame_height == 0:
        return None
    mid_y = (lh[1] + rh[1]) / 2
    return float(mid_y / frame_height)


def bbox_aspect_ratio(box: tuple[float, float, float, float]) -> float:
    """
    width / height of bounding box.
    Standing person: tall & narrow → ratio < 1
    Fallen person: wide & short → ratio > 1
    Robust to camera-angle foreshortening that breaks 2D skeleton-angle math.
    """
    x1, y1, x2, y2 = box
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    return float(w / h)


def centroid_from_keypoints(keypoints: np.ndarray) -> tuple[float, float] | None:
    """Mean (x, y) of all visible keypoints."""
    if keypoints is None or len(keypoints) == 0:
        return None
    visible = keypoints[keypoints[:, 2] >= MIN_CONF]
    if len(visible) == 0:
        return None
    return float(visible[:, 0].mean()), float(visible[:, 1].mean())
