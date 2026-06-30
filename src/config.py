from dataclasses import dataclass, field
from pathlib import Path
import json

ROOT = Path(__file__).parent.parent

# ── Model ──────────────────────────────────────────────────────────────────
POSE_MODEL = "yolo11n-pose.pt"   # swap to yolo26n-pose.pt when weights land
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.7

# ── Tracking ───────────────────────────────────────────────────────────────
TRACKER_CONFIG = "bytetrack.yaml"   # ultralytics built-in (ocsort not in this version)
MAX_DISAPPEARED = 60             # frames before track is dropped

# ── Fall detection FSM ─────────────────────────────────────────────────────
# Body axis angle (degrees from vertical) above which person is considered non-upright
FALL_ANGLE_THRESHOLD = 45.0
# Hip Y must be within this fraction of frame height from floor
FLOOR_PROXIMITY_FRACTION = 0.60
# Frames the "down" state must persist before alert fires
FALL_CONFIRM_FRAMES = 15         # ~0.5 s at 30 fps — real fall transitions are short
# Frames of upright posture needed to reset fall state
FALL_RESET_FRAMES = 15

# ── Inactivity timer ───────────────────────────────────────────────────────
# Max centroid displacement (px) to count as "still"
INACTIVITY_MOTION_THRESHOLD_PX = 30
# Frames of stillness before alert per zone (key = zone name)
INACTIVITY_THRESHOLDS: dict[str, int] = {
    "floor":    300,    # 10 s — floor stillness high priority but not instant
    "chair":    9000,   # 5 min
    "bed":      18000,  # 10 min (sleeping is normal)
    "default":  9000,   # 5 min
}

# ── Restricted zones ───────────────────────────────────────────────────────
RESTRICTED_ZONE_FILE = ROOT / "data" / "zones" / "zones.json"
RESTRICTED_HOURS = (22, 6)   # 10pm – 6am (start_hour, end_hour)

# ── Alert ──────────────────────────────────────────────────────────────────
ALERT_ACKNOWLEDGE_TIMEOUT_S = 60   # escalate if not ack'd
INCIDENTS_DB = ROOT / "incidents" / "incidents.db"

# ── Privacy ────────────────────────────────────────────────────────────────
STORE_SKELETON_CLIP = True   # render skeleton overlay, NOT raw video
STORE_RAW_CLIP = False
MAX_CLIP_SECONDS = 10
