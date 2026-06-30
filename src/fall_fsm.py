"""
Per-track fall detection finite state machine.

States:
  UPRIGHT  → normal
  SUSPECT  → angle/proximity thresholds crossed, counting down
  FALLEN   → confirmed fall, alert fired
  STANDING → recovery transition (counting up before returning to UPRIGHT)

Detection uses two independent signals, OR'd together:
  1. Body axis angle + floor proximity (skeleton-based)
  2. Bounding box aspect ratio (width/height) — robust to camera-angle
     foreshortening that breaks signal 1 when a body lies roughly in
     line with the camera's depth axis.

Noise tolerance: a single non-down frame does not reset the SUSPECT
counter — pose estimation is noisy frame-to-frame and real fall
transitions are short (15-25 frames), so one bad read shouldn't
discard the whole detection window.
"""
from enum import Enum, auto
from dataclasses import dataclass, field

from config import (
    FALL_ANGLE_THRESHOLD,
    FLOOR_PROXIMITY_FRACTION,
    FALL_CONFIRM_FRAMES,
    FALL_RESET_FRAMES,
)

ASPECT_RATIO_THRESHOLD = 1.3   # width/height above this = lying down
SUSPECT_GRACE_FRAMES   = 6     # consecutive non-down frames tolerated before reset
# NOTE: a standalone hip_y-only signal was tried (catches falls toward/away
# from camera where angle+bbox foreshorten) but caused too many false
# positives from people simply standing close to camera. Would need a
# per-track baseline delta rather than an absolute threshold to be safe —
# left as a future improvement, not implemented.


class FallState(Enum):
    UPRIGHT  = auto()
    SUSPECT  = auto()
    FALLEN   = auto()
    STANDING = auto()   # recovering


@dataclass
class FallFSM:
    track_id: int
    state: FallState = FallState.UPRIGHT
    _suspect_frames: int        = field(default=0, repr=False)
    _suspect_miss_streak: int   = field(default=0, repr=False)
    _standing_frames: int       = field(default=0, repr=False)
    alert_fired: bool = False

    def update(
        self,
        angle: float | None,
        hip_y_frac: float | None,
        aspect_ratio: float | None = None,
    ) -> tuple[FallState, bool]:
        """
        Feed latest pose features.
        Returns (new_state, alert_just_fired).
        """
        alert_just_fired = False
        is_down = self._is_down_pose(angle, hip_y_frac, aspect_ratio)

        match self.state:
            case FallState.UPRIGHT | FallState.STANDING:
                if is_down:
                    self.state = FallState.SUSPECT
                    self._suspect_frames = 1
                    self._suspect_miss_streak = 0
                    self._standing_frames = 0
                else:
                    self.state = FallState.UPRIGHT
                    self._standing_frames = 0

            case FallState.SUSPECT:
                if is_down:
                    self._suspect_frames += 1
                    self._suspect_miss_streak = 0
                    if self._suspect_frames >= FALL_CONFIRM_FRAMES and not self.alert_fired:
                        self.state = FallState.FALLEN
                        self.alert_fired = True
                        alert_just_fired = True
                else:
                    self._suspect_miss_streak += 1
                    if self._suspect_miss_streak >= SUSPECT_GRACE_FRAMES:
                        # Genuinely got back up — reset
                        self.state = FallState.UPRIGHT
                        self._suspect_frames = 0
                        self._suspect_miss_streak = 0
                    # else: tolerate the noisy/occluded frame, keep counting

            case FallState.FALLEN:
                if not is_down:
                    self.state = FallState.STANDING
                    self._standing_frames = 1
                    self.alert_fired = False   # reset so a second fall can alert

            case FallState.STANDING:
                if not is_down:
                    self._standing_frames += 1
                    if self._standing_frames >= FALL_RESET_FRAMES:
                        self.state = FallState.UPRIGHT
                        self._standing_frames = 0
                else:
                    self.state = FallState.FALLEN
                    self._standing_frames = 0

        return self.state, alert_just_fired

    @staticmethod
    def _is_down_pose(
        angle: float | None,
        hip_y_frac: float | None,
        aspect_ratio: float | None,
    ) -> bool:
        angle_signal = (
            angle is not None and hip_y_frac is not None
            and angle > FALL_ANGLE_THRESHOLD
            and hip_y_frac > FLOOR_PROXIMITY_FRACTION
        )
        bbox_signal = (
            aspect_ratio is not None
            and aspect_ratio > ASPECT_RATIO_THRESHOLD
        )
        return angle_signal or bbox_signal
