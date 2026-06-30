"""
ML-based fall detection — same state machine structure as fall_fsm.FallFSM,
but the per-frame "is this person down?" decision comes from a trained
classifier instead of hand-coded angle/aspect-ratio thresholds.

Why keep the same FSM shell: a single-frame model prediction is still noisy
frame-to-frame (just like the raw geometric signals were). Temporal
confirmation (require N consecutive "down" predictions before alerting)
is still necessary regardless of what produces the per-frame signal.
"""
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import warnings
import joblib
import pandas as pd

from config import ML_PREDICT_THRESHOLD
from fall_fsm import FallState, FALL_CONFIRM_FRAMES, FALL_RESET_FRAMES, SUSPECT_GRACE_FRAMES

MODEL_PATH = Path(__file__).parent.parent / "models" / "fall_classifier.joblib"
VELOCITY_WINDOW = 5

_model_cache = None


def _load_model():
    global _model_cache
    if _model_cache is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{MODEL_PATH} not found. Run scripts/build_training_data.py "
                f"then scripts/train_fall_classifier.py first."
            )
        _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


@dataclass
class MLFallFSM:
    track_id: int
    state: FallState = FallState.UPRIGHT
    _suspect_frames: int        = field(default=0, repr=False)
    _suspect_miss_streak: int   = field(default=0, repr=False)
    _standing_frames: int       = field(default=0, repr=False)
    _angle_hist: deque = field(default_factory=lambda: deque(maxlen=VELOCITY_WINDOW), repr=False)
    _hip_hist:   deque = field(default_factory=lambda: deque(maxlen=VELOCITY_WINDOW), repr=False)
    _ar_hist:    deque = field(default_factory=lambda: deque(maxlen=VELOCITY_WINDOW), repr=False)
    alert_fired: bool = False

    def update(
        self,
        angle: float | None,
        hip_y_frac: float | None,
        aspect_ratio: float | None = None,
    ) -> tuple[FallState, bool]:
        alert_just_fired = False
        is_down = self._predict_down(angle, hip_y_frac, aspect_ratio)

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
                        self.state = FallState.UPRIGHT
                        self._suspect_frames = 0
                        self._suspect_miss_streak = 0

            case FallState.FALLEN:
                if not is_down:
                    self.state = FallState.STANDING
                    self._standing_frames = 1
                    self.alert_fired = False

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

    def _predict_down(
        self,
        angle: float | None,
        hip_y_frac: float | None,
        aspect_ratio: float | None,
    ) -> bool:
        if angle is None or hip_y_frac is None or aspect_ratio is None:
            return False

        self._angle_hist.append(angle)
        self._hip_hist.append(hip_y_frac)
        self._ar_hist.append(aspect_ratio)

        angle_vel = self._angle_hist[-1] - self._angle_hist[0] if len(self._angle_hist) >= 2 else 0.0
        hip_vel   = self._hip_hist[-1]   - self._hip_hist[0]   if len(self._hip_hist)   >= 2 else 0.0
        ar_vel    = self._ar_hist[-1]    - self._ar_hist[0]    if len(self._ar_hist)    >= 2 else 0.0

        bundle = _load_model()
        clf, feature_cols = bundle["model"], bundle["feature_cols"]

        features = {
            "angle": angle, "hip_y": hip_y_frac, "aspect_ratio": aspect_ratio,
            "angle_vel": angle_vel, "hip_vel": hip_vel, "ar_vel": ar_vel,
        }
        X = pd.DataFrame([[features[c] for c in feature_cols]], columns=feature_cols)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob = clf.predict_proba(X)[0][1]  # P(fall)
        return prob >= ML_PREDICT_THRESHOLD
