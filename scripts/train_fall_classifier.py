"""
CareWatch AI — Train a fall classifier on extracted frame features.

Replaces the hand-coded angle/aspect-ratio threshold rules with a model
that learns the actual boundary from labeled real data (Le2i + MultiCam).

Splits by VIDEO, not by frame, to avoid data leakage — if frames from the
same video appeared in both train and test, the model would partly just
be memorizing that video rather than generalizing.

Usage:
    python scripts/train_fall_classifier.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import joblib

REPO_ROOT  = Path(__file__).parent.parent
FEATURES_CSV = REPO_ROOT / "data" / "training" / "frame_features.csv"
MODEL_DIR  = REPO_ROOT / "models"
MODEL_PATH = MODEL_DIR / "fall_classifier.joblib"

FEATURE_COLS = ["angle", "hip_y", "aspect_ratio", "angle_vel", "hip_vel", "ar_vel"]


def main() -> None:
    if not FEATURES_CSV.exists():
        print(f"[ERROR] {FEATURES_CSV} not found. Run scripts/build_training_data.py first.")
        sys.exit(1)

    df = pd.read_csv(FEATURES_CSV)
    print(f"[INFO] Loaded {len(df)} labeled frames from {df['video'].nunique()} videos")
    print(f"[INFO] Class balance: {df['label'].sum()} fall frames, "
          f"{(df['label']==0).sum()} non-fall frames "
          f"({df['label'].mean():.1%} positive)\n")

    # ── Video-level train/test split (prevents leakage) ─────────────────────
    videos = df["video"].unique()
    rng = np.random.RandomState(42)
    rng.shuffle(videos)
    split_idx = int(len(videos) * 0.8)
    train_videos = set(videos[:split_idx])
    test_videos  = set(videos[split_idx:])

    train_df = df[df["video"].isin(train_videos)]
    test_df  = df[df["video"].isin(test_videos)]
    print(f"[INFO] Train: {len(train_videos)} videos, {len(train_df)} frames")
    print(f"[INFO] Test:  {len(test_videos)} videos, {len(test_df)} frames\n")

    X_train, y_train = train_df[FEATURE_COLS], train_df["label"]
    X_test,  y_test  = test_df[FEATURE_COLS],  test_df["label"]

    # ── Train ─────────────────────────────────────────────────────────────
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",   # fall frames are the minority class
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ── Frame-level evaluation (informative, not the final word —
    #    video-level temporal confirmation happens downstream) ──────────────
    y_pred = clf.predict(X_test)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)

    print("=" * 60)
    print("FRAME-LEVEL HELD-OUT RESULTS (not the real test — see below)")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1:        {f1:.1%}")
    print(f"  Confusion matrix [[TN FP] [FN TP]]:\n{cm}")
    print("=" * 60)

    print("\nFeature importances (what the model actually learned to rely on):")
    for feat, imp in sorted(zip(FEATURE_COLS, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat:15} {imp:.3f}")

    # ── Save model ────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "feature_cols": FEATURE_COLS}, MODEL_PATH)
    print(f"\n[INFO] Model saved -> {MODEL_PATH.relative_to(REPO_ROOT)}")
    print(f"\n[NEXT] Run: python scripts/test_suite.py --detector ml --notes \"ML classifier v1\"")
    print(f"       Then: python scripts/test_suite.py --compare <rule_run_id> <ml_run_id>")


if __name__ == "__main__":
    main()
