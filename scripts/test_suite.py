"""
CareWatch AI — Standalone test suite.

Runs the detection pipeline against every video in the local test set
(Le2i + MultiCam, auto-discovered from data/le2i/ and data/multicam/),
and writes two CSVs:

  data/eval_results/test_runs/<run_id>_detailed.csv
      One row PER VIDEO: dataset, video name, expected label, predicted
      label, correct/incorrect, ground-truth fall window, detected frame,
      latency, false-positive count. This is the "what did it get wrong
      and why" sheet.

  data/eval_results/test_runs/summary.csv
      One row PER RUN: aggregate precision/recall/F1 per dataset and
      overall, so you can track whether a code change made things
      better or worse over time. Appends — never overwrites.

No Kaggle download, no network calls — assumes data/le2i/ and
data/multicam/ already exist locally (see README for how they were built).

Usage:
    python scripts/test_suite.py
    python scripts/test_suite.py --notes "after velocity gate revert"
    python scripts/test_suite.py --compare RUN_ID_1 RUN_ID_2
"""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ultralytics import YOLO

from config import POSE_MODEL
from evaluate import evaluate_video, parse_annotation
from evaluate_multicam import load_ground_truth as load_multicam_gt, write_temp_annotation

REPO_ROOT     = Path(__file__).parent.parent
LE2I_DIR      = REPO_ROOT / "data" / "le2i"
MULTICAM_DIR  = REPO_ROOT / "data" / "multicam"
MULTICAM_CSV  = MULTICAM_DIR / "data_tuple3.csv"
RUNS_DIR      = REPO_ROOT / "data" / "eval_results" / "test_runs"
SUMMARY_CSV   = RUNS_DIR / "summary.csv"


# ── Test set discovery ──────────────────────────────────────────────────────

def discover_le2i() -> list[dict]:
    """Returns [{video, ann, dataset}] for every Le2i video with a matching annotation."""
    items = []
    video_dir = LE2I_DIR / "Videos"
    ann_dir   = LE2I_DIR / "Annotation_files"
    if not video_dir.exists():
        return items
    for video_file in sorted(video_dir.glob("*.avi")):
        ann_file = ann_dir / f"{video_file.stem}.txt"
        if ann_file.exists():
            items.append({"video": video_file, "ann": ann_file, "dataset": "Le2i"})
    return items


def discover_multicam(cam: int = 2) -> list[dict]:
    """Returns [{video, ann, dataset}] for every MultiCam chute, writing temp
    annotation files in the Le2i 2-line format so evaluate_video() can read them."""
    items = []
    if not MULTICAM_CSV.exists():
        return items
    gt_map = load_multicam_gt(MULTICAM_CSV)
    tmp_ann_dir = MULTICAM_DIR / "_tmp_annotations"
    tmp_ann_dir.mkdir(exist_ok=True)

    for video_file in sorted(MULTICAM_DIR.glob(f"chute*_cam{cam}.avi")):
        chute = int(video_file.stem.split("_")[0].replace("chute", ""))
        segments = gt_map.get((chute, cam), [])
        ann_path = tmp_ann_dir / f"{video_file.stem}.txt"
        if segments:
            start = min(s for s, _ in segments)
            end   = max(e for _, e in segments)
            write_temp_annotation(start, end, ann_path)
        else:
            ann_path.write_text("")
        items.append({"video": video_file, "ann": ann_path, "dataset": "MultiCam"})
    return items


# ── Run + record ─────────────────────────────────────────────────────────────

def classify(r: dict) -> tuple[str, str, str, bool]:
    """Returns (expected_label, predicted_label, result_type, correct)."""
    expected = "Fall" if r["has_fall"] else "No Fall"

    if r["has_fall"]:
        predicted = "Fall" if r["tp"] else "No Fall"
        result_type = "TP" if r["tp"] else "FN"
        correct = bool(r["tp"])
    else:
        predicted = "Fall" if r["fp"] > 0 else "No Fall"
        result_type = "FP" if r["fp"] > 0 else "TN"
        correct = r["fp"] == 0

    return expected, predicted, result_type, correct


def run_suite(notes: str = "", detector: str = "rule") -> Path:
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"[INFO] Run ID: {run_id}")
    print(f"[INFO] Detector: {detector}")

    test_items = discover_le2i() + discover_multicam()
    if not test_items:
        print("[ERROR] No test videos found. Expected data/le2i/ and/or data/multicam/.")
        sys.exit(1)

    print(f"[INFO] Test set: {len(test_items)} videos "
          f"({sum(1 for i in test_items if i['dataset']=='Le2i')} Le2i, "
          f"{sum(1 for i in test_items if i['dataset']=='MultiCam')} MultiCam)\n")

    model = YOLO(POSE_MODEL)

    detailed_rows = []
    by_dataset: dict[str, dict] = {}

    for item in test_items:
        r = evaluate_video(item["video"], item["ann"], model, camera_id="test_suite", detector=detector)
        if "error" in r:
            print(f"  [SKIP] {item['video'].name}: {r['error']}")
            continue

        expected, predicted, result_type, correct = classify(r)

        ds = item["dataset"]
        by_dataset.setdefault(ds, {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "latencies": []})
        by_dataset[ds][result_type.lower()] = by_dataset[ds].get(result_type.lower(), 0) + 1
        # fp_count can be >0 even on a true-fall video (spurious extra alerts
        # outside the correct window) — must always be added to precision,
        # not just when the video itself has no fall.
        if r["has_fall"]:
            by_dataset[ds]["fp"] += r["fp"]
        if r["latency_frames"] is not None:
            by_dataset[ds]["latencies"].append(r["latency_frames"])

        flag = "OK" if correct else "WRONG"
        print(f"  [{flag:5}] {ds:10} {item['video'].name:25} "
              f"expected={expected:8} predicted={predicted:8} ({result_type})")

        detailed_rows.append({
            "run_id":          run_id,
            "dataset":         ds,
            "video":           item["video"].name,
            "expected":        expected,
            "predicted":       predicted,
            "result_type":     result_type,
            "correct":         correct,
            "gt_fall_start":   r["fall_start"],
            "detected_frame":  r["detected_frame"],
            "latency_frames":  r["latency_frames"],
            "fp_count":        r["fp"],
            "total_frames":    r["total_frames"],
        })

    # ── Write detailed CSV ───────────────────────────────────────────────────
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    detailed_path = RUNS_DIR / f"{run_id}_detailed.csv"
    with open(detailed_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detailed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detailed_rows)
    print(f"\n[INFO] Detailed results -> {detailed_path.relative_to(REPO_ROOT)}")

    # ── Compute + print + save summary ──────────────────────────────────────
    print("\n" + "=" * 80)
    summary_row = {"run_id": run_id, "timestamp": datetime.now().isoformat(),
                   "detector": detector, "notes": notes}

    overall_tp = overall_fn = overall_fp = 0
    for ds, counts in by_dataset.items():
        tp, fn, fp = counts.get("tp", 0), counts.get("fn", 0), counts.get("fp", 0)
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-9)
        avg_lat   = sum(counts["latencies"]) / len(counts["latencies"]) if counts["latencies"] else 0

        print(f"{ds}: TP={tp} FN={fn} FP={fp} | "
              f"Precision={precision:.1%} Recall={recall:.1%} F1={f1:.1%} | "
              f"AvgLatency={avg_lat:.0f}f")

        prefix = ds.lower()
        summary_row[f"{prefix}_videos"]    = tp + fn + counts.get("tn", 0) + fp
        summary_row[f"{prefix}_precision"] = round(precision * 100, 1)
        summary_row[f"{prefix}_recall"]    = round(recall * 100, 1)
        summary_row[f"{prefix}_f1"]        = round(f1 * 100, 1)

        overall_tp += tp
        overall_fn += fn
        overall_fp += fp

    overall_precision = overall_tp / max(overall_tp + overall_fp, 1)
    overall_recall    = overall_tp / max(overall_tp + overall_fn, 1)
    overall_f1        = 2 * overall_precision * overall_recall / max(overall_precision + overall_recall, 1e-9)

    print(f"\nOVERALL: Precision={overall_precision:.1%} Recall={overall_recall:.1%} F1={overall_f1:.1%}")
    print("=" * 80)

    summary_row["overall_precision"] = round(overall_precision * 100, 1)
    summary_row["overall_recall"]    = round(overall_recall * 100, 1)
    summary_row["overall_f1"]        = round(overall_f1 * 100, 1)

    write_header = not SUMMARY_CSV.exists()
    # Union of fieldnames across all historical runs (datasets may differ run to run)
    fieldnames = list(summary_row.keys())
    if SUMMARY_CSV.exists():
        with open(SUMMARY_CSV) as f:
            existing_fields = next(csv.reader(f), [])
        fieldnames = list(dict.fromkeys(existing_fields + fieldnames))

    with open(SUMMARY_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(summary_row)
    print(f"[INFO] Summary appended -> {SUMMARY_CSV.relative_to(REPO_ROOT)}")

    return detailed_path


# ── Compare two runs ─────────────────────────────────────────────────────────

def compare_runs(run_id_a: str, run_id_b: str) -> None:
    def load(run_id: str) -> dict[str, dict]:
        path = next(RUNS_DIR.glob(f"{run_id}*_detailed.csv"), None)
        if path is None:
            print(f"[ERROR] No detailed results found for run {run_id}")
            sys.exit(1)
        with open(path) as f:
            return {row["video"]: row for row in csv.DictReader(f)}

    a = load(run_id_a)
    b = load(run_id_b)

    print(f"\nComparing {run_id_a} -> {run_id_b}\n")
    print(f"{'Video':<28}{'Before':<10}{'After':<10}{'Change'}")
    print("-" * 70)

    changed = 0
    for video in sorted(set(a) | set(b)):
        before = a.get(video, {}).get("result_type", "—")
        after  = b.get(video, {}).get("result_type", "—")
        if before != after:
            changed += 1
            arrow = "IMPROVED" if (before in ("FN", "FP") and after in ("TP", "TN")) else \
                    "REGRESSED" if (before in ("TP", "TN") and after in ("FN", "FP")) else "CHANGED"
            print(f"{video:<28}{before:<10}{after:<10}{arrow}")

    if changed == 0:
        print("(no per-video changes between these two runs)")
    print(f"\n{changed} video(s) changed result.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes", default="", help="Free-text note for this run, e.g. 'after reverting velocity gate'")
    parser.add_argument("--detector", default="rule", choices=["rule", "ml"],
                        help="rule = hand-coded FallFSM, ml = trained classifier (needs models/fall_classifier.joblib)")
    parser.add_argument("--compare", nargs=2, metavar=("RUN_ID_A", "RUN_ID_B"),
                        help="Compare two previous run IDs instead of running a new test")
    args = parser.parse_args()

    if args.compare:
        compare_runs(*args.compare)
    else:
        run_suite(notes=args.notes, detector=args.detector)
