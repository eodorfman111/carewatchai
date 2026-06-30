"""
Save MultiCam evaluation results in the same format as run_eval.py uses for Le2i.
Run after evaluate_multicam.py to persist + commit results.

Usage:
    python src/evaluate_multicam.py --dataset data/multicam --csv data/multicam/data_tuple3.csv > /tmp/multicam_out.txt
    python scripts/save_multicam_eval.py --raw-output /tmp/multicam_out.txt --notes "first real-data run"
"""
import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT    = Path(__file__).parent.parent
RESULTS_DIR  = REPO_ROOT / "data" / "eval_results"
ALL_RUNS_CSV = RESULTS_DIR / "all_runs.csv"


def parse_summary(output: str) -> dict:
    def extract(pattern, text, cast=int):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else None

    return {
        "videos_evaluated":   extract(r"Videos evaluated\s*:\s*(\d+)", output),
        "true_positives":     extract(r"True Positives\s*:\s*(\d+)", output),
        "false_negatives":    extract(r"False Negatives\s*:\s*(\d+)", output),
        "false_positives":    extract(r"False Positives\s*:\s*(\d+)", output),
        "precision":          extract(r"Precision\s*:\s*([\d.]+)%", output, float),
        "recall":             extract(r"Recall\s*:\s*([\d.]+)%", output, float),
        "f1_score":           extract(r"F1 Score\s*:\s*([\d.]+)%", output, float),
        "avg_latency_frames": extract(r"Avg Latency\s*:\s*([\d.]+) frames", output, float),
    }


def save_results(dataset_name, kaggle_slug, notes, summary, raw_output) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M")

    result = {
        "timestamp":    now.isoformat(),
        "dataset_name": dataset_name,
        "kaggle_slug":  kaggle_slug,
        "subset":       "all",
        "notes":        notes,
        **summary,
        "raw_output":   raw_output,
    }

    slug_safe = dataset_name.replace("/", "-").replace(" ", "_")
    json_path = RESULTS_DIR / f"{timestamp}_{slug_safe}.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"[INFO] Results saved -> {json_path.relative_to(REPO_ROOT)}")

    csv_fields = [
        "timestamp", "dataset_name", "kaggle_slug", "subset", "notes",
        "videos_evaluated", "true_positives", "false_negatives", "false_positives",
        "precision", "recall", "f1_score", "avg_latency_frames",
    ]
    write_header = not ALL_RUNS_CSV.exists()
    with open(ALL_RUNS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(result)
    print(f"[INFO] Run appended -> {ALL_RUNS_CSV.relative_to(REPO_ROOT)}")

    return json_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-output", required=True, help="Path to captured stdout from evaluate_multicam.py")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    raw = Path(args.raw_output).read_text(encoding="utf-8", errors="replace")
    summary = parse_summary(raw)
    save_results(
        dataset_name="MultiCam-nursing-home-reenacted",
        kaggle_slug="soumicksarker/multiple-cameras-fall-dataset",
        notes=args.notes,
        summary=summary,
        raw_output=raw,
    )
