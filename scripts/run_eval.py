"""
CareWatch AI — Le2i batch evaluation runner.

Downloads the Le2i fall dataset from Kaggle (if not already present),
runs src/evaluate.py, and saves structured results to data/eval_results/.

Each run produces:
  data/eval_results/YYYY-MM-DD_HH-MM_<dataset>.json  — full metadata + stats
  data/eval_results/all_runs.csv                      — one row per run, easy to compare

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --subset Coffee_room_01
    python scripts/run_eval.py --skip-download
    python scripts/run_eval.py --dataset-name "Le2i-low-quality" --notes "720p compressed test"

Requirements:
    pip install kaggle
    Set KAGGLE_API_TOKEN env var, or save key to ~/.kaggle/access_token
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT      = Path(__file__).parent.parent
DATA_DIR       = REPO_ROOT / "data" / "videos"
RESULTS_DIR    = REPO_ROOT / "data" / "eval_results"
EVALUATE_PY    = REPO_ROOT / "src" / "evaluate.py"
ALL_RUNS_CSV   = RESULTS_DIR / "all_runs.csv"

KAGGLE_DATASET = "tuyenldvn/falldataset-imvia"


# ── Kaggle helpers ─────────────────────────────────────────────────────────────

def check_kaggle_credentials() -> bool:
    token_file = Path.home() / ".kaggle" / "access_token"
    env_token  = os.environ.get("KAGGLE_API_TOKEN")
    if token_file.exists() or env_token:
        return True
    print(
        "\n[ERROR] No Kaggle credentials found.\n"
        "Either:\n"
        "  export KAGGLE_API_TOKEN=your_token\n"
        "or:\n"
        "  mkdir -p ~/.kaggle && echo YOUR_TOKEN > ~/.kaggle/access_token "
        "&& chmod 600 ~/.kaggle/access_token\n"
    )
    return False


def setup_kaggle_json() -> None:
    """Write ~/.kaggle/kaggle.json from env var or access_token file if needed."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return

    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        access_token_file = Path.home() / ".kaggle" / "access_token"
        if access_token_file.exists():
            token = access_token_file.read_text().strip()

    if token:
        kaggle_json.parent.mkdir(parents=True, exist_ok=True)
        kaggle_json.write_text(f'{{"username":"kaggle","key":"{token}"}}')
        kaggle_json.chmod(0o600)


def download_dataset() -> bool:
    print(f"[INFO] Downloading Le2i dataset from Kaggle: {KAGGLE_DATASET}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    setup_kaggle_json()

    result = subprocess.run([
        sys.executable, "-m", "kaggle",
        "datasets", "download",
        KAGGLE_DATASET,
        "--unzip",
        "-p", str(DATA_DIR),
    ])
    if result.returncode != 0:
        print("[ERROR] Kaggle download failed. Check credentials and dataset slug.")
        return False

    print(f"[INFO] Dataset downloaded to {DATA_DIR}")
    return True


# ── Dataset discovery ──────────────────────────────────────────────────────────

def find_dataset_root() -> Path | None:
    """Find the Le2i root dir (the one containing subset folders with Annotation_files)."""
    if not any(DATA_DIR.rglob("Annotation_files")):
        return None
    for p in DATA_DIR.iterdir():
        if p.is_dir() and (p / "Annotation_files").exists():
            return p.parent
    return DATA_DIR


# ── Output parsing ─────────────────────────────────────────────────────────────

def parse_summary(output: str) -> dict:
    """Extract numeric results from evaluate.py's printed summary block."""
    def extract(pattern, text, cast=int):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else None

    return {
        "videos_evaluated": extract(r"Videos evaluated\s*:\s*(\d+)", output),
        "true_positives":   extract(r"True Positives\s*:\s*(\d+)", output),
        "false_negatives":  extract(r"False Negatives\s*:\s*(\d+)", output),
        "false_positives":  extract(r"False Positives\s*:\s*(\d+)", output),
        "precision":        extract(r"Precision\s*:\s*([\d.]+)%", output, float),
        "recall":           extract(r"Recall\s*:\s*([\d.]+)%", output, float),
        "f1_score":         extract(r"F1 Score\s*:\s*([\d.]+)%", output, float),
        "avg_latency_frames": extract(r"Avg Latency\s*:\s*([\d.]+) frames", output, float),
    }


# ── Results saving ─────────────────────────────────────────────────────────────

def save_results(
    dataset_name: str,
    kaggle_slug: str,
    subset: str | None,
    notes: str,
    summary: dict,
    raw_output: str,
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M")

    # ── Per-run JSON ──────────────────────────────────────────────────────────
    result = {
        "timestamp":    now.isoformat(),
        "dataset_name": dataset_name,
        "kaggle_slug":  kaggle_slug,
        "subset":       subset or "all",
        "notes":        notes,
        **summary,
        "raw_output":   raw_output,
    }
    slug_safe = dataset_name.replace("/", "-").replace(" ", "_")
    json_path = RESULTS_DIR / f"{timestamp}_{slug_safe}.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"\n[INFO] Results saved → {json_path}")

    # ── Append to all_runs.csv ────────────────────────────────────────────────
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
    print(f"[INFO] Run appended  → {ALL_RUNS_CSV}")

    return json_path


# ── Evaluation runner ──────────────────────────────────────────────────────────

def run_evaluation(dataset_root: Path, subset: str | None) -> tuple[int, str]:
    """Run evaluate.py and return (exit_code, captured_output)."""
    cmd = [sys.executable, str(EVALUATE_PY), "--dataset", str(dataset_root)]
    if subset:
        cmd += ["--subset", subset]

    print(f"\n[INFO] Running: {' '.join(cmd)}\n")

    # Tee output: show in terminal AND capture it
    lines = []
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
        lines.append(line)
    process.wait()

    return process.returncode, "".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download Le2i + run CareWatch evaluation")
    parser.add_argument("--subset",        default=None,
                        help="Only evaluate one subset, e.g. Coffee_room_01")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip Kaggle download (use if data already present)")
    parser.add_argument("--dataset-name",  default="Le2i",
                        help="Human label for this dataset, e.g. 'Le2i-low-quality'")
    parser.add_argument("--notes",         default="",
                        help="Free-text notes about this run, e.g. 'after tuning inactivity threshold'")
    args = parser.parse_args()

    # ── Download ──────────────────────────────────────────────────────────────
    if not args.skip_download:
        already_present = any(DATA_DIR.rglob("Annotation_files"))
        if already_present:
            print("[INFO] Le2i data already present, skipping download.")
        else:
            if not check_kaggle_credentials():
                sys.exit(1)
            if not download_dataset():
                sys.exit(1)

    # ── Find dataset ──────────────────────────────────────────────────────────
    dataset_root = find_dataset_root()
    if dataset_root is None:
        print(f"[ERROR] Could not find Le2i Annotation_files under {DATA_DIR}.")
        sys.exit(1)
    print(f"[INFO] Dataset root: {dataset_root}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    exit_code, raw_output = run_evaluation(dataset_root, args.subset)

    # ── Save results ──────────────────────────────────────────────────────────
    summary = parse_summary(raw_output)
    save_results(
        dataset_name=args.dataset_name,
        kaggle_slug=KAGGLE_DATASET,
        subset=args.subset,
        notes=args.notes,
        summary=summary,
        raw_output=raw_output,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()