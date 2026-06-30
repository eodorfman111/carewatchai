"""
CareWatch AI — Le2i batch evaluation runner.

Downloads Le2i to a temp directory (never touches the repo or your disk permanently),
runs src/evaluate.py, saves structured results to data/eval_results/, then
deletes the videos and auto-commits the results to git.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --subset Coffee_room_01
    python scripts/run_eval.py --skip-download   # if you already have videos in /tmp
    python scripts/run_eval.py --no-commit        # skip the auto git commit
    python scripts/run_eval.py --dataset-name "Le2i-coffee-room" --notes "baseline run"

Requirements:
    pip install kaggle
    export KAGGLE_API_TOKEN=your_token
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).parent.parent
RESULTS_DIR  = REPO_ROOT / "data" / "eval_results"
EVALUATE_PY  = REPO_ROOT / "src" / "evaluate.py"
ALL_RUNS_CSV = RESULTS_DIR / "all_runs.csv"

# Videos go here — outside the repo, cleaned up after every run
TMP_VIDEO_DIR = Path(tempfile.gettempdir()) / "carewatchai_videos"

KAGGLE_DATASET = "tuyenldvn/falldataset-imvia"


# ── Kaggle helpers ─────────────────────────────────────────────────────────────

def check_kaggle_credentials() -> bool:
    token_file = Path.home() / ".kaggle" / "access_token"
    env_token  = os.environ.get("KAGGLE_API_TOKEN")
    if token_file.exists() or env_token:
        return True
    print(
        "\n[ERROR] No Kaggle credentials found.\n"
        "Run:  export KAGGLE_API_TOKEN=your_token\n"
    )
    return False


def setup_kaggle_json() -> None:
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
    print(f"[INFO] Downloading Le2i to temp dir: {TMP_VIDEO_DIR}")
    print(f"[INFO] Videos will be deleted automatically after the eval.\n")
    TMP_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    setup_kaggle_json()

    result = subprocess.run([
        sys.executable, "-m", "kaggle",
        "datasets", "download",
        KAGGLE_DATASET,
        "--unzip",
        "-p", str(TMP_VIDEO_DIR),
    ])
    if result.returncode != 0:
        print("[ERROR] Kaggle download failed.")
        return False

    print(f"[INFO] Download complete.")
    return True


# ── Dataset discovery ──────────────────────────────────────────────────────────

def find_dataset_root(base: Path) -> Path | None:
    if not any(base.rglob("Annotation_files")):
        return None
    for p in base.iterdir():
        if p.is_dir() and (p / "Annotation_files").exists():
            return p.parent
    return base


# ── Output parsing ─────────────────────────────────────────────────────────────

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


def parse_failures(output: str) -> list[dict]:
    """
    Parse per-video results from evaluate.py output.
    Returns a list of dicts for every video, with extra detail on misses.
    """
    failures = []
    # Match lines like: "video (4).avi   True   MISSED   0   1   0   —   ❌ MISS"
    pattern = re.compile(
        r"^(video\s*\(\d+\)\.\w+)\s+"   # filename
        r"(\w+)\s+"                       # has_fall (True/False)
        r"(frame\s+\d+|MISSED)\s+"        # detected
        r"(\d+)\s+(\d+)\s+(\d+)",         # TP FN FP
        re.MULTILINE,
    )
    for m in pattern.finditer(output):
        video, has_fall, detected, tp, fn, fp = m.groups()
        missed = detected.strip() == "MISSED"
        false_positive = int(fp) > 0
        if missed or false_positive:
            failures.append({
                "video":          video.strip(),
                "has_fall":       has_fall == "True",
                "detected":       detected.strip(),
                "tp":             int(tp),
                "fn":             int(fn),
                "fp":             int(fp),
                "miss_type":      "false_negative" if missed else "false_positive",
                "download_cmd":   (
                    f'kaggle datasets download tuyenldvn/falldataset-imvia '
                    f'-f "Coffee_room_01/Coffee_room_01/{video.strip()}" '
                    f'-p /tmp/carewatchai_debug --unzip'
                ),
            })
    return failures


# ── Results saving ─────────────────────────────────────────────────────────────

def save_results(dataset_name, kaggle_slug, subset, notes, summary, raw_output) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M")

    failures = parse_failures(raw_output)

    result = {
        "timestamp":    now.isoformat(),
        "dataset_name": dataset_name,
        "kaggle_slug":  kaggle_slug,
        "subset":       subset or "all",
        "notes":        notes,
        **summary,
        "failures":     failures,
        "raw_output":   raw_output,
    }

    if failures:
        print(f"\n[INFO] {len(failures)} failure(s) recorded:")
        for f in failures:
            tag = "❌ MISS" if f["miss_type"] == "false_negative" else "⚠️  FALSE POS"
            print(f"  {tag}  {f['video']}")
        print(f"\n  To debug, download individual videos with the 'download_cmd' in the JSON.")

    slug_safe = dataset_name.replace("/", "-").replace(" ", "_")
    json_path = RESULTS_DIR / f"{timestamp}_{slug_safe}.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"\n[INFO] Results saved → {json_path.relative_to(REPO_ROOT)}")

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
    print(f"[INFO] Run appended  → {ALL_RUNS_CSV.relative_to(REPO_ROOT)}")

    return json_path


# ── Git commit ─────────────────────────────────────────────────────────────────

def commit_results(summary: dict, dataset_name: str) -> None:
    f1 = summary.get("f1_score")
    precision = summary.get("precision")
    recall = summary.get("recall")
    msg = (
        f"eval: {dataset_name} — "
        f"F1={f1}% P={precision}% R={recall}%"
    )
    subprocess.run(["git", "-C", str(REPO_ROOT), "add", str(RESULTS_DIR)], check=True)
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "commit", "-m", msg],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"[INFO] Results committed: \"{msg}\"")
        subprocess.run(["git", "-C", str(REPO_ROOT), "push"], check=False)
        print(f"[INFO] Pushed to remote.")
    else:
        print(f"[INFO] Nothing new to commit (results unchanged).")


# ── Cleanup ────────────────────────────────────────────────────────────────────

def cleanup_videos() -> None:
    if TMP_VIDEO_DIR.exists():
        print(f"\n[INFO] Deleting temp videos: {TMP_VIDEO_DIR}")
        shutil.rmtree(TMP_VIDEO_DIR)
        print(f"[INFO] Videos deleted. Your disk is clean.")


# ── Evaluation runner ──────────────────────────────────────────────────────────

def run_evaluation(dataset_root: Path, subset: str | None) -> tuple[int, str]:
    cmd = [sys.executable, str(EVALUATE_PY), "--dataset", str(dataset_root)]
    if subset:
        cmd += ["--subset", subset]

    print(f"\n[INFO] Running: {' '.join(cmd)}\n")
    lines = []
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
        lines.append(line)
    process.wait()
    return process.returncode, "".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run CareWatch evaluation — no videos stored locally")
    parser.add_argument("--subset",        default=None,
                        help="Only evaluate one subset, e.g. Coffee_room_01")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download (use if /tmp/carewatchai_videos already exists)")
    parser.add_argument("--no-commit",     action="store_true",
                        help="Skip auto git commit of results")
    parser.add_argument("--no-cleanup",    action="store_true",
                        help="Keep videos in /tmp after eval (useful for debugging)")
    parser.add_argument("--dataset-name",  default="Le2i",
                        help="Label for this run, e.g. 'Le2i-low-quality'")
    parser.add_argument("--notes",         default="",
                        help="Free-text notes, e.g. 'after tuning inactivity threshold'")
    args = parser.parse_args()

    try:
        # ── Download ──────────────────────────────────────────────────────────
        if not args.skip_download:
            already_present = TMP_VIDEO_DIR.exists() and any(TMP_VIDEO_DIR.rglob("Annotation_files"))
            if already_present:
                print(f"[INFO] Videos already in {TMP_VIDEO_DIR}, skipping download.")
            else:
                if not check_kaggle_credentials():
                    sys.exit(1)
                if not download_dataset():
                    sys.exit(1)

        # ── Find dataset ──────────────────────────────────────────────────────
        dataset_root = find_dataset_root(TMP_VIDEO_DIR)
        if dataset_root is None:
            print(f"[ERROR] Could not find Le2i Annotation_files in {TMP_VIDEO_DIR}.")
            sys.exit(1)
        print(f"[INFO] Dataset root: {dataset_root}")

        # ── Evaluate ──────────────────────────────────────────────────────────
        exit_code, raw_output = run_evaluation(dataset_root, args.subset)

        # ── Save + commit results ─────────────────────────────────────────────
        summary = parse_summary(raw_output)
        save_results(
            dataset_name=args.dataset_name,
            kaggle_slug=KAGGLE_DATASET,
            subset=args.subset,
            notes=args.notes,
            summary=summary,
            raw_output=raw_output,
        )

        if not args.no_commit:
            commit_results(summary, args.dataset_name)

    finally:
        # Always clean up videos unless --no-cleanup passed
        if not args.no_cleanup:
            cleanup_videos()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()