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
import zipfile
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


# Fixed CI benchmark: matches the existing historical baseline (15 videos,
# F1=84.6%, all_runs.csv row from 2026-06-30) rather than the full 48 videos
# actually present in Coffee_room_01 on Kaggle. Evaluating a different video
# count than the recorded baseline would make check_regression.py's
# comparison meaningless (different sample size, not a real regression), and
# downloading the full subset (3.1GB) or full dataset (16GB) on every PR is
# impractical for CI. Extend this map if other subsets get a CI job.
CI_BENCHMARK_VIDEOS = {
    "Coffee_room_01": list(range(1, 16)),
}


def _fetch_single_file(kaggle_path: str, dest: Path) -> bool:
    """Download one file from the dataset via `-f` and place it at `dest`.
    Kaggle wraps single-file downloads in a .zip that --unzip doesn't
    reliably expand, and URL-encodes spaces in the local filename, so we
    extract/rename manually rather than relying on kaggle's own unzip."""
    if dest.exists():
        return True

    stem = Path(kaggle_path).name.replace(" ", "%20")
    for leftover in TMP_VIDEO_DIR.glob(f"{stem}*"):
        leftover.unlink()

    result = subprocess.run([
        sys.executable, "-m", "kaggle", "datasets", "download",
        KAGGLE_DATASET, "-f", kaggle_path, "-p", str(TMP_VIDEO_DIR), "--force",
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Failed to download {kaggle_path}:\n{result.stderr}")
        return False

    candidates = list(TMP_VIDEO_DIR.glob(f"{stem}*"))
    if not candidates:
        print(f"[ERROR] Expected a downloaded file matching {stem}* in {TMP_VIDEO_DIR}, found none.")
        return False
    src = candidates[0]

    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as zf:
            inner = zf.namelist()[0]
            with zf.open(inner) as f_in, open(dest, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        src.unlink()
    else:
        shutil.move(str(src), str(dest))
    return True


def download_subset_files(subset: str) -> bool:
    """Download only the fixed CI_BENCHMARK_VIDEOS for `subset`, laid out in
    the flat structure evaluate.py expects (subset/*.avi +
    subset/Annotation_files/*.txt) — much cheaper than download_dataset()."""
    video_nums = CI_BENCHMARK_VIDEOS.get(subset)
    if not video_nums:
        print(f"[ERROR] No fixed CI file list defined for subset '{subset}'. "
              f"Known subsets: {list(CI_BENCHMARK_VIDEOS)}")
        return False

    dest_video_dir = TMP_VIDEO_DIR / subset
    dest_ann_dir   = dest_video_dir / "Annotation_files"
    dest_video_dir.mkdir(parents=True, exist_ok=True)
    dest_ann_dir.mkdir(parents=True, exist_ok=True)
    setup_kaggle_json()

    print(f"[INFO] Downloading {len(video_nums)} fixed videos for subset '{subset}' "
          f"(CI benchmark, matches historical baseline)...\n")

    for n in video_nums:
        name = f"video ({n})"
        ok_video = _fetch_single_file(
            f"{subset}/{subset}/Videos/{name}.avi", dest_video_dir / f"{name}.avi")
        ok_ann = _fetch_single_file(
            f"{subset}/{subset}/Annotation_files/{name}.txt", dest_ann_dir / f"{name}.txt")
        if not (ok_video and ok_ann):
            return False
        print(f"  fetched {name}")

    print(f"\n[INFO] Download complete: {dest_video_dir}")
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


def parse_failures(output: str, subset: str | None) -> list[dict]:
    """
    Pull per-video results (including miss diagnostics) from evaluate.py's
    structured RESULTS_JSON line, and return the failing ones (misses and
    false positives) with their full reasoning attached.
    """
    match = re.search(r"^RESULTS_JSON:(.+)$", output, re.MULTILINE)
    if not match:
        return []
    results = json.loads(match.group(1))

    subset_folder = subset or "Coffee_room_01"

    failures = []
    for r in results:
        if "error" in r:
            continue
        missed = bool(r["has_fall"]) and r["fn"] > 0
        false_positive = r["fp"] > 0
        if not (missed or false_positive):
            continue
        failures.append({
            "video":          r["video"],
            "has_fall":       r["has_fall"],
            "detected":       f"frame {r['detected_frame']}" if r["detected_frame"] else "MISSED",
            "tp":             r["tp"],
            "fn":             r["fn"],
            "fp":             r["fp"],
            "miss_type":      "false_negative" if missed else "false_positive",
            "miss_reason":    r.get("miss_reason"),
            "download_cmd":   (
                f'kaggle datasets download tuyenldvn/falldataset-imvia '
                f'-f "{subset_folder}/{subset_folder}/Videos/{r["video"]}" '
                f'-p /tmp/carewatchai_debug --unzip'
            ),
        })
    return failures


# ── Results saving ─────────────────────────────────────────────────────────────

def save_results(dataset_name, kaggle_slug, subset, notes, summary, raw_output) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M")

    failures = parse_failures(raw_output, subset)

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
            category = (f.get("miss_reason") or {}).get("category")
            suffix = f"  [{category}]" if category else ""
            print(f"  {tag}  {f['video']}{suffix}")
        print(f"\n  Full reasoning is saved per-video under 'failures[].miss_reason' in the JSON.")
        print(f"  To debug, download individual videos with the 'download_cmd' in the JSON.")

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
    parser.add_argument("--full-subset",   action="store_true",
                        help="Download the complete subset instead of the fixed CI "
                             "benchmark file list (slower, not comparable to the "
                             "existing history in all_runs.csv)")
    args = parser.parse_args()

    use_fixed_list = (args.subset in CI_BENCHMARK_VIDEOS) and not args.full_subset

    try:
        # ── Download ──────────────────────────────────────────────────────────
        if not args.skip_download:
            already_present = TMP_VIDEO_DIR.exists() and any(TMP_VIDEO_DIR.rglob("*.avi"))
            if already_present:
                print(f"[INFO] Videos already in {TMP_VIDEO_DIR}, skipping download.")
            elif use_fixed_list:
                if not check_kaggle_credentials():
                    sys.exit(1)
                if not download_subset_files(args.subset):
                    sys.exit(1)
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