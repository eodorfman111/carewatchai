"""
CareWatch AI — Le2i batch evaluation runner.

Downloads the Le2i fall dataset from Kaggle (if not already present),
then runs src/evaluate.py against all videos and saves a CSV summary.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --subset Coffee_room_01
    python scripts/run_eval.py --skip-download   # if data already present

Requirements:
    pip install kaggle
    Set KAGGLE_API_TOKEN env var, or save key to ~/.kaggle/access_token
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).parent.parent
DATA_DIR     = REPO_ROOT / "data" / "videos"
RESULTS_DIR  = REPO_ROOT / "data" / "eval_results"
EVALUATE_PY  = REPO_ROOT / "src" / "evaluate.py"

KAGGLE_DATASET = "tuyenldvn/falldataset-imvia"
SENTINEL_FILE  = DATA_DIR / "Annotation_files"   # exists once Le2i is unpacked


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
        "  mkdir -p ~/.kaggle && echo YOUR_TOKEN > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token\n"
    )
    return False


def download_dataset() -> bool:
    print(f"[INFO] Downloading Le2i dataset from Kaggle: {KAGGLE_DATASET}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Write token to kaggle.json if only env var is set
    token = os.environ.get("KAGGLE_API_TOKEN")
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if token and not kaggle_json.exists():
        kaggle_json.parent.mkdir(parents=True, exist_ok=True)
        kaggle_json.write_text(f'{{"username":"kaggle","key":"{token}"}}')
        kaggle_json.chmod(0o600)

    # Also support the simpler access_token format used by this project
    access_token_file = Path.home() / ".kaggle" / "access_token"
    if access_token_file.exists() and not kaggle_json.exists():
        key = access_token_file.read_text().strip()
        kaggle_json.parent.mkdir(parents=True, exist_ok=True)
        kaggle_json.write_text(f'{{"username":"kaggle","key":"{key}"}}')
        kaggle_json.chmod(0o600)

    result = subprocess.run(
        [
            sys.executable, "-m", "kaggle",
            "datasets", "download",
            KAGGLE_DATASET,
            "--unzip",
            "-p", str(DATA_DIR),
        ],
        capture_output=False,
    )
    if result.returncode != 0:
        print("[ERROR] Kaggle download failed. Check credentials and dataset slug.")
        return False

    print(f"[INFO] Dataset downloaded to {DATA_DIR}")
    return True


def find_dataset_root() -> Path | None:
    """
    Le2i unzips to a folder like data/videos/falldataset-imvia/
    or directly into data/videos/. Find the right root.
    """
    # Check if annotation files are directly under DATA_DIR
    if any(DATA_DIR.rglob("Annotation_files")):
        # Find top-level dir that contains Annotation_files
        for p in DATA_DIR.iterdir():
            if p.is_dir() and (p / "Annotation_files").exists():
                return p.parent  # return parent so evaluate.py can scan subsets
        return DATA_DIR

    return None


def run_evaluation(dataset_root: Path, subset: str | None) -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(EVALUATE_PY),
        "--dataset", str(dataset_root),
    ]
    if subset:
        cmd += ["--subset", subset]

    print(f"\n[INFO] Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Le2i and run CareWatch eval")
    parser.add_argument("--subset",        default=None,
                        help="Only evaluate one Le2i subset, e.g. Coffee_room_01")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip Kaggle download (use if data already present)")
    args = parser.parse_args()

    # ── Download ──────────────────────────────────────────────────────────────
    if not args.skip_download:
        already_present = any(DATA_DIR.rglob("Annotation_files"))
        if already_present:
            print("[INFO] Le2i data already present, skipping download.")
            print("       Pass --skip-download to suppress this check.")
        else:
            if not check_kaggle_credentials():
                sys.exit(1)
            if not download_dataset():
                sys.exit(1)

    # ── Find dataset root ─────────────────────────────────────────────────────
    dataset_root = find_dataset_root()
    if dataset_root is None:
        print(
            f"[ERROR] Could not find Le2i Annotation_files under {DATA_DIR}.\n"
            "Make sure the dataset downloaded correctly."
        )
        sys.exit(1)

    print(f"[INFO] Dataset root: {dataset_root}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    exit_code = run_evaluation(dataset_root, args.subset)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()