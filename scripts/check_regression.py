"""
CareWatch AI — eval regression gate.

Compares the most recent row in data/eval_results/all_runs.csv against the
best F1 score seen among prior rows of the *same benchmark*, and exits 1
(failing CI) if F1 dropped by more than the allowed threshold, or if the
latest run has no F1 at all (a failed/incomplete eval run should never pass
silently).

"Same benchmark" is matched on kaggle_slug + videos_evaluated rather than
dataset_name/subset — those two columns are labeled inconsistently across
runs in practice (e.g. the original manual baseline recorded
dataset_name="Le2i-Coffee_room_01", subset="all", while CI records
dataset_name="CI", subset="Coffee_room_01" — same 15 videos, different
labels). Comparing across genuinely different benchmarks (a different video
count, or a different Kaggle dataset entirely) is meaningless, so the pool
of "historical" runs must be scoped to the ones this run is actually
comparable to.

Usage:
    python scripts/check_regression.py
    python scripts/check_regression.py --threshold 5.0   # allow bigger drops
"""
import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT    = Path(__file__).parent.parent
ALL_RUNS_CSV = REPO_ROOT / "data" / "eval_results" / "all_runs.csv"

DEFAULT_THRESHOLD_PCT = 2.0   # percentage points of F1 allowed to drop

# Known tradeoff: this gates on F1, which weights precision/recall equally.
# ROADMAP.md's own targets don't (bed exit wants >95% recall specifically).
# See "Known Issues / Tech Debt" in ROADMAP.md.


def load_runs(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        print(f"[ERROR] {csv_path} does not exist — nothing to check.")
        sys.exit(1)
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail if the latest eval run regressed on F1")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_PCT,
                         help=f"Max allowed F1 drop in percentage points (default {DEFAULT_THRESHOLD_PCT})")
    args = parser.parse_args()

    rows = load_runs(ALL_RUNS_CSV)
    if not rows:
        print(f"[ERROR] {ALL_RUNS_CSV} has no rows — nothing to check.")
        sys.exit(1)

    latest = rows[-1]
    history = rows[:-1]

    latest_f1_raw = latest.get("f1_score", "").strip()
    if not latest_f1_raw:
        print(f"[FAIL] Latest run ({latest.get('timestamp', '?')}) has no F1 score — "
              f"the eval run itself likely failed (see its raw_output). Blocking.")
        sys.exit(1)
    latest_f1 = float(latest_f1_raw)
    latest_slug = latest.get("kaggle_slug", "").strip()
    latest_n    = latest.get("videos_evaluated", "").strip()

    def is_comparable(r: dict) -> bool:
        if not r.get("f1_score", "").strip():
            return False
        return (r.get("kaggle_slug", "").strip() == latest_slug
                and r.get("videos_evaluated", "").strip() == latest_n)

    comparable_history = [r for r in history if is_comparable(r)]
    historical_f1s = [float(r["f1_score"]) for r in comparable_history]

    if not historical_f1s:
        print(f"[PASS] No prior runs on the same benchmark "
              f"(kaggle_slug={latest_slug!r}, videos_evaluated={latest_n}) to compare "
              f"against — treating this as the baseline (F1={latest_f1:.1f}%).")
        sys.exit(0)

    best_historical_f1 = max(historical_f1s)
    drop = best_historical_f1 - latest_f1

    print(f"Comparing against  : {len(historical_f1s)} prior run(s) with "
          f"kaggle_slug={latest_slug!r}, videos_evaluated={latest_n}")
    print(f"Latest F1          : {latest_f1:.1f}%  ({latest.get('timestamp', '?')})")
    print(f"Best historical F1 : {best_historical_f1:.1f}%")
    print(f"Drop               : {drop:.1f} points  (threshold: {args.threshold:.1f})")

    if drop > args.threshold:
        print(f"\n[FAIL] F1 regressed by {drop:.1f} points, more than the "
              f"{args.threshold:.1f}-point threshold. Blocking.")
        sys.exit(1)

    print(f"\n[PASS] F1 within threshold of best historical run.")
    sys.exit(0)


if __name__ == "__main__":
    main()
