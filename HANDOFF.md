# CareWatch AI — Handoff

> For any agent (or human) picking up work on this repo. Read this first.
> Keep this current as work happens — update "Read This First" and "In
> Progress" every session; trim "Recently Completed" down to a few bullets
> instead of letting it grow forever. This is a living doc, not a diary.
>
> **Two people/agents are working on this repo in parallel** (confirmed
> 2026-07-01). Expect concurrent changes on `master` between sessions —
> `git fetch` both remotes and check recent commits before assuming the
> state you last saw is still current.

Last updated: 2026-07-01

---

## Read This First

**Two remotes, and they mean different things:**
- `origin` → `samanthadorfman/carewatchai` — personal fork, used for
  pushing feature branches.
- `upstream` → `eodorfman111/carewatchai` — where PRs actually get merged.
  This is the canonical branch history.

`origin/master` is kept manually synced to `upstream/master` (merge, not
rebase — see 2026-07-01 entry below). Don't assume `origin/master` is
current without fetching upstream and diffing first.

**Eval regression gate PR (#1) merged into `upstream/master`.** But the
`eval.yml` workflow it added has **never actually run** — a
`pull_request`-triggered workflow only uses the version already committed
to the *base* branch, and this workflow was introduced in the same PR that
was supposed to trigger it, so GitHub Actions skipped it for PR #1. It's
live now and *will* trigger starting with the next PR. A follow-up PR
(updating this doc) is being opened specifically to confirm it actually
passes on GitHub Actions for the first time — check its Actions tab before
trusting the gate is real.

**A second workstream landed on `upstream/master` in parallel**: an ML
RandomForest fall classifier (`src/ml_fall_fsm.py`, `evaluate_multicam.py`,
`scripts/train_fall_classifier.py`, etc.), evaluated separately against a
MultiCam nursing-home-reenacted dataset (F1=54.2%, much harder than Le2i).
It hooks into `evaluate.py` via an optional `detector="ml"` param
(defaults to `"rule"` = the hand-coded FSM this session's work is built
around) — additive, not a replacement, as of this writing. Confirmed by
diffing the actual merged file contents (not just the commit graph) that
none of the eval-regression-gate work below was clobbered.

## In Progress / Blocked On

- **Confirming the CI workflow actually passes** — a PR updating this doc
  is being opened right now specifically to test it. Check its Actions run.
- **Phase 2 (production self-health monitoring)** not started. Paused
  before starting because it touches the live safety-critical pipeline
  (`src/pipeline.py`) and one piece — a metadata-only heartbeat so ops can
  see camera health remotely — has no defined destination (no
  server/endpoint exists in this repo to receive it). Needs a decision
  before building: what receives the heartbeat, and how that's exposed
  without violating the "video never leaves the building" promise.
- **Phase 3 (wire miss categories into `all_runs.csv` itself)** not done as
  originally scoped. Miss categories (`never_triggered` /
  `suspect_not_sustained` / `track_id_churn`) currently live only in each
  run's per-video JSON (`data/eval_results/*.json` → `failures[].miss_reason`),
  not as columns in `all_runs.csv` — a single run can have multiple
  different failure categories across its videos, which doesn't fit
  cleanly as scalar CSV columns. The dashboard's Eval Metrics page works
  around this by aggregating across all the JSON files directly. Revisit
  if `all_runs.csv`-only tooling ever needs this without reading JSON.

## Recently Completed

- **Eval regression gating** (merged via PR #1 into `upstream/master`):
  - `src/evaluate.py` — missed falls now log *why*: body angle at
    fall_start, YOLO tracking-loss %, longest FSM SUSPECT streak vs.
    `FALL_CONFIRM_FRAMES`, hip_y/aspect-ratio ranges, and a categorized
    diagnosis. Emits a `RESULTS_JSON:` line for tooling to consume.
  - `scripts/check_regression.py` — fails CI if the latest run's F1 drops
    >2 points below the best historical run *on the same benchmark*
    (matched on `kaggle_slug` + `videos_evaluated`, not `dataset_name`/
    `subset` — those are labeled inconsistently across runs in practice).
  - `scripts/run_eval.py` — added a fixed-file downloader
    (`download_subset_files`) so CI fetches only the 15 videos the
    existing baseline (F1=84.6%) was measured on, instead of the full
    16GB Le2i dataset (or even the full 3.1GB Coffee_room_01 subset,
    which turned out to have 48 videos, not 15 — see Gotchas).
  - `.github/workflows/eval.yml` — runs the above on every PR into master.
  - `dashboard/app.py` — new "Eval Metrics" page: F1/precision/recall
    trend, pass/fail vs. ROADMAP.md targets, miss-category breakdown.
  - `ROADMAP.md` tech debt table — noted that gating on F1 (equal
    precision/recall weight) could block a change that trades precision
    for recall on a feature where recall is explicitly prioritized (bed
    exit wants >95% recall specifically).
- **2026-07-01: reconciled `origin`/`upstream` divergence.** PR #1 merged
  into `upstream/master` while `origin/master` only had this doc's initial
  commit. Merged `upstream/master` into `origin/master` (clean, no
  conflicts — verified by diffing final file contents, not just trusting
  the merge commit). `origin/master` now matches `upstream/master` plus
  this doc.

## Known Gotchas

- **`pull_request`-triggered GitHub Actions workflows don't run on the PR
  that introduces them** — they use the base branch's copy of the
  workflow file. Bit everyone on `eval.yml`/PR #1 (see above).
- **Le2i's real Kaggle layout doesn't match what `evaluate.py` expects.**
  Kaggle has `{subset}/{subset}/Videos/*.avi` and
  `{subset}/{subset}/Annotation_files/*.txt` (double-nested, videos in a
  subfolder); `evaluate.py`'s scan expects a flat `{subset}/*.avi` +
  `{subset}/Annotation_files/*.txt`. `download_subset_files()` in
  `run_eval.py` downloads into the flat structure to paper over this —
  don't be surprised if a manual `kaggle datasets download -f ...` lands
  somewhere evaluate.py won't find it.
- **Kaggle's single-file `-f` download wraps the file in a `.zip`** even
  for a single `.avi`/`.txt`, and doesn't reliably honor `--unzip` for
  single-file downloads — `_fetch_single_file()` in `run_eval.py` extracts
  manually.
- **Coffee_room_01 actually has 48 videos on Kaggle**, not 15 — the
  original manual baseline was run against an incomplete local copy.
  `CI_BENCHMARK_VIDEOS` in `run_eval.py` deliberately pins to videos 1-15
  to stay comparable to that baseline; don't casually "fix" this to use
  all 48 without also deciding whether to re-baseline `all_runs.csv`.
- **`check_regression.py` assumes the physically last row in
  `all_runs.csv` is "this run"** — it's meant to run immediately after
  `run_eval.py` appends a fresh row in the same job, not as a standalone
  historical query. Running it manually against the full committed
  history will check whatever run happens to be last in the file (which
  may not be the benchmark you care about) — it's working as intended, not
  a bug, but confusing if you forget this.
- **There's an untracked nested `carewatchai/` directory** at the repo
  root (its own git repo, `?? carewatchai/` in git status) — unexplained,
  not touched, not part of any of the above work. Ask before deleting or
  investigating; could be someone's in-progress clone.
- **No `gh` CLI on this machine**, and Homebrew is broken here (unsupported
  macOS version for this brew install) — PRs get pushed via `git push` and
  opened manually via the compare-link GitHub prints, not `gh pr create`.

## Where to Look

- Product strategy, feature priority, target metrics: `ROADMAP.md`
- Eval pipeline (FSM path): `src/evaluate.py` (per-video),
  `scripts/run_eval.py` (batch runner + Kaggle download),
  `scripts/check_regression.py` (gate)
- Eval pipeline (ML path): `src/ml_fall_fsm.py`, `src/evaluate_multicam.py`,
  `scripts/train_fall_classifier.py`
- Runtime detection pipeline: `src/pipeline.py`, `src/fall_fsm.py`
- Dashboard: `dashboard/app.py`
