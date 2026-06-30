# CareWatch AI

Real-time overhead AI monitoring for senior living facilities. Detects falls, prolonged inactivity, nighttime wandering, and restricted zone breaches using existing IP cameras.

**No wearables. No cloud. No video leaves the building.**

## What it does

- 🔴 **Fall detection** — body axis angle + hip position, confirmed over 1.5 seconds
- 🟡 **Prolonged inactivity** — zone-aware stillness timer (floor: 10s, chair: 5min, bed: 10min)
- 🔵 **Nighttime wandering** — motion detection between 10pm–6am
- 🟠 **Restricted zone breach** — polygon zones per camera, time-gated

## Development in GitHub Codespaces

All development happens in Codespaces — no large video files on your local machine.

### New Codespace

Dependencies install automatically via `.devcontainer/devcontainer.json`. Once the Codespace finishes loading:

```bash
# 1. Add your Kaggle token to GitHub Secrets (one-time, do this once ever):
#    Your fork → Settings → Secrets and variables → Codespaces → New secret
#    Name: KAGGLE_API_TOKEN  Value: your_token

# 2. Download Le2i dataset + run full evaluation
python scripts/run_eval.py

# 3. Or just test the pipeline on the sample videos (no download needed)
python src/pipeline.py --source data/videos/sample_1.mp4 --camera hallway_01
```

### Existing Codespace (resuming work)

Videos are still present — no re-download needed:

```bash
# Run evaluation (skips download since data is already there)
python scripts/run_eval.py --skip-download

# Or test pipeline on sample video
python src/pipeline.py --source data/videos/sample_1.mp4 --camera hallway_01
```

### Pushing changes

```bash
git add .
git commit -m "describe what you changed"
git push
```

Then open a pull request on GitHub to merge into the main repo.

> **Tip:** Suspend your Codespace (just close the tab) rather than deleting it. Videos persist for up to 30 days, saving you the re-download.

---

## Stack

- YOLO26-pose / YOLOv11-pose — 17-point skeleton keypoints
- ByteTrack — multi-person tracking
- OpenCV — video processing
- Streamlit — staff dashboard + marketing site
- SQLite — incident logging

## Run locally

```bash
pip install -r requirements.txt
python src/pipeline.py --source your_video.mp4 --camera cam_01
python -m streamlit run dashboard/app.py --server.port 8501
python -m streamlit run website/app.py --server.port 8502
```

## Project structure
src/

pipeline.py           — main inference loop

fall_fsm.py           — fall detection state machine

inactivity_timer.py   — stillness detection

wandering_detector.py — nighttime wandering

zone_manager.py       — polygon zone logic

pose_utils.py         — keypoint math

gait_tracker.py       — walking speed estimation

overlay.py            — visualization

incident_log.py       — SQLite logging

evaluate.py           — Le2i dataset evaluator

scripts/

run_eval.py           — download Le2i + run evaluation

dashboard/

app.py                — staff alert dashboard

website/

app.py                — marketing site

data/

zones/zones.json      — camera zone configuration

videos/               — gitignored, download via run_eval.py


## Live demo

[carewatchai.com](https://carewatchai.com)
