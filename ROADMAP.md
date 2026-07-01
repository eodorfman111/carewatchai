# CareWatch AI — Strategy & Roadmap

> Living document. Update this as we learn, ship, and talk to customers.
> Last updated: June 2026

---

## What We're Building

AI safety monitoring for senior living facilities (40–80 residents) using their existing IP cameras. Runs entirely on-device — no cloud, no video storage, no wearables. Staff get real-time alerts on a dashboard when something is wrong.

**The core insight:** every competitor either stores video in the cloud, requires new hardware, or both. We don't. That's what gets us in the door.

---

## Competitive Landscape

### Who we actually compete with (senior care)

| Competitor | Approach | Weakness vs. us |
|---|---|---|
| **SafelyYou** | AI camera, $100M+ funded | Cloud-dependent, video leaves facility, expensive |
| **KamiCare (Kami Vision)** | AI camera | Cloud video storage, privacy concerns |
| **Vayyar** | 4D radar sensors | No visual context, requires new hardware install |
| **Nobi** | Smart lamp per room | $6K/room/year, requires hardware everywhere |
| **AltumView (Sentinare)** | Stick-figure privacy cam | Requires their proprietary camera hardware |
| **EyeWatch LIVE** | Human + AI monitoring | Humans in the loop = expensive, not scalable |

### Who we do NOT compete with (different market)

**i3 International** — their slip/fall detection targets Walmart, Burger King, Kroger. It's a liability protection tool for retailers, not a care tool for seniors. Completely different buyer, different problem, different sales motion.

### Our differentiation (in order of importance)
1. **On-device inference** — video never leaves the building. HIPAA-friendly by architecture.
2. **No new hardware** — works on whatever IP cameras the facility already has.
3. **Published pricing** — only competitor with a public price ($1,000–1,500/month flat). Builds trust with small operators who've been burned by enterprise sales.
4. **Pre-incident detection** — we're building toward detecting events *before* a fall, not after. Bed exit, agitation, elopement. Competitors mostly detect the fall itself.

---

## Detection Use Cases — Prioritized

### Tier 1: Ship now / already in pipeline

**1. Fall detection** ✅ *Working*
- Body axis angle + hip position, confirmed over 1.5 seconds via FSM
- Tested on AI-generated CCTV footage — fires correctly in ~3 seconds
- Next: run Le2i evaluation to get precision/recall/F1 baseline

**2. Prolonged inactivity** ✅ *Working, needs tuning*
- Zone-aware stillness timer (floor: 10s, chair: 5min, bed: 10min)
- Known issue: fires too fast for people standing still in hallways (300-frame threshold too aggressive)
- Fix: raise floor threshold, add velocity check before triggering

**3. Nighttime wandering** ✅ *Working*
- Zone crossing detection between 10pm–6am
- Good for memory care where residents leave their room at night

**4. Restricted zone breach** ✅ *Working*
- Polygon zones per camera, time-gated
- Useful for medication rooms, stairwells, utility areas

---

### Tier 2: High priority — build next

**5. Bed exit / Out-of-Bed (OOB) detection** 🔴 *Highest priority gap*

*Why:* This is the #1 feature facilities ask about after fall detection. SafelyYou and KamiCare both lead with it. More importantly it's *preventive* — catching a resident getting out of bed at 3am lets staff intervene before a fall happens. The value story shifts from "we documented your fall" to "we prevented the fall."

*How to build:* We already have the pose keypoints. Bed exit is a lying→sitting→standing transition:
- Frame 0: person horizontal (body axis angle ~0°, hip_y high)
- Frame N: person vertical (body axis angle ~90°, hip_y low)
- Trigger: transition happens in <30 seconds, during nighttime hours
- The `fall_fsm.py` logic is actually close — we need an inverse FSM (LYING → RISING → STANDING)

*Dataset:* No clean Kaggle dataset. Use:
- [Elderly Activity Dataset](https://www.kaggle.com/datasets/saswatsethda/elderly-activity-dataset) — contains lying/sitting/standing
- [Activity Recognition with Healthy Older People](https://www.kaggle.com/datasets/marklvl/activity-recognition-with-healthy-older-people)
- Our own `generate_test_video.py` to create synthetic bed exit videos

*Files to create/modify:*
- `src/bed_exit_fsm.py` — new FSM mirroring fall_fsm.py but for lying→standing
- `src/pipeline.py` — add bed exit detection alongside fall detection

---

**6. Elopement detection** 🔴 *Critical for memory care*

*Why:* Memory care facilities get fined and face lawsuits when dementia residents leave the building unsupervised. Elopement is distinct from wandering — it's specifically about a resident reaching/breaching an exit. Facilities pay a lot for wander management systems (door sensors, RFID) — we can replace that with a camera.

*How to build:* We already have zone breach detection. Elopement is:
- Zone = exit door polygon
- Alert fires when resident enters the exit zone AND is moving toward it (velocity vector pointing at door)
- Higher urgency than standard zone breach — different alert type, sound alarm not just dashboard notification
- Should override nighttime hours check (elopement is dangerous 24/7)

*Implementation:* Extend `zone_manager.py` with an `is_elopement_zone()` flag in `zones.json`. Add elopement as a separate incident type in `incident_log.py`.

---

**7. Agitation detection** 🟡 *High value for memory care*

*Why:* Agitation is an early warning sign before a behavioral incident (resident-to-resident aggression, self-harm, falls caused by confusion). Staff catching agitation early can de-escalate. No competitor currently offers this.

*How to build:* Agitation manifests as:
- Rapid arm movement (high variance in wrist keypoint velocity)
- Pacing (high centroid displacement, short zone-crossing intervals)
- Posture changes (repeated standing/sitting cycles)
- We already track gait speed in `gait_tracker.py` — agitation detection extends this

*Dataset:* 
- Search Kaggle for "Real Life Violence Situations Dataset" — contains pre-aggression behavior
- [Human Activity Recognition Video Dataset](https://www.kaggle.com/datasets/sharjeelmazhar/human-activity-recognition-video-dataset)

*Files to create:*
- `src/agitation_detector.py` — velocity variance + pacing pattern detection

---

### Tier 3: Build after Tier 2

**8. Staff response time tracking**

*Why:* Facilities need to prove to regulators that staff responded within X minutes of an incident. We already log incidents with timestamps. If we can detect when a second person (staff) enters the camera frame after an alert, we can automatically log response time. This turns into a compliance report that facilities desperately need.

*How to build:* Multi-person tracking is already working (ByteTrack). After a fall/bed exit alert for Track ID X, monitor how long until a second tracked person (staff) comes within 1 meter of Track X.

**9. Distress posture detection**

*Why:* Choking, clutching chest, sudden collapse that isn't a full fall. Supplements fall detection for medical events.

*How to build:* Specific keypoint patterns — hands-to-throat (wrist keypoints near neck keypoints), doubled-over posture (shoulder keypoints below hip keypoints). Add as a new alert type in `pipeline.py`.

**10. Crowding / density detection**

*Why:* Infection control in dining rooms, common areas. Also useful for detecting when a resident is isolated (opposite of crowding) — alone in a room for too long triggers inactivity, but crowding in a hallway could indicate an incident.

*How to build:* Count active track IDs per zone. Alert if count exceeds threshold for the zone type.

---

## Evaluation & Benchmarking

### Current benchmark: Le2i dataset
- 143 annotated fall videos, 4 environments (coffee room, home, office, lecture room)
- Ground truth: frame-level fall start/end annotations
- Metrics: Precision, Recall, F1, Avg detection latency
- Run: `python scripts/run_eval.py`
- Results saved to: `data/eval_results/all_runs.csv`

### Datasets to add as we build new features

| Feature | Dataset | Kaggle slug |
|---|---|---|
| Fall detection | Le2i | `tuyenldvn/falldataset-imvia` |
| Bed exit | Elderly Activity | `saswatsethda/elderly-activity-dataset` |
| Activity recognition | HAR Video | `sharjeelmazhar/human-activity-recognition-video-dataset` |
| Older adults activity | Healthy Older People | `marklvl/activity-recognition-with-healthy-older-people` |
| Agitation | Real Life Violence | Search Kaggle |

### What good looks like
- Fall detection: >90% precision, >85% recall, <2s average latency
- Bed exit: >95% recall (missing one is worse than a false alarm at 3am)
- Elopement: >99% recall (zero tolerance for misses)

---

## Known Issues / Tech Debt

| Issue | Impact | Fix |
|---|---|---|
| Inactivity threshold fires too fast in hallways | High false positive rate | Raise floor threshold, add velocity check |
| Pink/green YOLO color cycling bleeds through overlay | Visual quality | Force all colors through our state-based overlay only |
| No RTSP stream tested on real IP camera | Unvalidated for real deployment | Test against a real camera ASAP |
| Dashboard CSS fighting Streamlit default theme | Looks unpolished | Override Streamlit theme in `.streamlit/config.toml` |
| evaluate.py only tested on synthetic runs | F1 score unvalidated | Run Le2i eval and record baseline |
| CI regression gate (`scripts/check_regression.py`) blocks on F1 dropping, but F1 weights precision/recall equally while our own targets don't (bed exit wants >95% recall specifically, "missing one is worse than a false alarm at 3am") | A change that trades some precision for meaningfully better recall — a genuinely good tradeoff for a safety product — could still fail the gate | Gate on recall directly (per feature target) instead of/in addition to F1, once bed exit and elopement have their own eval scripts |

---

## Next Steps (ordered)

1. **Get baseline F1** — run `python scripts/run_eval.py` on Le2i, record results in `data/eval_results/`
2. **Fix inactivity threshold** — tune `inactivity_timer.py` to reduce hallway false positives
3. **Build bed exit FSM** — new `src/bed_exit_fsm.py`, wire into pipeline
4. **Build elopement detection** — extend zone manager with elopement zone type
5. **Test on real RTSP stream** — borrow an IP camera or use a phone as RTSP source
6. **Build agitation detector** — `src/agitation_detector.py`
7. **Staff response time logging** — extend `incident_log.py`
8. **Talk to a facility** — everything above should be validated with a real operator before we build more

---

## Business Context

- **Target customer:** Small independent assisted living and memory care, 40–80 residents
- **Pricing:** ~$1,000–1,500/month flat per facility
- **Sales motion:** Facilities are tired of enterprise sales with hidden pricing. Lead with transparent pricing and a 30-day trial.
- **Regulatory angle:** Falls are the #1 cause of injury-related deaths in adults 65+. CMS (Medicare/Medicaid) requires facilities to have fall prevention programs. We make compliance easier to document.
- **First real customer profile:** Director of nursing or administrator at an independent memory care facility. They don't have a tech team. The product needs to work out of the box.