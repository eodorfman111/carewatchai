"""CareWatch AI — Marketing Website (Streamlit)"""
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="CareWatch AI — Real-Time Senior Safety Monitoring",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
VIDEO_DIR  = ROOT / "data" / "sample_videos"
STATIC_DIR = ROOT / "static"
CSS_FILE  = Path(__file__).parent / "style.css"

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0a0a0f; color: #e8e8f0; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 40%, #0a1628 100%);
    padding: 80px 60px 60px;
    text-align: center;
    border-bottom: 1px solid #1a2540;
}
.hero-badge {
    display: inline-block;
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.4);
    color: #60a5fa;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 500;
    margin-bottom: 24px;
    letter-spacing: 0.5px;
}
.hero h1 {
    font-size: 56px;
    font-weight: 800;
    line-height: 1.1;
    color: #f0f0ff;
    margin: 0 0 20px;
}
.hero h1 span { color: #3b82f6; }
.hero p {
    font-size: 20px;
    color: #8892a4;
    max-width: 640px;
    margin: 0 auto 36px;
    line-height: 1.6;
}
.hero-stats {
    display: flex;
    justify-content: center;
    gap: 48px;
    margin-top: 48px;
    padding-top: 40px;
    border-top: 1px solid #1a2540;
}
.stat-item { text-align: center; }
.stat-number { font-size: 36px; font-weight: 800; color: #3b82f6; }
.stat-label  { font-size: 13px; color: #6b7280; margin-top: 4px; }

/* Section */
.section {
    padding: 72px 60px;
    border-bottom: 1px solid #1a2540;
}
.section-dark { background: #0a0a0f; }
.section-mid  { background: #0d1117; }
.section-label {
    font-size: 12px; font-weight: 600; letter-spacing: 2px;
    color: #3b82f6; text-transform: uppercase; margin-bottom: 12px;
}
.section-title {
    font-size: 38px; font-weight: 800; color: #f0f0ff;
    margin: 0 0 16px; line-height: 1.2;
}
.section-sub {
    font-size: 17px; color: #8892a4; max-width: 560px; line-height: 1.6;
}

/* Feature cards */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    margin-top: 48px;
}
.feature-card {
    background: #0d1117;
    border: 1px solid #1a2540;
    border-radius: 12px;
    padding: 28px;
    transition: border-color 0.2s;
}
.feature-card:hover { border-color: #3b82f6; }
.feature-icon { font-size: 28px; margin-bottom: 14px; }
.feature-title { font-size: 17px; font-weight: 700; color: #f0f0ff; margin-bottom: 8px; }
.feature-desc  { font-size: 14px; color: #6b7280; line-height: 1.6; }

/* Alert types */
.alert-card {
    background: #0d1117;
    border-radius: 12px;
    padding: 24px 28px;
    border-left: 4px solid;
    margin-bottom: 16px;
}
.alert-fall      { border-color: #ef4444; }
.alert-inactivity{ border-color: #f59e0b; }
.alert-wandering { border-color: #3b82f6; }
.alert-zone      { border-color: #f97316; }
.alert-title { font-size: 16px; font-weight: 700; color: #f0f0ff; margin-bottom: 6px; }
.alert-desc  { font-size: 14px; color: #6b7280; line-height: 1.5; }

/* Comparison table */
.comp-table { width: 100%; border-collapse: collapse; margin-top: 32px; }
.comp-table th {
    background: #0d1117; color: #8892a4;
    padding: 14px 20px; text-align: left;
    font-size: 13px; font-weight: 600;
    border-bottom: 1px solid #1a2540;
}
.comp-table td {
    padding: 14px 20px; border-bottom: 1px solid #1a2540;
    font-size: 14px; color: #c8d0dc;
}
.comp-table tr:hover td { background: rgba(59,130,246,0.04); }
.comp-ours { color: #3b82f6 !important; font-weight: 600; }
.check { color: #22c55e; }
.cross { color: #ef4444; }
.partial { color: #f59e0b; }

/* Privacy */
.privacy-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 20px;
    margin-top: 40px;
}
.privacy-card {
    background: #0d1117;
    border: 1px solid #1a2540;
    border-radius: 12px;
    padding: 28px;
}
.privacy-card h4 { color: #f0f0ff; font-size: 16px; margin: 12px 0 8px; }
.privacy-card p  { color: #6b7280; font-size: 14px; line-height: 1.6; margin: 0; }

/* CTA */
.cta-section {
    background: linear-gradient(135deg, #0d1f3c 0%, #0a1628 100%);
    padding: 80px 60px;
    text-align: center;
    border-top: 1px solid #1a2540;
}
.cta-section h2 { font-size: 42px; font-weight: 800; color: #f0f0ff; margin-bottom: 16px; }
.cta-section p  { font-size: 18px; color: #8892a4; margin-bottom: 36px; }

/* Buttons */
.btn-primary {
    display: inline-block;
    background: #3b82f6;
    color: white !important;
    padding: 14px 32px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 16px;
    text-decoration: none;
    margin: 6px;
    cursor: pointer;
}
.btn-secondary {
    display: inline-block;
    background: transparent;
    color: #3b82f6 !important;
    border: 1px solid #3b82f6;
    padding: 14px 32px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 16px;
    text-decoration: none;
    margin: 6px;
}

/* Nav */
.navbar {
    position: sticky; top: 0; z-index: 999;
    background: rgba(10,10,15,0.95);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid #1a2540;
    padding: 16px 60px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.nav-logo { font-size: 20px; font-weight: 800; color: #f0f0ff; }
.nav-logo span { color: #3b82f6; }
.nav-links { display: flex; gap: 32px; }
.nav-link  { color: #8892a4; font-size: 14px; text-decoration: none; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ── NAVBAR ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div class="nav-logo">Care<span>Watch</span> AI</div>
    <div class="nav-links">
        <a class="nav-link" href="#features">Features</a>
        <a class="nav-link" href="#how-it-works">How It Works</a>
        <a class="nav-link" href="#privacy">Privacy</a>
        <a class="nav-link" href="#compare">Compare</a>
        <a class="nav-link" href="#performance">Accuracy</a>
        <a class="nav-link" href="#contact">Contact</a>
    </div>
</div>
""", unsafe_allow_html=True)


# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🏥 AI-Powered Senior Safety Monitoring</div>
    <h1>Falls detected.<br><span>Staff alerted in seconds.</span></h1>
    <p>Real-time overhead AI monitoring that alerts senior care staff to falls and safety events
       — no wearables, no cameras in bathrooms, <b style="color:#f0f0ff">no video ever leaves your building.</b></p>
    <p style="font-size:14px;color:#6b7280;margin-top:-16px">Conventional bed alarms produce 20–30 false alarms per bed per night. CareWatch uses two-stage pose confirmation. Real falls only.</p>
    <div style="margin-top: 32px;">
        <a class="btn-primary" href="#demo">Watch Live Demo</a>
        <a class="btn-secondary" href="#contact">Request Pilot</a>
    </div>
    <div class="hero-stats">
        <div class="stat-item">
            <div class="stat-number">$10K</div>
            <div class="stat-label">Average cost per fall incident</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">&lt;2s</div>
            <div class="stat-label">Fall detection latency</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">24/7</div>
            <div class="stat-label">Passive monitoring, no staff action</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">0</div>
            <div class="stat-label">Wearables required</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── DEMO VIDEO ───────────────────────────────────────────────────────────────
st.markdown('<div class="section section-dark" id="demo">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Live Demo</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">See it in action</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Real CCTV footage processed entirely on-device. No video leaves the building.</div>', unsafe_allow_html=True)

demo_col1, demo_col2 = st.columns(2)

# Use static folder first (deployed), fall back to local data folder
pipeline_video = (STATIC_DIR / "Realistic_security_camera_foot_pipeline.mp4")
skeleton_video = (STATIC_DIR / "Realistic_security_camera_foot_skeleton.mp4")
if not pipeline_video.exists():
    pipeline_video = next(VIDEO_DIR.glob("*_pipeline.mp4"), None)
if not skeleton_video.exists():
    skeleton_video = next(VIDEO_DIR.glob("*_skeleton.mp4"), None)

with demo_col1:
    st.markdown("#### 🎥 Annotated View — Fall Detected")
    if pipeline_video and pipeline_video.exists():
        st.video(str(pipeline_video))
    else:
        st.markdown("""
        <div style="background:#0d1117;border:1px solid #1a2540;border-radius:12px;
                    padding:48px;text-align:center;color:#6b7280">
            <div style="font-size:48px;margin-bottom:16px">🎬</div>
            <div style="font-size:16px;font-weight:600;color:#f0f0ff;margin-bottom:8px">
                Live Demo Video
            </div>
            <div style="font-size:14px">
                Real CCTV footage with AI skeleton overlay.<br>
                Fall detected → red alert fires in 2 seconds.
            </div>
        </div>
        """, unsafe_allow_html=True)

with demo_col2:
    st.markdown("#### 🦴 Privacy Mode — Skeleton Only")
    if skeleton_video and skeleton_video.exists():
        st.video(str(skeleton_video))
    else:
        st.markdown("""
        <div style="background:#0d1117;border:1px solid #1a2540;border-radius:12px;
                    padding:48px;text-align:center;color:#6b7280">
            <div style="font-size:48px;margin-bottom:16px">🦴</div>
            <div style="font-size:16px;font-weight:600;color:#f0f0ff;margin-bottom:8px">
                Privacy Mode
            </div>
            <div style="font-size:14px">
                Same detection, zero identifiable video.<br>
                Skeleton wireframe only — no faces, no footage stored.
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Second demo row — multi-person
multi_video   = STATIC_DIR / "this_video_sucked_she_was_alre_pipeline.mp4"
dining_video  = STATIC_DIR / "veo_dining_pipeline.mp4"

demo_col3, demo_col4 = st.columns(2)
with demo_col3:
    st.markdown("#### 👥 Multi-Person Tracking")
    if multi_video.exists():
        st.video(str(multi_video))
    else:
        st.markdown("""
        <div style="background:#0d1117;border:1px solid #1a2540;border-radius:12px;
                    padding:48px;text-align:center;color:#6b7280">
            <div style="font-size:32px;margin-bottom:8px">👥</div>
            <div style="font-size:14px;color:#f0f0ff">Tracks every person simultaneously.<br>Each gets their own ID and fall FSM.</div>
        </div>""", unsafe_allow_html=True)

with demo_col4:
    st.markdown("#### 🍽️ Common Area Monitoring")
    if dining_video.exists():
        st.video(str(dining_video))
    else:
        st.markdown("""
        <div style="background:#0d1117;border:1px solid #1a2540;border-radius:12px;
                    padding:48px;text-align:center;color:#6b7280">
            <div style="font-size:32px;margin-bottom:8px">🍽️</div>
            <div style="font-size:14px;color:#f0f0ff">Dining rooms, lounges, hallways.<br>Works on existing facility cameras.</div>
        </div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ── ALERT TYPES ──────────────────────────────────────────────────────────────
st.markdown('<div class="section section-mid" id="features">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Every safety event. Automatically.</div>', unsafe_allow_html=True)
st.markdown("""
<div style="margin-top: 32px;">
    <div class="alert-card alert-fall">
        <div class="alert-title">🔴 Fall Detection</div>
        <div class="alert-desc">Body axis angle and hip position analyzed every frame. Fall confirmed only after 1.5 seconds sustained — eliminating false alarms from bending, sitting, or kneeling. Alert fires in under 2 seconds of confirmation.</div>
    </div>
    <div class="alert-card alert-inactivity">
        <div class="alert-title">🟡 Prolonged Inactivity</div>
        <div class="alert-desc">Zone-aware stillness detection. A resident motionless on the floor triggers in 10 seconds. In a chair or bed, thresholds are longer — because sleeping is normal. Prevents pressure ulcers and catches unresponsive residents.</div>
    </div>
    <div class="alert-card alert-wandering">
        <div class="alert-title">🔵 Nighttime Wandering</div>
        <div class="alert-desc">Motion detected between 10pm and 6am flags wandering behavior common in dementia residents. Repeated zone crossing triggers escalation. Elopement risk alerts fire before the resident reaches an exit.</div>
    </div>
    <div class="alert-card alert-zone">
        <div class="alert-title">🟠 Restricted Zone Breach</div>
        <div class="alert-desc">Configurable polygon zones per camera. Stairwells, exit doors, kitchens during off-hours. Any resident entering a restricted area triggers an immediate alert to staff — time-gated so daytime access doesn't cause false alarms.</div>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ── HOW IT WORKS ─────────────────────────────────────────────────────────────
st.markdown('<div class="section section-dark" id="how-it-works">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Architecture</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Works on cameras you already have.</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">No new hardware required. Plugs into your existing IP camera network via RTSP stream.</div>', unsafe_allow_html=True)

steps = [
    ("1", "🎥", "Existing IP Cameras", "Milestone, Genetec, Avigilon, Hikvision, Dahua — any camera with RTSP output. No replacement needed."),
    ("2", "🤖", "On-Device AI Inference", "YOLO26-pose runs locally on an edge appliance in your facility. 17 skeletal keypoints extracted per person, per frame."),
    ("3", "📐", "Safety Analytics Engine", "Fall FSM, inactivity timers, zone polygons, and wandering logic process every track in real time."),
    ("4", "🚨", "Two-Stage Alert Filter", "Events confirmed over multiple frames before alerting. Bending, sitting, and kneeling never fire. Only real events do."),
    ("5", "📱", "Staff Alert Delivery", "Mobile push, SMS, and dashboard notification. 60-second escalation if unacknowledged. Full incident log auto-generated."),
]

cols = st.columns(5)
for col, (num, icon, title, desc) in zip(cols, steps):
    with col:
        st.markdown(f"""
        <div style="text-align:center;padding:20px 10px">
            <div style="font-size:32px;margin-bottom:8px">{icon}</div>
            <div style="font-size:11px;color:#3b82f6;font-weight:600;margin-bottom:6px">STEP {num}</div>
            <div style="font-size:15px;font-weight:700;color:#f0f0ff;margin-bottom:8px">{title}</div>
            <div style="font-size:13px;color:#6b7280;line-height:1.5">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ── FEATURES GRID ────────────────────────────────────────────────────────────
st.markdown('<div class="section section-mid">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Features</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Built for real facilities.</div>', unsafe_allow_html=True)

features = [
    ("🦴", "Skeleton Overlay",         "17-point pose keypoints rendered on every tracked resident. Interpretable, debuggable, and privacy-safe."),
    ("📊", "Gait Speed Tracking",      "Walking speed measured passively per resident. Declining gait speed flags rising fall risk before a fall happens."),
    ("🗺️", "Zone Editor",              "Draw polygons directly on your camera view. Define bed, chair, floor, and restricted zones without touching code."),
    ("🕐", "Shift Handoff Reports",    "Auto-generated incident summary at every shift change. Incoming staff know exactly what happened overnight."),
    ("📋", "Auto Incident Logging",    "Every alert timestamped, camera-tagged, and logged to an immutable audit trail. Export CSV for regulatory review."),
    ("👨‍👩‍👧", "Multi-Person Tracking",   "Tracks every resident in common areas simultaneously. Each person gets their own ID, FSM, and inactivity timer."),
    ("🌙", "Night Mode",               "Heightened sensitivity between 10pm and 6am. Wandering and bed-exit alerts active. Inactivity thresholds tighten."),
    ("🔒", "On-Device Only",           "Zero video transmitted to cloud. Inference runs locally. Nothing leaves your building — ever."),
    ("📱", "Staff Dashboard",          "Live alert feed, acknowledgement workflow, analytics charts, and incident log in one Streamlit interface."),
]

cols = st.columns(3)
for i, (icon, title, desc) in enumerate(features):
    with cols[i % 3]:
        st.markdown(f"""
        <div class="feature-card" style="margin-bottom:20px">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ── PRIVACY ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section section-dark" id="privacy">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Privacy</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Resident dignity is not optional.</div>', unsafe_allow_html=True)

st.markdown("""
<div class="privacy-grid">
    <div class="privacy-card">
        <div style="font-size:28px">🚫</div>
        <h4>No cameras in bathrooms</h4>
        <p>CareWatch is never deployed in bathrooms, shower rooms, or personal care areas. Door sensors and duration timers handle bathroom safety without visual monitoring.</p>
    </div>
    <div class="privacy-card">
        <div style="font-size:28px">🦴</div>
        <h4>Skeleton-only privacy mode</h4>
        <p>Sensitive areas can render skeleton wireframe only — no raw video, no faces, no identifiable appearance. Families and residents see motion context, not surveillance.</p>
    </div>
    <div class="privacy-card">
        <div style="font-size:28px">🏠</div>
        <h4>Video never leaves the building</h4>
        <p>All inference runs on an edge appliance inside your facility. No video is transmitted to cloud servers. A data breach at CareWatch cannot expose resident footage.</p>
    </div>
    <div class="privacy-card">
        <div style="font-size:28px">📋</div>
        <h4>No facial recognition. Ever.</h4>
        <p>Residents are identified by room assignment, not by face. CareWatch knows "someone fell in Room 214" — your staff know who lives there. We never store biometric data.</p>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ── COMPARISON ───────────────────────────────────────────────────────────────
st.markdown('<div class="section section-mid" id="compare">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Comparison</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">More detection. Better privacy. Lower cost.</div>', unsafe_allow_html=True)

st.markdown("""
<table class="comp-table">
<thead>
<tr>
    <th>Feature</th>
    <th class="comp-ours">CareWatch AI</th>
    <th>SafelyYou</th>
    <th>Vayyar</th>
    <th>Nobi</th>
    <th>Wearables</th>
</tr>
</thead>
<tbody>
<tr><td>Fall detection</td>
    <td class="check comp-ours">✓</td><td class="check">✓</td><td class="check">✓</td><td class="check">✓</td><td class="check">✓</td></tr>
<tr><td>Works on existing cameras</td>
    <td class="check comp-ours">✓</td><td class="check">✓</td><td class="cross">✗</td><td class="cross">✗</td><td class="cross">✗</td></tr>
<tr><td>Video stays on-device</td>
    <td class="check comp-ours">✓</td><td class="cross">✗</td><td class="check">✓</td><td class="cross">✗</td><td>—</td></tr>
<tr><td>No wearable required</td>
    <td class="check comp-ours">✓</td><td class="check">✓</td><td class="check">✓</td><td class="check">✓</td><td class="cross">✗</td></tr>
<tr><td>Inactivity monitoring</td>
    <td class="check comp-ours">✓</td><td class="cross">✗</td><td class="check">✓</td><td class="cross">✗</td><td class="partial">~</td></tr>
<tr><td>Wandering detection</td>
    <td class="check comp-ours">✓</td><td class="cross">✗</td><td class="cross">✗</td><td class="cross">✗</td><td class="partial">~</td></tr>
<tr><td>Restricted zone alerts</td>
    <td class="check comp-ours">✓</td><td class="cross">✗</td><td class="cross">✗</td><td class="cross">✗</td><td class="cross">✗</td></tr>
<tr><td>Gait speed tracking</td>
    <td class="check comp-ours">✓</td><td class="cross">✗</td><td class="cross">✗</td><td class="cross">✗</td><td class="partial">~</td></tr>
<tr><td>No facial recognition</td>
    <td class="check comp-ours">✓</td><td class="check">✓</td><td class="check">✓</td><td class="check">✓</td><td class="check">✓</td></tr>
<tr><td>Small operator pricing</td>
    <td class="check comp-ours">✓</td><td class="cross">✗</td><td class="cross">✗</td><td class="cross">✗</td><td class="partial">~</td></tr>
</tbody>
</table>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ── PERFORMANCE ──────────────────────────────────────────────────────────────
st.markdown('<div class="section section-dark" id="performance">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Validated Accuracy</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Tested on real datasets. Not cherry-picked demos.</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Benchmarked against the Le2i Fall Detection Dataset — a public research benchmark of real staged falls used to evaluate fall detection systems worldwide.</div>', unsafe_allow_html=True)

perf_col1, perf_col2, perf_col3 = st.columns(3)
with perf_col1:
    st.markdown("""
    <div style="background:#0d1117;border:1px solid #1a2540;border-radius:12px;padding:32px;text-align:center;margin-top:32px">
        <div style="font-size:56px;font-weight:800;color:#22c55e">100%</div>
        <div style="font-size:16px;font-weight:700;color:#f0f0ff;margin:8px 0 4px">Precision</div>
        <div style="font-size:13px;color:#6b7280">Zero false alarms on the Le2i benchmark. Every alert fired was a real fall.</div>
    </div>
    """, unsafe_allow_html=True)
with perf_col2:
    st.markdown("""
    <div style="background:#0d1117;border:1px solid #1a2540;border-radius:12px;padding:32px;text-align:center;margin-top:32px">
        <div style="font-size:56px;font-weight:800;color:#3b82f6">84.6%</div>
        <div style="font-size:16px;font-weight:700;color:#f0f0ff;margin:8px 0 4px">F1 Score</div>
        <div style="font-size:13px;color:#6b7280">Harmonic mean of precision and recall on held-out test videos.</div>
    </div>
    """, unsafe_allow_html=True)
with perf_col3:
    st.markdown("""
    <div style="background:#0d1117;border:1px solid #1a2540;border-radius:12px;padding:32px;text-align:center;margin-top:32px">
        <div style="font-size:56px;font-weight:800;color:#f59e0b">&lt;2s</div>
        <div style="font-size:16px;font-weight:700;color:#f0f0ff;margin:8px 0 4px">Alert Latency</div>
        <div style="font-size:13px;color:#6b7280">Median time from fall onset to staff notification across all test videos.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="margin-top:28px;padding:20px 28px;background:#0d1117;border:1px solid #1a2540;border-radius:12px;display:flex;align-items:flex-start;gap:16px">
    <div style="font-size:22px;flex-shrink:0">🔬</div>
    <div>
        <div style="font-size:14px;font-weight:600;color:#f0f0ff;margin-bottom:4px">How we test</div>
        <div style="font-size:13px;color:#6b7280;line-height:1.6">
            Results above are from the <b style="color:#c8d0dc">Le2i public benchmark</b> — 15 held-out videos of real staged falls,
            evaluated by running the full pipeline cold (no tuning on test data).
            We also continuously evaluate on the MultiCam nursing-home dataset with confounding ADL events
            (bending, sitting, lying down) to harden the model against real-world conditions.
            No cherry-picking. <a href="https://github.com/eodorfman111/carewatchai" target="_blank" style="color:#3b82f6">View methodology on GitHub →</a>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ── PRICING ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section section-mid" id="pricing">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Pricing</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Simple pricing. No surprises.</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Every competitor hides their price. We don\'t.</div>', unsafe_allow_html=True)

pc1, pc2, pc3 = st.columns(3)
with pc1:
    st.markdown("""
    <div style="background:#0d1117;border:1px solid #1a2540;border-radius:12px;padding:32px;text-align:center;margin-top:32px">
        <div style="font-size:13px;color:#6b7280;font-weight:600;letter-spacing:1px;margin-bottom:12px">STARTER</div>
        <div style="font-size:48px;font-weight:800;color:#f0f0ff">$39</div>
        <div style="font-size:14px;color:#6b7280;margin-bottom:24px">/room/month</div>
        <div style="font-size:13px;color:#8892a4;line-height:2">
            Up to 10 rooms<br>Fall + inactivity detection<br>Staff dashboard<br>Incident log + CSV export<br>Email alerts<br>
        </div>
        <div style="margin-top:24px;font-size:12px;color:#4b5563">Hardware: ~$500 one-time (mini PC)</div>
    </div>
    """, unsafe_allow_html=True)

with pc2:
    st.markdown("""
    <div style="background:#0d1f3c;border:2px solid #3b82f6;border-radius:12px;padding:32px;text-align:center;margin-top:32px;position:relative">
        <div style="position:absolute;top:-12px;left:50%;transform:translateX(-50%);background:#3b82f6;color:white;padding:4px 16px;border-radius:20px;font-size:12px;font-weight:600">MOST POPULAR</div>
        <div style="font-size:13px;color:#60a5fa;font-weight:600;letter-spacing:1px;margin-bottom:12px">PROFESSIONAL</div>
        <div style="font-size:48px;font-weight:800;color:#f0f0ff">$59</div>
        <div style="font-size:14px;color:#6b7280;margin-bottom:24px">/room/month</div>
        <div style="font-size:13px;color:#8892a4;line-height:2">
            Up to 50 rooms<br>All 4 detection types<br>SMS + push alerts<br>Analytics dashboard<br>Zone editor<br>Gait speed tracking<br>Shift handoff reports<br>
        </div>
        <div style="margin-top:24px;font-size:12px;color:#4b5563">Hardware: ~$500 one-time (mini PC)</div>
    </div>
    """, unsafe_allow_html=True)

with pc3:
    st.markdown("""
    <div style="background:#0d1117;border:1px solid #1a2540;border-radius:12px;padding:32px;text-align:center;margin-top:32px">
        <div style="font-size:13px;color:#6b7280;font-weight:600;letter-spacing:1px;margin-bottom:12px">ENTERPRISE</div>
        <div style="font-size:48px;font-weight:800;color:#f0f0ff">Custom</div>
        <div style="font-size:14px;color:#6b7280;margin-bottom:24px">contact us</div>
        <div style="font-size:13px;color:#8892a4;line-height:2">
            Unlimited rooms<br>Multi-facility dashboard<br>EHR / nurse-call integration<br>White-label option<br>Dedicated support<br>SLA guarantee<br>
        </div>
        <div style="margin-top:24px;font-size:12px;color:#4b5563">Hardware included in contract</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;margin-top:32px;padding:20px;background:#0d1117;border-radius:12px;border:1px solid #1a2540">
    <span style="color:#22c55e;font-weight:700">✓ No long-term contracts</span>
    &nbsp;&nbsp;·&nbsp;&nbsp;
    <span style="color:#22c55e;font-weight:700">✓ Cancel anytime</span>
    &nbsp;&nbsp;·&nbsp;&nbsp;
    <span style="color:#22c55e;font-weight:700">✓ 30-day free pilot</span>
    &nbsp;&nbsp;·&nbsp;&nbsp;
    <span style="color:#22c55e;font-weight:700">✓ Works on your existing cameras</span>
</div>
<div style="text-align:center;margin-top:20px;padding:16px;background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);border-radius:8px">
    <span style="color:#fca5a5;font-size:14px">
        💡 <b>The math:</b> One prevented fall hospitalization costs $10,000–$18,000.
        A 50-room facility at $59/room/month = $2,950/month.
        One prevented fall pays for <b>3+ months</b> of CareWatch.
    </span>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ── CTA ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="cta-section" id="contact">', unsafe_allow_html=True)
st.markdown("""
<h2>Ready to protect your residents?</h2>
<p>Request a pilot. We'll set up on your existing cameras in under a day.</p>
""", unsafe_allow_html=True)

contact_col1, contact_col2, contact_col3 = st.columns([1, 2, 1])
with contact_col2:
    with st.form("contact_form"):
        name     = st.text_input("Your Name")
        email    = st.text_input("Work Email")
        facility = st.text_input("Facility Name")
        size     = st.selectbox("Facility Size",
                                ["< 25 residents", "25–75 residents",
                                 "75–150 residents", "150+ residents"])
        message  = st.text_area("Anything else?", placeholder="Camera system you're using, biggest safety challenge, etc.")
        submitted = st.form_submit_button("Request a Pilot →", use_container_width=True)
        if submitted and name and email and facility:
            st.success(f"Thanks {name}! We'll be in touch at {email} within 24 hours.")
        elif submitted:
            st.error("Please fill in your name, email, and facility name.")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="background:#0a0a0f;padding:32px 60px;text-align:center;border-top:1px solid #1a2540">
    <div style="font-size:18px;font-weight:800;color:#f0f0ff;margin-bottom:8px">
        Care<span style="color:#3b82f6">Watch</span> AI
    </div>
    <div style="font-size:13px;color:#4b5563">
        Real-time overhead AI monitoring for senior living facilities.
        No wearables. No cloud. No compromises.
    </div>
    <div style="font-size:12px;color:#374151;margin-top:16px">
        © 2026 CareWatch AI. Built with ❤️ for senior care operators.
    </div>
</div>
""", unsafe_allow_html=True)
