"""CareWatch AI — Streamlit dashboard."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import time
import json
import sqlite3
import streamlit as st
import pandas as pd
from collections import Counter
from datetime import datetime, timedelta
from incident_log import recent_incidents, acknowledge, init_db

EVAL_RESULTS_DIR = Path(__file__).parent.parent / "data" / "eval_results"
ALL_RUNS_CSV     = EVAL_RESULTS_DIR / "all_runs.csv"

# Mirrors "What good looks like" in ROADMAP.md — update both together.
EVAL_TARGETS = {
    "Fall detection": {"precision": 90.0, "recall": 85.0, "dataset_name_prefix": None},
    "Bed exit":        {"precision": None, "recall": 95.0, "dataset_name_prefix": "BedExit"},
    "Elopement":       {"precision": None, "recall": 99.0, "dataset_name_prefix": "Elopement"},
}

st.set_page_config(
    page_title="CareWatch AI",
    page_icon="🏥",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #080810 !important;
    border-right: 1px solid #1a2540;
}
[data-testid="stSidebar"] * { color: #c8d0dc !important; }

/* Main background */
.main { background: #0a0a0f; }
.block-container { padding: 2rem 2rem 2rem !important; }

/* Hide streamlit chrome */
#MainMenu, footer { visibility: hidden; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #0d1117;
    border: 1px solid #1a2540;
    border-radius: 10px;
    padding: 16px 20px;
}

/* Alert cards */
.alert-row {
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
    border-left: 4px solid;
    font-family: 'Inter', sans-serif;
}

/* Buttons */
[data-testid="stButton"] button {
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    border: 1px solid #1a2540 !important;
    background: #0d1117 !important;
    color: #c8d0dc !important;
    transition: all 0.15s !important;
}
[data-testid="stButton"] button:hover {
    border-color: #3b82f6 !important;
    color: #3b82f6 !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: #0d1117 !important;
    border: 1px solid #1a2540 !important;
    border-radius: 8px !important;
}

/* Divider */
hr { border-color: #1a2540 !important; }

/* Title */
h1 { color: #f0f0ff !important; font-weight: 800 !important; }
h2, h3 { color: #e0e0f0 !important; font-weight: 700 !important; }

/* Radio buttons in sidebar */
[data-testid="stRadio"] label { color: #8892a4 !important; }

/* Charts */
[data-testid="stArrowVegaLiteChart"] { background: #0d1117 !important; border-radius: 8px; }

/* Dataframe */
[data-testid="stDataFrame"] { background: #0d1117 !important; }

/* Logo in sidebar */
.sidebar-logo {
    font-size: 22px; font-weight: 800; color: #f0f0ff;
    padding: 8px 0 16px; letter-spacing: -0.5px;
}
.sidebar-logo span { color: #3b82f6; }
</style>
""", unsafe_allow_html=True)

EVENT_ICONS = {
    "fall":            "🔴 FALL DETECTED",
    "inactivity":      "🟡 PROLONGED INACTIVITY",
    "restricted_zone": "🟠 RESTRICTED ZONE",
    "wandering":       "🔵 WANDERING",
}

EVENT_COLORS = {
    "fall":            "#ff4444",
    "inactivity":      "#ffaa00",
    "restricted_zone": "#ff8800",
    "wandering":       "#4488ff",
}

init_db()

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">Care<span>Watch</span> AI</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px;color:#4b5563;margin-bottom:20px">Real-Time Senior Safety Monitor</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigate", ["🚨 Live Alerts", "📊 Analytics", "📈 Eval Metrics", "🗺️ Zone Editor", "📋 Incident Log"])
    st.markdown("---")
    auto_refresh = st.checkbox("Auto-refresh (5s)", value=False)
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.rerun()
    st.markdown("---")
    st.markdown('<div style="font-size:11px;color:#374151;line-height:1.8">✓ On-device inference<br>✓ No video leaves building<br>✓ HIPAA-ready architecture</div>', unsafe_allow_html=True)

rows = recent_incidents(limit=500)
df_all = pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()

if not df_all.empty:
    df_all["datetime"] = pd.to_datetime(df_all["timestamp"], unit="s")
    df_all["date"]     = df_all["datetime"].dt.date
    df_all["hour"]     = df_all["datetime"].dt.hour
    df_all["time_str"] = df_all["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_all["pending"]  = df_all["ack_time"].isna()


# ════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE ALERTS
# ════════════════════════════════════════════════════════════════════════════
if page == "🚨 Live Alerts":
    st.title("CareWatch AI — Live Alert Feed")

    unacked = df_all[df_all["pending"]] if not df_all.empty else pd.DataFrame()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔴 Active Alerts", len(unacked))
    col2.metric("📋 Total Today",
                len(df_all[df_all["date"] == datetime.now().date()]) if not df_all.empty else 0)
    col3.metric("✅ Resolved Today",
                len(df_all[(df_all["date"] == datetime.now().date()) & (~df_all["pending"])]) if not df_all.empty else 0)

    if not df_all.empty and not unacked.empty:
        avg_resp = df_all.dropna(subset=["ack_time"])
        if not avg_resp.empty:
            avg_s = (avg_resp["ack_time"] - avg_resp["timestamp"]).mean()
            col4.metric("⚡ Avg Response", f"{avg_s:.0f}s")

    st.divider()

    if unacked.empty:
        st.success("✅ No active alerts — all clear")
    else:
        st.error(f"⚠️  {len(unacked)} alert(s) require acknowledgement")

    # Show unacked first, then recent acked
    display_rows = rows[:50]
    for r in display_rows:
        ev   = r["event_type"]
        icon = EVENT_ICONS.get(ev, ev.upper())
        col  = EVENT_COLORS.get(ev, "#888888")
        ts   = datetime.fromtimestamp(r["timestamp"]).strftime("%H:%M:%S")
        pending = r["ack_time"] is None

        border = f"border-left: 4px solid {col};"
        bg     = "background:#1a1a1a;" if pending else "background:#111111;"
        st.markdown(
            f'<div style="{border}{bg}padding:8px 12px;margin:4px 0;border-radius:4px">'
            f'<b style="color:{col}">{icon}</b> &nbsp; '
            f'Camera <b>{r["camera_id"]}</b> | Track <b>{r["track_id"]}</b> | '
            f'Zone <b>{r["zone"]}</b> | <span style="color:#aaa">{ts}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if pending:
            b1, b2, b3, _ = st.columns([1, 1, 1, 3])
            if b1.button("✓ False Alarm",      key=f"fa_{r['id']}"):
                acknowledge(r["id"], "false_alarm");  st.rerun()
            if b2.button("🤝 Non-injury Assist", key=f"ni_{r['id']}"):
                acknowledge(r["id"], "assist");       st.rerun()
            if b3.button("🚑 Clinical Concern",  key=f"cc_{r['id']}"):
                acknowledge(r["id"], "clinical");     st.rerun()

        if r["clip_path"] and Path(r["clip_path"]).exists():
            with st.expander("View snapshot"):
                st.image(r["clip_path"], width=400)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.title("Analytics")

    if df_all.empty:
        st.info("No incidents logged yet — run the pipeline on a video first.")
        st.stop()

    # Date range filter
    days = st.slider("Show last N days", 1, 30, 7)
    cutoff = datetime.now() - timedelta(days=days)
    df = df_all[df_all["datetime"] >= cutoff].copy()

    if df.empty:
        st.info(f"No incidents in the last {days} days.")
        st.stop()

    # ── KPI row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Incidents", len(df))
    c2.metric("Falls", len(df[df["event_type"] == "fall"]))
    c3.metric("Inactivity", len(df[df["event_type"] == "inactivity"]))
    c4.metric("Wandering", len(df[df["event_type"] == "wandering"]))
    fa_rate = len(df[df["disposition"] == "false_alarm"]) / max(len(df.dropna(subset=["disposition"])), 1)
    c5.metric("False Alarm Rate", f"{fa_rate:.0%}")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Incidents by Type")
        type_counts = df["event_type"].value_counts().reset_index()
        type_counts.columns = ["Event Type", "Count"]
        st.bar_chart(type_counts.set_index("Event Type"))

    with col_b:
        st.subheader("Incidents by Hour of Day")
        hour_counts = df.groupby("hour").size().reset_index(name="Count")
        st.bar_chart(hour_counts.set_index("hour"))

    st.subheader("Incidents Over Time")
    daily = df.groupby(["date", "event_type"]).size().unstack(fill_value=0)
    st.bar_chart(daily)

    st.subheader("Response Time Distribution")
    resp_df = df.dropna(subset=["ack_time"]).copy()
    if not resp_df.empty:
        resp_df["response_s"] = resp_df["ack_time"] - resp_df["timestamp"]
        st.bar_chart(resp_df["response_s"].value_counts().sort_index())
        avg = resp_df["response_s"].mean()
        st.caption(f"Average response time: **{avg:.1f} seconds**")
    else:
        st.info("No acknowledged incidents yet.")

    st.subheader("Incidents by Camera")
    cam_counts = df.groupby(["camera_id", "event_type"]).size().unstack(fill_value=0)
    st.dataframe(cam_counts, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: EVAL METRICS
# ════════════════════════════════════════════════════════════════════════════
elif page == "📈 Eval Metrics":
    st.title("Eval Metrics")
    st.caption("Detection accuracy over time, from data/eval_results/ "
               "(produced by scripts/run_eval.py).")

    if not ALL_RUNS_CSV.exists():
        st.info("No eval runs recorded yet. Run `python scripts/run_eval.py` first.")
        st.stop()

    runs = pd.read_csv(ALL_RUNS_CSV)
    runs["timestamp"] = pd.to_datetime(runs["timestamp"])
    runs = runs.sort_values("timestamp")

    scored = runs.dropna(subset=["f1_score"]).copy()
    failed_run_count = len(runs) - len(scored)

    if scored.empty:
        st.warning("Every recorded run failed before producing a score "
                    "(no F1/precision/recall) — nothing to plot yet.")
        st.stop()

    latest = scored.iloc[-1]

    # ── KPI row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest F1",        f"{latest['f1_score']:.1f}%")
    c2.metric("Latest Precision", f"{latest['precision']:.1f}%")
    c3.metric("Latest Recall",    f"{latest['recall']:.1f}%")
    c4.metric("Runs Recorded",    len(runs),
              delta=f"{failed_run_count} failed" if failed_run_count else None,
              delta_color="inverse")

    st.caption(f"Latest run: **{latest['dataset_name']}** / {latest['subset']} "
               f"— {int(latest['videos_evaluated'])} videos, "
               f"{latest['timestamp']:%Y-%m-%d %H:%M}")

    st.divider()

    # ── Trend chart ──────────────────────────────────────────────────────────
    st.subheader("Precision / Recall / F1 Over Time")
    trend = scored.set_index("timestamp")[["precision", "recall", "f1_score"]]
    trend.columns = ["Precision", "Recall", "F1"]
    st.line_chart(trend)

    st.divider()

    # ── Pass/fail against ROADMAP targets ───────────────────────────────────
    st.subheader("Status vs. ROADMAP Targets")
    target_rows = []
    for feature, target in EVAL_TARGETS.items():
        prefix = target["dataset_name_prefix"]
        feature_runs = (scored[scored["dataset_name"].str.startswith(prefix, na=False)]
                         if prefix else scored)
        if feature_runs.empty:
            target_rows.append({
                "Feature": feature, "Latest Precision": "—", "Latest Recall": "—",
                "Target": f"P>{target['precision']}% R>{target['recall']}%"
                          if target["precision"] else f"R>{target['recall']}%",
                "Status": "⏳ not yet measured",
            })
            continue
        r = feature_runs.iloc[-1]
        meets_p = target["precision"] is None or r["precision"] >= target["precision"]
        meets_r = r["recall"] >= target["recall"]
        target_rows.append({
            "Feature": feature,
            "Latest Precision": f"{r['precision']:.1f}%",
            "Latest Recall": f"{r['recall']:.1f}%",
            "Target": f"P>{target['precision']}% R>{target['recall']}%"
                      if target["precision"] else f"R>{target['recall']}%",
            "Status": "✅ meets target" if (meets_p and meets_r) else "❌ below target",
        })
    st.dataframe(pd.DataFrame(target_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── Miss category breakdown (from per-run JSON, not the summary CSV) ────
    st.subheader("Miss Reasons (False Negatives)")
    st.caption("Why detections were missed, categorized by evaluate.py's "
               "diagnostics — pulled from every saved run's JSON.")

    category_counts: Counter = Counter()
    fp_total = 0
    for jf in EVAL_RESULTS_DIR.glob("*.json"):
        try:
            data = json.loads(jf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for failure in data.get("failures", []):
            if failure.get("miss_type") == "false_positive":
                fp_total += 1
                continue
            reason = failure.get("miss_reason") or {}
            category_counts[reason.get("category", "uncategorized")] += 1

    col_fn, col_fp = st.columns(2)
    with col_fn:
        if category_counts:
            cat_df = pd.DataFrame(
                sorted(category_counts.items(), key=lambda kv: -kv[1]),
                columns=["Category", "Count"],
            )
            st.bar_chart(cat_df.set_index("Category"))
        else:
            st.info("No categorized misses recorded yet.")
    with col_fp:
        st.metric("Total False Positives (all runs)", fp_total)

    if failed_run_count:
        st.divider()
        st.caption(f"⚠️ {failed_run_count} recorded run(s) produced no score at all "
                   f"(download/setup failure) and are excluded from the charts above.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: ZONE EDITOR
# ════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Zone Editor":
    st.title("Zone Editor")
    st.caption("Define bed, chair, floor, and restricted zones per camera. "
               "Zones are normalised 0–1 coordinates (x=left→right, y=top→bottom).")

    ZONES_FILE = Path(__file__).parent.parent / "data" / "zones" / "zones.json"
    ZONES_FILE.parent.mkdir(parents=True, exist_ok=True)
    zones_data = json.loads(ZONES_FILE.read_text()) if ZONES_FILE.exists() else {}

    camera_id = st.text_input("Camera ID", value="cam_01")
    cam_zones  = zones_data.get(camera_id, [])

    st.subheader(f"Zones for {camera_id}")

    # Show existing zones
    if cam_zones:
        for idx, z in enumerate(cam_zones):
            with st.expander(f"{z['name']} ({z['type']})", expanded=False):
                c1, c2 = st.columns([3, 1])
                c1.json(z["poly"])
                if c2.button("🗑 Delete", key=f"del_{idx}"):
                    cam_zones.pop(idx)
                    zones_data[camera_id] = cam_zones
                    ZONES_FILE.write_text(json.dumps(zones_data, indent=2))
                    st.rerun()
    else:
        st.info("No zones defined for this camera yet.")

    st.divider()
    st.subheader("Add New Zone")
    st.caption("Enter polygon as normalised coordinates — 4 corners, e.g. [[0.1,0.2],[0.5,0.2],[0.5,0.8],[0.1,0.8]]")

    with st.form("add_zone"):
        z_name = st.text_input("Zone Name", placeholder="bed_area")
        z_type = st.selectbox("Zone Type", ["bed", "chair", "floor", "restricted", "default"])
        z_poly = st.text_area("Polygon (JSON)", value="[[0.1,0.1],[0.5,0.1],[0.5,0.9],[0.1,0.9]]")
        submitted = st.form_submit_button("Add Zone")

        if submitted:
            try:
                poly = json.loads(z_poly)
                if len(poly) < 3:
                    st.error("Polygon needs at least 3 points.")
                else:
                    cam_zones.append({"name": z_name, "type": z_type, "poly": poly})
                    zones_data[camera_id] = cam_zones
                    ZONES_FILE.write_text(json.dumps(zones_data, indent=2))
                    st.success(f"Zone '{z_name}' added.")
                    st.rerun()
            except json.JSONDecodeError:
                st.error("Invalid JSON for polygon.")

    st.divider()
    st.subheader("Raw zones.json")
    st.json(zones_data)

    if st.button("💾 Export zones.json"):
        st.download_button("Download", ZONES_FILE.read_text(),
                           file_name="zones.json", mime="application/json")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: INCIDENT LOG
# ════════════════════════════════════════════════════════════════════════════
elif page == "📋 Incident Log":
    st.title("Incident Log")

    if df_all.empty:
        st.info("No incidents logged yet.")
        st.stop()

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    filter_type = col_f1.selectbox("Event Type", ["All"] + list(df_all["event_type"].unique()))
    filter_cam  = col_f2.selectbox("Camera",     ["All"] + list(df_all["camera_id"].unique()))
    filter_disp = col_f3.selectbox("Disposition",["All", "false_alarm", "assist", "clinical", "pending"])

    df_log = df_all.copy()
    if filter_type != "All":
        df_log = df_log[df_log["event_type"] == filter_type]
    if filter_cam != "All":
        df_log = df_log[df_log["camera_id"] == filter_cam]
    if filter_disp == "pending":
        df_log = df_log[df_log["pending"]]
    elif filter_disp != "All":
        df_log = df_log[df_log["disposition"] == filter_disp]

    st.caption(f"Showing {len(df_log)} incidents")

    display_cols = ["time_str", "camera_id", "track_id", "event_type",
                    "zone", "disposition", "ack_time"]
    available = [c for c in display_cols if c in df_log.columns]
    st.dataframe(df_log[available].sort_values("time_str", ascending=False),
                 use_container_width=True, height=500)

    # Export
    csv = df_log[available].to_csv(index=False)
    st.download_button("📥 Export CSV", csv, "carewatch_incidents.csv", "text/csv")


# ── Auto-refresh ─────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(5)
    st.rerun()
