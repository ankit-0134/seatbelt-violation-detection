import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from pathlib import Path
from detector import VehicleAnalyzer

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle Safety Monitor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Custom CSS  (dark industrial theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background: #0d0f14;
    color: #e0e4ed;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111520;
    border-right: 1px solid #1e2535;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: #151a27;
    border: 1px solid #252d42;
    border-radius: 8px;
    padding: 12px 16px;
}
div[data-testid="metric-container"] label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: #6b7fa3;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2rem;
    color: #00e5ff;
}

/* Sliders accent */
div[data-baseweb="slider"] [data-testid="stThumbValue"] {
    color: #00e5ff;
}

/* Buttons */
button[kind="primary"], .stButton > button {
    background: linear-gradient(135deg, #00b4d8, #0077b6);
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 0.08em;
    font-size: 0.85rem;
    padding: 0.5rem 1.2rem;
}

/* Header strip */
.header-strip {
    background: linear-gradient(90deg, #0d1117 0%, #0a2540 50%, #0d1117 100%);
    border-bottom: 2px solid #00e5ff33;
    padding: 1rem 1.5rem;
    margin-bottom: 1.5rem;
    border-radius: 4px;
}
.header-strip h1 {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.6rem;
    color: #00e5ff;
    margin: 0;
    letter-spacing: 0.06em;
}
.header-strip p {
    color: #6b7fa3;
    margin: 0.25rem 0 0;
    font-size: 0.85rem;
}

/* Legend pill */
.legend-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-family: 'Share Tech Mono', monospace;
    margin: 2px 4px;
}
.zone-info {
    background: #151a27;
    border: 1px solid #1e2535;
    border-radius: 8px;
    padding: 12px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    color: #8fa3c8;
    line-height: 1.8;
}

/* Progress / status */
.status-processing {
    color: #ffd166;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-strip">
    <h1>🚗 VEHICLE SAFETY MONITOR</h1>
    <p>Seatbelt detection · Number plate recognition · Zone-based triggering</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Sidebar — configuration
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    # ── Detection Zone Lines ──────────────────
    st.markdown("#### 📏 Detection Zone Lines")
    st.caption("Cars whose centre falls **between** the two lines are analysed for seatbelt & plate.")

    line1_pct = st.slider(
        "Line 1 — top boundary (%)",
        min_value=5, max_value=90, value=35, step=1,
        help="Position of the top trigger line as % of frame height."
    )
    line2_pct = st.slider(
        "Line 2 — bottom boundary (%)",
        min_value=10, max_value=95, value=65, step=1,
        help="Position of the bottom trigger line as % of frame height."
    )

    if line1_pct >= line2_pct:
        st.error("⚠️ Line 1 must be above Line 2.")
        st.stop()

    st.markdown(
        f"""<div class="zone-info">
        ZONE HEIGHT&nbsp;&nbsp;: {line2_pct - line1_pct}%<br>
        LINE 1 (top)&nbsp;&nbsp;&nbsp;: {line1_pct / 100:.2f} of frame<br>
        LINE 2 (bottom): {line2_pct / 100:.2f} of frame
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Confidence thresholds ─────────────────
    st.markdown("#### 🎯 Confidence Thresholds")
    car_conf = st.slider("Car detection", 0.1, 0.95, 0.40, 0.05)
    sb_conf = st.slider("Seatbelt detection", 0.1, 0.95, 0.35, 0.05)
    pl_conf = st.slider("Plate detection", 0.1, 0.95, 0.35, 0.05)

    st.markdown("---")

    # ── Model paths ───────────────────────────
    st.markdown("#### 📂 Model Paths")
    car_model_path = st.text_input("Car detector model", value="models/car_detector.pt")
    sb_model_path = st.text_input("Seatbelt model", value="models/seatbelt.pt")
    pl_model_path = st.text_input("Plate model", value="models/best_plate.pt")

    st.markdown("---")

    # ── Output ────────────────────────────────
    st.markdown("#### 💾 Output")
    save_output = st.checkbox("Save processed video", value=True)
    show_stats_overlay = st.checkbox("Live stats overlay on video", value=True)
    output_fps = st.slider("Output FPS", 5, 60, 25, 5)


# ─────────────────────────────────────────────
#  Main area
# ─────────────────────────────────────────────
col_upload, col_legend = st.columns([2, 1])

with col_upload:
    st.markdown("#### 📤 Upload Video")
    uploaded_file = st.file_uploader(
        "Drop a .mp4 / .avi / .mov file",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
    )

with col_legend:
    st.markdown("#### 🎨 Colour Legend")
    st.markdown(
        """
        <span class="legend-pill" style="background:#FF8C0022;border:1px solid #FF8C00;color:#FF8C00;">🟧 Car / Vehicle</span>
        <span class="legend-pill" style="background:#00FF0022;border:1px solid #00FF00;color:#00FF00;">🟩 Seatbelt ✓</span>
        <span class="legend-pill" style="background:#FF000022;border:1px solid #FF4444;color:#FF4444;">🟥 No Seatbelt</span>
        <span class="legend-pill" style="background:#FFFF0022;border:1px solid #FFFF00;color:#FFFF00;">🟨 Number Plate</span>
        <span class="legend-pill" style="background:#00FFFF22;border:1px solid #00FFFF;color:#00FFFF;">〰️ Detection Zone</span>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ─────────────────────────────────────────────
#  Processing
# ─────────────────────────────────────────────
if uploaded_file is not None:
    run_btn = st.button("▶  Run Analysis", type="primary")

    if run_btn:
        # Save upload to temp file
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Initialise detector
        with st.spinner("Loading models …"):
            try:
                analyzer = VehicleAnalyzer(
                    car_model_path=car_model_path,
                    seatbelt_model_path=sb_model_path,
                    plate_model_path=pl_model_path,
                    line1_y=line1_pct / 100,
                    line2_y=line2_pct / 100,
                    car_conf=car_conf,
                    seatbelt_conf=sb_conf,
                    plate_conf=pl_conf,
                )
            except Exception as e:
                st.error(f"❌ Failed to load models: {e}")
                st.stop()

        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25

        st.info(f"📹 Video: {vid_w}×{vid_h} px · {total_frames} frames · {src_fps:.1f} FPS")

        # Output video writer
        out_path = None
        writer = None
        if save_output:
            out_path = os.path.join(tempfile.gettempdir(), "vsm_output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, output_fps, (vid_w, vid_h))

        # UI placeholders
        progress_bar = st.progress(0)
        status_text = st.empty()
        frame_display = st.empty()

        # Stats columns
        sc1, sc2, sc3, sc4 = st.columns(4)
        m_cars = sc1.empty()
        m_belt = sc2.empty()
        m_nobelt = sc3.empty()
        m_plates = sc4.empty()

        # Cumulative totals
        totals = {"cars_in_zone": 0, "seatbelt": 0, "no_seatbelt": 0, "plates": 0}
        frame_idx = 0
        t_start = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed, stats = analyzer.process_frame(frame)

            # Accumulate
            for k in totals:
                totals[k] += stats[k]

            # Stats overlay on frame
            if show_stats_overlay:
                overlay_text = [
                    f"FRAME {frame_idx+1}/{total_frames}",
                    f"ZONE CARS  : {stats['cars_in_zone']}",
                    f"SEATBELT   : {stats['seatbelt']}",
                    f"NO BELT    : {stats['no_seatbelt']}",
                    f"PLATES     : {stats['plates']}",
                ]
                for i, txt in enumerate(overlay_text):
                    cv2.putText(
                        processed, txt, (10, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 230, 255), 1, cv2.LINE_AA,
                    )

            if writer:
                writer.write(processed)

            # Show every 3rd frame in Streamlit (performance)
            if frame_idx % 3 == 0:
                rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                frame_display.image(rgb, channels="RGB", use_container_width=True)

                elapsed = time.time() - t_start
                fps_live = (frame_idx + 1) / max(elapsed, 1e-6)
                status_text.markdown(
                    f'<p class="status-processing">⚙️ Processing frame {frame_idx+1}/{total_frames} '
                    f'· {fps_live:.1f} fps · {elapsed:.1f}s elapsed</p>',
                    unsafe_allow_html=True,
                )

                m_cars.metric("🚗 Cars in zone", totals["cars_in_zone"])
                m_belt.metric("✅ Seatbelt", totals["seatbelt"])
                m_nobelt.metric("❌ No Seatbelt", totals["no_seatbelt"])
                m_plates.metric("🔢 Plates found", totals["plates"])

            progress_bar.progress(min((frame_idx + 1) / max(total_frames, 1), 1.0))
            frame_idx += 1

        cap.release()
        if writer:
            writer.release()

        elapsed_total = time.time() - t_start
        status_text.success(
            f"✅ Done! Processed {frame_idx} frames in {elapsed_total:.1f}s "
            f"({frame_idx/elapsed_total:.1f} fps avg)"
        )

        # Final summary
        st.markdown("### 📊 Final Summary")
        fa, fb, fc, fd = st.columns(4)
        fa.metric("Total cars in zone", totals["cars_in_zone"])
        fb.metric("Seatbelt detections", totals["seatbelt"])
        fc.metric("No-seatbelt detections", totals["no_seatbelt"])
        fd.metric("Number plates found", totals["plates"])

        # Download processed video
        if save_output and out_path and os.path.exists(out_path):
            with open(out_path, "rb") as f:
                st.download_button(
                    label="⬇️  Download Processed Video",
                    data=f,
                    file_name="vehicle_safety_output.mp4",
                    mime="video/mp4",
                )

        # Cleanup
        os.unlink(tmp_path)

else:
    st.markdown(
        """
        <div style="text-align:center;padding:3rem;background:#111520;
                    border:1px dashed #252d42;border-radius:12px;color:#4a5a7a;">
            <div style="font-size:3rem;">🎬</div>
            <p style="font-family:'Share Tech Mono',monospace;font-size:1rem;margin-top:0.5rem;">
                Upload a video to get started
            </p>
            <p style="font-size:0.8rem;margin-top:0.25rem;">
                Supported: MP4 · AVI · MOV · MKV
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
