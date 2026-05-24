from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.rtt.pipeline import split_audio_file
from src.rtt.segmentation import SegmentationConfig


st.set_page_config(page_title="Ripped Records -> Tracks", page_icon="vinyl", layout="wide")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
      :root {
        --bg-1: #f7efe5;
        --bg-2: #f1d6a8;
        --accent: #14532d;
        --accent-2: #9a3412;
        --text: #1f2937;
      }
      .stApp {
        background: radial-gradient(circle at 20% 10%, var(--bg-2), transparent 45%),
                    radial-gradient(circle at 80% 90%, #dbeafe, transparent 45%),
                    linear-gradient(130deg, var(--bg-1), #fff7ed);
      }
      html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--text);
      }
      .metric-box {
        border: 1px solid rgba(20, 83, 45, 0.2);
        background: rgba(255,255,255,0.7);
        padding: 0.85rem;
        border-radius: 0.8rem;
        backdrop-filter: blur(5px);
      }
      .smallmono {
        font-family: 'IBM Plex Mono', monospace;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Ripped Records -> Tracks")
st.caption("Upload one long MP3 and split it with layered heuristics: silence, BPM shifts, tonality jumps, and spectral novelty.")

with st.sidebar:
    st.header("Segmentation Rules")
    sensitivity = st.slider("Global sensitivity", min_value=0.0, max_value=1.0, value=0.55, step=0.01)
    min_track = st.slider("Min track length (s)", min_value=10.0, max_value=240.0, value=40.0, step=1.0)
    max_track = st.slider("Max track length (s)", min_value=60.0, max_value=1200.0, value=600.0, step=5.0)
    silence_db = st.slider("Silence threshold (dB)", min_value=-70.0, max_value=-10.0, value=-36.0, step=1.0)
    silence_min = st.slider("Min silence window (s)", min_value=0.2, max_value=4.0, value=1.2, step=0.1)

    st.subheader("Rule Weights")
    w_silence = st.slider("Silence", 0.0, 1.0, 0.35, 0.01)
    w_bpm = st.slider("BPM Change", 0.0, 1.0, 0.20, 0.01)
    w_tonal = st.slider("Tonality Change", 0.0, 1.0, 0.20, 0.01)
    w_spec = st.slider("Spectral Novelty", 0.0, 1.0, 0.25, 0.01)

uploaded = st.file_uploader("Drop your .mp3 here", type=["mp3", "wav", "flac", "ogg"])

if uploaded is not None:
    with tempfile.TemporaryDirectory(prefix="rtt_") as tmp:
        in_path = Path(tmp) / uploaded.name
        in_path.write_bytes(uploaded.getbuffer())

        output_dir = Path(tmp) / "tracks"
        cfg = SegmentationConfig(
            min_track_len_s=min_track,
            max_track_len_s=max_track,
            silence_db_threshold=silence_db,
            silence_min_len_s=silence_min,
            sensitivity=sensitivity,
            weight_silence=w_silence,
            weight_bpm_change=w_bpm,
            weight_tonality_change=w_tonal,
            weight_spectral_novelty=w_spec,
        )

        with st.spinner("Analyzing recording and detecting boundaries..."):
            result = split_audio_file(file_path=in_path, output_dir=output_dir, cfg=cfg)

        boundaries = result.segmentation.boundaries_s
        tracks = []
        for idx in range(len(boundaries) - 1):
            tracks.append(
                {
                    "track": idx + 1,
                    "start_s": boundaries[idx],
                    "end_s": boundaries[idx + 1],
                    "duration_s": round(boundaries[idx + 1] - boundaries[idx], 2),
                }
            )

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='metric-box'><b>Detected tracks</b><br><span class='smallmono'>{len(tracks)}</span></div>", unsafe_allow_html=True)
        c2.markdown(
            f"<div class='metric-box'><b>Boundaries</b><br><span class='smallmono'>{len(boundaries)}</span></div>",
            unsafe_allow_html=True,
        )
        c3.markdown(
            f"<div class='metric-box'><b>Total duration</b><br><span class='smallmono'>{round(result.segmentation.duration_s, 2)} s</span></div>",
            unsafe_allow_html=True,
        )

        st.subheader("Track Timeline")
        st.dataframe(pd.DataFrame(tracks), use_container_width=True)

        st.subheader("Diagnostics")
        diag = result.segmentation.diagnostics
        chart_df = pd.DataFrame(
            {
                "time_s": diag["time_s"],
                "combo_score": diag["combo_score"],
                "bpm_score": diag["bpm_score"],
                "tonality_score": diag["tonality_score"],
                "novelty_score": diag["novelty_score"],
                "silence_score": diag["silence_score"],
            }
        ).set_index("time_s")
        st.line_chart(chart_df)

        zip_bytes = result.zip_path.read_bytes()
        st.download_button(
            label="Download split tracks as ZIP",
            data=zip_bytes,
            file_name=result.zip_path.name,
            mime="application/zip",
            use_container_width=True,
        )
else:
    st.info("Upload a file to begin segmentation.")
