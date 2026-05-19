import streamlit as st
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from io import BytesIO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.analysis import (
    load_masks, filter_masks, extract_colors,
    compute_iridescence_metrics, interpret_iridescence,
)
from utils.video import export_overlay_video

st.set_page_config(page_title="Analyze", layout="wide")

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data"

# ── Species selector ───────────────────────────────────────────────────────────
species_list = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
if not species_list:
    st.warning("No species found. Add one on the Home page.")
    st.stop()

active = st.session_state.get("active_species", species_list[0])
if active not in species_list:
    active = species_list[0]

with st.sidebar:
    st.header("Analyze")
    active = st.selectbox("Species", species_list,
                          index=species_list.index(active) if active in species_list else 0)
    st.session_state.active_species = active
    st.divider()
    _thr_pct = st.slider(
        "Mask quality threshold",
        min_value=5, max_value=50, value=25, step=5,
        help="Drop masks smaller than this fraction of their median size.",
        format="%d%%",
    )
    area_threshold = _thr_pct / 100.0

SPECIES_DIR = DATA_DIR / active
FRAMES_DIR  = SPECIES_DIR / "frames"
MASKS_PATH  = SPECIES_DIR / "masks.npz"
OUTPUT_DIR  = SPECIES_DIR / "output"
PARTS_PATH  = SPECIES_DIR / "parts.json"

st.title(f"Analyze — {active}")

if not MASKS_PATH.exists():
    st.warning("masks.npz not found. Complete the Segment step first.")
    st.stop()

if not FRAMES_DIR.exists() or not any(FRAMES_DIR.glob("*.jpg")):
    st.warning("No frames found. Run the Annotate step first (it extracts frames automatically).")
    st.stop()

OUTPUT_DIR.mkdir(exist_ok=True)

# ── Load part colors ───────────────────────────────────────────────────────────
def _hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[0:2], 16) / 255.0,
                     int(h[2:4], 16) / 255.0,
                     int(h[4:6], 16) / 255.0])

part_color_map: dict = {}
if PARTS_PATH.exists():
    with open(PARTS_PATH) as f:
        parts_meta = json.load(f)
    part_color_map = {p["name"]: p["color"] for p in parts_meta}

DEFAULT_MPL = ["tomato", "mediumseagreen", "steelblue", "gold", "orchid", "coral"]

# ── Cached analysis ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Running color analysis...")
def run_analysis(masks_path_str: str, frames_dir_str: str, threshold: float):
    video_segments, parts, fps, fw, fh = load_masks(Path(masks_path_str))
    video_segments, thresholds, area_data, n_filtered = filter_masks(
        video_segments, parts, threshold
    )
    df = extract_colors(Path(frames_dir_str), video_segments, parts)
    metrics = compute_iridescence_metrics(df, parts)
    interp  = interpret_iridescence(metrics)
    return df, metrics, interp, video_segments, parts, fps, area_data, thresholds, n_filtered

df, metrics, interp, video_segments, parts, fps, area_data, thresholds, n_filtered = run_analysis(
    str(MASKS_PATH), str(FRAMES_DIR), area_threshold
)

total_slots = sum(len(v) for v in video_segments.values())
st.caption(f"{len(df)} frames analyzed  |  {n_filtered}/{total_slots} mask slots filtered  |  {fps:.1f} FPS")

# ── Iridescence Metrics table ──────────────────────────────────────────────────
st.subheader("Iridescence Metrics")

if metrics:
    metric_df = pd.DataFrame(metrics).T
    metric_df.index.name = "Part"
    st.dataframe(
        metric_df.style.background_gradient(subset=["iridescence_index"], cmap="YlOrRd"),
        use_container_width=True,
    )
else:
    st.info("No valid mask data found. Check that masks.npz contains usable frames.")

st.divider()

# ── Interpretation ─────────────────────────────────────────────────────────────
st.subheader("Iridescence Interpretation")

if interp:
    level_colors = {"Strong": "#ff4444", "Moderate": "#ff8c00",
                    "Low-moderate": "#d4a017", "Low": "#4caf50"}
    cols = st.columns(max(len(interp), 1))
    for col, (part, result) in zip(cols, interp.items()):
        color = level_colors.get(result["level"], "#888")
        with col:
            st.markdown(
                f'<div style="border-left:4px solid {color};padding:10px;border-radius:4px">'
                f'<b>{part}</b><br>'
                f'<span style="color:{color};font-size:13px">{result["level"]} iridescence</span><br>'
                f'<small><i>{result["mechanism"]}</i></small><br><br>'
                f'<small>{result["detail"]}</small><br>'
                f'<small>Index: <b>{result["iridescence_index"]}</b></small>'
                f'</div>',
                unsafe_allow_html=True,
            )

st.divider()

# ── Plots ──────────────────────────────────────────────────────────────────────
st.subheader("Color Dynamics")

def _mpl_color(part, i):
    if part in part_color_map:
        return part_color_map[part]
    return DEFAULT_MPL[i % len(DEFAULT_MPL)]

valid_parts = [p for p in parts if f"{p}_hue" in df.columns]

tab_time, tab_ab, tab_area = st.tabs(["Hue / Chroma / Brightness", "AB Trajectory", "Mask Quality"])

with tab_time:
    if valid_parts:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        for i, part in enumerate(valid_parts):
            c     = _mpl_color(part, i)
            L_col = f"{part}_L_norm" if f"{part}_L_norm" in df.columns else f"{part}_L"
            axes[0].plot(df.index, df[f"{part}_hue"],    color=c, alpha=0.8, label=part, linewidth=0.8)
            axes[1].plot(df.index, df[f"{part}_chroma"], color=c, alpha=0.8, label=part, linewidth=0.8)
            axes[2].plot(df.index, df[L_col],            color=c, alpha=0.8, label=part, linewidth=0.8)
        axes[0].set_ylabel("Hue (°)")
        axes[0].set_title("Iridescent Hue Shift")
        axes[1].set_ylabel("Chroma")
        axes[1].set_title("Color Saturation (higher = more vivid)")
        axes[2].set_ylabel("L* (normalised)")
        axes[2].set_title("Brightness (lighting-corrected)")
        axes[2].set_xlabel("Frame")
        for ax in axes:
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        plt.suptitle(f"{active} — Color Dynamics", fontsize=13, fontweight="bold")
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        st.image(buf, use_container_width=True)
        plt.savefig(str(OUTPUT_DIR / "color_shifts.png"), dpi=150)
        plt.close()
    else:
        st.info("No hue data available.")

with tab_ab:
    if valid_parts:
        fig, ax = plt.subplots(figsize=(7, 7))
        markers = ["o", "s", "^", "D", "v", "P"]
        for i, part in enumerate(valid_parts):
            A_col, B_col = f"{part}_A", f"{part}_B"
            valid = df[[A_col, B_col]].dropna()
            if valid.empty:
                continue
            sc = ax.scatter(
                valid[A_col], valid[B_col],
                c=valid.index, cmap="plasma",
                label=part, alpha=0.7, s=12,
                marker=markers[i % len(markers)],
            )
        plt.colorbar(sc, ax=ax, label="Frame number")
        ax.set_xlabel("A*  (green — red)")
        ax.set_ylabel("B*  (blue — yellow)")
        ax.set_title("AB Color Trajectory — Iridescence Fingerprint", fontsize=11)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.legend(fontsize=8)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        st.image(buf, use_container_width=True)
        plt.savefig(str(OUTPUT_DIR / "ab_trajectory.png"), dpi=150)
        plt.close()

with tab_area:
    fig, axes = plt.subplots(len(parts), 1,
                             figsize=(14, 3 * len(parts)), sharex=True)
    if len(parts) == 1:
        axes = [axes]
    for ax, part in zip(axes, parts):
        areas = pd.Series(area_data.get(part, {})).sort_index()
        c = _mpl_color(part, parts.index(part))
        ax.fill_between(areas.index, areas.values, alpha=0.35, color=c)
        ax.plot(areas.index, areas.values, color=c, linewidth=0.8)
        thr = thresholds.get(part, 0)
        ax.axhline(thr, color="red", linestyle="--", linewidth=1.2,
                   label=f"threshold ({area_threshold:.0%} of median)")
        ax.set_ylabel("Mask area (px)")
        ax.set_title(f"{part}  —  threshold: {thr:.0f} px")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Frame")
    plt.suptitle("Mask Area Over Time", fontsize=12, fontweight="bold")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.savefig(str(OUTPUT_DIR / "mask_area.png"), dpi=150)
    plt.close()

st.divider()

# ── Export ─────────────────────────────────────────────────────────────────────
st.subheader("Export")

col_csv, col_vid = st.columns(2)

with col_csv:
    csv_path = OUTPUT_DIR / "color_data.csv"
    df.to_csv(str(csv_path))
    with open(csv_path, "rb") as f:
        st.download_button(
            "Download color_data.csv",
            data=f.read(),
            file_name=f"{active}_color_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

with col_vid:
    video_out = OUTPUT_DIR / "mask_overlay.mp4"
    if st.button("Export overlay video", use_container_width=True):
        part_rgb_colors = {
            part: _hex_to_rgb(part_color_map.get(part, "#ffff00"))
            for part in parts
        }
        with st.spinner(f"Rendering {len(list(FRAMES_DIR.glob('*.jpg')))} frames..."):
            export_overlay_video(FRAMES_DIR, video_out, fps, video_segments, part_rgb_colors)
        st.success(f"Saved to Data/{active}/output/mask_overlay.mp4")

    if video_out.exists():
        size_mb = video_out.stat().st_size / 1e6
        st.caption(f"mask_overlay.mp4 already exists ({size_mb:.1f} MB)")
