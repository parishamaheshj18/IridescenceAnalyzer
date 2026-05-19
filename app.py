import streamlit as st
import json
import shutil
from pathlib import Path

st.set_page_config(page_title="Iridescence Lab", layout="wide")

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data"
DATA_DIR.mkdir(exist_ok=True)

st.title("Iridescence Lab")
st.caption("Multi-species structural color analysis  |  Annotate → Segment → Analyze")

# ── Migration notice for existing hummingbird project ─────────────────────────
legacy_video  = ROOT / "Hummingbird.mov"
legacy_masks  = ROOT / "hummingbird_masks.npz"
legacy_kf     = ROOT / "hummingbird_keyframes.json"
legacy_frames = ROOT / "frames"
target_dir    = DATA_DIR / "Hummingbird"

if legacy_video.exists() and not target_dir.exists():
    st.info(
        "Existing project detected (Hummingbird.mov in root folder). "
        "Click below to migrate it into the Data/ folder so it works with this app."
    )
    if st.button("Migrate existing Hummingbird project"):
        target_dir.mkdir(exist_ok=True)
        shutil.copy2(str(legacy_video), str(target_dir / legacy_video.name))
        if legacy_masks.exists():
            shutil.copy2(str(legacy_masks), str(target_dir / "masks.npz"))
        if legacy_kf.exists():
            shutil.copy2(str(legacy_kf), str(target_dir / "keyframes.json"))
        if legacy_frames.exists() and not (target_dir / "frames").exists():
            (target_dir / "frames").symlink_to(legacy_frames.resolve())
        st.success("Migrated to Data/Hummingbird/")
        st.rerun()
    st.divider()

# ── Species library ────────────────────────────────────────────────────────────
species_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])

st.subheader("Species Library")

if not species_dirs:
    st.info("No species yet. Add one below.")
else:
    cols = st.columns(min(len(species_dirs), 4))
    for col, sdir in zip(cols, species_dirs):
        video_files = list(sdir.glob("*.mov")) + list(sdir.glob("*.mp4")) + \
                      list(sdir.glob("*.avi")) + list(sdir.glob("*.mkv"))
        has_video   = len(video_files) > 0
        has_kf      = (sdir / "keyframes.json").exists()
        has_masks   = (sdir / "masks.npz").exists()
        has_csv     = (sdir / "output" / "color_data.csv").exists()

        with col:
            st.markdown(f"**{sdir.name}**")
            st.caption(
                f"{'OK' if has_video else '--'} Video  "
                f"{'OK' if has_kf else '--'} Annotated  "
                f"{'OK' if has_masks else '--'} Segmented  "
                f"{'OK' if has_csv else '--'} Analyzed"
            )
            if st.button("Open", key=f"open_{sdir.name}"):
                st.session_state.active_species = sdir.name
                st.switch_page("pages/1_Annotate.py")

# ── Add new species ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Add New Species")

with st.form("new_species"):
    name       = st.text_input("Species name", placeholder="e.g. peacock_spider")
    video_file = st.file_uploader("Upload video", type=["mov", "mp4", "avi", "mkv"])
    submitted  = st.form_submit_button("Create")

    if submitted:
        if not name:
            st.error("Enter a species name.")
        elif not video_file:
            st.error("Upload a video file.")
        else:
            safe_name = name.strip().replace(" ", "_")
            sdir = DATA_DIR / safe_name
            sdir.mkdir(exist_ok=True)
            video_path = sdir / video_file.name
            with open(video_path, "wb") as f:
                f.write(video_file.read())
            st.session_state.active_species = safe_name
            st.success(f"Created {safe_name}. Go to Annotate to get started.")
            st.rerun()
