import streamlit as st
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.colab_gen import generate_notebook

st.set_page_config(page_title="Segment", layout="wide")

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
    st.header("Segment")
    active = st.selectbox("Species", species_list,
                          index=species_list.index(active) if active in species_list else 0)
    st.session_state.active_species = active

SPECIES_DIR = DATA_DIR / active
KF_PATH     = SPECIES_DIR / "keyframes.json"
MASKS_PATH  = SPECIES_DIR / "masks.npz"

video_files = (list(SPECIES_DIR.glob("*.mov")) + list(SPECIES_DIR.glob("*.mp4")) +
               list(SPECIES_DIR.glob("*.avi")) + list(SPECIES_DIR.glob("*.mkv")))

st.title(f"Segment {active}")

# ── Status summary ─────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Video", "Found" if video_files else "Missing")
with c2:
    if KF_PATH.exists():
        with open(KF_PATH) as f:
            kfs = json.load(f)
        st.metric("Keyframes", len(kfs))
    else:
        st.metric("Keyframes", "Not annotated yet")
with c3:
    if MASKS_PATH.exists():
        size_mb = MASKS_PATH.stat().st_size / 1e6
        st.metric("masks.npz", f"{size_mb:.1f} MB")
    else:
        st.metric("masks.npz", "Not yet uploaded")

st.divider()

# ── How SAM 2 works (one-time explainer) ──────────────────────────────────────
with st.expander("How does segmentation work?"):
    st.markdown("""
SAM 2 (Segment Anything Model 2) is a general-purpose video segmentation model that does not
need to be retrained for each species. You provide a few painted masks from the Annotate step,
and SAM 2 tracks and propagates those regions across every frame of the video.

**Why Google Colab?** SAM 2 requires a GPU. The free T4 GPU on Google Colab handles a
2600-frame video in about 15 minutes using the chunked approach built into the generated notebook.

**Workflow:**
1. Generate the notebook and download it
2. Upload your video, `keyframes.json`, and `annotation_masks.npz` to Google Drive under `MyDrive/Iridescence Lab/{species}/`
3. Open the notebook in Colab, switch the runtime to a T4 GPU, and run all cells
4. Download `masks.npz` from Drive when it finishes, then upload it back here
""")

# ── Step 1: Generate Colab notebook ───────────────────────────────────────────
st.subheader("Step 1: Generate your Colab Notebook")

if not KF_PATH.exists():
    st.warning("keyframes.json not found. Complete annotation first.")
elif not video_files:
    st.warning("No video found in Data/" + active + "/")
else:
    video_filename = video_files[0].name
    drive_folder   = st.text_input(
        "Google Drive folder name",
        value="Iridescence Lab",
        help="Notebook expects MyDrive/<this folder>/" + active + "/"
    )

    notebook = generate_notebook(
        species_name=active,
        video_filename=video_filename,
        drive_folder=drive_folder,
    )
    notebook_bytes = json.dumps(notebook, indent=2).encode()

    st.download_button(
        label="Download sam2_" + active + ".ipynb",
        data=notebook_bytes,
        file_name="sam2_" + active + ".ipynb",
        mime="application/json",
        type="primary",
    )

    ann_masks_exists = (SPECIES_DIR / "annotation_masks.npz").exists()
    st.info(
        f"Before running in Colab, upload these three files to "
        f"`MyDrive/{drive_folder}/{active}/`:  \n"
        f"· `{video_filename}`  \n"
        f"· `keyframes.json`  \n"
        f"· `annotation_masks.npz`"
        + ("  ✓ ready" if ann_masks_exists else "  ⚠ not painted yet — finish annotating first")
    )

st.divider()

# ── Step 2: Upload masks.npz ──────────────────────────────────────────────────
st.subheader("Step 2: Upload your masks")

uploaded = st.file_uploader(
    "Upload masks.npz downloaded from Google Drive",
    type=["npz"],
    key=f"masks_upload_{active}",
)
if uploaded is not None:
    SPECIES_DIR.mkdir(exist_ok=True)
    with open(MASKS_PATH, "wb") as f:
        f.write(uploaded.read())
    size_mb = MASKS_PATH.stat().st_size / 1e6
    st.success(f"Saved masks.npz ({size_mb:.1f} MB) to Data/{active}/")

if MASKS_PATH.exists():
    st.success(f"masks.npz is present ({MASKS_PATH.stat().st_size / 1e6:.1f} MB). "
               "Proceed to the Analyze tab.")
