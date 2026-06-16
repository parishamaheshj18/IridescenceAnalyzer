import json
import cv2
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.normal_colab_gen import generate_normal_notebook
from utils.normal_estimator import check_dependencies, run_local_normal_estimation, active_method

import streamlit as st
from PIL import Image as PILImage

st.set_page_config(page_title="Surface Geometry", layout="wide")

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
    st.header("Surface Geometry")
    active = st.selectbox("Species", species_list,
                          index=species_list.index(active) if active in species_list else 0)
    st.session_state.active_species = active
    st.divider()
    frame_stride = st.number_input(
        "Frame stride",
        min_value=1, max_value=30, value=1,
        help="Process every Nth frame. stride=5 is 5× faster with minor accuracy loss.",
    )
    n_gallery = st.slider("Gallery samples", min_value=4, max_value=30, value=12,
                          help="How many frames to save and show as normal maps.")

SPECIES_DIR  = DATA_DIR / active
FRAMES_DIR   = SPECIES_DIR / "frames"
MASKS_PATH   = SPECIES_DIR / "masks.npz"
NORMALS_PATH = SPECIES_DIR / "normal_data.csv"
MAPS_DIR     = SPECIES_DIR / "normal_maps"
PARTS_PATH   = SPECIES_DIR / "parts.json"

video_files = (list(SPECIES_DIR.glob("*.mov")) + list(SPECIES_DIR.glob("*.mp4")) +
               list(SPECIES_DIR.glob("*.avi")) + list(SPECIES_DIR.glob("*.mkv")))

st.title(f"Surface Normals: {active}")
st.caption(
    "RGB-encoded surface normals via StableNormal YOSO.  "
    "Blue = facing camera · Red = left/right tilt · Green = up/down tilt."
)

# ── Status ─────────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    n_frames = len(list(FRAMES_DIR.glob("*.jpg"))) if FRAMES_DIR.exists() else 0
    st.metric("Frames", n_frames if n_frames else "Missing")
with c2:
    st.metric("masks.npz", "Ready" if MASKS_PATH.exists() else "Missing")
with c3:
    n_maps = len(list(MAPS_DIR.glob("*_normal.png"))) if MAPS_DIR.exists() else 0
    st.metric("Normal maps saved", n_maps if n_maps else "None yet")

st.divider()

# ── Run locally ────────────────────────────────────────────────────────────────
st.subheader("Run Normal Estimation")

deps_ok, deps_msg = check_dependencies()
method_now = active_method()

if method_now == "stablenormal":
    st.success("StableNormal YOSO detected — will produce colorful normal map images.")
elif method_now == "depth":
    st.info("Depth Anything V2 detected — normal maps derived from depth gradients.")
else:
    st.warning(
        "StableNormal not installed. Install it for visual normal maps:  \n"
        "```\npip install git+https://github.com/Stable-X/StableNormal.git diffusers\n"
        "pip install --upgrade torch torchvision\n```"
    )

if not FRAMES_DIR.exists() or n_frames == 0:
    st.warning("No frames found. Complete the Analyze step first.")
elif not MASKS_PATH.exists():
    st.warning("masks.npz not found. Complete the Segment step first.")
else:
    st.info(
        f"Will process **{len(list(FRAMES_DIR.glob('*.jpg'))[::int(frame_stride)])} frames** "
        f"(stride={int(frame_stride)}) and save **{n_gallery} sample normal maps**."
    )

    if st.button("Run Locally", type="primary", key="run_normals"):
        progress_bar = st.progress(0.0)
        status_text  = st.empty()

        def _cb(current: int, total: int, msg: str) -> None:
            progress_bar.progress(min(current / max(total, 1), 1.0))
            status_text.text(msg)

        try:
            run_local_normal_estimation(
                frames_dir=FRAMES_DIR,
                masks_path=MASKS_PATH,
                output_path=NORMALS_PATH,
                frame_stride=int(frame_stride),
                progress_callback=_cb,
                n_sample_maps=int(n_gallery),
            )
            progress_bar.progress(1.0)
            status_text.success("Done. Scroll down to see the normal maps.")
            st.rerun()
        except Exception as exc:
            st.error(f"Estimation failed: {exc}")

# ── Colab fallback ─────────────────────────────────────────────────────────────
with st.expander("Or run on Google Colab"):
    if video_files and MASKS_PATH.exists():
        drive_folder = st.text_input("Google Drive folder", value="Iridescence Lab",
                                     key="normals_drive_folder")
        notebook = generate_normal_notebook(
            species_name=active,
            video_filename=video_files[0].name,
            drive_folder=drive_folder,
            frame_stride=int(frame_stride),
        )
        st.download_button(
            label=f"Download normal_maps_{active}.ipynb",
            data=json.dumps(notebook, indent=2).encode(),
            file_name=f"normal_maps_{active}.ipynb",
            mime="application/json",
        )
        uploaded_csv = st.file_uploader("Upload normal_data.csv from Colab", type=["csv"],
                                        key=f"normals_upload_{active}")
        if uploaded_csv is not None:
            SPECIES_DIR.mkdir(exist_ok=True)
            with open(NORMALS_PATH, "wb") as fh:
                fh.write(uploaded_csv.read())
            st.success("Saved normal_data.csv.")
            st.rerun()

        uploaded_zip = st.file_uploader("Upload normal_maps.zip from Colab", type=["zip"],
                                        key=f"maps_upload_{active}")
        if uploaded_zip is not None:
            import zipfile, io
            MAPS_DIR.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(uploaded_zip.read())) as zf:
                zf.extractall(str(MAPS_DIR))
            n = len(list(MAPS_DIR.glob("*_normal.png")))
            st.success(f"Extracted {n} normal map images.")
            st.rerun()
    else:
        st.warning("Video and masks.npz must be present to generate a notebook.")

# ── Gallery ────────────────────────────────────────────────────────────────────
map_files = sorted(MAPS_DIR.glob("*_normal.png")) if MAPS_DIR.exists() else []

if not map_files:
    st.stop()

st.divider()
st.subheader("Normal Map Gallery")
st.caption(
    "Each pair shows the original frame (left) and its StableNormal output (right).  \n"
    "Body-part outlines are drawn in their annotation colors."
)

# Load body-part colors
part_colors: dict = {}
if PARTS_PATH.exists():
    with open(PARTS_PATH) as fh:
        for p in json.load(fh):
            part_colors[p["name"]] = p["color"]

# Load masks once
_mask_data = np.load(str(MASKS_PATH), allow_pickle=False) if MASKS_PATH.exists() else None
_video_segs: dict = {}
if _mask_data is not None:
    for _key in _mask_data.files:
        if _key.startswith("_"):
            continue
        _fi, _pt = _key.split("_", 1)
        _video_segs.setdefault(int(_fi), {})[_pt] = _mask_data[_key].astype(bool)

for fpath in map_files:
    fidx      = int(fpath.stem.split("_")[0])
    orig_path = FRAMES_DIR / f"{fidx:05d}.jpg"
    if not orig_path.exists():
        continue

    orig_img   = np.array(PILImage.open(str(orig_path)))
    normal_img = np.array(PILImage.open(str(fpath)))

    # Draw body-part outlines on the normal map
    overlay = normal_img.copy()
    for part, mask in _video_segs.get(fidx, {}).items():
        hex_c = part_colors.get(part, "#ffffff")
        r = int(hex_c[1:3], 16)
        g = int(hex_c[3:5], 16)
        b = int(hex_c[5:7], 16)
        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (r, g, b), 2)

    col_orig, col_norm = st.columns(2)
    with col_orig:
        st.image(orig_img,  caption=f"Frame {fidx}", use_container_width=True)
    with col_norm:
        st.image(overlay,   caption=f"Frame {fidx} — normal map", use_container_width=True)

    st.divider()
