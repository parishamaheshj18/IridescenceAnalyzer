import streamlit as st
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

# ── Compatibility shim: streamlit-drawable-canvas calls
#    streamlit.elements.image.image_to_url(image, width_int, ...) which was
#    moved to streamlit.elements.lib.image_utils in Streamlit ≥ 1.36 and its
#    signature changed (width_int → LayoutConfig).  Patch the old name back in
#    so the canvas package finds it, then delegate to the real new function.
import streamlit.elements.image as _st_img_mod
if not hasattr(_st_img_mod, "image_to_url"):
    from streamlit.elements.lib.image_utils import (
        image_to_url as _new_image_to_url,
    )
    from streamlit.elements.lib.layout_utils import create_layout_config as _mk_layout

    def _image_to_url_shim(image, width=-1, clamp=False, channels="RGB",
                           output_format="auto", image_id="", allow_emoji=False):
        layout = _mk_layout(width=width)
        return _new_image_to_url(image, layout, clamp, channels, output_format, image_id)

    _st_img_mod.image_to_url = _image_to_url_shim

try:
    from streamlit_drawable_canvas import st_canvas
    _HAS_CANVAS = True
except ImportError:
    _HAS_CANVAS = False

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.video import extract_frames, get_video_info, read_frame_rgb

st.set_page_config(page_title="Annotate", layout="wide")

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data"

DEFAULT_COLORS = ["#ff5555", "#55ff55", "#55ffff", "#ffff55", "#ff88cc", "#aaaaff"]
DISPLAY_WIDTH  = 700

# ── Species selector ───────────────────────────────────────────────────────────
species_list = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
if not species_list:
    st.warning("No species found. Add one on the Home page.")
    st.stop()

active = st.session_state.get("active_species", species_list[0])
if active not in species_list:
    active = species_list[0]

with st.sidebar:
    st.header("Annotate")
    active = st.selectbox("Species", species_list,
                          index=species_list.index(active) if active in species_list else 0)
    st.session_state.active_species = active

SPECIES_DIR    = DATA_DIR / active
FRAMES_DIR     = SPECIES_DIR / "frames"
KF_PATH        = SPECIES_DIR / "keyframes.json"
PARTS_PATH     = SPECIES_DIR / "parts.json"
ANN_MASKS_PATH = SPECIES_DIR / "annotation_masks.npz"

# ── Load video + extract frames ───────────────────────────────────────────────
video_files = (list(SPECIES_DIR.glob("*.mov")) + list(SPECIES_DIR.glob("*.mp4")) +
               list(SPECIES_DIR.glob("*.avi")) + list(SPECIES_DIR.glob("*.mkv")))
if not video_files:
    st.error(f"No video found in Data/{active}/")
    st.stop()

video_path = video_files[0]

with st.sidebar:
    with st.spinner("Extracting frames..."):
        n_frames = extract_frames(video_path, FRAMES_DIR)
    info = get_video_info(video_path)
    st.caption(f"{info['width']}x{info['height']} @ {info['fps']:.1f} FPS  |  {n_frames} frames")

# ── Annotation mask helpers ───────────────────────────────────────────────────
def _load_ann_masks() -> dict:
    if not ANN_MASKS_PATH.exists():
        return {}
    data = np.load(str(ANN_MASKS_PATH))
    return {k: data[k] for k in data.files}

def _save_ann_mask(frame_idx: int, part_name: str, mask: np.ndarray):
    existing = _load_ann_masks()
    existing[f"{frame_idx}_{part_name}"] = mask.astype(bool)
    np.savez_compressed(str(ANN_MASKS_PATH), **existing)

def _delete_ann_mask(frame_idx: int, part_name: str):
    existing = _load_ann_masks()
    key = f"{frame_idx}_{part_name}"
    if key in existing:
        del existing[key]
        np.savez_compressed(str(ANN_MASKS_PATH), **existing)

# ── State initialization ───────────────────────────────────────────────────────
def _init_state():
    if "parts" not in st.session_state or st.session_state.get("_parts_species") != active:
        if PARTS_PATH.exists():
            with open(PARTS_PATH) as f:
                st.session_state.parts = json.load(f)
        else:
            if KF_PATH.exists():
                with open(KF_PATH) as f:
                    kfs = json.load(f)
                if kfs:
                    part_names = [k for k in kfs[0] if k != "frame"]
                    st.session_state.parts = [
                        {"name": n, "color": DEFAULT_COLORS[i % len(DEFAULT_COLORS)]}
                        for i, n in enumerate(part_names)
                    ]
                else:
                    st.session_state.parts = [{"name": "part_1", "color": DEFAULT_COLORS[0]}]
            else:
                st.session_state.parts = [
                    {"name": "crown", "color": DEFAULT_COLORS[0]},
                    {"name": "back",  "color": DEFAULT_COLORS[1]},
                    {"name": "tail",  "color": DEFAULT_COLORS[2]},
                ]
        st.session_state._parts_species = active

    if "keyframes" not in st.session_state or st.session_state.get("_kf_species") != active:
        if KF_PATH.exists():
            with open(KF_PATH) as f:
                st.session_state.keyframes = json.load(f)
        else:
            st.session_state.keyframes = []
        st.session_state._kf_species = active

    if "active_kf_idx" not in st.session_state:
        st.session_state.active_kf_idx = None
    if "canvas_version" not in st.session_state:
        st.session_state.canvas_version = 0

_init_state()

# ── Display helpers ────────────────────────────────────────────────────────────
def _make_overlay_image(
    frame_rgb: np.ndarray,
    parts: list,
    ann_masks: dict,
    frame_idx: int,
    display_width: int = DISPLAY_WIDTH,
) -> tuple[Image.Image, float]:
    """Resize frame and overlay saved masks. Returns (PIL Image, scale_factor)."""
    import cv2 as _cv2
    orig_h, orig_w = frame_rgb.shape[:2]
    scale      = display_width / orig_w
    display_h  = max(1, int(orig_h * scale))

    small   = _cv2.resize(frame_rgb, (display_width, display_h))
    overlay = small.copy().astype(float)

    for p in parts:
        key = f"{frame_idx}_{p['name']}"
        if key not in ann_masks:
            continue
        orig_mask = ann_masks[key]
        disp_mask = _cv2.resize(orig_mask.astype(np.uint8), (display_width, display_h)) > 0
        hx = p["color"].lstrip("#")
        r, g, b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
        overlay[disp_mask, 0] = overlay[disp_mask, 0] * 0.45 + r * 0.55
        overlay[disp_mask, 1] = overlay[disp_mask, 1] * 0.45 + g * 0.55
        overlay[disp_mask, 2] = overlay[disp_mask, 2] * 0.45 + b * 0.55

    img = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
    return img, scale

def _browse_image(
    frame_rgb: np.ndarray,
    kf: dict,
    parts: list,
    ann_masks: dict,
    frame_idx: int,
) -> Image.Image:
    """Browse mode: show masks + legacy dot annotations."""
    img, scale = _make_overlay_image(frame_rgb, parts, ann_masks, frame_idx)
    draw = ImageDraw.Draw(img)
    for p in parts:
        coords = kf.get(p["name"])
        if not isinstance(coords, list):
            continue
        x, y, r = int(coords[0] * scale), int(coords[1] * scale), 8
        draw.ellipse([x - r, y - r, x + r, y + r], fill=p["color"], outline="white", width=2)
        draw.text((x + 12, y - 8), p["name"], fill=p["color"])
    return img

# ── Apply pending slider jump BEFORE slider renders ───────────────────────────
_slider_key  = f"fslider_{active}"
_pending_key = f"_pending_frame_{active}"
if _pending_key in st.session_state:
    st.session_state[_slider_key] = st.session_state.pop(_pending_key)

# ── Layout ─────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([2, 3])

with col_left:
    st.subheader("Parts")
    if not _HAS_CANVAS:
        st.warning("Run `pip install streamlit-drawable-canvas` to enable painting.")

    parts_to_delete = []
    for i, part in enumerate(st.session_state.parts):
        c1, c2, c3 = st.columns([3, 2, 1])
        with c1:
            new_name = st.text_input("Name", value=part["name"], key=f"pname_{active}_{i}",
                                     label_visibility="collapsed")
        with c2:
            new_color = st.color_picker("Color", value=part["color"], key=f"pcolor_{active}_{i}",
                                        label_visibility="collapsed")
        with c3:
            if st.button("X", key=f"pdel_{active}_{i}"):
                parts_to_delete.append(i)
        if new_name != part["name"] or new_color != part["color"]:
            st.session_state.parts[i] = {"name": new_name, "color": new_color}

    for i in reversed(parts_to_delete):
        st.session_state.parts.pop(i)
        st.rerun()

    if st.button("+ Add part"):
        idx = len(st.session_state.parts)
        st.session_state.parts.append({
            "name": f"part_{idx + 1}",
            "color": DEFAULT_COLORS[idx % len(DEFAULT_COLORS)],
        })
        st.rerun()

    st.divider()

    # ── Keyframe list ──────────────────────────────────────────────────────────
    st.subheader("Keyframes")

    frame_num = st.slider("Browse frames", 0, max(n_frames - 1, 0), 0, key=_slider_key)

    c_add, c_num = st.columns([2, 1])
    with c_num:
        add_frame = st.number_input("Frame #", 0, n_frames - 1, frame_num,
                                    key=f"add_frame_{active}", label_visibility="collapsed")
    with c_add:
        if st.button("Add keyframe", use_container_width=True):
            existing_frames = [kf["frame"] for kf in st.session_state.keyframes]
            if add_frame in existing_frames:
                st.warning(f"Frame {add_frame} already added.")
            else:
                new_kf = {"frame": add_frame}
                for p in st.session_state.parts:
                    new_kf[p["name"]] = None
                st.session_state.keyframes.append(new_kf)
                st.session_state.keyframes.sort(key=lambda x: x["frame"])
                new_idx = next(i for i, kf in enumerate(st.session_state.keyframes)
                               if kf["frame"] == add_frame)
                st.session_state.active_kf_idx = new_idx
                st.session_state[_pending_key] = add_frame
                st.rerun()

    for i, kf in enumerate(st.session_state.keyframes):
        ann_masks_check = _load_ann_masks()
        parts_done = sum(
            1 for p in st.session_state.parts
            if kf.get(p["name"]) is not None or f"{kf['frame']}_{p['name']}" in ann_masks_check
        )
        total_parts = len(st.session_state.parts)
        status = f"{parts_done}/{total_parts} painted"

        c_frame, c_btn, c_del = st.columns([3, 2, 1])
        with c_frame:
            is_active = (st.session_state.active_kf_idx == i)
            label = f"**Frame {kf['frame']}**" if is_active else f"Frame {kf['frame']}"
            st.markdown(f"{label}  \n_{status}_")
        with c_btn:
            if st.button("Paint" if not is_active else "Re-paint", key=f"ann_btn_{i}"):
                st.session_state.active_kf_idx = i
                st.session_state.canvas_version += 1
                st.session_state[_pending_key] = kf["frame"]
                st.rerun()
        with c_del:
            if st.button("X", key=f"kf_del_{i}"):
                st.session_state.keyframes.pop(i)
                if st.session_state.active_kf_idx == i:
                    st.session_state.active_kf_idx = None
                st.rerun()

    st.divider()

    if st.button("Save keyframes.json", type="primary", use_container_width=True):
        with open(KF_PATH, "w") as f:
            json.dump(st.session_state.keyframes, f, indent=2)
        with open(PARTS_PATH, "w") as f:
            json.dump(st.session_state.parts, f, indent=2)
        st.success(f"Saved {len(st.session_state.keyframes)} keyframes to Data/{active}/")

# ── Right column ───────────────────────────────────────────────────────────────
with col_right:
    active_kf_idx = st.session_state.active_kf_idx
    frame_rgb     = read_frame_rgb(FRAMES_DIR, frame_num)

    ann_kf = (st.session_state.keyframes[active_kf_idx]
              if active_kf_idx is not None and active_kf_idx < len(st.session_state.keyframes)
              else None)
    in_ann = ann_kf is not None and ann_kf["frame"] == frame_num

    if in_ann:
        kf    = ann_kf
        parts = st.session_state.parts
        fidx  = kf["frame"]

        c_hdr, c_stop = st.columns([5, 1])
        with c_hdr:
            st.subheader(f"Painting frame {fidx}")
        with c_stop:
            if st.button("Stop", key="stop_ann"):
                st.session_state.active_kf_idx = None
                st.rerun()

        if frame_rgb is None:
            st.error(f"Frame {fidx} not found in frames/")
        elif not _HAS_CANVAS:
            st.error("Install streamlit-drawable-canvas: `pip install streamlit-drawable-canvas`")
        else:
            ann_masks = _load_ann_masks()

            # Part selector
            sel_idx = st.radio(
                "Part to paint:",
                range(len(parts)),
                format_func=lambda i: parts[i]["name"],
                horizontal=True,
                key=f"sel_part_{active}_{fidx}",
            )
            current_part = parts[sel_idx]

            # Status chips
            status_cols = st.columns(max(len(parts), 1))
            for ci, p in enumerate(parts):
                with status_cols[ci]:
                    has_mask = f"{fidx}_{p['name']}" in ann_masks
                    has_val  = kf.get(p["name"]) is not None
                    if has_mask or has_val:
                        st.markdown(
                            f'<span style="color:{p["color"]}">✓ {p["name"]}</span>',
                            unsafe_allow_html=True,
                        )
                    elif ci == sel_idx:
                        st.markdown(
                            f'<span style="color:{p["color"]}"><b>→ {p["name"]}</b></span>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<span style="color:gray">○ ' + p["name"] + '</span>',
                            unsafe_allow_html=True,
                        )

            # Background: frame with previously saved masks overlaid
            bg_img, display_scale = _make_overlay_image(
                frame_rgb, parts, ann_masks, fidx, DISPLAY_WIDTH
            )

            st.info(
                f"Paint **{current_part['name'].upper()}** using the brush below, "
                f"then click **Confirm**. Switch parts with the radio buttons above."
            )

            brush_size = st.slider("Brush size", 5, 60, 25, key=f"brush_{active}_{fidx}")

            canvas_key = (
                f"canvas_{active}_{fidx}_{sel_idx}_{st.session_state.canvas_version}"
            )
            canvas_result = st_canvas(
                background_image=bg_img,
                height=bg_img.height,
                width=bg_img.width,
                drawing_mode="freedraw",
                stroke_color=current_part["color"],
                stroke_width=brush_size,
                fill_color="rgba(0,0,0,0)",
                key=canvas_key,
                display_toolbar=False,
            )

            # Cache canvas data in session state so Confirm can read it
            pending_cv_key = f"_cv_data_{canvas_key}"
            if (canvas_result.image_data is not None and
                    int(canvas_result.image_data[:, :, 3].max()) > 0):
                st.session_state[pending_cv_key] = canvas_result.image_data

            c_confirm, c_clear, c_skip = st.columns(3)

            with c_confirm:
                if st.button(f"Confirm: {current_part['name']}", type="primary",
                              use_container_width=True):
                    img_data = st.session_state.get(pending_cv_key)
                    if img_data is None:
                        img_data = canvas_result.image_data
                    if img_data is not None and int(img_data[:, :, 3].max()) > 0:
                        import cv2 as _cv2
                        disp_mask = img_data[:, :, 3] > 0
                        orig_h, orig_w = frame_rgb.shape[:2]
                        orig_mask = _cv2.resize(
                            disp_mask.astype(np.uint8), (orig_w, orig_h)
                        ) > 0
                        _save_ann_mask(fidx, current_part["name"], orig_mask)
                        kf[current_part["name"]] = "masked"
                        st.session_state.keyframes[active_kf_idx] = kf
                        st.session_state.pop(pending_cv_key, None)
                        st.session_state.canvas_version += 1
                        st.rerun()
                    else:
                        st.warning("Paint the region first, then confirm.")

            with c_clear:
                if st.button("Clear canvas", use_container_width=True):
                    st.session_state.pop(pending_cv_key, None)
                    st.session_state.canvas_version += 1
                    st.rerun()

            with c_skip:
                if st.button("Skip (not visible)", use_container_width=True):
                    kf[current_part["name"]] = None
                    st.session_state.keyframes[active_kf_idx] = kf
                    # Explicitly mark as skipped with sentinel so it counts as done
                    kf[current_part["name"]] = "skip"
                    st.session_state.canvas_version += 1
                    st.rerun()

            # Erase button for a specific part
            st.divider()
            st.caption("Remove a saved mask:")
            erase_cols = st.columns(max(len(parts), 1))
            for ci, p in enumerate(parts):
                with erase_cols[ci]:
                    has_mask = f"{fidx}_{p['name']}" in ann_masks
                    if has_mask:
                        if st.button(f"Erase {p['name']}", key=f"erase_{active}_{fidx}_{ci}"):
                            _delete_ann_mask(fidx, p["name"])
                            kf[p["name"]] = None
                            st.session_state.keyframes[active_kf_idx] = kf
                            st.rerun()

    else:
        # Browse mode
        if ann_kf is not None:
            st.caption(
                f"Browsing frame {frame_num}  |  "
                f"painting frame {ann_kf['frame']} — "
                f"move slider there to continue"
            )
        else:
            st.subheader("Frame browser")

        if frame_rgb is not None:
            ann_masks = _load_ann_masks()
            kf_for_frame = next(
                (kf for kf in st.session_state.keyframes if kf["frame"] == frame_num), {}
            )
            img = _browse_image(
                frame_rgb, kf_for_frame, st.session_state.parts, ann_masks, frame_num
            )
            st.image(img, caption=f"Frame {frame_num}", use_container_width=True)
        else:
            st.info("Select a frame using the slider on the left.")
