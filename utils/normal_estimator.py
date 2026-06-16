"""
Local surface normal estimation from video frames.

Method priority (selected automatically based on what's installed):

  stablenormal  StableNormal YOSO — fast single-pass model, produces
                colorful XYZ-encoded normal maps.  Requires:
                  pip install git+https://github.com/Stable-X/StableNormal.git diffusers
                  pip install torch>=2.0 torchvision

  depth         Depth Anything V2 Small — derives normals from depth via
                Sobel gradients.  Requires torch>=2.4 + transformers>=5.

  gradient      Pure OpenCV Sobel — no torch at all.  Works immediately
                with the existing venv.  Roughness only (no normal images).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd


# ── Capability detection ───────────────────────────────────────────────────────

def _stablenormal_available() -> bool:
    try:
        from stablenormal.pipeline_yoso_normal import YOSONormalsPipeline  # noqa
        import torch  # noqa
        return True
    except Exception:
        return False


def _torch_available() -> bool:
    try:
        import torch
        import transformers  # noqa
        major, minor = (int(x) for x in torch.__version__.split(".")[:2])
        return (major, minor) >= (2, 4)
    except Exception:
        return False


def _best_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def active_method() -> str:
    if _stablenormal_available():
        return "stablenormal"
    if _torch_available():
        return "depth"
    return "gradient"


def check_dependencies() -> tuple[bool, str]:
    method = active_method()
    if method == "stablenormal":
        return True, "stablenormal"
    if method == "depth":
        return True, "depth"
    return True, (
        "gradient — StableNormal not installed.  \n"
        "For full normal-map visualization, install:\n"
        "```\npip install git+https://github.com/Stable-X/StableNormal.git diffusers\n"
        "pip install torch torchvision\n```"
    )


# ── StableNormal YOSO ──────────────────────────────────────────────────────────

def _load_stablenormal(device: str):
    import torch
    from stablenormal.pipeline_yoso_normal import YOSONormalsPipeline
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    pipe = YOSONormalsPipeline.from_pretrained(
        "Stable-X/yoso-normal-v0-3",
        trust_remote_code=True,
        variant="fp16" if dtype == torch.float16 else None,
        torch_dtype=dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _normal_img_to_array(normal_img, orig_w: int, orig_h: int) -> np.ndarray:
    """Convert StableNormal output (PIL or tensor) to (H, W, 3) float32 in [-1, 1]."""
    from PIL import Image as PILImage
    if hasattr(normal_img, "cpu"):              # torch tensor → PIL
        normal_img = normal_img.cpu()
        if hasattr(normal_img, "numpy"):
            arr = normal_img.numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = arr.transpose(1, 2, 0)   # CHW → HWC
            normal_img = PILImage.fromarray((arr * 127.5 + 127.5).clip(0, 255).astype(np.uint8))
    if not isinstance(normal_img, np.ndarray):
        normal_img = PILImage.fromarray(np.array(normal_img)) if not hasattr(normal_img, "size") else normal_img
        if normal_img.size != (orig_w, orig_h):
            normal_img = normal_img.resize((orig_w, orig_h), PILImage.BILINEAR)
        normal_img = np.array(normal_img)
    arr = normal_img.astype(np.float32)
    if arr.max() > 1.5:                        # 0-255 scale → normalise
        arr = arr / 127.5 - 1.0
    # resize if needed
    if arr.shape[0] != orig_h or arr.shape[1] != orig_w:
        arr = cv2.resize(arr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    # ensure unit normals
    norms = np.linalg.norm(arr, axis=2, keepdims=True)
    arr /= np.maximum(norms, 1e-8)
    return arr


def _part_stats_from_normals(normals: np.ndarray, mask: np.ndarray) -> dict:
    pix    = normals[mask]
    mean_n = pix.mean(axis=0)
    mean_n /= max(float(np.linalg.norm(mean_n)), 1e-8)
    dots   = np.clip((pix * mean_n).sum(axis=1), -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    return {
        "nx":        round(float(mean_n[0]), 5),
        "ny":        round(float(mean_n[1]), 5),
        "nz":        round(float(mean_n[2]), 5),
        "roughness": round(float(np.std(angles)), 3),
    }


# ── Depth-based normals (Depth Anything V2) ────────────────────────────────────

def _depth_to_normals(depth: np.ndarray) -> np.ndarray:
    d = depth.astype(np.float32)
    dmin, dmax = d.min(), d.max()
    if dmax > dmin:
        d = (d - dmin) / (dmax - dmin)
    dzdx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=5)
    dzdy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=5)
    normal = np.dstack((-dzdx, -dzdy, np.ones_like(d)))
    norms  = np.linalg.norm(normal, axis=2, keepdims=True)
    normal /= np.maximum(norms, 1e-8)
    return normal


# ── Gradient-based roughness (no torch) ───────────────────────────────────────

def _gradient_stats(bgr: np.ndarray, mask: np.ndarray) -> dict:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag  = np.sqrt(gx**2 + gy**2)
    pix  = mag[mask]
    if len(pix) == 0:
        return {"nx": np.nan, "ny": np.nan, "nz": np.nan, "roughness": np.nan}
    mean_gx = float(gx[mask].mean())
    mean_gy = float(gy[mask].mean())
    nx, ny, nz = -mean_gx, -mean_gy, 255.0
    n = max(float(np.linalg.norm([nx, ny, nz])), 1e-8)
    return {
        "nx":        round(nx / n, 5),
        "ny":        round(ny / n, 5),
        "nz":        round(nz / n, 5),
        "roughness": round(float(np.std(pix)), 3),
    }


# ── Main estimation function ───────────────────────────────────────────────────

def run_local_normal_estimation(
    frames_dir: Path,
    masks_path: Path,
    output_path: Path,
    frame_stride: int = 1,
    progress_callback: Callable[[int, int, str], None] | None = None,
    n_sample_maps: int = 20,
) -> pd.DataFrame:
    """
    Estimate surface normals for every body part in every frame.

    When StableNormal is installed, also saves coloured normal-map PNG images
    to <output_path.parent>/normal_maps/ for visual inspection in the app.

    Parameters
    ----------
    frames_dir       : directory with 00000.jpg, 00001.jpg, …
    masks_path       : masks.npz from the Segment step
    output_path      : where to write normal_data.csv
    frame_stride     : process every Nth frame
    progress_callback: optional fn(current, total, message)
    n_sample_maps    : how many evenly-spaced normal-map images to save
    """
    from PIL import Image as PILImage

    method = active_method()

    def _progress(i: int, n: int, msg: str = "") -> None:
        if progress_callback:
            progress_callback(i, n, msg)

    # ── Load masks ──────────────────────────────────────────────────────────
    _progress(0, 1, "Loading masks…")
    data = np.load(str(masks_path), allow_pickle=False)
    video_segments: dict = {}
    parts: set = set()
    for key in data.files:
        if key.startswith("_"):
            continue
        fidx_str, part = key.split("_", 1)
        video_segments.setdefault(int(fidx_str), {})[part] = data[key].astype(bool)
        parts.add(part)
    parts_sorted = sorted(parts)

    frame_idxs = sorted(video_segments.keys())
    if frame_stride > 1:
        frame_idxs = frame_idxs[::frame_stride]

    # Indices at which to save normal-map images
    if n_sample_maps > 0 and len(frame_idxs) > 0:
        step = max(1, len(frame_idxs) // n_sample_maps)
        save_set = set(frame_idxs[::step][:n_sample_maps])
    else:
        save_set = set()

    # Normal maps output directory
    maps_dir = output_path.parent / "normal_maps"
    if save_set:
        maps_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────
    pipe = None
    depth_pipe = None

    if method == "stablenormal":
        device = _best_device()
        _progress(0, 1,
                  f"Loading StableNormal YOSO on {device.upper()}…  "
                  "(first run downloads the model — cached afterwards)")
        pipe = _load_stablenormal(device)

    elif method == "depth":
        import torch  # noqa: F401 — needed in scope for depth branch
        from transformers import pipeline as hf_pipeline
        device = _best_device()
        _progress(0, 1,
                  f"Loading Depth Anything V2 Small on {device.upper()}…  "
                  "(first run downloads ~400 MB — cached afterwards)")
        depth_pipe = hf_pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=device,
        )

    else:
        _progress(0, 1, "Using gradient method (install StableNormal for visual normal maps).")

    # ── Process frames ──────────────────────────────────────────────────────
    total = len(frame_idxs)
    rows: list[dict] = []

    for i, fidx in enumerate(frame_idxs):
        frame_path = frames_dir / f"{fidx:05d}.jpg"
        if not frame_path.exists():
            _progress(i + 1, total, f"Frame {fidx} not found, skipping")
            continue

        _progress(i, total, f"Frame {fidx}  ({i + 1}/{total})")

        bgr    = cv2.imread(str(frame_path))
        if bgr is None:
            continue
        orig_h, orig_w = bgr.shape[:2]

        row    = {"frame": fidx, "method": method}
        segs   = video_segments.get(fidx, {})
        normals: np.ndarray | None = None

        # ── StableNormal ────────────────────────────────────────────────────
        if method == "stablenormal":
            img_pil = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            result  = pipe(img_pil)
            # Pipeline may return PIL image directly or wrapped in a list/dict
            if isinstance(result, (list, tuple)):
                result = result[0]
            if isinstance(result, dict):
                result = result.get("normal_map") or result.get("image") or list(result.values())[0]
            normals = _normal_img_to_array(result, orig_w, orig_h)

            if fidx in save_set:
                # Save the RGB-encoded normal map image
                normal_rgb = ((normals + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                PILImage.fromarray(normal_rgb).save(str(maps_dir / f"{fidx:05d}_normal.png"))

        # ── Depth Anything V2 ───────────────────────────────────────────────
        elif method == "depth":
            img_pil   = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            result    = depth_pipe(img_pil)
            depth_raw = result["depth"]
            if hasattr(depth_raw, "cpu"):
                depth_raw = depth_raw.cpu()
            if hasattr(depth_raw, "numpy"):
                depth_arr = depth_raw.numpy().astype(np.float32)
            elif isinstance(depth_raw, np.ndarray):
                depth_arr = depth_raw.astype(np.float32)
            else:
                depth_arr = np.array(depth_raw).astype(np.float32)
            if depth_arr.shape[0] != orig_h or depth_arr.shape[1] != orig_w:
                depth_arr = cv2.resize(depth_arr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            normals = _depth_to_normals(depth_arr)

        # ── Compute per-part stats ──────────────────────────────────────────
        for part in parts_sorted:
            mask = segs.get(part)
            if mask is None or not mask.any():
                for s in ("nx", "ny", "nz", "roughness"):
                    row[f"{part}_{s}"] = np.nan
            elif normals is not None:
                for s, v in _part_stats_from_normals(normals, mask).items():
                    row[f"{part}_{s}"] = v
            else:                             # gradient fallback
                for s, v in _gradient_stats(bgr, mask).items():
                    row[f"{part}_{s}"] = v

        rows.append(row)

    _progress(total, total, "Saving results…")

    df = pd.DataFrame(rows).set_index("frame")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(output_path))

    n_maps = len(list(maps_dir.glob("*_normal.png"))) if maps_dir.exists() else 0
    _progress(total, total,
              f"Done — {len(df)} frames processed"
              + (f", {n_maps} normal map images saved" if n_maps else ""))
    return df
