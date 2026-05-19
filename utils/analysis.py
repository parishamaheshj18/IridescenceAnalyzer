import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from skimage import color as skcolor


def load_masks(masks_path: Path):
    """Returns (video_segments, parts, fps, frame_width, frame_height)."""
    data = np.load(str(masks_path), allow_pickle=False)
    fps = float(data["_fps"][0]) if "_fps" in data.files else 25.0
    fw = int(data["_frame_wh"][0]) if "_frame_wh" in data.files else 0
    fh = int(data["_frame_wh"][1]) if "_frame_wh" in data.files else 0
    video_segments: dict = {}
    for key in data.files:
        if key.startswith("_"):
            continue
        fidx_str, part = key.split("_", 1)
        video_segments.setdefault(int(fidx_str), {})[part] = data[key].astype(bool)
    parts = sorted({p for segs in video_segments.values() for p in segs})
    return video_segments, parts, fps, fw, fh


def filter_masks(video_segments: dict, parts: list, threshold_pct: float = 0.25):
    """Zero out masks smaller than threshold_pct of median area. Returns updated segments + stats."""
    area_data = {part: {} for part in parts}
    for fidx, segs in video_segments.items():
        for part, mask in segs.items():
            area_data[part][fidx] = int(mask.sum())

    thresholds = {}
    for part in parts:
        areas = pd.Series(area_data[part])
        valid = areas[areas > 0]
        thresholds[part] = float(valid.median() * threshold_pct) if len(valid) > 0 else 0.0

    filtered = 0
    for fidx in video_segments:
        for part in list(video_segments[fidx].keys()):
            if area_data[part].get(fidx, 0) < thresholds.get(part, 0):
                video_segments[fidx][part] = np.zeros_like(video_segments[fidx][part])
                filtered += 1

    return video_segments, thresholds, area_data, filtered


def extract_colors(
    frames_dir: Path,
    video_segments: dict,
    parts: list,
    use_bg_patch: bool = False,
    bg_coords: tuple = (10, 10, 60, 60),
) -> pd.DataFrame:
    rows = []
    bg_L_by_frame: dict = {}

    for fidx in sorted(video_segments.keys()):
        frame_path = frames_dir / f"{fidx:05d}.jpg"
        if not frame_path.exists():
            continue
        bgr = cv2.imread(str(frame_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        lab = skcolor.rgb2lab(rgb)

        if use_bg_patch:
            x1, y1, x2, y2 = bg_coords
            bg_L_by_frame[fidx] = float(lab[y1:y2, x1:x2, 0].mean())
        else:
            bg_L_by_frame[fidx] = float(np.median(lab[:, :, 0]))

        row: dict = {"frame": fidx}
        for part in parts:
            mask = video_segments[fidx].get(part)
            if mask is None or not mask.any():
                for col in [f"{part}_L", f"{part}_A", f"{part}_B", f"{part}_hue", f"{part}_chroma"]:
                    row[col] = np.nan
            else:
                L = float(lab[:, :, 0][mask].mean())
                A = float(lab[:, :, 1][mask].mean())
                B = float(lab[:, :, 2][mask].mean())
                row[f"{part}_L"] = L
                row[f"{part}_A"] = A
                row[f"{part}_B"] = B
                row[f"{part}_hue"] = float(np.degrees(np.arctan2(B, A)) % 360)
                row[f"{part}_chroma"] = float(np.sqrt(A**2 + B**2))
        rows.append(row)

    df = pd.DataFrame(rows).set_index("frame")
    bg_L = pd.Series(bg_L_by_frame)
    L_corr = bg_L.median() - bg_L
    for part in parts:
        if f"{part}_L" in df.columns:
            df[f"{part}_L_norm"] = df[f"{part}_L"] + L_corr.reindex(df.index)
    return df


def compute_iridescence_metrics(df: pd.DataFrame, parts: list) -> dict:
    metrics: dict = {}
    for part in parts:
        if f"{part}_hue" not in df.columns:
            continue
        hue = df[f"{part}_hue"].dropna()
        chroma = df[f"{part}_chroma"].dropna()
        L_col = f"{part}_L_norm" if f"{part}_L_norm" in df.columns else f"{part}_L"
        L = df[L_col].dropna() if L_col in df.columns else pd.Series(dtype=float)
        if len(hue) < 5:
            continue

        # Circular hue velocity (mean |frame-to-frame change|)
        hue_rad = np.radians(hue.values)
        diffs = np.diff(hue_rad)
        diffs = (diffs + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]
        hue_velocity = float(np.abs(np.degrees(diffs)).mean())

        # Circular hue spread (analogous to std on a circle)
        sin_m, cos_m = np.sin(hue_rad).mean(), np.cos(hue_rad).mean()
        R = float(np.clip(np.sqrt(sin_m**2 + cos_m**2), 1e-9, 1.0))
        hue_spread = float(np.degrees(np.sqrt(-2 * np.log(R))))
        hue_mean = float(np.degrees(np.arctan2(sin_m, cos_m)) % 360)

        chroma_mean = float(chroma.mean()) if len(chroma) > 0 else 0.0
        chroma_cv = float(chroma.std() / chroma_mean) if chroma_mean > 0 else 0.0

        metrics[part] = {
            "hue_mean_deg": round(hue_mean, 1),
            "hue_spread_deg": round(hue_spread, 1),
            "hue_velocity_deg_per_frame": round(hue_velocity, 3),
            "chroma_mean": round(chroma_mean, 2),
            "chroma_variability_pct": round(chroma_cv * 100, 1),
            "brightness_L_mean": round(float(L.mean()) if len(L) > 0 else 0.0, 1),
            "iridescence_index": round(hue_spread * chroma_mean / 100.0, 3),
            "frames_analyzed": int(len(hue)),
        }
    return metrics


def interpret_iridescence(metrics: dict) -> dict:
    results: dict = {}
    for part, m in metrics.items():
        ii = m["iridescence_index"]
        hs = m["hue_spread_deg"]
        ch = m["chroma_mean"]
        vel = m["hue_velocity_deg_per_frame"]

        if ii > 8 and hs > 40:
            level = "Strong"
            mechanism = "Thin-film interference / photonic crystal array"
            detail = (
                "Wide hue spread with high saturation is the hallmark of ordered nanostructures "
                "(e.g. hummingbird barbule platelets) producing constructive thin-film interference. "
                "Color shifts strongly with viewing angle."
            )
        elif ii > 4 and ch > 15:
            level = "Moderate"
            mechanism = "Thin-film interference (likely)"
            detail = (
                "Moderate hue variation with good saturation suggests single-layer thin-film interference, "
                "common in iridescent gorgets and butterfly wing scales."
            )
        elif vel > 1.5 and ch > 10:
            level = "Moderate"
            mechanism = "Angle-dependent structural coloration"
            detail = (
                "Rapid frame-to-frame hue shifts indicate strong viewing-angle sensitivity, "
                "consistent with platelet-based structural coloration where each movement "
                "changes the interference condition."
            )
        elif ch > 20:
            level = "Low-moderate"
            mechanism = "Pigment-based with possible structural enhancement"
            detail = (
                "High saturation but stable hue points to rich pigmentation (carotenoids, pterins). "
                "Minor structural contribution may be present but is not the dominant mechanism."
            )
        else:
            level = "Low"
            mechanism = "Primarily pigment-based coloration"
            detail = (
                "Low hue variation and moderate chroma are consistent with pigment-dominated "
                "coloration (melanin, carotenoids) with minimal structural contribution."
            )

        results[part] = {
            "level": level,
            "mechanism": mechanism,
            "detail": detail,
            "iridescence_index": m["iridescence_index"],
        }
    return results
