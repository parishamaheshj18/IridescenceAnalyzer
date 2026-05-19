#!/usr/bin/env python3
"""
hummingbird_analysis.py  —  local color analysis
Run: python hummingbird_analysis.py

Requires hummingbird_masks.npz in the same folder.
Saves all figures + CSV to output/
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from pathlib import Path

try:
    from skimage import color as skcolor
except ImportError:
    sys.exit('Missing dependency: pip install scikit-image')

# ── Config ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
VIDEO_PATH  = PROJECT_DIR / 'Hummingbird.mov'
FRAMES_DIR  = PROJECT_DIR / 'frames'
MASKS_PATH  = PROJECT_DIR / 'hummingbird_masks.npz'
OUTPUT_DIR  = PROJECT_DIR / 'output'

# Lighting normalisation: set True + adjust coordinates to use a static
# background patch instead of whole-frame median
USE_BACKGROUND_PATCH  = False
BG_X1, BG_Y1 = 10, 10
BG_X2, BG_Y2 = 60, 60

# Drop masks smaller than this fraction of their per-part median area
AREA_THRESHOLD_PCT = 0.25

MASK_COLORS = {'crown': np.array([1, 0, 0]),
               'back':  np.array([0, 1, 0]),
               'tail':  np.array([0, 1, 1])}
COLOR_MAP   = {'crown': 'tomato', 'back': 'mediumseagreen', 'tail': 'steelblue'}

FRAMES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

assert VIDEO_PATH.exists(), f'Video not found: {VIDEO_PATH}'
assert MASKS_PATH.exists(), (
    f'Masks not found: {MASKS_PATH}\n'
    '  → Run hummingbird_sam2_segment.ipynb in Colab first,\n'
    '    then download hummingbird_masks.npz into this folder.'
)

# ── 1. Extract frames (skip if already done) ───────────────────────────────────
existing = sorted(FRAMES_DIR.glob('*.jpg'))
cap = cv2.VideoCapture(str(VIDEO_PATH))
fps = cap.get(cv2.CAP_PROP_FPS)

if existing:
    cap.release()
    print(f'Frames already extracted: {len(existing)} at {fps:.1f} FPS')
else:
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Extracting {total} frames  ({w}x{h} @ {fps:.1f} FPS)...')
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(FRAMES_DIR / f'{saved:05d}.jpg'), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1
    cap.release()
    print(f'Extracted {saved} frames → {FRAMES_DIR}')

# ── 2. Load masks ──────────────────────────────────────────────────────────────
print(f'\nLoading {MASKS_PATH.name} ...')
data = np.load(str(MASKS_PATH), allow_pickle=False)

fps  = float(data['_fps'][0])   if '_fps'      in data.files else fps
fw   = int(data['_frame_wh'][0]) if '_frame_wh' in data.files else 0
fh   = int(data['_frame_wh'][1]) if '_frame_wh' in data.files else 0
print(f'FPS: {fps:.1f}  |  frame size: {fw}x{fh}')

video_segments: dict[int, dict[str, np.ndarray]] = {}
for key in data.files:
    if key.startswith('_'):
        continue
    fidx_str, part = key.split('_', 1)
    video_segments.setdefault(int(fidx_str), {})[part] = data[key].astype(bool)

parts_available = sorted({p for segs in video_segments.values() for p in segs})
print(f'Loaded {len(video_segments)} frames  |  parts: {parts_available}')

# ── 3. Verify masks ────────────────────────────────────────────────────────────
frames_list = sorted(video_segments.keys())
n           = len(frames_list)
samples     = [frames_list[int(i * (n - 1) / 4)] for i in range(5)]

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for ax, fidx in zip(axes, samples):
    bgr = cv2.imread(str(FRAMES_DIR / f'{fidx:05d}.jpg'))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(float) / 255.0
    overlay = rgb.copy()
    for part, mask in video_segments[fidx].items():
        c = MASK_COLORS.get(part, np.array([1, 1, 0]))
        overlay[mask] = overlay[mask] * 0.5 + c * 0.5
    ax.imshow(overlay)
    ax.set_title(f'Frame {fidx}', fontsize=9)
    ax.axis('off')

patches = [mpatches.Patch(color=MASK_COLORS[p], label=p)
           for p in parts_available if p in MASK_COLORS]
fig.legend(handles=patches, loc='lower center', ncol=3)
plt.suptitle('SAM 2 Masks', fontsize=13)
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / 'mask_preview.png'), dpi=150)
plt.close()
print('Saved: output/mask_preview.png')

# ── 4. Mask quality filter ─────────────────────────────────────────────────────
area_data = {part: {} for part in parts_available}
for fidx, segs in video_segments.items():
    for part, mask in segs.items():
        area_data[part][fidx] = int(mask.sum())

thresholds = {}
fig, axes = plt.subplots(len(parts_available), 1,
                         figsize=(16, 3 * len(parts_available)), sharex=True)
if len(parts_available) == 1:
    axes = [axes]

for ax, part in zip(axes, parts_available):
    areas       = pd.Series(area_data[part]).sort_index()
    median_area = areas[areas > 0].median()
    threshold   = median_area * AREA_THRESHOLD_PCT
    thresholds[part] = threshold
    ax.fill_between(areas.index, areas.values,
                    alpha=0.4, color=COLOR_MAP.get(part, 'gray'))
    ax.plot(areas.index, areas.values,
            color=COLOR_MAP.get(part, 'gray'), linewidth=0.8)
    ax.axhline(threshold, color='red', linewidth=1.5, linestyle='--',
               label=f'threshold ({AREA_THRESHOLD_PCT:.0%} of median)')
    ax.set_ylabel('Mask area (px)')
    ax.set_title(f'{part}  —  median: {median_area:.0f} px, threshold: {threshold:.0f} px')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

axes[-1].set_xlabel('Frame')
plt.suptitle('Mask Area Over Time', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / 'mask_area.png'), dpi=150)
plt.close()
print('Saved: output/mask_area.png')

filtered = 0
for fidx in video_segments:
    for part in list(video_segments[fidx].keys()):
        if area_data[part].get(fidx, 0) < thresholds.get(part, 0):
            video_segments[fidx][part] = np.zeros_like(video_segments[fidx][part])
            filtered += 1

total_slots = sum(len(v) for v in video_segments.values())
print(f'Filtered {filtered} / {total_slots} mask slots as too small')

# ── 5. Color extraction + lighting normalisation (single pass) ─────────────────
print(f'\nExtracting colour from {len(video_segments)} frames...')

rows        = []
bg_L_by_frame = {}

for fidx in sorted(video_segments.keys()):
    frame_path = FRAMES_DIR / f'{fidx:05d}.jpg'
    if not frame_path.exists():
        continue

    bgr = cv2.imread(str(frame_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    lab = skcolor.rgb2lab(rgb)

    # Lighting reference for this frame
    if USE_BACKGROUND_PATCH:
        bg_L_by_frame[fidx] = float(lab[BG_Y1:BG_Y2, BG_X1:BG_X2, 0].mean())
    else:
        bg_L_by_frame[fidx] = float(np.median(lab[:, :, 0]))

    row = {'frame': fidx}
    for part in parts_available:
        mask = video_segments[fidx].get(part)
        if mask is None or not mask.any():
            row[f'{part}_L']      = np.nan
            row[f'{part}_A']      = np.nan
            row[f'{part}_B']      = np.nan
            row[f'{part}_hue']    = np.nan
            row[f'{part}_chroma'] = np.nan
        else:
            L = float(lab[:, :, 0][mask].mean())
            A = float(lab[:, :, 1][mask].mean())
            B = float(lab[:, :, 2][mask].mean())
            row[f'{part}_L']      = L
            row[f'{part}_A']      = A
            row[f'{part}_B']      = B
            row[f'{part}_hue']    = float(np.degrees(np.arctan2(B, A)) % 360)
            row[f'{part}_chroma'] = float(np.sqrt(A**2 + B**2))
    rows.append(row)

df = pd.DataFrame(rows).set_index('frame')
print(f'Colour data: {len(df)} frames')

# Apply lighting correction
bg_L         = pd.Series(bg_L_by_frame)
L_correction = bg_L.median() - bg_L

for part in parts_available:
    if f'{part}_L' in df.columns:
        df[f'{part}_L_norm'] = df[f'{part}_L'] + L_correction.reindex(df.index)

print(f'L* correction range: {L_correction.min():.1f} to {L_correction.max():.1f}')

# ── 6. Save CSV ────────────────────────────────────────────────────────────────
csv_out = OUTPUT_DIR / 'hummingbird_color_data.csv'
df.to_csv(str(csv_out))
print(f'Saved: {csv_out}')

# ── 7. Plot: hue / chroma / brightness over time ──────────────────────────────
parts = [p for p in ['crown', 'back', 'tail'] if f'{p}_hue' in df.columns]

fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
for part in parts:
    c     = COLOR_MAP.get(part, 'gray')
    L_col = f'{part}_L_norm' if f'{part}_L_norm' in df.columns else f'{part}_L'
    axes[0].plot(df.index, df[f'{part}_hue'],    label=part, color=c, alpha=0.8)
    axes[1].plot(df.index, df[f'{part}_chroma'], label=part, color=c, alpha=0.8)
    axes[2].plot(df.index, df[L_col],            label=part, color=c, alpha=0.8)

axes[0].set_ylabel('Hue angle (°)')
axes[0].set_title('Iridescent Hue Shift')
axes[1].set_ylabel('Chroma')
axes[1].set_title('Color Saturation')
axes[2].set_ylabel('L* (normalised)')
axes[2].set_title('Brightness (lighting-corrected)')
axes[2].set_xlabel('Frame')
for ax in axes:
    ax.legend()
    ax.grid(alpha=0.3)

plt.suptitle('Hummingbird Body Part Colour Dynamics', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / 'color_shifts.png'), dpi=150)
print('Saved: output/color_shifts.png')

# ── 8. Plot: AB colour trajectory ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
markers = {'crown': 'o', 'back': 's', 'tail': '^'}
sc = None
for part in parts:
    valid = df[[f'{part}_A', f'{part}_B']].dropna()
    if valid.empty:
        continue
    sc = ax.scatter(valid[f'{part}_A'], valid[f'{part}_B'],
                    c=valid.index, cmap='plasma', label=part,
                    alpha=0.7, s=12, marker=markers.get(part, 'o'))

if sc is not None:
    plt.colorbar(sc, ax=ax, label='Frame number')
ax.set_xlabel('A*  (green ← → red)')
ax.set_ylabel('B*  (blue ← → yellow)')
ax.set_title('AB Colour Trajectory — Iridescence Fingerprint', fontsize=12)
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.legend()
ax.set_aspect('equal')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / 'ab_trajectory.png'), dpi=150)
print('Saved: output/ab_trajectory.png')

plt.show()

# ── 9. Export mask overlay video ──────────────────────────────────────────────
VIDEO_OUT = OUTPUT_DIR / 'mask_overlay.mp4'
print(f'\nExporting mask overlay video → {VIDEO_OUT}')

# Determine frame size from first available frame
all_frame_idxs = sorted(
    int(p.stem) for p in FRAMES_DIR.glob('*.jpg')
)

sample_bgr = cv2.imread(str(FRAMES_DIR / f'{all_frame_idxs[0]:05d}.jpg'))
h_out, w_out = sample_bgr.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264, reliable on macOS
writer = cv2.VideoWriter(str(VIDEO_OUT), fourcc, fps, (w_out, h_out))
if not writer.isOpened():
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # fallback
    writer = cv2.VideoWriter(str(VIDEO_OUT), fourcc, fps, (w_out, h_out))

OVERLAY_ALPHA = 0.5  # mask opacity

for fidx in all_frame_idxs:
    bgr = cv2.imread(str(FRAMES_DIR / f'{fidx:05d}.jpg'))
    if bgr is None:
        continue

    segs = video_segments.get(fidx)
    if segs:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(float) / 255.0
        overlay = rgb.copy()
        for part, mask in segs.items():
            if mask.any():
                c = MASK_COLORS.get(part, np.array([1, 1, 0]))
                overlay[mask] = overlay[mask] * (1 - OVERLAY_ALPHA) + c * OVERLAY_ALPHA
        out_bgr = cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        out_bgr = bgr

    writer.write(out_bgr)

writer.release()
print(f'Saved: output/mask_overlay.mp4  ({len(all_frame_idxs)} frames @ {fps:.1f} FPS)')
