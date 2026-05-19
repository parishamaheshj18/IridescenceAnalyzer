#!/usr/bin/env python3
"""
hummingbird_annotate.py  —  local annotation tool
Run from the project folder:  python hummingbird_annotate.py

Phase 1 — Frame browser:  shows a grid of 20 frames; close the window when done.
Phase 2 — Type frames:    enter the frame numbers you want to annotate.
Phase 3 — Annotation:     one frame at a time; click each body part in order.
                           Press  N  = not visible   |   D or Enter = next frame

Saves: hummingbird_keyframes.json  (upload this to Google Drive, then run Colab)
"""

import json
import sys
import cv2
import numpy as np
from pathlib import Path
from matplotlib import patheffects
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR    = Path(__file__).resolve().parent
VIDEO_PATH     = PROJECT_DIR / 'Hummingbird.mov'
FRAMES_DIR     = PROJECT_DIR / 'frames'
KEYFRAMES_PATH = PROJECT_DIR / 'hummingbird_keyframes.json'

PARTS  = ['crown', 'back', 'tail']
COLORS = {'crown': '#ff5555', 'back': '#55ff55', 'tail': '#55ffff'}
_stroke = [patheffects.withStroke(linewidth=2, foreground='black')]

# ── Sanity check ──────────────────────────────────────────────────────────────
if not VIDEO_PATH.exists():
    sys.exit(f'ERROR: Video not found at {VIDEO_PATH}')

# ── 1. Extract frames ─────────────────────────────────────────────────────────
FRAMES_DIR.mkdir(exist_ok=True)
existing = sorted(FRAMES_DIR.glob('*.jpg'))

cap          = cv2.VideoCapture(str(VIDEO_PATH))
fps          = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if existing:
    cap.release()
    print(f'Frames already extracted: {len(existing)}  (delete frames/ to re-extract)')
else:
    print(f'Extracting {total_frames} frames from {VIDEO_PATH.name} ...')
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(FRAMES_DIR / f'{saved:05d}.jpg'), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1
    cap.release()
    print(f'Extracted {saved} frames.')

# ── 2. Frame browser ──────────────────────────────────────────────────────────
print('\nOpening frame browser — close the window when you have noted your frame numbers.')

N_SAMPLES   = 20
step        = max(1, total_frames // N_SAMPLES)
sample_idxs = list(range(0, total_frames, step))[:N_SAMPLES]

cols = 5
rows = -(-len(sample_idxs) // cols)
fig_b, axes = plt.subplots(rows, cols, figsize=(18, rows * 3.5))
fig_b.patch.set_facecolor('#111827')
fig_b.canvas.manager.set_window_title('Frame Browser — note frame numbers, then close')

for ax, fidx in zip(axes.flatten(), sample_idxs):
    bgr = cv2.imread(str(FRAMES_DIR / f'{fidx:05d}.jpg'))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ax.imshow(rgb)
    ax.set_title(str(fidx), fontsize=11, fontweight='bold', color='white', pad=4)
    ax.axis('off')
for ax in axes.flatten()[len(sample_idxs):]:
    ax.set_visible(False)

plt.suptitle('Frame overview — note 3–5 frame numbers at different bird orientations, then close',
             fontsize=12, color='white', y=1.01)
plt.tight_layout()
plt.show(block=True)

# ── 3. Ask which frames to annotate ──────────────────────────────────────────
print('\nEnter the frame numbers to annotate, separated by spaces.')
suggested = ' '.join(str(sample_idxs[i]) for i in range(0, len(sample_idxs), 4))
print(f'  e.g.: {suggested}')

raw = input('Frame numbers: ').strip()
if not raw:
    sys.exit('No frames entered. Exiting.')

frames_to_annotate = []
for token in raw.split():
    try:
        frames_to_annotate.append(int(token))
    except ValueError:
        print(f'  Skipping non-integer: {token!r}')

if not frames_to_annotate:
    sys.exit('No valid frame numbers. Exiting.')

print(f'\nAnnotating {len(frames_to_annotate)} frames: {frames_to_annotate}')
print('Controls:  click = mark part   |   N key = not visible   |   D or Enter = next frame\n')

# ── 4. Annotation loop ────────────────────────────────────────────────────────
KEYFRAMES = []

for fi, fidx in enumerate(frames_to_annotate):
    path = FRAMES_DIR / f'{fidx:05d}.jpg'
    if not path.exists():
        print(f'  WARNING: frame {fidx} not found, skipping.')
        continue

    kf  = {'frame': fidx}
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor('#111827')
    fig.canvas.manager.set_window_title(
        f'Annotate frame {fidx}  ({fi+1}/{len(frames_to_annotate)})'
    )
    ax.imshow(rgb)
    ax.axis('off')

    # Mutable state for this frame
    state = {'pi': 0}

    def _set_title(state=state, fidx=fidx, fi=fi):
        pi = state['pi']
        n  = len(frames_to_annotate)
        if pi < len(PARTS):
            p  = PARTS[pi]
            c  = COLORS[p]
            msg = (f'Frame {fidx}  ({fi+1}/{n})  —  '
                   f'click  {p.upper()}  |  N = not visible  |  D / Enter = done')
            fig.suptitle(msg, fontsize=10, color=c)
        else:
            fig.suptitle(
                f'Frame {fidx} — done. Press D or Enter (or close window) to continue.',
                fontsize=10, color='#66ff66'
            )
        fig.canvas.draw_idle()

    _set_title()

    def _on_click(event, state=state, kf=kf, ax=ax, fig=fig):
        if event.inaxes != ax or state['pi'] >= len(PARTS):
            return
        part = PARTS[state['pi']]
        x, y = int(round(event.xdata)), int(round(event.ydata))
        kf[part] = [x, y]
        dot = ax.plot(x, y, 'o', color=COLORS[part], ms=13, mec='white', mew=2, zorder=5)
        t   = ax.text(x + 7, y - 7, part, color=COLORS[part],
                      fontsize=11, fontweight='bold', zorder=5)
        t.set_path_effects(_stroke)
        state['pi'] += 1
        _set_title()

    def _on_key(event, state=state, kf=kf, ax=ax, fig=fig):
        key = (event.key or '').lower()
        if key == 'n' and state['pi'] < len(PARTS):
            part = PARTS[state['pi']]
            kf[part] = None
            row = state['pi']
            t   = ax.text(0.02, 0.97 - row * 0.07, f'✕  {part}: not visible',
                          color=COLORS[part], fontsize=10, fontweight='bold',
                          transform=ax.transAxes, va='top', zorder=5)
            t.set_path_effects(_stroke)
            state['pi'] += 1
            _set_title()
        elif key in ('d', 'enter', ' '):
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', _on_click)
    fig.canvas.mpl_connect('key_press_event', _on_key)
    plt.tight_layout()
    plt.show(block=True)

    # Any parts not clicked → None
    for p in PARTS:
        kf.setdefault(p, None)

    KEYFRAMES.append(kf)
    summary = ', '.join(
        f'{p}={kf[p]}' if kf[p] is not None else f'{p}=hidden'
        for p in PARTS
    )
    print(f'  Saved frame {fidx}: {summary}')

# ── 5. Save JSON ──────────────────────────────────────────────────────────────
if not KEYFRAMES:
    sys.exit('No frames annotated. Nothing saved.')

with open(KEYFRAMES_PATH, 'w') as f:
    json.dump(KEYFRAMES, f, indent=2)

print(f'\nSaved {len(KEYFRAMES)} keyframes → {KEYFRAMES_PATH}')
print()
print('Next steps:')
print('  1. Upload hummingbird_keyframes.json to Google Drive')
print('     (drag it into the "HummingBird Project" folder on drive.google.com)')
print('  2. Run hummingbird_sam2_segment.ipynb in Colab')
