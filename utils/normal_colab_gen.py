def _code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": source}


def _md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def generate_normal_notebook(species_name: str, video_filename: str,
                             drive_folder: str = "Iridescence Lab",
                             frame_stride: int = 1) -> dict:
    cell_paths = (
        "# ── Paths ──────────────────────────────────────────────────────────────────\n"
        "SPECIES       = '" + species_name + "'\n"
        "DRIVE_FOLDER  = '/content/drive/MyDrive/" + drive_folder + "/" + species_name + "'\n"
        "VIDEO_PATH    = DRIVE_FOLDER + '/" + video_filename + "'\n"
        "MASKS_PATH    = DRIVE_FOLDER + '/masks.npz'\n"
        "NORMALS_OUT   = DRIVE_FOLDER + '/normal_data.csv'\n"
        "FRAME_STRIDE  = " + str(frame_stride) + "  # process every Nth frame (1 = all)\n\n"
        "assert os.path.exists(VIDEO_PATH),  'Video not found: '  + VIDEO_PATH\n"
        "assert os.path.exists(MASKS_PATH),  'masks.npz not found — run segmentation first'\n"
        "print('Ready:', SPECIES)"
    )

    cells = [
        _md_cell(
            "# Surface Geometry (Normal Maps) — " + species_name + "\n\n"
            "Estimates per-pixel surface normals using **StableNormal (YOSO)** from HuggingFace, "
            "then computes per-body-part roughness statistics using the SAM 2 masks.\n\n"
            "**Runtime:** Runtime > Change runtime type > **T4 GPU**\n\n"
            "**Before running, make sure these files are in "
            "`MyDrive/" + drive_folder + "/" + species_name + "/`:**\n"
            "1. The video file (`" + video_filename + "`)\n"
            "2. `masks.npz` (from the Segment tab)\n\n"
            "**After running:** Download `normal_data.csv` and upload it in the "
            "Surface Geometry tab."
        ),

        _code_cell(
            "# ── Install StableNormal ────────────────────────────────────────────────────\n"
            "# Run this cell first. It will restart the runtime automatically.\n"
            "# After restart, run all remaining cells (Runtime > Run all / Ctrl+F9).\n"
            "import importlib, sys\n\n"
            "if 'stablenormal' not in sys.modules:\n"
            "    !pip install -q git+https://github.com/Stable-X/StableNormal.git\n"
            "    !pip install -q diffusers accelerate\n"
            "    print('Install complete — restarting runtime...')\n"
            "    import os; os.kill(os.getpid(), 9)   # auto-restart\n"
            "else:\n"
            "    print('StableNormal already loaded.')"
        ),

        _code_cell(
            "# ── Imports ─────────────────────────────────────────────────────────────────\n"
            "import os, cv2\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import torch\n"
            "from PIL import Image\n\n"
            "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
            "print(f'PyTorch {torch.__version__}  |  device: {device}')"
        ),

        _code_cell(
            "# ── Mount Drive ─────────────────────────────────────────────────────────────\n"
            "from google.colab import drive\n"
            "drive.mount('/content/drive')"
        ),

        _code_cell(cell_paths),

        _code_cell(
            "# ── Extract frames from video ───────────────────────────────────────────────\n"
            "FRAMES_DIR = '/content/frames'\n"
            "os.makedirs(FRAMES_DIR, exist_ok=True)\n\n"
            "cap    = cv2.VideoCapture(VIDEO_PATH)\n"
            "fps    = cap.get(cv2.CAP_PROP_FPS)\n"
            "width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))\n"
            "height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))\n"
            "total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n"
            "print(f'{width}x{height} @ {fps:.1f} FPS — {total} frames')\n\n"
            "saved = 0\n"
            "while True:\n"
            "    ret, frame = cap.read()\n"
            "    if not ret: break\n"
            "    cv2.imwrite(f'{FRAMES_DIR}/{saved:05d}.jpg', frame,\n"
            "                [cv2.IMWRITE_JPEG_QUALITY, 95])\n"
            "    saved += 1\n"
            "cap.release()\n"
            "print(f'Extracted {saved} frames.')"
        ),

        _code_cell(
            "# ── Load masks.npz ───────────────────────────────────────────────────────────\n"
            "masks_data = np.load(MASKS_PATH, allow_pickle=False)\n\n"
            "video_segments = {}\n"
            "parts = set()\n"
            "for key in masks_data.files:\n"
            "    if key.startswith('_'):\n"
            "        continue\n"
            "    fidx_str, part = key.split('_', 1)\n"
            "    video_segments.setdefault(int(fidx_str), {})[part] = \\\n"
            "        masks_data[key].astype(bool)\n"
            "    parts.add(part)\n\n"
            "parts = sorted(parts)\n"
            "print(f'Parts: {parts}')\n"
            "print(f'Frames with masks: {len(video_segments)}')"
        ),

        _code_cell(
            "# ── Load StableNormal (YOSO — fast, single-pass) ───────────────────────────\n"
            "from stablenormal.pipeline_yoso_normal import YOSONormalsPipeline\n\n"
            "pipe = YOSONormalsPipeline.from_pretrained(\n"
            "    'Stable-X/yoso-normal-v0-3',\n"
            "    trust_remote_code=True,\n"
            "    variant='fp16',\n"
            "    torch_dtype=torch.float16,\n"
            ").to(device)\n"
            "pipe.set_progress_bar_config(disable=True)\n"
            "print('StableNormal (YOSO) loaded.')"
        ),

        _code_cell(
            "# ── Estimate normals and compute per-part statistics ────────────────────────\n"
            "#\n"
            "# For each frame: run YOSO → (H,W,3) normal map in [-1,1]\n"
            "# Apply each body-part mask → compute mean normal + angular roughness\n"
            "#   roughness = std of per-pixel angle deviations from the mean normal\n"
            "#   A flat mirror surface → roughness ≈ 0°\n"
            "#   A rough matte surface → roughness > 20°\n"
            "\n"
            "frame_idxs = sorted(video_segments.keys())\n"
            "if FRAME_STRIDE > 1:\n"
            "    frame_idxs = frame_idxs[::FRAME_STRIDE]\n"
            "    print(f'Stride={FRAME_STRIDE}: processing {len(frame_idxs)} of '\n"
            "          f'{len(video_segments)} frames')\n"
            "else:\n"
            "    print(f'Processing all {len(frame_idxs)} frames')\n\n"
            "rows = []\n\n"
            "for i, fidx in enumerate(frame_idxs):\n"
            "    frame_path = f'{FRAMES_DIR}/{fidx:05d}.jpg'\n"
            "    if not os.path.exists(frame_path):\n"
            "        continue\n\n"
            "    img_pil = Image.open(frame_path).convert('RGB')\n"
            "    orig_w, orig_h = img_pil.size\n\n"
            "    with torch.no_grad():\n"
            "        normal_img = pipe(img_pil)   # PIL Image, RGB = packed XYZ\n\n"
            "    # Decode: RGB 0-255 → normal XYZ in [-1, 1]\n"
            "    normal_np = np.array(normal_img.resize((orig_w, orig_h),\n"
            "                                           Image.BILINEAR)).astype(np.float32)\n"
            "    normal_np = normal_np / 127.5 - 1.0   # (H, W, 3)\n\n"
            "    # Renormalise each pixel to unit length\n"
            "    norms = np.linalg.norm(normal_np, axis=2, keepdims=True)\n"
            "    normal_np /= np.maximum(norms, 1e-8)\n\n"
            "    segs = video_segments.get(fidx, {})\n"
            "    row  = {'frame': fidx}\n\n"
            "    for part in parts:\n"
            "        mask = segs.get(part)\n"
            "        if mask is None or not mask.any():\n"
            "            for s in ['nx', 'ny', 'nz', 'roughness']:\n"
            "                row[f'{part}_{s}'] = np.nan\n"
            "            continue\n\n"
            "        pix = normal_np[mask]           # (N, 3)\n\n"
            "        mean_n = pix.mean(axis=0)\n"
            "        mean_n /= max(np.linalg.norm(mean_n), 1e-8)\n\n"
            "        # Angular deviation from mean (roughness in degrees)\n"
            "        dots   = np.clip((pix * mean_n).sum(axis=1), -1.0, 1.0)\n"
            "        angles = np.degrees(np.arccos(dots))\n\n"
            "        row[f'{part}_nx']        = round(float(mean_n[0]), 5)\n"
            "        row[f'{part}_ny']        = round(float(mean_n[1]), 5)\n"
            "        row[f'{part}_nz']        = round(float(mean_n[2]), 5)\n"
            "        row[f'{part}_roughness'] = round(float(np.std(angles)), 3)\n\n"
            "    rows.append(row)\n\n"
            "    if (i + 1) % 25 == 0 or i == len(frame_idxs) - 1:\n"
            "        print(f'  {i+1}/{len(frame_idxs)} frames done')\n\n"
            "print(f'\\nProcessed {len(rows)} frames.')"
        ),

        _code_cell(
            "# ── Save normal_data.csv to Google Drive ─────────────────────────────────────\n"
            "df = pd.DataFrame(rows).set_index('frame')\n"
            "df['method'] = 'stablenormal'\n"
            "df.to_csv(NORMALS_OUT)\n"
            "print(f'Saved {len(df)} rows  →  {NORMALS_OUT}')"
        ),

        _code_cell(
            "# ── Save normal map images to Google Drive ───────────────────────────────────\n"
            "# Save one coloured normal-map PNG per sampled frame.\n"
            "# RGB encodes XYZ orientation:\n"
            "#   Red   = surface tilting left/right\n"
            "#   Green = surface tilting up/down\n"
            "#   Blue  = surface facing the camera\n"
            "\n"
            "N_SAMPLE_MAPS = 20   # how many frames to save as images\n"
            "MAPS_OUT      = DRIVE_FOLDER + '/normal_maps'\n"
            "os.makedirs(MAPS_OUT, exist_ok=True)\n\n"
            "step       = max(1, len(frame_idxs) // N_SAMPLE_MAPS)\n"
            "sample_idx = frame_idxs[::step][:N_SAMPLE_MAPS]\n\n"
            "print(f'Saving {len(sample_idx)} normal-map images...')\n\n"
            "for fidx in sample_idx:\n"
            "    frame_path = f'{FRAMES_DIR}/{fidx:05d}.jpg'\n"
            "    if not os.path.exists(frame_path):\n"
            "        continue\n\n"
            "    img_pil = Image.open(frame_path).convert('RGB')\n"
            "    orig_w, orig_h = img_pil.size\n\n"
            "    with torch.no_grad():\n"
            "        normal_img = pipe(img_pil)\n\n"
            "    # Resize to original frame resolution and save\n"
            "    if hasattr(normal_img, 'resize'):\n"
            "        normal_img = normal_img.resize((orig_w, orig_h), Image.BILINEAR)\n"
            "    out_path = f'{MAPS_OUT}/{fidx:05d}_normal.png'\n"
            "    normal_img.save(out_path)\n\n"
            "print(f'Saved {len(sample_idx)} images to {MAPS_OUT}')\n\n"
            "# Zip for easy download\n"
            "import shutil\n"
            "ZIP_OUT = DRIVE_FOLDER + '/normal_maps'\n"
            "shutil.make_archive(ZIP_OUT, 'zip', MAPS_OUT)\n"
            "zip_mb = os.path.getsize(ZIP_OUT + '.zip') / 1e6\n"
            "print(f'Zipped  →  {ZIP_OUT}.zip  ({zip_mb:.1f} MB)')\n"
            "print()\n"
            "print('Download both files from Google Drive and upload in the Surface Geometry tab:')\n"
            "print('  · normal_data.csv')\n"
            "print('  · normal_maps.zip')"
        ),
    ]

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3", "language": "python", "name": "python3"
            },
            "language_info": {"name": "python", "version": "3.10.0"},
            "accelerator": "GPU",
            "colab": {
                "name": "normal_maps_" + species_name + ".ipynb",
                "provenance": [],
            },
        },
        "cells": cells,
    }
