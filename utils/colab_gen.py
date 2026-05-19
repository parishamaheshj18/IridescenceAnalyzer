def _code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": source}


def _md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def generate_notebook(species_name: str, video_filename: str,
                      drive_folder: str = "Iridescence Lab") -> dict:
    cell_paths = (
        "# ── Paths — adjust DRIVE_FOLDER if your Drive layout differs ──────────────\n"
        "SPECIES         = '" + species_name + "'\n"
        "DRIVE_FOLDER    = '/content/drive/MyDrive/" + drive_folder + "/" + species_name + "'\n"
        "VIDEO_PATH      = DRIVE_FOLDER + '/" + video_filename + "'\n"
        "KF_PATH         = DRIVE_FOLDER + '/keyframes.json'\n"
        "ANN_MASKS_PATH  = DRIVE_FOLDER + '/annotation_masks.npz'\n"
        "MASKS_OUT       = DRIVE_FOLDER + '/masks.npz'\n\n"
        "assert os.path.exists(VIDEO_PATH), 'Video not found: ' + VIDEO_PATH\n"
        "assert os.path.exists(KF_PATH),    'keyframes.json not found: ' + KF_PATH\n"
        "print('Ready:', SPECIES)\n"
        "print('Painted masks available:', os.path.exists(ANN_MASKS_PATH))"
    )

    cells = [
        _md_cell(
            "# SAM 2 Segmentation — " + species_name + "\n\n"
            "**Runtime:** Runtime > Change runtime type > **T4 GPU**\n\n"
            "**Before running, upload these files to "
            "`MyDrive/" + drive_folder + "/" + species_name + "/`:**\n"
            "1. The video file (`" + video_filename + "`)\n"
            "2. `keyframes.json`\n"
            "3. `annotation_masks.npz` (the painted masks — from the Annotate tab)\n\n"
            "**After running:** Download `masks.npz` from Drive and upload it in the app "
            "(Segment tab)."
        ),

        _code_cell(
            "# ── Install SAM 2 ──────────────────────────────────────────────────────────\n"
            "!pip install -q git+https://github.com/facebookresearch/sam2.git\n\n"
            "import os\n"
            "if not os.path.exists('/content/sam2_hiera_small.pt'):\n"
            "    !wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
            "sam2_hiera_small.pt -P /content/\n"
            "    print('Checkpoint downloaded.')\n"
            "else:\n"
            "    print('Checkpoint already present.')"
        ),

        _code_cell(
            "# ── Imports ─────────────────────────────────────────────────────────────────\n"
            "import cv2, json, gc, os, shutil\n"
            "import numpy as np\n"
            "import torch\n"
            "from pathlib import Path\n"
            "from contextlib import nullcontext\n\n"
            "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
            "print(f'PyTorch {torch.__version__}  |  device: {device}')\n\n"
            "def autocast_ctx(dev):\n"
            "    return torch.autocast('cuda', dtype=torch.bfloat16) "
            "if dev == 'cuda' else nullcontext()"
        ),

        _code_cell(
            "# ── Mount Drive ─────────────────────────────────────────────────────────────\n"
            "from google.colab import drive\n"
            "drive.mount('/content/drive')"
        ),

        _code_cell(cell_paths),

        _code_cell(
            "# ── Extract frames ──────────────────────────────────────────────────────────\n"
            "FRAMES_DIR = '/content/frames'\n"
            "os.makedirs(FRAMES_DIR, exist_ok=True)\n\n"
            "cap    = cv2.VideoCapture(VIDEO_PATH)\n"
            "fps    = cap.get(cv2.CAP_PROP_FPS)\n"
            "width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))\n"
            "height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))\n"
            "total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n"
            "print(f'{width}x{height} @ {fps:.1f} FPS — {total} frames ({total/fps:.1f}s)')\n\n"
            "saved = 0\n"
            "while True:\n"
            "    ret, frame = cap.read()\n"
            "    if not ret: break\n"
            "    cv2.imwrite(f'{FRAMES_DIR}/{saved:05d}.jpg', frame, "
            "[cv2.IMWRITE_JPEG_QUALITY, 95])\n"
            "    saved += 1\n"
            "cap.release()\n"
            "print(f'Extracted {saved} frames.')"
        ),

        _code_cell(
            "# ── Load keyframes + painted annotation masks ────────────────────────────────\n"
            "import json as _json\n"
            "with open(KF_PATH) as f:\n"
            "    KEYFRAMES = _json.load(f)\n\n"
            "all_parts = []\n"
            "for kf in KEYFRAMES:\n"
            "    for k in kf:\n"
            "        if k != 'frame' and k not in all_parts:\n"
            "            all_parts.append(k)\n\n"
            "PART_IDS   = {part: i+1 for i, part in enumerate(all_parts)}\n"
            "ID_TO_PART = {v: k for k, v in PART_IDS.items()}\n\n"
            "# Load painted masks (preferred over point prompts)\n"
            "ANN_MASKS = {}\n"
            "if os.path.exists(ANN_MASKS_PATH):\n"
            "    _d = np.load(ANN_MASKS_PATH)\n"
            "    ANN_MASKS = {k: _d[k] for k in _d.files}\n"
            "    print(f'Loaded {len(ANN_MASKS)} painted mask entries')\n"
            "else:\n"
            "    print('No annotation_masks.npz found — using point prompts (less precise)')\n\n"
            "print('Parts:', PART_IDS)\n"
            "print(f'Loaded {len(KEYFRAMES)} keyframes')"
        ),

        _code_cell(
            "# ── Load SAM 2 ──────────────────────────────────────────────────────────────\n"
            "from sam2.build_sam import build_sam2_video_predictor\n\n"
            "predictor = build_sam2_video_predictor(\n"
            "    config_file='sam2_hiera_s.yaml',\n"
            "    ckpt_path='/content/sam2_hiera_small.pt',\n"
            "    device=device,\n"
            ")\n"
            "print(f'SAM 2 loaded on {device}.')"
        ),

        _code_cell(
            "# ── Chunked propagation (300 frames/chunk to stay within Colab RAM) ─────────\n"
            "CHUNK_SIZE = 300\n"
            "CHUNK_DIR  = '/content/frames_chunk'\n\n"
            "kf_by_frame = {kf['frame']: kf for kf in KEYFRAMES}\n"
            "all_frames  = sorted(int(f.stem) for f in Path(FRAMES_DIR).glob('*.jpg'))\n"
            "chunks      = [all_frames[i:i+CHUNK_SIZE] for i in range(0, len(all_frames), CHUNK_SIZE)]\n\n"
            "video_segments = {}\n"
            "carry_masks    = None\n\n"
            "print(f'{len(all_frames)} frames  ->  {len(chunks)} chunks of <={CHUNK_SIZE}\\n')\n\n"
            "for ci, chunk in enumerate(chunks):\n"
            "    chunk_start = chunk[0]\n"
            "    chunk_kfs   = [f for f in chunk if f in kf_by_frame]\n"
            "    anchor_orig = chunk_kfs[0] if chunk_kfs else chunk[0]\n"
            "    anchor_rel  = anchor_orig - chunk_start\n\n"
            "    label = f'keyframes {chunk_kfs}' if chunk_kfs else 'carry-over'\n"
            "    print(f'Chunk {ci+1}/{len(chunks)}: frames {chunk[0]}-{chunk[-1]}  ({label})')\n\n"
            "    tmp = Path(CHUNK_DIR)\n"
            "    if tmp.exists(): shutil.rmtree(tmp)\n"
            "    tmp.mkdir()\n"
            "    for i, fidx in enumerate(chunk):\n"
            "        os.symlink(f'{FRAMES_DIR}/{fidx:05d}.jpg', f'{CHUNK_DIR}/{i:05d}.jpg')\n\n"
            "    with torch.inference_mode(), autocast_ctx(device):\n"
            "        state = predictor.init_state(video_path=CHUNK_DIR, offload_video_to_cpu=True)\n"
            "        predictor.reset_state(state)\n"
            "        prompts_added = False\n\n"
            "        if chunk_kfs:\n"
            "            for fidx in chunk_kfs:\n"
            "                kf  = kf_by_frame[fidx]\n"
            "                rel = fidx - chunk_start\n"
            "                for part, obj_id in PART_IDS.items():\n"
            "                    val      = kf.get(part)\n"
            "                    mask_key = f'{fidx}_{part}'\n"
            "                    if val in (None, 'skip'):\n"
            "                        # Part not visible — negative center point\n"
            "                        predictor.add_new_points_or_box(\n"
            "                            inference_state=state, frame_idx=rel,\n"
            "                            obj_id=obj_id,\n"
            "                            points=np.array(\n"
            "                                [[width//2, height//2]], dtype=np.float32),\n"
            "                            labels=np.array([0], dtype=np.int32),\n"
            "                        )\n"
            "                    elif mask_key in ANN_MASKS:\n"
            "                        # Painted mask prompt — most precise\n"
            "                        predictor.add_new_mask(\n"
            "                            inference_state=state, frame_idx=rel,\n"
            "                            obj_id=obj_id,\n"
            "                            mask=torch.from_numpy(\n"
            "                                ANN_MASKS[mask_key].astype(bool)),\n"
            "                        )\n"
            "                    elif isinstance(val, list):\n"
            "                        # Legacy point prompt (backward compat)\n"
            "                        predictor.add_new_points_or_box(\n"
            "                            inference_state=state, frame_idx=rel,\n"
            "                            obj_id=obj_id,\n"
            "                            points=np.array([val], dtype=np.float32),\n"
            "                            labels=np.array([1], dtype=np.int32),\n"
            "                        )\n"
            "            prompts_added = True\n\n"
            "        elif carry_masks is not None:\n"
            "            for part, obj_id in PART_IDS.items():\n"
            "                mask = carry_masks.get(part)\n"
            "                if mask is not None and mask.any():\n"
            "                    predictor.add_new_mask(\n"
            "                        inference_state=state, frame_idx=0, obj_id=obj_id,\n"
            "                        mask=torch.from_numpy(mask),\n"
            "                    )\n"
            "                    prompts_added = True\n\n"
            "        if not prompts_added:\n"
            "            print('  -> no subject visible, skipping.')\n"
            "            predictor.reset_state(state); del state\n"
            "            gc.collect(); torch.cuda.empty_cache()\n"
            "            shutil.rmtree(CHUNK_DIR)\n"
            "            carry_masks = None\n"
            "            continue\n\n"
            "        for rel, obj_ids, logits in predictor.propagate_in_video(\n"
            "                state, start_frame_idx=anchor_rel):\n"
            "            orig = chunk_start + rel\n"
            "            video_segments[orig] = {\n"
            "                ID_TO_PART[oid]: (logits[i] > 0.0).cpu().numpy().astype(bool).squeeze()\n"
            "                for i, oid in enumerate(obj_ids)\n"
            "            }\n"
            "            del logits\n\n"
            "        if anchor_rel > 0:\n"
            "            for rel, obj_ids, logits in predictor.propagate_in_video(\n"
            "                    state, start_frame_idx=anchor_rel, reverse=True):\n"
            "                orig = chunk_start + rel\n"
            "                video_segments[orig] = {\n"
            "                    ID_TO_PART[oid]: (logits[i] > 0.0).cpu().numpy().astype(bool).squeeze()\n"
            "                    for i, oid in enumerate(obj_ids)\n"
            "                }\n"
            "                del logits\n\n"
            "        last_orig = chunk[-1]\n"
            "        if last_orig in video_segments:\n"
            "            last_masks = video_segments[last_orig]\n"
            "            carry_masks = (\n"
            "                {p: last_masks[p].copy() for p in last_masks}\n"
            "                if any(m.any() for m in last_masks.values()) else None\n"
            "            )\n\n"
            "        predictor.reset_state(state); del state\n\n"
            "    gc.collect(); torch.cuda.empty_cache(); shutil.rmtree(CHUNK_DIR)\n"
            "    n_done = sum(1 for f in chunk if f in video_segments)\n"
            "    print(f'  -> {n_done}/{len(chunk)} frames done  |  total: {len(video_segments)}')\n\n"
            "print(f'\\nDone: {len(video_segments)} / {len(all_frames)} frames segmented.')"
        ),

        _code_cell(
            "# ── Save masks to Google Drive ──────────────────────────────────────────────\n"
            "mask_dict = {}\n"
            "for fidx, parts in video_segments.items():\n"
            "    for part, mask in parts.items():\n"
            "        mask_dict[f'{fidx:05d}_{part}'] = mask\n\n"
            "mask_dict['_fps']       = np.array([fps])\n"
            "mask_dict['_frame_wh']  = np.array([width, height])\n"
            "mask_dict['_ann_frame'] = np.array([KEYFRAMES[0]['frame']])\n\n"
            "np.savez_compressed(MASKS_OUT, **mask_dict)\n"
            "size_mb = os.path.getsize(MASKS_OUT) / 1e6\n"
            "print(f'Saved {len(video_segments)} frames -> {MASKS_OUT}')\n"
            "print(f'File size: {size_mb:.1f} MB')\n"
            "print()\n"
            "print('Download masks.npz from Google Drive, then upload it in the app (Segment tab).')"
        ),
    ]

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
            "colab": {"name": "sam2_" + species_name + ".ipynb", "provenance": []},
        },
        "cells": cells,
    }
