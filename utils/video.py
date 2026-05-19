import cv2
import numpy as np
from pathlib import Path


def extract_frames(video_path: Path, frames_dir: Path, quality: int = 95) -> int:
    frames_dir.mkdir(exist_ok=True)
    existing = sorted(frames_dir.glob("*.jpg"))
    if existing:
        return len(existing)
    cap = cv2.VideoCapture(str(video_path))
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(
            str(frames_dir / f"{saved:05d}.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        saved += 1
    cap.release()
    return saved


def get_video_info(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def count_frames(frames_dir: Path) -> int:
    return len(sorted(frames_dir.glob("*.jpg")))


def read_frame_rgb(frames_dir: Path, fidx: int):
    path = frames_dir / f"{fidx:05d}.jpg"
    if not path.exists():
        return None
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def export_overlay_video(
    frames_dir: Path,
    output_path: Path,
    fps: float,
    video_segments: dict,
    part_colors: dict,
    alpha: float = 0.5,
) -> None:
    all_idxs = sorted(int(p.stem) for p in frames_dir.glob("*.jpg"))
    if not all_idxs:
        return
    sample = cv2.imread(str(frames_dir / f"{all_idxs[0]:05d}.jpg"))
    h, w = sample.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    for fidx in all_idxs:
        bgr = cv2.imread(str(frames_dir / f"{fidx:05d}.jpg"))
        if bgr is None:
            continue
        segs = video_segments.get(fidx)
        if segs:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(float) / 255.0
            overlay = rgb.copy()
            for part, mask in segs.items():
                if mask.any():
                    c = part_colors.get(part, np.array([1.0, 1.0, 0.0]))
                    overlay[mask] = overlay[mask] * (1 - alpha) + c * alpha
            bgr = cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()
