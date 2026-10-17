"""Video probing and browser-friendly (H.264) transcoding."""

import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, Optional

import cv2

try:
    import imageio_ffmpeg
    FFMPEG_EXE: Optional[str] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = shutil.which("ffmpeg")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}
PROXY_MAX_HEIGHT = 720

_TIME_RE = re.compile(r"out_time_us=(\d+)")


def probe(path: Path) -> Dict[str, float]:
    """Returns fps, frame count, size and duration as decoded by OpenCV."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path.name}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 1 or fps > 240:
        fps = 25.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frames <= 0:
        # Containers like webm often report no frame count: count by decoding
        frames = 0
        while cap.grab():
            frames += 1
    cap.release()
    if frames <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Video has no decodable frames: {path.name}")
    return {
        "fps": round(float(fps), 3),
        "frames": frames,
        "width": width,
        "height": height,
        "duration": round(frames / fps, 3),
    }


def _run_ffmpeg(args, duration_s: float, progress: Optional[Callable[[float], None]]) -> None:
    cmd = [FFMPEG_EXE, "-hide_banner", "-loglevel", "error", "-y", *args[:-1],
           "-progress", "pipe:1", "-nostats", args[-1]]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in proc.stdout:
        m = _TIME_RE.match(line.strip())
        if m and progress and duration_s > 0:
            progress(min(1.0, int(m.group(1)) / 1e6 / duration_s))
    _, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {err.strip()[-400:]}")


def make_proxy(src: Path, dst: Path, progress: Optional[Callable[[float], None]] = None) -> None:
    """
    Re-encodes `src` into a constant-frame-rate H.264 MP4 (max 720p) that every browser plays.
    All analysis runs on this proxy, so frame index i maps exactly to time i / fps in the player.
    """
    if FFMPEG_EXE is None:
        shutil.copyfile(src, dst)
        return
    info = probe(src)
    _run_ffmpeg([
        "-i", str(src),
        "-vf", f"scale=-2:'min({PROXY_MAX_HEIGHT},ih)':flags=bicubic,format=yuv420p",
        "-r", f"{info['fps']}", "-fps_mode", "cfr",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-movflags", "+faststart", "-an",
        str(dst),
    ], info["duration"], progress)


def to_h264(src: Path, dst: Path, duration_s: float, progress: Optional[Callable[[float], None]] = None) -> None:
    """Converts an OpenCV-written (mp4v) file into browser-playable H.264."""
    if FFMPEG_EXE is None:
        shutil.move(str(src), str(dst))
        return
    _run_ffmpeg([
        "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-an",
        str(dst),
    ], duration_s, progress)
    src.unlink(missing_ok=True)


def first_frame_jpeg(path: Path, max_width: int = 320) -> bytes:
    """Grabs a frame ~10% into the video as a JPEG thumbnail."""
    cap = cv2.VideoCapture(str(path))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frames > 10:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames // 10)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return b""
    h, w = frame.shape[:2]
    if w > max_width:
        frame = cv2.resize(frame, (max_width, int(h * max_width / w)))
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes() if ok else b""
