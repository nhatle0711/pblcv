"""
Person search engine for the web demo.

Splits the work into a one-off ANALYZE step per video (detect -> track -> crop -> encode
V_app/V_mot/V_video per track, all cached) and a cheap SEARCH step per prompt (decompose into
Q_app/Q_mot with TextDecomposer, encode T_full/T_app/T_mot, three cosine scores per track).
Fusion weights are applied client-side so they can be tuned live; the defaults match
train.py::evaluate (0.50 joint / 0.35 app / 0.15 mot).
"""

import base64
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from data.text_decomposer import TextDecomposer
from models.damr_main import DAMRModel

ProgressCb = Callable[[str, float, float], None]

DEFAULT_WEIGHTS = {"joint": 0.50, "app": 0.35, "mot": 0.15}
NUM_TRACK_FRAMES = 16
CROP_SIZE = 224
ENCODE_BATCH = 4
MAX_TRACK_GAP = 15          # frames a track survives without a matching detection
MIN_TRACK_DETECTIONS = 3
SOFTMAX_TAU = 0.07          # same temperature as the InfoNCE training loss

# Distinct rank colours shared by the rendered MP4 (BGR) and the web UI (hex)
RANK_COLORS = [
    ((127, 255, 0), "#00ff7f"),
    ((255, 210, 0), "#00d2ff"),
    ((0, 184, 255), "#ffb800"),
    ((133, 42, 255), "#ff2a85"),
    ((247, 85, 168), "#a855f7"),
    ((74, 107, 255), "#ff6b4a"),
    ((0, 255, 204), "#ccff00"),
    ((212, 245, 0), "#00f5d4"),
    ((248, 189, 56), "#38bdf8"),
    ((72, 29, 225), "#e11d48"),
]
OTHER_COLOR = ((184, 163, 148), "#94a3b8")
DARK_TEXT_RANKS = {1, 2, 3, 7, 8, 9}

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def rank_color(rank: int) -> Tuple[Tuple[int, int, int], str]:
    return RANK_COLORS[rank - 1] if 1 <= rank <= len(RANK_COLORS) else OTHER_COLOR


def _jpeg(img_bgr: np.ndarray, max_size: int, quality: int = 85) -> bytes:
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_size:
        s = max_size / float(max(h, w))
        img_bgr = cv2.resize(img_bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else b""


def _data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def _iou(a: List[int], b: List[int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(union, 1e-6)


@dataclass
class Track:
    id: int
    boxes: Dict[int, List[int]]          # frame -> [x1, y1, x2, y2], incl. interpolated gaps
    det_frames: List[int]                # frames with a real detection
    clip: Optional[np.ndarray] = None    # [16, 224, 224, 3] RGB uint8, model input
    thumb: bytes = b""                   # JPEG, representative crop
    film: List[bytes] = field(default_factory=list)  # 4 JPEG crops across the track

    @property
    def start(self) -> int:
        return min(self.boxes)

    @property
    def end(self) -> int:
        return max(self.boxes)


@dataclass
class Analysis:
    id: str
    video_id: str
    params: Dict[str, Any]
    fps: float
    width: int
    height: int
    frames_processed: int
    frames_total: int
    tracks: Dict[int, Track]
    ckpt_key: str = ""
    emb: Dict[str, torch.Tensor] = field(default_factory=dict)  # V_app/V_mot/V_video [N, D], L2-normalised
    track_order: List[int] = field(default_factory=list)          # row order of emb


class SearchEngine:
    def __init__(self, root: Path):
        self.root = root
        self.checkpoints_dir = root / "checkpoints"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_lock = threading.RLock()
        self.decomposer = TextDecomposer(language="en")
        self.detector = None
        self.detector_backend = "opencv"
        self.checkpoint_info: Dict[str, Any] = {"loaded": False, "name": None}
        print(f"[*] SearchEngine on device: {self.device}")

        # ImageNet ViT weights are fully overwritten by any checkpoint, so skip downloading them then
        self.model = DAMRModel(
            embed_dim=768,
            text_model_name="bert-base-uncased",
            motion_frame_num=NUM_TRACK_FRAMES,
            vit_depth=6,
            vit_heads=12,
            drop_rate=0.0,
            pretrained_vit=not self.scan_checkpoints(),
        ).to(self.device).eval()
        if getattr(self.model.text_encoder, "fallback", False):
            print("[!] WARNING: bert-base-uncased could not be loaded; the text encoder fell back to an "
                  "untrained model and checkpoints will not match. Check HF_HOME / network access.")
        self._load_default_checkpoint()
        self._init_detector()

    # ------------------------------------------------------------------ checkpoints
    def scan_checkpoints(self) -> List[Dict[str, Any]]:
        if not self.checkpoints_dir.exists():
            return []
        found = []
        for cp in sorted(self.checkpoints_dir.glob("**/*.pth")):
            if cp.is_file() and cp.stat().st_size > 1024 * 1024:
                found.append({
                    "name": str(cp.relative_to(self.checkpoints_dir)),
                    "size_mb": round(cp.stat().st_size / (1024 * 1024), 1),
                })
        return found

    def _load_default_checkpoint(self) -> None:
        def priority(c: Dict[str, Any]) -> int:
            name = Path(c["name"]).name.lower()
            if name == "best_weights.pth":
                return 0
            if "best" in name:
                return 1
            if "latest" in name:
                return 2
            return 3

        for c in sorted(self.scan_checkpoints(), key=priority):
            try:
                self.load_checkpoint(c["name"])
                return
            except Exception as e:
                print(f"[!] Skipping checkpoint {c['name']}: {e}")
        print("[!] No usable checkpoint in checkpoints/. Using ImageNet ViT + BERT initial weights.")

    def load_checkpoint(self, name: str) -> Dict[str, Any]:
        """Loads checkpoints/<name>. Rejects paths outside checkpoints/ and partial matches."""
        base = self.checkpoints_dir.resolve()
        cp = (base / name).resolve()
        if base not in cp.parents or not cp.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {name}")

        ckpt = torch.load(cp, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        with self.model_lock:
            model_state = self.model.state_dict()
            matched = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
            if len(matched) < len(model_state):
                raise ValueError(
                    f"Architecture mismatch: {len(matched)}/{len(model_state)} layers match"
                )
            self.model.load_state_dict(matched)

        metrics = ckpt.get("metrics", {}) if isinstance(ckpt, dict) else {}
        cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        self.checkpoint_info = {
            "loaded": True,
            "name": str(cp.relative_to(base)),
            "layers": len(matched),
            "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
            "sub_dataset": (cfg.get("data") or {}).get("sub_dataset"),
            "metrics": {k: round(float(v), 2) for k, v in metrics.items() if isinstance(v, (int, float))},
        }
        print(f"[+] Loaded checkpoint {self.checkpoint_info['name']} ({len(matched)} layers, "
              f"epoch {self.checkpoint_info['epoch']})")
        return self.checkpoint_info

    @property
    def ckpt_key(self) -> str:
        return str(self.checkpoint_info.get("name"))

    # ------------------------------------------------------------------ detection
    def _init_detector(self) -> None:
        try:
            from ultralytics import YOLO
            yolo_path = self.root / "yolov8n.pt"
            self.detector = YOLO(str(yolo_path) if yolo_path.exists() else "yolov8n.pt")
            self.detector_backend = "yolo"
            print("[+] Person detector: Ultralytics YOLOv8n")
            return
        except Exception as e:
            print(f"[*] YOLOv8 unavailable ({e}), trying torchvision...")
        try:
            import torchvision.models.detection as tv_det
            weights = tv_det.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
            self.detector = tv_det.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights).to(self.device).eval()
            self.detector_backend = "torchvision"
            print("[+] Person detector: torchvision Faster R-CNN MobileNetV3")
            return
        except Exception:
            pass
        self.detector = cv2.HOGDescriptor()
        self.detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.detector_backend = "opencv"
        print("[+] Person detector: OpenCV HOG")

    def detect(self, frame: np.ndarray, conf: float) -> List[List[int]]:
        """Person boxes [x1, y1, x2, y2] in a BGR frame."""
        h, w = frame.shape[:2]
        raw: List[Tuple[float, float, float, float]] = []
        if self.detector_backend == "yolo":
            res = self.detector(frame, classes=[0], conf=conf, verbose=False)
            if res and res[0].boxes is not None:
                raw = [tuple(b) for b in res[0].boxes.xyxy.cpu().numpy().tolist()]
        elif self.detector_backend == "torchvision":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(self.device)
            with torch.no_grad():
                p = self.detector([t])[0]
            for box, label, score in zip(p["boxes"].cpu().numpy(), p["labels"].cpu().numpy(), p["scores"].cpu().numpy()):
                if label == 1 and score >= conf:
                    raw.append(tuple(box))
        else:
            found, _ = self.detector.detectMultiScale(frame, winStride=(8, 8), padding=(4, 4), scale=1.05)
            raw = [(x, y, x + bw, y + bh) for (x, y, bw, bh) in found]

        boxes = []
        for x1, y1, x2, y2 in raw:
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if (x2 - x1) > 15 and (y2 - y1) > 30:
                boxes.append([x1, y1, x2, y2])
        return boxes

    # ------------------------------------------------------------------ analysis
    def analyze(self, video_path: Path, video_id: str, analysis_id: str, params: Dict[str, Any],
                progress: ProgressCb) -> Analysis:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Cannot open video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        max_frames = int(params["max_frames"]) or total
        n_frames = min(total, max_frames) if total else max_frames
        stride = max(1, int(params.get("stride", 1)))
        conf = float(params.get("conf", 0.35))

        # Pass 1: detect + associate. Only boxes are kept, never full frames.
        tracks: Dict[int, Dict[str, Any]] = {}
        next_tid = 1
        idx = 0
        while idx < n_frames:
            ok, frame = cap.read()
            if not ok:
                break
            detections = self.detect(frame, conf) if idx % stride == 0 else []
            matched = set()
            for tid, t in tracks.items():
                if idx - t["last"] > MAX_TRACK_GAP:
                    continue
                last = t["boxes"][t["last"]]
                best, best_d = 0.0, -1
                for d_i, det in enumerate(detections):
                    if d_i in matched:
                        continue
                    iou = _iou(last, det)
                    dist = math.hypot((last[0] + last[2] - det[0] - det[2]) / 2,
                                      (last[1] + last[3] - det[1] - det[3]) / 2)
                    diag = math.hypot(det[2] - det[0], det[3] - det[1])
                    score = iou + max(0.0, 1.0 - dist / max(diag, 20))
                    if (iou > 0.15 or dist < diag * 0.9) and score > best:
                        best, best_d = score, d_i
                if best_d >= 0:
                    det = detections[best_d]
                    matched.add(best_d)
                    gap = idx - t["last"]
                    for gf in range(t["last"] + 1, idx):
                        a = (gf - t["last"]) / float(gap)
                        t["boxes"][gf] = [int(last[k] + a * (det[k] - last[k])) for k in range(4)]
                    t["boxes"][idx] = det
                    t["dets"].append(idx)
                    t["last"] = idx
            for d_i, det in enumerate(detections):
                if d_i not in matched:
                    tracks[next_tid] = {"boxes": {idx: det}, "dets": [idx], "last": idx}
                    next_tid += 1
            idx += 1
            if idx % 5 == 0 or idx == n_frames:
                progress("detect", idx, n_frames)
        cap.release()
        frames_processed = idx

        valid = [t for t in tracks.values() if len(t["dets"]) >= MIN_TRACK_DETECTIONS]
        valid.sort(key=lambda t: min(t["boxes"]))
        result_tracks = {
            i: Track(id=i, boxes=t["boxes"], det_frames=t["dets"]) for i, t in enumerate(valid, start=1)
        }

        analysis = Analysis(
            id=analysis_id, video_id=video_id, params=params, fps=float(fps), width=width, height=height,
            frames_processed=frames_processed, frames_total=total or frames_processed, tracks=result_tracks,
        )
        if result_tracks:
            self._extract_crops(video_path, analysis, progress)
            self.encode_tracks(analysis, progress)
        return analysis

    def _extract_crops(self, video_path: Path, analysis: Analysis, progress: ProgressCb) -> None:
        """Pass 2: re-reads only the frames needed for each track's 16-frame clip and previews."""
        wanted: Dict[int, List[Tuple[int, int]]] = {}  # frame -> [(track_id, slot)]
        for t in analysis.tracks.values():
            picks = np.linspace(0, len(t.det_frames) - 1, NUM_TRACK_FRAMES).round().astype(int)
            t.clip = np.zeros((NUM_TRACK_FRAMES, CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
            for slot, p in enumerate(picks):
                wanted.setdefault(t.det_frames[p], []).append((t.id, slot))
        film_slots = {0, 5, 10, 15}
        thumb_slot = NUM_TRACK_FRAMES // 2
        film: Dict[int, Dict[int, bytes]] = {tid: {} for tid in analysis.tracks}

        cap = cv2.VideoCapture(str(video_path))
        last_needed = max(wanted)
        idx = 0
        while idx <= last_needed:
            ok, frame = cap.read()
            if not ok:
                break
            for tid, slot in wanted.get(idx, []):
                t = analysis.tracks[tid]
                x1, y1, x2, y2 = t.boxes[idx]
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                t.clip[slot] = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (CROP_SIZE, CROP_SIZE),
                                          interpolation=cv2.INTER_LINEAR)
                if slot == thumb_slot:
                    t.thumb = _jpeg(crop, 256)
                if slot in film_slots:
                    film[tid][slot] = _jpeg(crop, 256)
            idx += 1
            if idx % 10 == 0:
                progress("crops", idx, last_needed + 1)
        cap.release()
        for tid, t in analysis.tracks.items():
            t.film = [film[tid][s] for s in sorted(film[tid])]
        progress("crops", 1, 1)

    def encode_tracks(self, analysis: Analysis, progress: Optional[ProgressCb] = None) -> None:
        """Encodes every track clip with SpaceTimeViT. Cached per checkpoint."""
        order = sorted(analysis.tracks)
        outs: Dict[str, List[torch.Tensor]] = {"V_app": [], "V_mot": [], "V_video": []}
        for b in range(0, len(order), ENCODE_BATCH):
            ids = order[b:b + ENCODE_BATCH]
            clips = torch.from_numpy(np.stack([analysis.tracks[i].clip for i in ids]))  # [B,T,H,W,3]
            x = clips.permute(0, 1, 4, 2, 3).float().div(255.0)                          # [B,T,3,H,W]
            x = ((x.flatten(0, 1) - _MEAN) / _STD).view_as(x).to(self.device)
            with self.model_lock, torch.no_grad():
                v_app, v_mot, v_video = self.model.encode_video(x)
            outs["V_app"].append(v_app.float().cpu())
            outs["V_mot"].append(v_mot.float().cpu())
            outs["V_video"].append(v_video.float().cpu())
            if progress:
                progress("encode", b + len(ids), len(order))
        analysis.emb = {k: F.normalize(torch.cat(v), dim=-1) for k, v in outs.items()}
        analysis.track_order = order
        analysis.ckpt_key = self.ckpt_key

    # ------------------------------------------------------------------ search
    def search(self, analysis: Analysis, prompt: str) -> Dict[str, Any]:
        """Three cosine scores per track; ranking/fusion happens in the client."""
        prompt = prompt.strip()
        concepts = self.decomposer.extract_concepts(prompt)
        if analysis.tracks and analysis.ckpt_key != self.ckpt_key:
            self.encode_tracks(analysis)

        result = {
            "prompt": prompt,
            "q_app": concepts["q_app"],
            "q_mot": concepts["q_mot"],
            "app_keywords": concepts["app_keywords"],
            "mot_keywords": concepts["mot_keywords"],
            "scores": [],
        }
        if not analysis.tracks:
            return result

        with self.model_lock, torch.no_grad():
            t = self.model.encode_text([prompt, concepts["q_app"], concepts["q_mot"]], device=self.device)
        t = F.normalize(t.float().cpu(), dim=-1)
        sim_joint = (analysis.emb["V_video"] @ t[0]).tolist()
        sim_app = (analysis.emb["V_app"] @ t[1]).tolist()
        sim_mot = (analysis.emb["V_mot"] @ t[2]).tolist()
        result["scores"] = [
            {"track_id": tid, "joint": round(sim_joint[i], 5), "app": round(sim_app[i], 5), "mot": round(sim_mot[i], 5)}
            for i, tid in enumerate(analysis.track_order)
        ]
        return result

    @staticmethod
    def rank(scores: List[Dict[str, Any]], weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Fuses the three scores with normalised weights and adds softmax match probability."""
        wsum = sum(max(0.0, float(weights.get(k, 0.0))) for k in DEFAULT_WEIGHTS) or 1.0
        w = {k: max(0.0, float(weights.get(k, 0.0))) / wsum for k in DEFAULT_WEIGHTS}
        ranked = [dict(s, fused=sum(w[k] * s[k] for k in w)) for s in scores]
        ranked.sort(key=lambda s: s["fused"], reverse=True)
        if ranked:
            m = max(s["fused"] for s in ranked)
            exps = [math.exp((s["fused"] - m) / SOFTMAX_TAU) for s in ranked]
            z = sum(exps)
            for r, (s, e) in enumerate(zip(ranked, exps), start=1):
                s["rank"] = r
                s["prob"] = e / z
        return ranked

    # ------------------------------------------------------------------ render
    def render(self, video_path: Path, analysis: Analysis, ranked: List[Dict[str, Any]], prompt: str,
               top_k: int, out_path: Path, progress: ProgressCb) -> None:
        """Draws top-K boxes, labels and a HUD onto the proxy video (mp4v; caller converts to H.264)."""
        cap = cv2.VideoCapture(str(video_path))
        w, h = analysis.width, analysis.height
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), analysis.fps, (w, h))
        top = ranked[:top_k]
        draw_order = list(reversed(top))  # #1 drawn last, on top
        target = top[0] if top else None
        trail: List[Tuple[int, int]] = []
        scale = max(0.4, min(1.0, w / 1280.0))
        n = analysis.frames_processed

        for f_idx in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            for s in draw_order:
                box = analysis.tracks[s["track_id"]].boxes.get(f_idx)
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                color = rank_color(s["rank"])[0]
                thick = 3 if s["rank"] == 1 else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
                tag = f"#{s['rank']} P{s['track_id']}  {s['fused']:.3f} ({s['prob'] * 100:.0f}%)"
                fs = 0.5 * scale + 0.1
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
                ty = max(th + 8, y1)
                cv2.rectangle(frame, (x1, ty - th - 8), (min(w - 1, x1 + tw + 8), ty), color, -1)
                txt = (0, 0, 0) if s["rank"] in DARK_TEXT_RANKS else (255, 255, 255)
                cv2.putText(frame, tag, (x1 + 4, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, txt, 1, cv2.LINE_AA)
                if s["rank"] == 1:
                    trail.append(((x1 + x2) // 2, y2))
                    trail = trail[-30:]
            for i in range(1, len(trail)):
                cv2.line(frame, trail[i - 1], trail[i], RANK_COLORS[0][0], 2, cv2.LINE_AA)

            hud_h = int(36 * scale + 10)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, hud_h), (20, 16, 12), -1)
            frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
            max_chars = max(20, int(w / (11 * scale)) - 40)
            q = prompt if len(prompt) <= max_chars else prompt[:max_chars - 1] + "…"
            hud = f'"{q}"   #1: P{target["track_id"]}   {f_idx + 1}/{n}' if target else f"{f_idx + 1}/{n}"
            cv2.putText(frame, hud, (12, int(hud_h * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale + 0.08,
                        (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(frame)
            if f_idx % 10 == 0:
                progress("render", f_idx + 1, n)
        cap.release()
        writer.release()

    # ------------------------------------------------------------------ serialisation
    def tracks_payload(self, analysis: Analysis) -> Dict[str, Any]:
        """Compact per-frame boxes and previews for the browser overlay/timeline."""
        tracks = []
        for tid in sorted(analysis.tracks):
            t = analysis.tracks[tid]
            frames = sorted(t.boxes)
            tracks.append({
                "id": tid,
                "start": t.start,
                "end": t.end,
                "n_det": len(t.det_frames),
                "boxes": [[f, *t.boxes[f]] for f in frames],
                "thumb": _data_url(t.thumb) if t.thumb else "",
                "film": [_data_url(j) for j in t.film],
            })
        return {
            "analysis_id": analysis.id,
            "video_id": analysis.video_id,
            "params": analysis.params,
            "fps": analysis.fps,
            "width": analysis.width,
            "height": analysis.height,
            "frames_processed": analysis.frames_processed,
            "frames_total": analysis.frames_total,
            "tracks": tracks,
        }
