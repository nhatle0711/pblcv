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
import os
import re
import tempfile
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

DEFAULT_WEIGHTS = {"joint": 0.30, "app": 0.20, "mot": 0.05, "clip": 0.45}
DSCA_WEIGHTS = {"joint": 0.50, "app": 0.35, "mot": 0.15}   # defaults when CLIP is unavailable (= train.py::evaluate)
CLIP_MODEL = "openai/clip-vit-base-patch16"
CLIP_CROPS = 3              # preview crops per track (thumb + first/last film crop); ~165 ms each on CPU
CLIP_BATCH = 16
CLIP_PREFIX = "a photo of "
NUM_TRACK_FRAMES = 16
CROP_SIZE = 224
ENCODE_BATCH = 4
SOFTMAX_TAU = 0.07          # same temperature as the InfoNCE training loss

# --- tracking (YOLO + BoT-SORT with native ReID) ---
TRACKER_VERSION = "botsort-reid-v5-split-clip"   # part of the analysis cache key; bump when analysis output changes
DETECT_IMGSZ = 960          # upscaled inference finds small, distant people in crowds (Accurate mode)
DETECT_IMGSZ_FAST = 640     # Balanced/Fast modes: skips tiny far-away people -> fewer tracks to encode
DETECT_CONF_LOW = 0.1       # low-score boxes feed BoT-SORT's second association stage
TRACK_BUFFER_S = 2.0        # seconds a lost track is kept for re-association
MAX_INTERP_GAP_S = 2.0      # fill boxes across gaps up to this long
MIN_TRACK_DETECTIONS = 12   # drop tracks seen fewer times than this (~0.5s at 25 fps)
MIN_TRACK_HEIGHT = 40       # px; median box height below this gives useless crops

# --- tracklet stitching with the model's own appearance embedding (V_app) ---
STITCH_MAX_GAP_S = 8.0      # max time between the end of A and the start of B (occlusion length)
STITCH_SPEED = 1.5          # max movement in box-heights per second (plus half a height of slack)
STITCH_MIN_SIM = 0.92       # fallback without the re-ID model: cosine(V_app_A, V_app_B) (different people peak ~0.92)
REID_MODEL = "person-reidentification-retail-0277"   # OpenVINO person re-ID, 256-D embedding of a 256x128 crop
REID_MIN_SIM = 0.60         # cosine of mean re-ID embeddings to join tracklets; on a crowded station clip different
                            # people p99 ~0.40, the same person median ~0.79; 0.52-0.56 produced visible wrong joins
REID_SAMPLE_S = 0.25        # during tracking, one re-ID sample per track every this many seconds...
REID_SAMPLE_MAX_OCCL = 0.3  # ...only from detections with at most this much overlap with other boxes
SPLIT_MAX_SIM = 0.40        # split a track where the mean re-ID embeddings before/after differ this much
SPLIT_MIN_SAMPLES = 2       # samples needed on each side of a split
STITCH_GENDER_GAP = 0.6     # never join tracklets whose P(male) differ by more than this

# --- pedestrian attributes (gender etc.) with OpenVINO person-attributes-recognition-crossroad-0230 ---
ATTR_MODEL = "person-attributes-recognition-crossroad-0230"
OMZ_URL = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/{0}/FP32/{0}"
ATTR_NAMES = ["male", "bag", "backpack", "hat", "longsleeves", "longpants", "longhair", "coat"]
MALE_WORDS = {"man", "men", "male", "males", "guy", "guys", "boy", "boys", "gentleman", "gentlemen", "he", "him", "his"}
FEMALE_WORDS = {"woman", "women", "female", "females", "lady", "ladies", "girl", "girls", "she", "her", "hers"}
POSE_MODEL = "yolov8n-pose.pt"  # shoulders/hips/knees locate the top & bottom garment inside each crop

# Query constraints matched against the attribute model; fused -= weight * (1 - p_match). Gender 0.3 put 42/48
# top-8 results on the right gender in tests; the others are smaller because those attributes are less reliable.
CONSTRAINT_WEIGHTS = {"gender": 0.30, "top_color": 0.25, "bottom_color": 0.20, "backpack": 0.15, "bag": 0.10,
                      "hat": 0.15, "coat": 0.10, "longsleeves": 0.10, "longpants": 0.10, "longhair": 0.15}
LOOSE_COLOR_WEIGHT = 0.15   # "a man in red" (color without a garment) -> top_color with this weight
ATTR_PATTERNS = {           # key -> (regex meaning "yes", regex meaning "no")
    "backpack": (r"\b(backpacks?|rucksacks?)\b", None),
    "bag": (r"\b(bags?|handbags?|purses?|totes?)\b", None),
    "hat": (r"\b(hats?|caps?|beanies?|helmets?)\b", None),
    "coat": (r"\b(coats?|jackets?|overcoats?|parkas?|blazers?)\b", None),
    "longsleeves": (r"\blong[- ]?sleeve[ds]?\b", r"\b(short[- ]?sleeve[ds]?|sleeveless|t-?shirts?|tees?|tank tops?)\b"),
    "longpants": (r"\b(pants|trousers|jeans|leggings|sweatpants)\b", r"\bshorts\b"),
    "longhair": (r"\b(long[- ]hair(ed)?|ponytail)\b", r"\b(short[- ]hair(ed)?|bald)\b"),
}
TOP_GARMENTS = {"shirt", "shirts", "t-shirt", "tshirt", "tee", "top", "jacket", "coat", "sweater", "hoodie", "blouse",
                "vest", "dress", "suit", "cardigan", "blazer", "jumper", "sweatshirt", "polo", "parka", "overcoat"}
BOTTOM_GARMENTS = {"pants", "trousers", "jeans", "shorts", "skirt", "leggings", "sweatpants"}
OTHER_ITEMS = {"bag", "bags", "backpack", "handbag", "purse", "hat", "cap", "shoes", "sneakers", "boots", "hair",
               "umbrella", "suitcase", "glasses", "sunglasses", "helmet", "beanie", "scarf", "mask", "socks"}
COLOR_ALIASES = {"black": "black", "white": "white", "grey": "grey", "gray": "grey", "silver": "grey",
                 "red": "red", "maroon": "red", "burgundy": "red", "orange": "orange", "yellow": "yellow",
                 "gold": "yellow", "green": "green", "olive": "green", "blue": "blue", "navy": "blue",
                 "denim": "blue", "turquoise": "blue", "teal": "blue",
                 "cyan": "blue", "purple": "purple", "violet": "purple", "pink": "pink", "brown": "brown",
                 "beige": "beige", "khaki": "beige", "tan": "beige", "cream": "beige"}
COLOR_MODIFIERS = {"dark", "light", "bright", "pale", "deep", "and", "or", "/"}
# A few sRGB prototypes per colour name; pixels are assigned to the nearest one in (weighted) Lab space.
COLOR_PROTOS = {
    "black": [(20, 20, 20), (45, 45, 50)],
    "white": [(240, 240, 240), (210, 210, 215)],
    "grey": [(128, 128, 128), (95, 95, 100), (170, 170, 172)],
    "red": [(200, 30, 35), (135, 20, 30)],
    "orange": [(240, 130, 30)],
    "yellow": [(235, 210, 40)],
    "green": [(40, 140, 60), (75, 85, 45)],
    "blue": [(40, 80, 190), (30, 40, 90), (110, 150, 205),
             (55, 70, 100), (140, 160, 190), (60, 165, 175)],   # + dark/light denim, turquoise
    "purple": [(120, 50, 140), (90, 60, 120)],
    "pink": [(240, 140, 180), (215, 90, 140)],
    "brown": [(110, 70, 40), (80, 55, 40)],
    "beige": [(215, 195, 160), (185, 165, 125)],
}
COLOR_NAMES = list(COLOR_PROTOS)
COLOR_SIM = {("grey", "black"): 0.5, ("grey", "white"): 0.5, ("brown", "beige"): 0.5, ("brown", "orange"): 0.4,
             ("pink", "red"): 0.5, ("purple", "pink"): 0.4, ("purple", "blue"): 0.4, ("beige", "white"): 0.4,
             ("yellow", "orange"): 0.5, ("beige", "yellow"): 0.3, ("brown", "black"): 0.3, ("red", "orange"): 0.3}

# Legacy greedy tracker (only for the non-YOLO fallback detectors)
MAX_TRACK_GAP = 15

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


def _max_overlap(xyxy: np.ndarray) -> np.ndarray:
    """For each box, the largest fraction of its area covered by any other box in the same frame."""
    n = len(xyxy)
    if n < 2:
        return np.zeros(n)
    x1, y1, x2, y2 = (xyxy[:, i] for i in range(4))
    iw = np.clip(np.minimum(x2[:, None], x2[None]) - np.maximum(x1[:, None], x1[None]), 0, None)
    ih = np.clip(np.minimum(y2[:, None], y2[None]) - np.maximum(y1[:, None], y1[None]), 0, None)
    cover = iw * ih / np.maximum((x2 - x1) * (y2 - y1), 1e-6)[:, None]
    np.fill_diagonal(cover, 0)
    return np.clip(cover.max(axis=1), 0, 1)


def _lab(rgb: np.ndarray) -> np.ndarray:
    """[N, 3] uint8 RGB -> [N, 3] float Lab (L 0..100), with L down-weighted so shading matters less than hue."""
    lab = cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab).reshape(-1, 3)
    return lab * np.array([0.6, 1.0, 1.0], dtype=np.float32)


_PROTO_LAB = _lab(np.array([c for n in COLOR_NAMES for c in COLOR_PROTOS[n]], dtype=np.uint8))
_PROTO_IDX = np.array([i for i, n in enumerate(COLOR_NAMES) for _ in COLOR_PROTOS[n]])


def _color_hist(region_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Fraction of the region's pixels nearest to each colour name (len(COLOR_NAMES),), or None if too small."""
    if region_bgr is None or region_bgr.shape[0] < 3 or region_bgr.shape[1] < 3:
        return None
    if region_bgr.shape[0] * region_bgr.shape[1] > 900:
        region_bgr = cv2.resize(region_bgr, (30, 30), interpolation=cv2.INTER_AREA)
    lab = _lab(cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3))
    nearest = _PROTO_IDX[((lab[:, None] - _PROTO_LAB[None]) ** 2).sum(-1).argmin(1)]
    return np.bincount(nearest, minlength=len(COLOR_NAMES)) / len(nearest)


def _garment_regions(kp: Optional[np.ndarray], w: int, h: int, top_pt, bottom_pt):
    """
    Top/bottom garment boxes inside a crop of size (w, h). kp: COCO keypoints [17, 3] in crop coordinates.
    Uses shoulders -> hips for the top and hips -> knees for the bottom; the bottom is None when the hips are
    not visible (crop shows only the upper body). Falls back to the attribute model's colour points.
    """
    def box(x1, y1, x2, y2):
        x1, x2 = int(max(0, min(x1, x2))), int(min(w, max(x1, x2)))
        y1, y2 = int(max(0, min(y1, y2))), int(min(h, max(y1, y2)))
        return (x1, y1, x2, y2) if x2 - x1 >= 3 and y2 - y1 >= 3 else None

    vis = (lambda i: kp is not None and kp[i, 2] > 0.5)
    top = bottom = None
    if vis(5) and vis(6):
        sx1, sx2 = sorted((kp[5, 0], kp[6, 0]))
        sw = max(sx2 - sx1, 0.2 * w)
        sy = (kp[5, 1] + kp[6, 1]) / 2
        hips = [i for i in (11, 12) if vis(i)]
        hy = float(np.mean([kp[i, 1] for i in hips])) if hips else min(h, sy + 1.5 * sw)
        L = max(hy - sy, 1.0)
        top = box(sx1 + 0.15 * sw, sy + 0.15 * L, sx2 - 0.15 * sw, hy - 0.1 * L)
        if hips:
            hx = [kp[i, 0] for i in hips]
            hx1, hx2 = min(hx), max(hx)
            hw = max(hx2 - hx1, 0.5 * sw)
            cx = (hx1 + hx2) / 2
            knees = [i for i in (13, 14) if vis(i)]
            ky = float(np.mean([kp[i, 1] for i in knees])) if knees else min(h, hy + 0.9 * L)
            if ky - hy >= 0.4 * L:
                Lb = ky - hy
                bottom = box(cx - 0.6 * hw, hy + 0.1 * Lb, cx + 0.6 * hw, ky - 0.05 * Lb)
    elif top_pt is not None:
        tx, ty = top_pt[0] * w, top_pt[1] * h
        top = box(tx - 0.15 * w, ty - 0.06 * h, tx + 0.15 * w, ty + 0.06 * h)
    return top, bottom


def _parse_colors(words: List[str], i: int) -> List[str]:
    """Colour names directly before position i ("dark blue", "black and white")."""
    out, j = [], i - 1
    while j >= 0 and (words[j] in COLOR_ALIASES or words[j] in COLOR_MODIFIERS):
        if words[j] in COLOR_ALIASES:
            out.append(COLOR_ALIASES[words[j]])
        j -= 1
    return out


def query_attributes(prompt: str) -> List[Dict[str, Any]]:
    """
    Structured constraints found in the prompt: gender, garment colours and attribute-model attributes.
    Each is {"key", "value", "weight"}; value is "male"/"female", a list of colour names, or True/False.
    """
    text = prompt.lower()
    words = re.findall(r"[a-z]+(?:-[a-z]+)?|/", text)
    wset = set(words)
    out: List[Dict[str, Any]] = []

    male, female = bool(wset & MALE_WORDS), bool(wset & FEMALE_WORDS)
    if male != female:
        out.append({"key": "gender", "value": "male" if male else "female", "weight": CONSTRAINT_WEIGHTS["gender"]})

    top_c, bottom_c, loose, used = [], [], [], set()
    for i, wd in enumerate(words):
        if wd in TOP_GARMENTS or wd in BOTTOM_GARMENTS or wd in OTHER_ITEMS:
            cols = _parse_colors(words, i)
            for j in range(max(0, i - 4), i):
                used.add(j)
            (top_c if wd in TOP_GARMENTS else bottom_c if wd in BOTTOM_GARMENTS else []).extend(cols)
    for i, wd in enumerate(words):
        if wd in COLOR_ALIASES and i not in used:
            loose.append(COLOR_ALIASES[wd])
    if top_c:
        out.append({"key": "top_color", "value": sorted(set(top_c)), "weight": CONSTRAINT_WEIGHTS["top_color"]})
    elif loose:
        out.append({"key": "top_color", "value": sorted(set(loose)), "weight": LOOSE_COLOR_WEIGHT})
    if bottom_c:
        out.append({"key": "bottom_color", "value": sorted(set(bottom_c)), "weight": CONSTRAINT_WEIGHTS["bottom_color"]})

    for key, (yes, no) in ATTR_PATTERNS.items():
        if re.search(yes, text):
            out.append({"key": key, "value": True, "weight": CONSTRAINT_WEIGHTS[key]})
        elif no and re.search(no, text):
            out.append({"key": key, "value": False, "weight": CONSTRAINT_WEIGHTS[key]})
    return out


def _color_match(dist: Optional[Dict[str, float]], wanted: List[str]) -> Optional[float]:
    """How well a colour distribution matches any wanted colour, 0..1 (1 = the dominant colour)."""
    if not dist:
        return None
    top = max(dist.values()) or 1.0
    best = 0.0
    for c in wanted:
        for k, v in dist.items():
            sim = 1.0 if k == c else COLOR_SIM.get((c, k), COLOR_SIM.get((k, c), 0.0))
            best = max(best, sim * v / top)
    return min(1.0, best)


def constraint_p(track: "Track", c: Dict[str, Any]) -> Optional[float]:
    """Probability that the track satisfies one query constraint (None when the track has no such information)."""
    key, val = c["key"], c["value"]
    if key == "gender":
        p = track.attrs.get("male")
        return None if p is None else (p if val == "male" else 1.0 - p)
    if key in ("top_color", "bottom_color"):
        return _color_match(track.attrs.get(key), val)
    p = track.attrs.get(key)
    return None if p is None else (p if val else 1.0 - p)


@dataclass
class Track:
    id: int
    boxes: Dict[int, List[int]]          # frame -> [x1, y1, x2, y2], incl. interpolated gaps
    det_frames: List[int]                # frames with a real detection
    quality: Dict[int, float] = field(default_factory=dict)  # det frame -> crop quality (conf, occlusion, size)
    attrs: Dict[str, Any] = field(default_factory=dict)      # male/backpack/... -> prob; top_color/bottom_color -> {colour: frac}
    reid: Optional[np.ndarray] = None    # L2-normalised mean re-ID embedding (tracking samples or preview crops)
    reid_n: int = 0                      # number of samples behind `reid` (weights merges)
    clip_emb: Optional[torch.Tensor] = None  # L2-normalised mean CLIP image embedding of preview crops
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
    track_emb: Dict[int, Dict[str, torch.Tensor]] = field(default_factory=dict)  # per-track V_* rows
    stitch_log: List[Dict[str, Any]] = field(default_factory=list)  # re-ID merges (for inspection)


class SearchEngine:
    def __init__(self, root: Path, use_clip: bool = True):
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
        self._init_attr_model()
        self._init_clip(use_clip)

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

    # ------------------------------------------------------------------ attributes
    def _ov_model(self, name: str, folder: str, size_mb: int, dynamic_batch: bool = False):
        """Compiles an Open Model Zoo IR from checkpoints/<folder>/, downloading it once. Raises on failure."""
        import openvino as ov
        xml = self.checkpoints_dir / folder / f"{name}.xml"
        if not xml.exists() or not xml.with_suffix(".bin").exists():
            import urllib.request
            xml.parent.mkdir(parents=True, exist_ok=True)
            print(f"[*] Downloading {name} (~{size_mb} MB)...")
            for ext in (".xml", ".bin"):
                tmp = xml.with_suffix(ext + ".part")
                urllib.request.urlretrieve(OMZ_URL.format(name) + ext, tmp)
                tmp.replace(xml.with_suffix(ext))
        core = ov.Core()
        model = core.read_model(str(xml))
        if dynamic_batch:
            shape = list(model.input(0).shape)
            model.reshape([-1, *shape[1:]])
        device = "GPU" if "GPU" in core.available_devices else "CPU"
        return core.compile_model(model, device), device

    def _init_attr_model(self) -> None:
        """OpenVINO pedestrian-attribute and re-ID models, pose model for garment regions. All optional."""
        self.attr_model = self.reid_model = self.pose_model = None
        try:
            import openvino  # noqa: F401
        except ImportError:
            print("[!] openvino not installed: attribute recognition and re-ID disabled.")
            return
        try:
            self.reid_model, device = self._ov_model(REID_MODEL, "reid", 8, dynamic_batch=True)
            print(f"[+] Re-ID model: {REID_MODEL} on {device}")
        except Exception as e:
            print(f"[!] Re-ID model unavailable ({e}): tracklet stitching falls back to V_app similarity.")
        try:
            self.attr_model, device = self._ov_model(ATTR_MODEL, "attributes", 3)
            self.attr_output = self.attr_model.output("453")
            self.attr_points = (self.attr_model.output("456"), self.attr_model.output("459"))
            print(f"[+] Attribute model: {ATTR_MODEL} on {device}")
        except Exception as e:
            print(f"[!] Attribute model unavailable ({e}): gender/attribute recognition disabled.")
            self.attr_model = None
            return
        try:
            from ultralytics import YOLO
            self.pose_model = YOLO(str(self.checkpoints_dir / POSE_MODEL))
            print(f"[+] Pose model: {POSE_MODEL} (garment colour regions)")
        except Exception as e:
            print(f"[!] Pose model unavailable ({e}): colours fall back to the attribute model's colour points.")

    def _embed_reid(self, crops: List[np.ndarray]) -> np.ndarray:
        """[N, 256] L2-normalised re-ID embeddings of BGR crops."""
        x = np.stack([cv2.resize(c, (128, 256)).transpose(2, 0, 1) for c in crops]).astype(np.float32)
        with self.model_lock:
            v = self.reid_model([x])[self.reid_model.output(0)]
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)

    def _reid(self, analysis: Analysis, tids: List[int]) -> None:
        """Fallback re-ID embedding from the preview crops, for tracks without samples taken during tracking."""
        if self.reid_model is None:
            return
        for tid in tids:
            t = analysis.tracks[tid]
            if t.reid is not None:
                continue
            crops = [img for j in t.film + ([t.thumb] if t.thumb else [])
                     if (img := cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR)) is not None]
            if not crops:
                continue
            m = self._embed_reid(crops).mean(axis=0)
            t.reid, t.reid_n = m / max(float(np.linalg.norm(m)), 1e-6), len(crops)

    def _init_clip(self, enabled: bool) -> None:
        """Optional open-vocabulary scorer (fourth fused score). Downloaded once to the HF cache (~600 MB)."""
        self.clip_model = self.clip_processor = None
        if not enabled:
            print("[*] CLIP disabled (--no_clip).")
            return
        try:
            from transformers import CLIPModel, CLIPProcessor
            self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(self.device).eval()
            self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
            print(f"[+] CLIP: {CLIP_MODEL}")
        except Exception as e:
            print(f"[!] CLIP unavailable ({e}): ranking uses the DSCA scores only.")
            self.clip_model = None

    @property
    def default_weights(self) -> Dict[str, float]:
        return dict(DEFAULT_WEIGHTS) if self.clip_model is not None else dict(DSCA_WEIGHTS)

    @staticmethod
    def _letterbox(img_bgr: np.ndarray) -> np.ndarray:
        """Pads a (tall) person crop to a square RGB image so CLIP's centre crop keeps head and feet."""
        h, w = img_bgr.shape[:2]
        side = max(h, w)
        out = np.full((side, side, 3), 127, np.uint8)
        y, x = (side - h) // 2, (side - w) // 2
        out[y:y + h, x:x + w] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return out

    def _clip_embed(self, analysis: Analysis, tids: List[int], progress: Optional[ProgressCb] = None) -> None:
        """Mean CLIP image embedding over up to CLIP_CROPS preview crops of each track."""
        if self.clip_model is None or not tids:
            return
        items = []  # (tid, letterboxed RGB crop)
        for tid in tids:
            t = analysis.tracks[tid]
            jpegs = ([t.thumb] if t.thumb else []) + ([t.film[0], t.film[-1]] if len(t.film) >= 2 else t.film)
            for j in jpegs[:CLIP_CROPS]:
                img = cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    items.append((tid, self._letterbox(img)))
        feats: Dict[int, List[torch.Tensor]] = {}
        for b in range(0, len(items), CLIP_BATCH):
            chunk = items[b:b + CLIP_BATCH]
            x = self.clip_processor(images=[im for _, im in chunk], return_tensors="pt")["pixel_values"].to(self.device)
            with self.model_lock, torch.inference_mode():
                f = F.normalize(self.clip_model.get_image_features(pixel_values=x).float().cpu(), dim=-1)
            for (tid, _), v in zip(chunk, f):
                feats.setdefault(tid, []).append(v)
            if progress:
                progress("clip", b + len(chunk), len(items))
        for tid, vs in feats.items():
            analysis.tracks[tid].clip_emb = F.normalize(torch.stack(vs).mean(0), dim=-1)

    def _pose_keypoints(self, crops: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """COCO keypoints [17, 3] (crop coordinates) of the person filling each crop, or None."""
        if getattr(self, "pose_model", None) is None or not crops:
            return [None] * len(crops)
        padded, pads = [], []
        for c in crops:
            pad = int(0.3 * c.shape[0])   # context so a tightly cropped person is still detected
            padded.append(cv2.copyMakeBorder(c, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(114, 114, 114)))
            pads.append(pad)
        with self.model_lock:
            results = self.pose_model(padded, imgsz=320, verbose=False)
        out = []
        for c, pad, r in zip(crops, pads, results):
            if r.keypoints is None or not len(r.boxes):
                out.append(None)
                continue
            h, w = c.shape[:2]
            target = [pad, pad, pad + w, pad + h]
            ious = [_iou([int(v) for v in b], target) for b in r.boxes.xyxy.cpu().numpy()]
            k = r.keypoints.data[int(np.argmax(ious))].cpu().numpy().copy()
            k[:, :2] -= pad
            out.append(k)
        return out

    def _attributes(self, analysis: Analysis, tids: List[int]) -> None:
        """
        Per track, over its preview crops (least-occluded frames): averages the 8 attribute probabilities and
        builds top/bottom garment colour distributions from pose-located regions.
        """
        if self.attr_model is None:
            return
        for tid in tids:
            t = analysis.tracks[tid]
            crops = [img for j in t.film + ([t.thumb] if t.thumb else [])
                     if (img := cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR)) is not None]
            if not crops:
                continue
            probs, hists = [], {"top_color": [], "bottom_color": []}
            for img, kp in zip(crops, self._pose_keypoints(crops)):
                x = cv2.resize(img, (80, 160)).transpose(2, 0, 1)[None].astype(np.float32)  # BGR, NCHW, 0..255
                res = self.attr_model([x])
                probs.append(res[self.attr_output].reshape(-1))
                h, w = img.shape[:2]
                regions = _garment_regions(kp, w, h, res[self.attr_points[0]].reshape(-1), res[self.attr_points[1]].reshape(-1))
                for key, r in zip(("top_color", "bottom_color"), regions):
                    hist = _color_hist(img[r[1]:r[3], r[0]:r[2]]) if r else None
                    if hist is not None:
                        hists[key].append(hist)
            t.attrs = {k: float(v) for k, v in zip(ATTR_NAMES, np.mean(probs, axis=0))}
            for key, hs in hists.items():
                if hs:
                    t.attrs[key] = {c: round(float(v), 4) for c, v in zip(COLOR_NAMES, np.mean(hs, axis=0))}

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

        # Pass 1: detect + track. Only boxes and per-detection quality are kept, never frames.
        if self.detector_backend == "yolo":
            raw, frames_processed = self._track_botsort(cap, n_frames, stride, conf, fps, progress)
        else:
            raw, frames_processed = self._track_greedy(cap, n_frames, stride, conf, progress)
        cap.release()
        raw = self._split_by_appearance(raw)

        max_gap = int(MAX_INTERP_GAP_S * fps)
        kept = []
        for t in raw.values():
            dets = sorted(t["boxes"])
            heights = [t["boxes"][f][3] - t["boxes"][f][1] for f in dets]
            if len(dets) < MIN_TRACK_DETECTIONS or float(np.median(heights)) < MIN_TRACK_HEIGHT:
                continue
            boxes = dict(t["boxes"])
            for f0, f1 in zip(dets, dets[1:]):
                if 1 < f1 - f0 <= max_gap:
                    b0, b1 = boxes[f0], boxes[f1]
                    for gf in range(f0 + 1, f1):
                        a = (gf - f0) / float(f1 - f0)
                        boxes[gf] = [int(b0[k] + a * (b1[k] - b0[k])) for k in range(4)]
            track = Track(id=0, boxes=boxes, det_frames=dets, quality=t["q"])
            if t.get("emb"):
                m = np.mean([v for _, v in t["emb"]], axis=0)
                track.reid, track.reid_n = m / max(float(np.linalg.norm(m)), 1e-6), len(t["emb"])
            kept.append(track)
        kept.sort(key=lambda t: t.start)
        tracks = {}
        for i, t in enumerate(kept, start=1):
            t.id = i
            tracks[i] = t

        analysis = Analysis(
            id=analysis_id, video_id=video_id, params=params, fps=float(fps), width=width, height=height,
            frames_processed=frames_processed, frames_total=total or frames_processed, tracks=tracks,
        )
        if tracks:
            self._extract_crops(video_path, analysis, list(tracks), progress)
            self._attributes(analysis, list(tracks))
            self._reid(analysis, list(tracks))
            self._clip_embed(analysis, list(tracks), progress)
            self.encode_tracks(analysis, progress=progress)
            self._stitch(video_path, analysis, progress)
        return analysis

    @staticmethod
    def _split_by_appearance(raw: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Fixes identity switches inside a track: when the tracker's box jumps to another person (typically
        while they cross and occlude each other), the re-ID samples before and after the switch disagree.
        Recursively cuts each track at the sample boundary with the lowest cosine between the mean embeddings
        of the two sides, while that cosine is below SPLIT_MAX_SIM. The pieces are re-joined by _stitch.
        """
        def norm_mean(vs):
            m = np.mean(vs, axis=0)
            return m / max(float(np.linalg.norm(m)), 1e-6)

        def cuts(samples):
            n = len(samples)
            if n < 2 * SPLIT_MIN_SAMPLES:
                return []
            vs = np.stack([v for _, v in samples])
            best_k, best_sim = None, SPLIT_MAX_SIM
            for k in range(SPLIT_MIN_SAMPLES, n - SPLIT_MIN_SAMPLES + 1):
                sim = float(norm_mean(vs[:k]) @ norm_mean(vs[k:]))
                if sim < best_sim:
                    best_k, best_sim = k, sim
            if best_k is None:
                return []
            return cuts(samples[:best_k]) + [samples[best_k][0]] + cuts(samples[best_k:])

        out: Dict[int, Dict[str, Any]] = {}
        next_id = max(raw, default=0) + 1
        for tid, t in raw.items():
            starts = cuts(t.get("emb", []))
            if not starts:
                out[tid] = t
                continue
            bounds = [-1] + starts + [10 ** 9]
            for lo, hi in zip(bounds, bounds[1:]):
                part = {"boxes": {f: b for f, b in t["boxes"].items() if lo <= f < hi},
                        "q": {f: q for f, q in t["q"].items() if lo <= f < hi},
                        "emb": [(f, v) for f, v in t["emb"] if lo <= f < hi]}
                if part["boxes"]:
                    out[next_id] = part
                    next_id += 1
        return out

    def _track_botsort(self, cap, n_frames: int, stride: int, conf: float, fps: float,
                       progress: ProgressCb) -> Tuple[Dict[int, Dict[str, Any]], int]:
        """YOLO + BoT-SORT: Kalman motion, two-stage (high/low score) Hungarian matching, native ReID."""
        cfg = (
            "tracker_type: botsort\n"
            f"track_high_thresh: {conf}\n"
            f"track_low_thresh: {DETECT_CONF_LOW}\n"
            f"new_track_thresh: {conf}\n"
            # BYTETracker scales track_buffer by frame_rate / 30; calls happen every `stride` frames
            f"track_buffer: {max(10, int(TRACK_BUFFER_S * 30 / stride))}\n"
            "match_thresh: 0.8\n"
            "fuse_score: True\n"
            "gmc_method: none\n"           # surveillance cameras are static
            "proximity_thresh: 0.5\n"
            "appearance_thresh: 0.8\n"
            "with_reid: True\n"
            "model: auto\n"                # ReID features straight from the detector, no extra model
        )
        fd, cfg_path = tempfile.mkstemp(suffix=".yaml", prefix="botsort_")
        with os.fdopen(fd, "w") as f:
            f.write(cfg)

        tracks: Dict[int, Dict[str, Any]] = {}
        idx, first = 0, True
        sample_every = max(stride, int(round(REID_SAMPLE_S * fps)))
        try:
            while idx < n_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % stride == 0:
                    res = self.detector.track(frame, persist=not first, tracker=cfg_path, conf=DETECT_CONF_LOW,
                                              imgsz=DETECT_IMGSZ if stride == 1 else DETECT_IMGSZ_FAST,
                                              classes=[0], verbose=False)[0]
                    first = False
                    b = res.boxes
                    if b is not None and b.id is not None and len(b):
                        xyxy = b.xyxy.cpu().numpy()
                        ids = b.id.int().cpu().tolist()
                        scores = b.conf.cpu().numpy()
                        occl = _max_overlap(xyxy)
                        h_img, w_img = frame.shape[:2]
                        to_embed = []
                        for (x1, y1, x2, y2), tid, s, oc in zip(xyxy, ids, scores, occl):
                            box = [max(0, int(x1)), max(0, int(y1)), min(w_img, int(x2)), min(h_img, int(y2))]
                            if box[2] - box[0] < 8 or box[3] - box[1] < 16:
                                continue
                            t = tracks.setdefault(tid, {"boxes": {}, "q": {}, "emb": []})
                            t["boxes"][idx] = box
                            t["q"][idx] = float(s) * (1.0 - float(oc)) * min(1.0, (box[3] - box[1]) / 120.0)
                            if (self.reid_model is not None and oc <= REID_SAMPLE_MAX_OCCL
                                    and box[3] - box[1] >= MIN_TRACK_HEIGHT
                                    and (not t["emb"] or idx - t["emb"][-1][0] >= sample_every)):
                                to_embed.append((tid, frame[box[1]:box[3], box[0]:box[2]]))
                        if to_embed:
                            for (tid, _), v in zip(to_embed, self._embed_reid([c for _, c in to_embed])):
                                tracks[tid]["emb"].append((idx, v))
                idx += 1
                if idx % 5 == 0 or idx == n_frames:
                    progress("detect", idx, n_frames)
        finally:
            os.unlink(cfg_path)
        return tracks, idx

    def _track_greedy(self, cap, n_frames: int, stride: int, conf: float,
                      progress: ProgressCb) -> Tuple[Dict[int, Dict[str, Any]], int]:
        """Fallback IoU/centre tracker for the torchvision and HOG detectors (no track ids of their own)."""
        tracks: Dict[int, Dict[str, Any]] = {}
        next_tid, idx = 1, 0
        while idx < n_frames:
            ok, frame = cap.read()
            if not ok:
                break
            detections = self.detect(frame, conf) if idx % stride == 0 else []
            matched = set()
            for t in tracks.values():
                if idx - t["last"] > MAX_TRACK_GAP:
                    continue
                last = t["boxes"][t["last"]]
                best, best_d = 0.0, -1
                for d_i, det in enumerate(detections):
                    if d_i in matched:
                        continue
                    iou = _iou(last, det)
                    if iou > 0.3 and iou > best:
                        best, best_d = iou, d_i
                if best_d >= 0:
                    matched.add(best_d)
                    t["boxes"][idx] = detections[best_d]
                    t["q"][idx] = 1.0
                    t["last"] = idx
            for d_i, det in enumerate(detections):
                if d_i not in matched:
                    tracks[next_tid] = {"boxes": {idx: det}, "q": {idx: 1.0}, "last": idx}
                    next_tid += 1
            idx += 1
            if idx % 5 == 0 or idx == n_frames:
                progress("detect", idx, n_frames)
        return tracks, idx

    def _pick_clip_frames(self, t: Track) -> List[int]:
        """16 frames spread over the track; in each time bin take the best-quality (least occluded) detection."""
        dets = t.det_frames
        if len(dets) <= NUM_TRACK_FRAMES:
            return [dets[i] for i in np.linspace(0, len(dets) - 1, NUM_TRACK_FRAMES).round().astype(int)]
        bins = np.array_split(np.array(dets), NUM_TRACK_FRAMES)
        return [int(max(b, key=lambda f: t.quality.get(int(f), 0.0))) for b in bins]

    def _extract_crops(self, video_path: Path, analysis: Analysis, tids: List[int], progress: ProgressCb) -> None:
        """Pass 2: re-reads only the frames needed for the given tracks' 16-frame clips and previews."""
        wanted: Dict[int, List[Tuple[int, int]]] = {}  # frame -> [(track_id, slot)]
        for tid in tids:
            t = analysis.tracks[tid]
            t.clip = np.zeros((NUM_TRACK_FRAMES, CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
            for slot, f in enumerate(self._pick_clip_frames(t)):
                wanted.setdefault(f, []).append((tid, slot))
        film_slots = {0, 5, 10, 15}
        thumb_slot = NUM_TRACK_FRAMES // 2
        film: Dict[int, Dict[int, bytes]] = {tid: {} for tid in tids}

        cap = cv2.VideoCapture(str(video_path))
        first_needed, last_needed = min(wanted), max(wanted)
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_needed)
        idx = first_needed
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
                progress("crops", idx - first_needed, last_needed - first_needed + 1)
        cap.release()
        for tid in tids:
            analysis.tracks[tid].film = [film[tid][s] for s in sorted(film[tid])]
        progress("crops", 1, 1)

    def encode_tracks(self, analysis: Analysis, tids: Optional[List[int]] = None,
                      progress: Optional[ProgressCb] = None, stage: str = "encode") -> None:
        """Encodes track clips with SpaceTimeViT (all tracks by default) and rebuilds the search matrices."""
        if tids is None:
            tids = sorted(analysis.tracks)
            analysis.track_emb = {}
        for b in range(0, len(tids), ENCODE_BATCH):
            ids = tids[b:b + ENCODE_BATCH]
            clips = torch.from_numpy(np.stack([analysis.tracks[i].clip for i in ids]))  # [B,T,H,W,3]
            x = clips.permute(0, 1, 4, 2, 3).float().div(255.0)                          # [B,T,3,H,W]
            x = ((x.flatten(0, 1) - _MEAN) / _STD).view_as(x).to(self.device)
            with self.model_lock, torch.no_grad():
                v_app, v_mot, v_video = self.model.encode_video(x)
            for j, tid in enumerate(ids):
                analysis.track_emb[tid] = {
                    "V_app": F.normalize(v_app[j].float().cpu(), dim=-1),
                    "V_mot": F.normalize(v_mot[j].float().cpu(), dim=-1),
                    "V_video": F.normalize(v_video[j].float().cpu(), dim=-1),
                }
            if progress:
                progress(stage, b + len(ids), len(tids))
        self._rebuild_matrices(analysis)
        analysis.ckpt_key = self.ckpt_key

    @staticmethod
    def _rebuild_matrices(analysis: Analysis) -> None:
        order = sorted(analysis.tracks)
        analysis.track_order = order
        analysis.emb = {k: torch.stack([analysis.track_emb[t][k] for t in order]) for k in ("V_app", "V_mot", "V_video")}
        if order and all(analysis.tracks[t].clip_emb is not None for t in order):
            analysis.emb["CLIP"] = torch.stack([analysis.tracks[t].clip_emb for t in order])

    def _stitch(self, video_path: Path, analysis: Analysis, progress: ProgressCb) -> None:
        """
        Re-ID across gaps (occlusions): joins tracklet B onto A when B starts at most STITCH_MAX_GAP_S after A
        ends, near where A was last seen, the attribute model does not see a different gender, and the OpenVINO
        re-ID embeddings agree (V_app similarity is the fallback when the re-ID model is unavailable). Merged tracks are re-cropped and re-encoded,
        then ids are renumbered by start time.
        """
        fps = analysis.fps
        tr = analysis.tracks
        center = lambda b: ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
        cands = []
        for a in tr.values():
            a_box = a.boxes[a.det_frames[-1]]
            for b in tr.values():
                gap = b.det_frames[0] - a.det_frames[-1]
                if gap <= 0 or gap > STITCH_MAX_GAP_S * fps:
                    continue
                b_box = b.boxes[b.det_frames[0]]
                h = max(a_box[3] - a_box[1], b_box[3] - b_box[1])
                (ax, ay), (bx, by) = center(a_box), center(b_box)
                if math.hypot(ax - bx, ay - by) > h * (STITCH_SPEED * gap / fps + 0.5):
                    continue
                if "male" in a.attrs and "male" in b.attrs and abs(a.attrs["male"] - b.attrs["male"]) > STITCH_GENDER_GAP:
                    continue
                if a.reid is not None and b.reid is not None:
                    sim = float(a.reid @ b.reid)
                    ok = sim >= REID_MIN_SIM
                else:
                    sim = float(analysis.track_emb[a.id]["V_app"] @ analysis.track_emb[b.id]["V_app"])
                    ok = sim >= STITCH_MIN_SIM
                if ok:
                    cands.append((sim, a.id, b.id, gap))

        succ: Dict[int, int] = {}
        pred: Dict[int, int] = {}
        analysis.stitch_log = []
        for sim, a, b, gap in sorted(cands, reverse=True):
            if a in succ or b in pred:
                continue
            succ[a], pred[b] = b, a
            analysis.stitch_log.append({"from": a, "to": b, "sim": round(sim, 4), "gap_s": round(gap / fps, 2)})
        if not succ:
            progress("stitch", 1, 1)
            return

        merged = []
        head_of: Dict[int, int] = {}
        for head in [t for t in tr if t not in pred and t in succ]:
            cur = head
            head_of[head] = head
            while cur in succ:
                head_of[succ[cur]] = head
                nxt = tr.pop(succ[cur])
                h = tr[head]
                h.boxes.update(nxt.boxes)
                h.quality.update(nxt.quality)
                h.det_frames = sorted(set(h.det_frames) | set(nxt.det_frames))
                if h.reid is not None and nxt.reid is not None:
                    m = h.reid * h.reid_n + nxt.reid * nxt.reid_n
                    h.reid, h.reid_n = m / max(float(np.linalg.norm(m)), 1e-6), h.reid_n + nxt.reid_n
                analysis.track_emb.pop(nxt.id, None)
                cur = nxt.id
            merged.append(head)
        progress("stitch", 0, 1)
        self._extract_crops(video_path, analysis, merged, lambda *a: None)
        self._attributes(analysis, merged)
        self._reid(analysis, merged)
        self._clip_embed(analysis, merged)
        self.encode_tracks(analysis, tids=merged, progress=progress, stage="stitch")

        # Renumber 1..N by start time so ids read naturally in the UI
        order = sorted(tr.values(), key=lambda t: t.start)
        remap = {t.id: i for i, t in enumerate(order, start=1)}
        analysis.tracks = {remap[t.id]: t for t in order}
        analysis.track_emb = {remap[k]: v for k, v in analysis.track_emb.items()}
        for t in order:
            t.id = remap[t.id]
        for s in analysis.stitch_log:
            s["into"] = remap[head_of[s["from"]]]
        self._rebuild_matrices(analysis)

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
            "constraints": query_attributes(prompt) if self.attr_model is not None else [],
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
        clip_raw = clip_norm = None
        if self.clip_model is not None and "CLIP" in analysis.emb:
            tok = self.clip_processor(text=[CLIP_PREFIX + prompt], return_tensors="pt", padding=True,
                                      truncation=True, max_length=77)
            with self.model_lock, torch.inference_mode():
                ct = F.normalize(self.clip_model.get_text_features(**{k: v.to(self.device) for k, v in tok.items()})
                                 .float().cpu(), dim=-1)[0]
            raw = analysis.emb["CLIP"] @ ct
            # CLIP cosines live in a narrow band (~0.2-0.35); map them per query onto the joint score's
            # mean/std so the weight slider is meaningful and the softmax temperature still applies.
            j = torch.tensor(sim_joint)
            z = (raw - raw.mean()) / raw.std().clamp_min(1e-6) if len(raw) > 1 else raw * 0
            clip_raw, clip_norm = raw.tolist(), (j.mean() + j.std().nan_to_num(0.0) * z).tolist()
        result["scores"] = []
        for i, tid in enumerate(analysis.track_order):
            attr_p = {c["key"]: constraint_p(analysis.tracks[tid], c) for c in result["constraints"]}
            known = [(c["weight"], attr_p[c["key"]]) for c in result["constraints"] if attr_p[c["key"]] is not None]
            result["scores"].append({
                "track_id": tid, "joint": round(sim_joint[i], 5), "app": round(sim_app[i], 5),
                "mot": round(sim_mot[i], 5),
                **({"clip": round(clip_norm[i], 5), "clip_raw": round(clip_raw[i], 5)} if clip_norm else {}),
                "attr_p": {k: (None if v is None else round(v, 4)) for k, v in attr_p.items()},
                "penalty": round(sum(w * (1.0 - p) for w, p in known), 5),      # subtracted from the fused score
                "attr_min": round(min(p for _, p in known), 4) if known else None,  # used by the attribute filter
            })
        return result

    @staticmethod
    def rank(scores: List[Dict[str, Any]], weights: Dict[str, float], attr_filter: bool = False) -> List[Dict[str, Any]]:
        """
        Fuses the DSCA (+ CLIP) scores with normalised weights, subtracts the attribute penalty and adds softmax match
        probability. attr_filter drops tracks failing any query constraint (p < 0.5).
        Mirrored by static/js/app.js::rerank, keep the two in sync.
        """
        keys = [k for k in DEFAULT_WEIGHTS if not scores or k in scores[0]]  # "clip" is absent without CLIP
        wsum = sum(max(0.0, float(weights.get(k, 0.0))) for k in keys) or 1.0
        w = {k: max(0.0, float(weights.get(k, 0.0))) / wsum for k in keys}
        if attr_filter:
            scores = [s for s in scores if s.get("attr_min") is None or s["attr_min"] >= 0.5]
        ranked = [dict(s, fused=sum(w[k] * s[k] for k in w) - s.get("penalty", 0.0)) for s in scores]
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
                "attrs": {k: (v if isinstance(v, dict) else round(v, 3)) for k, v in t.attrs.items()},
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
