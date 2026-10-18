"""
Training-free inference helpers shared by test.py and the web demo.

  - fuse:            weighted sum of the three DSCA similarities (joint / app / mot).
  - hub_scores +
    debias:          hubness reduction. A gallery item that is close to *every* caption of a reference
                     bank gets its scores lowered: S' = S - beta * (r_g - mean(r)), where r_g is the mean
                     top-k cosine of item g to the bank. This is a simplified CSLS; it keeps the cosine scale,
                     so downstream softmax temperatures and weights still apply, and never looks at the
                     queries being evaluated.
  - synthetic_bank:  deterministic captions built from the TextDecomposer vocabulary, used when no training
                     captions are available (the web demo).
  - encode_bank:     encodes a caption bank into normalised (T_full, T_app, T_mot).
"""

import random
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

DEFAULT_WEIGHTS = (0.50, 0.35, 0.15)

_SUBJECTS = ["man", "woman", "person", "young man", "young woman", "guy", "lady", "boy", "girl"]
_COLORS = ["black", "white", "red", "blue", "green", "yellow", "brown", "grey", "dark", "light",
           "orange", "pink", "purple", "beige", "navy", "khaki", "striped"]
_TOPS = ["coat", "jacket", "shirt", "t-shirt", "top", "blouse", "sweater", "hoodie", "vest", "suit", "dress"]
_BOTTOMS = ["pants", "trousers", "jeans", "shorts", "skirt", "leggings"]
_SHOES = ["shoes", "sneakers", "boots", "sandals"]
_CARRY = ["a backpack", "a bag", "a handbag", "a shoulder bag", "a plastic bag", "a suitcase", "an umbrella"]
_HEAD = ["a hat", "a cap", "glasses", "sunglasses"]
_ACTIONS = ["walking", "walking slowly", "walking quickly", "running", "standing", "turning around",
            "walking towards the camera", "walking away from the camera", "walking to the left",
            "walking to the right", "crossing the street", "moving forward"]


def synthetic_bank(n: int = 300, seed: int = 0) -> List[str]:
    """Deterministic pedestrian captions in the style of TVPReid annotations."""
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < n:
        parts = [f"A {rng.choice(_SUBJECTS)} wearing a {rng.choice(_COLORS)} {rng.choice(_TOPS)}"]
        if rng.random() < 0.8:
            parts.append(f"and {rng.choice(_COLORS)} {rng.choice(_BOTTOMS)}")
        if rng.random() < 0.3:
            parts.append(f"with {rng.choice(_COLORS)} {rng.choice(_SHOES)}")
        if rng.random() < 0.3:
            parts.append(f"and {rng.choice(_HEAD)}")
        if rng.random() < 0.4:
            parts.append(f"is carrying {rng.choice(_CARRY)} while")
        else:
            parts.append("is")
        parts.append(rng.choice(_ACTIONS) + ".")
        text = " ".join(parts)
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


@torch.no_grad()
def encode_bank(model, captions: Sequence[str], decomposer, device, batch_size: int = 64) -> Dict[str, torch.Tensor]:
    """Encodes captions (and their DSCA sub-prompts) into L2-normalised T_full / T_app / T_mot on CPU."""
    acc = {"T_full": [], "T_app": [], "T_mot": []}
    captions = list(captions)
    for i in range(0, len(captions), batch_size):
        chunk = captions[i:i + batch_size]
        q_app, q_mot = zip(*(decomposer.decompose(c) for c in chunk))
        for key, texts in (("T_full", chunk), ("T_app", list(q_app)), ("T_mot", list(q_mot))):
            acc[key].append(F.normalize(model.encode_text(list(texts), device=device).float().cpu(), dim=-1))
    return {k: torch.cat(v) for k, v in acc.items()}


def hub_scores(V: torch.Tensor, T_bank: torch.Tensor, k: int = 10) -> torch.Tensor:
    """r_g: mean of the top-k cosines between each (normalised) gallery item and the bank. Returns [G]."""
    sims = F.normalize(V.float(), dim=-1) @ F.normalize(T_bank.float(), dim=-1).t()
    return sims.topk(min(k, sims.size(1)), dim=1).values.mean(dim=1)


def debias(S: torch.Tensor, r: torch.Tensor, beta: float) -> torch.Tensor:
    """S [..., G] minus the centred hubness of each gallery item."""
    if beta == 0:
        return S
    return S - beta * (r - r.mean())


def fuse(s_joint: torch.Tensor, s_app: torch.Tensor, s_mot: torch.Tensor, weights: Sequence[float] = DEFAULT_WEIGHTS):
    wj, wa, wm = weights
    return wj * s_joint + wa * s_app + wm * s_mot
