"""
Evaluation & Inference Script for Spatio-Temporal DAMR (srccv)
Pure ViT + BERT Architecture for Text-to-Video Person Retrieval (TVPR)

Usage:
  1. Evaluate Checkpoint on Sub-dataset:
     python test.py --sub_dataset TVPReid-Duke
  2. Evaluate specific checkpoint file:
     python test.py --checkpoint checkpoints/TVPReid-Duke/best.pth
  3. Single Query Retrieval:
     python test.py --query "A woman in a red jacket walking towards the gate"
"""

import os
import sys
import warnings
import logging

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try: sys.stderr.reconfigure(encoding='utf-8')
    except Exception: pass

import argparse
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from utils.paths import paths, normalize_sub_dataset_name
from data.dataset import create_dataloaders
from models.damr_main import DAMRModel
from utils.metrics import compute_similarity_matrix, evaluate_rank_metrics
from utils.rerank import DEFAULT_WEIGHTS, debias, encode_bank, fuse, hub_scores, synthetic_bank
from utils.plot_benchmarks import plot_comprehensive_confusion_matrix, plot_training_dashboard


@torch.no_grad()
def collect_embeddings(model: torch.nn.Module, val_loader: torch.utils.data.DataLoader, device: torch.device,
                       tta_flip: bool = False) -> dict:
    """
    Encodes every caption (queries) and every unique video (gallery, deduplicated by video_id).
    With tta_flip, also encodes the horizontally flipped clips into E["G_flip"].
    All embeddings are L2-normalised and kept on CPU.
    """
    model.eval()
    norm = lambda x: F.normalize(x.float().cpu(), p=2, dim=-1)
    texts = {"T_full": [], "T_app": [], "T_mot": []}
    gal = {"V_video": [], "V_app": [], "V_mot": []}
    gal_flip = {"V_video": [], "V_app": [], "V_mot": []}
    query_pids, gallery_pids, visited_videos = [], [], set()

    print("🔍 Running Evaluation on Dataset..." + (" (TTA: + horizontal flip)" if tta_flip else ""))
    for batch in val_loader:
        captions = batch.get("caption", batch.get("q_app"))
        q_app = batch.get("q_app", None)
        q_mot = batch.get("q_mot", None)
        video_frames = batch.get("video_frames", batch.get("motion_frames", batch.get("keyframes"))).to(device, non_blocking=True)
        video_ids = batch.get("video_id")
        pids = batch["person_id"].numpy()

        out = model(captions, video_frames, q_app=q_app, q_mot=q_mot)
        texts["T_full"].append(norm(out["T"]))
        texts["T_app"].append(norm(out["T_app"]))
        texts["T_mot"].append(norm(out["T_mot"]))
        query_pids.extend(pids)

        # Gallery: exactly 1 entry per unique video track
        keep = []
        for i in range(len(pids)):
            vid = video_ids[i] if video_ids is not None else pids[i]
            if vid not in visited_videos:
                visited_videos.add(vid)
                keep.append(i)
                gallery_pids.append(pids[i])
        if not keep:
            continue
        for k in gal:
            gal[k].append(norm(out[k])[keep])
        if tta_flip:
            v_app, v_mot, v_video = model.encode_video(torch.flip(video_frames[keep], dims=[-1]))
            for k, v in (("V_video", v_video), ("V_app", v_app), ("V_mot", v_mot)):
                gal_flip[k].append(norm(v))

    E = {k: torch.cat(v) for k, v in texts.items()}
    E["G"] = {k: torch.cat(v) for k, v in gal.items()}
    E["G_flip"] = {k: torch.cat(v) for k, v in gal_flip.items()} if tta_flip else None
    E["pids_text"] = np.array(query_pids)
    E["pids_video"] = np.array(gallery_pids)
    return E


def score_matrix(E: dict, weights=DEFAULT_WEIGHTS, tta_flip: bool = False, bank: dict = None,
                 hub_beta: float = 0.0, hub_k: int = 10) -> np.ndarray:
    """Fused [N_text, N_video] similarity with optional flip-TTA and bank-based hubness reduction."""
    G = E["G"]
    if tta_flip:
        G = {k: F.normalize(G[k] + E["G_flip"][k], dim=-1) for k in G}
    sims = []
    for t_key, v_key, b_key in (("T_full", "V_video", "T_full"), ("T_app", "V_app", "T_app"), ("T_mot", "V_mot", "T_mot")):
        S = E[t_key] @ G[v_key].t()
        if bank is not None and hub_beta:
            S = debias(S, hub_scores(G[v_key], bank[b_key], hub_k), hub_beta)
        sims.append(S)
    return fuse(*sims, weights=weights).numpy()


@torch.no_grad()
def run_evaluation(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    alpha: float = 0.5,
    confusion_matrix: bool = False,
    cm_size: int = 10,
    output_dir: str = None,
    history: dict = None,
    export_dashboard: bool = False,
    weights=DEFAULT_WEIGHTS,
    tta_flip: bool = False,
    bank: dict = None,
    hub_beta: float = 0.0
):
    """
    Evaluates Text-to-Video Person Retrieval following standard TVPR benchmark protocol:
      - All N_queries text queries evaluated against unique video gallery.
      - Gallery deduplicated by video_id (37 videos for iLIDs, 302 for Duke).
      - Multi-level Disentangled Concept Scoring.
    """
    E = collect_embeddings(model, val_loader, device, tta_flip=tta_flip)
    pids_text, pids_video = E["pids_text"], E["pids_video"]
    sim_matrix = score_matrix(E, weights=weights, tta_flip=tta_flip, bank=bank, hub_beta=hub_beta)
    metrics = evaluate_rank_metrics(sim_matrix, pids_text, pids_video)

    print("\n==========================================")
    print("📊 SPACE-TIME DAMR RETRIEVAL BENCHMARK")
    print("==========================================")
    print(f"Rank@1  : {metrics.get('R@1', 0.0):.2f}%")
    print(f"Rank@5  : {metrics.get('R@5', 0.0):.2f}%")
    print(f"Rank@10 : {metrics.get('R@10', 0.0):.2f}%")
    print(f"Rank@50 : {metrics.get('R@50', 0.0):.2f}%")
    print(f"MdR     : {metrics.get('MdR', 0.0):.1f}")
    print(f"mAP     : {metrics.get('mAP', 0.0):.2f}%")
    print("==========================================\n")

    out_dir = output_dir or str(paths.reports_dir)
    if confusion_matrix:
        print(f"🎨 Generating Comprehensive Cross-Modal Confusion Matrices in: {out_dir}")
        cm_outputs = plot_comprehensive_confusion_matrix(
            sim_matrix=sim_matrix,
            pids_text=pids_text,
            pids_video=pids_video,
            output_dir=out_dir,
            max_identities=cm_size
        )

    if (confusion_matrix or export_dashboard) and history:
        try:
            print(f"📈 Regenerating Training Dashboard with Evaluated Similarity Matrix...")
            hist_copy = dict(history)
            hist_copy["sim_matrix"] = sim_matrix
            hist_copy["pids"] = pids_video
            hist_copy["pids_text"] = pids_text
            out_dash = os.path.join(out_dir, "training_dashboard.png")
            plot_training_dashboard(hist_copy, output_path=out_dash)
        except Exception as e:
            print(f"[!] Warning: Could not regenerate training dashboard: {e}")

    return metrics


SWEEP_WEIGHTS = [(0.50, 0.35, 0.15), (0.60, 0.40, 0.00), (0.55, 0.40, 0.05), (0.40, 0.50, 0.10), (1.00, 0.00, 0.00)]
SWEEP_HUB = [("none", 0.0), ("train", 0.5), ("train", 1.0), ("synthetic", 0.5), ("synthetic", 1.0)]


def train_caption_bank(train_loader, max_n: int = 2000, seed: int = 0):
    """Unique training captions (never the evaluated queries), subsampled deterministically."""
    caps = sorted({str(s["caption"]) for s in train_loader.dataset.samples if s.get("caption")})
    random.Random(seed).shuffle(caps)
    return caps[:max_n]


@torch.no_grad()
def run_sweep(model: torch.nn.Module, train_loader, val_loader, device: torch.device, output_dir: str = None):
    """
    Training-free inference sweep: encodes the val set once (plus flipped clips and two caption banks),
    then reports Rank@K for every (TTA, hubness bank/beta, fusion weights) combination.
    """
    E = collect_embeddings(model, val_loader, device, tta_flip=True)
    decomposer = val_loader.dataset.decomposer
    banks = {
        "train": encode_bank(model, train_caption_bank(train_loader), decomposer, device),
        "synthetic": encode_bank(model, synthetic_bank(), decomposer, device),
    }
    print(f"[+] Gallery {len(E['pids_video'])} videos | queries {len(E['pids_text'])} | "
          f"bank train {len(banks['train']['T_full'])} / synthetic {len(banks['synthetic']['T_full'])}")

    rows = []
    for tta in (False, True):
        for bank_name, beta in SWEEP_HUB:
            for w in SWEEP_WEIGHTS:
                sim = score_matrix(E, weights=w, tta_flip=tta, bank=banks.get(bank_name), hub_beta=beta)
                m = evaluate_rank_metrics(sim, E["pids_text"], E["pids_video"])
                rows.append({"tta": tta, "hub": bank_name, "beta": beta, "weights": list(w),
                             **{k: float(m.get(k, 0.0)) for k in ("R@1", "R@5", "R@10", "mAP", "MdR")}})

    base = rows[0]["R@1"]
    print("\n TTA  hub        beta  weights(j/a/m)     R@1     R@5    R@10     mAP   ΔR@1")
    for r in rows:
        w = "/".join(f"{x:.2f}" for x in r["weights"])
        print(f" {'on ' if r['tta'] else 'off'}  {r['hub']:<10} {r['beta']:.1f}   {w:<16} {r['R@1']:6.2f}  {r['R@5']:6.2f}  "
              f"{r['R@10']:6.2f}  {r['mAP']:6.2f}  {r['R@1'] - base:+6.2f}")

    out_dir = output_dir or str(paths.reports_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "inference_sweep.json")
    with open(out_path, "w") as f:
        json.dump({"sub_dataset": getattr(paths, "sub_dataset", "all"), "n_queries": int(len(E["pids_text"])),
                   "n_gallery": int(len(E["pids_video"])), "results": rows}, f, indent=2)
    print(f"\n[+] Saved sweep to {out_path}")
    return rows


@torch.no_grad()
def run_query_inference(model: torch.nn.Module, text_query: str, val_loader: torch.utils.data.DataLoader, device: torch.device, alpha: float = 0.85):
    model.eval()
    print(f"📝 Original Query : {text_query}")

    # Encode full natural language query with BERT
    T = model.encode_text([text_query], device=device)

    all_v = []
    all_v_app = []
    video_ids = []

    for batch in val_loader:
        video_frames = batch.get("video_frames", batch.get("motion_frames", batch.get("keyframes"))).to(device, non_blocking=True)
        V_app, V_mot, V_video = model.encode_video(video_frames)
        all_v.append(V_video.cpu())
        all_v_app.append(V_app.cpu())
        video_ids.extend(batch["video_id"])

    V = torch.cat(all_v, dim=0)
    V_app = torch.cat(all_v_app, dim=0)

    sim_matrix = compute_similarity_matrix(T.cpu(), V, V_app, beta=alpha)
    scores = sim_matrix[0]
    top_indices = np.argsort(-scores)[:5]

    print("\n🏆 TOP-5 RETRIEVED PERSON VIDEOS:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  Rank #{rank}: Video ID = {video_ids[idx]} | Similarity Score = {scores[idx]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SpaceTime-DAMR Model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pth file")
    parser.add_argument("--query", type=str, default=None, help="Text query string for single-query retrieval demo")
    parser.add_argument("--sub_dataset", type=str, default=None, help="Sub-dataset name (e.g. TVPReid-Duke, TVPReid-iLIDs, TVPReid-PRID, all)")
    parser.add_argument("--data_root", type=str, default=None, help="Path to dataset root folder")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--alpha", type=float, default=0.85, help="Weight blend between fused video and appearance (default: 0.85)")
    parser.add_argument("--confusion_matrix", "--cm", action="store_true", help="Generate and export comprehensive cross-modal confusion matrices")
    parser.add_argument("--cm_size", type=int, default=10, help="Number of identities to display in confusion matrix heatmap (default: 10)")
    parser.add_argument("--dashboard", action="store_true", help="Regenerate training dashboard from checkpoint history")
    parser.add_argument("--weights", type=float, nargs=3, default=list(DEFAULT_WEIGHTS), metavar=("J", "A", "M"),
                        help="Fusion weights for sim(T_full,V_video) / sim(T_app,V_app) / sim(T_mot,V_mot)")
    parser.add_argument("--tta_flip", action="store_true", help="Average gallery embeddings of original + horizontally flipped clips")
    parser.add_argument("--hub", choices=["none", "train", "synthetic"], default="none",
                        help="Hubness reduction with a caption bank: training captions or synthetic captions")
    parser.add_argument("--hub_beta", type=float, default=1.0, help="Strength of hubness reduction (used when --hub != none)")
    parser.add_argument("--sweep", action="store_true", help="Training-free sweep over TTA / hubness / fusion weights")
    parser.add_argument("--reports_dir", type=str, default=None, help="Custom output directory to save reports and confusion matrix plots")
    args, unknown = parser.parse_known_args()

    # Configure sub-dataset paths if provided
    sub_data = args.sub_dataset or getattr(paths, "sub_dataset", "all")
    if hasattr(paths, "set_sub_dataset"):
        paths.set_sub_dataset(sub_data)
    else:
        paths.sub_dataset = normalize_sub_dataset_name(sub_data)

    if args.data_root and hasattr(paths, "set_data_root"):
        paths.set_data_root(args.data_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DAMRModel(
        embed_dim=768,
        text_model_name="bert-base-uncased",
        motion_frame_num=16,
        vit_depth=6,
        vit_heads=12
    ).to(device)

    checkpoint_path = args.checkpoint or str(paths.get_checkpoint_path("best.pth"))
    history = None
    if os.path.exists(checkpoint_path):
        print(f"📦 Loading Checkpoint from {checkpoint_path}")
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        history = ckpt.get("history", None)
    else:
        print(f"⚠️ No checkpoint found at {checkpoint_path}. Running evaluation with initialized weights.")

    active_sub = getattr(paths, "sub_dataset", "all")
    train_loader, val_loader = create_dataloaders(
        data_root=str(paths.data_root),
        sub_dataset=active_sub,
        batch_size=args.batch_size,
        num_frames=16
    )

    if args.sweep:
        run_sweep(model, train_loader, val_loader, device, output_dir=args.reports_dir)
        return

    bank = None
    if args.hub != "none":
        caps = train_caption_bank(train_loader) if args.hub == "train" else synthetic_bank()
        bank = encode_bank(model, caps, val_loader.dataset.decomposer, device)

    if args.query:
        run_query_inference(model, args.query, val_loader, device, alpha=args.alpha)
    else:
        run_evaluation(
            model=model,
            val_loader=val_loader,
            device=device,
            alpha=args.alpha,
            confusion_matrix=args.confusion_matrix,
            cm_size=args.cm_size,
            output_dir=args.reports_dir,
            history=history,
            export_dashboard=args.dashboard,
            weights=tuple(args.weights),
            tta_flip=args.tta_flip,
            bank=bank,
            hub_beta=args.hub_beta if bank is not None else 0.0
        )


if __name__ == "__main__":
    main()
