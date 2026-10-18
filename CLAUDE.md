# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SpaceTime-DSCA (code also calls it "SpaceTime-DAMR"/`DAMRModel`): PyTorch Text-to-Video Person Retrieval on the TVPReid benchmark (`TVPReid-Duke`, `TVPReid-PRID`, `TVPReid-iLIDs`). Pure ViT + BERT. It deliberately avoids CLIP pretraining so comparison with the MFGF (ACM MM '24) baseline is fair. README.md is in Vietnamese and holds the research rationale, ablation tables and results.

There is no test suite, linter or build step. `pbl6remake_cvclip.ipynb` is the Colab runner.

## Commands

```bash
# Environment: the system Python has no pip/ensurepip, so use uv (~/.local/bin/uv).
# The local machine has no NVIDIA GPU, so the CPU torch wheel is used.
~/.local/bin/uv venv .venv --python 3.12
~/.local/bin/uv pip install --python .venv/bin/python -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
source .venv/bin/activate


# Train (CLI flags override configs/default.yaml)
python train.py --sub_dataset TVPReid-PRID --epochs 35 --lr 0.0001 --batch_size 16
python train.py --sub_dataset TVPReid-PRID --resume latest      # or 'best', or a .pth path
python train.py --data_root /path/to/dataset --no_pretrained_vit

# Evaluate (defaults to checkpoints/<sub_dataset>/best.pth)
python test.py --sub_dataset TVPReid-PRID --checkpoint checkpoints/TVPReid-PRID/best.pth --confusion_matrix --cm_size 10
python test.py --sub_dataset TVPReid-PRID --dashboard          # regenerate training dashboard from ckpt history

# Training-free inference tricks (utils/rerank.py): fusion weights, flip-TTA, hubness reduction with a caption bank.
# Without these flags test.py reproduces the original numbers exactly.
python test.py --sub_dataset TVPReid-PRID --weights 0.5 0.35 0.15 --tta_flip --hub train --hub_beta 0.5
python test.py --sub_dataset TVPReid-PRID --sweep              # one encode, full grid -> reports/<sub>/inference_sweep.json

# Single text query against the val gallery
python test.py --sub_dataset TVPReid-PRID --query "A man in a dark grey jacket walking away"

# Web demo (FastAPI + static UI); sample videos are read from ./samples by default
python web_demo.py --port 8000 --no-browser --samples_dir samples
```

If there are no captions under the data root, `TVPRDataset` silently builds a dummy index (32 train / 16 val samples). That is fine for a smoke test, but if metrics look meaningless, check this first.

## Paths and data layout (`utils/paths.py`)

- `paths` is a module-level singleton. `paths.set_sub_dataset(name)` creates `checkpoints/<sub>/`, `reports/<sub>/` and `outputs/<sub>/` inside the repo (`all` when no sub-dataset is given). `normalize_sub_dataset_name` maps loose names such as "prid" or "duke" to their canonical names.
- The data root is resolved in this order: `--data_root` → the `DATA_ROOT` env var → the first existing path among `/content/dataset`, `/content/TVPReid_dataset`, `C:/dataset`, `../dataset`, `../TVPReid_dataset` and `./dataset`. The default dataset location is next to the repo, not inside it.
- The dataset scanner finds any directory with a `captions/` subfolder. It joins `captions/{train,val,test}.csv` (`video_id` column) with `captions/*.json` (video_id → list of captions). It also accepts a CSV with a direct `caption` column, or JSON-only annotations.
- Checkpoints (`*.pth`) are gitignored. `reports/<sub>/` holds committed `train_history.json` and dashboard PNGs.

## Architecture

Data flow: `data/dataset.py` → `models/damr_main.py::DAMRModel` → `loss/criterion.py::DAMRCriterion`, with evaluation in `utils/metrics.py`.

- **Text decomposition (DSCA)**: `data/text_decomposer.py::TextDecomposer` runs inside `__getitem__`. It splits each caption into `q_app` (clothing/colors/accessories) and `q_mot` (motion verbs/direction). It is vocabulary-rule based, with an optional `llm_cache` dict override. Each batch dict carries `caption`, `q_app`, `q_mot`, `video_frames` `[B,T,C,H,W]` (T=16, 224²), `person_id` and `video_id`.
- **Text encoder**: `models/text_encoder.py` is a single shared BERT-base-uncased that encodes `T_full`, `T_app` and `T_mot`, all 768-D.
- **Video encoder**: `models/spacetime_vit.py::SpaceTimeViT` is a TimeSformer-style divided space-time ViT (6 blocks, 12 heads). Its spatial layers load ImageNet-1K weights from torchvision `vit_b_16`, falling back to HF `google/vit-base-patch16-224` if that fails, and its temporal attention `out_proj` is zero-initialized. It returns `(V_app, V_mot, V_video)`: attention-pooled appearance, a Bi-GRU motion stream, and an MLP fusion of the two.
- **Loss**: `L = InfoNCE(T_full, V_video) + λ_app·InfoNCE(T_app, V_app) + λ_mot·InfoNCE(T_mot, V_mot) + λ_mim·L_MIM`. `L_MIM` is the cosine orthogonality between V_app and V_mot (`models/mim_loss.py`); it is computed in the model forward pass and returned as `l_mim`.
- **Eval protocol** (`train.py::evaluate`): every caption is a query. The gallery is **deduplicated by `video_id`**, because each video has 2 captions; this deduplication is essential for correct Rank@K/MdR. The score is `0.50·sim(T_full,V_video) + 0.35·sim(T_app,V_app) + 0.15·sim(T_mot,V_mot)`. These weights are hardcoded and the `alpha` argument is ignored here. `test.py` instead goes through `compute_similarity_matrix` using `--alpha` (default 0.85), so numbers from the two paths can differ.

## Web demo (`web_demo.py` → `webapp/`)

`web_demo.py` is only a launcher. The app is split into a one-off **analyse** step per video and a cheap **search** step per prompt:

- **Prepare** (`webapp/video_io.py`): every input video is re-encoded to a constant-frame-rate H.264 `proxy.mp4` (max 720p) using the ffmpeg bundled with `imageio-ffmpeg`. *All* analysis and rendering read the proxy, so frame index `i` equals player time `i / fps` in the browser overlay. Do not analyse the original file.
- **Analyse** (`webapp/engine.py::SearchEngine.analyze`):
  - Pass 1 runs `YOLO.track` with ultralytics **BoT-SORT**: Kalman filter, two-stage high/low-score Hungarian matching, and native ReID features (`with_reid`, `model: auto`). The tracker yaml is written to a temp file for each run. The UI `conf` is used as `track_high_thresh`; the detector runs at `DETECT_CONF_LOW`. `imgsz` is 960 in Accurate mode and 640 in Balanced/Fast mode. This pass keeps only boxes plus a per-detection quality score (conf × (1 − occlusion) × size). The greedy tracker `_track_greedy` is only a fallback for non-YOLO detectors.
  - Tracks with fewer than `MIN_TRACK_DETECTIONS` detections or a median height under `MIN_TRACK_HEIGHT` are dropped.
  - Pass 2 re-reads only the frames each 16-frame clip needs. In each of the 16 time bins it picks the best-quality (least occluded) detection.
  - Each clip is batch-encoded into `V_app/V_mot/V_video`. On CPU this costs about 2s per track, which dominates analysis time in crowded scenes.
  - **Re-ID (split + stitch).** OpenVINO `person-reidentification-retail-0277` (about 8MB, `checkpoints/reid/`) is a real person re-ID model. Do **not** use `V_app` for re-ID: it is a text-retrieval embedding, and different people reach about 0.92 there. It is only the fallback when the re-ID model is missing.
    - During tracking, a re-ID sample is taken per track every `REID_SAMPLE_S` (0.25s), only from boxes with overlap ≤ `REID_SAMPLE_MAX_OCCL`.
    - `_split_by_appearance` cuts a track where the mean embeddings before and after differ (cosine < `SPLIT_MAX_SIM` 0.40). This fixes BoT-SORT jumping to the person who occludes the tracked one.
    - `_stitch` joins tracklet B onto A when all of these hold:
      - B starts at most `STITCH_MAX_GAP_S` (8s) after A ends;
      - B starts near A's last position;
      - the genders are not clearly different;
      - the cosine of the sample-mean embeddings `Track.reid` is ≥ `REID_MIN_SIM` 0.60.
    - Calibration on a crowded station clip: different people reach a p99 of about 0.40, the same person a median of about 0.79, and 0.52–0.56 produced visible wrong joins.
    - On that clip, mixed-identity tracks fell from 37% to 21% and 27 tracks span more than 10s. Merged tracks are re-cropped and re-encoded, and ids are renumbered by start time.
  - The result is cached in memory as an `Analysis`, keyed by `(video_id, max_frames, stride, conf, TRACKER_VERSION)`. Only the first `max_frames` frames are processed. Bump `TRACKER_VERSION` when tracking logic changes.
- **Search** returns three raw cosines per track: `T_full↔V_video`, `T_app↔V_app` and `T_mot↔V_mot`. `T_app`/`T_mot` come from `TextDecomposer.extract_concepts`. Fusion weights (default 0.50/0.35/0.15, the same as `train.py::evaluate`) and the softmax(τ=0.07) "match probability" are computed **in the browser** (`static/js/app.js::rerank`). The same logic is mirrored in `SearchEngine.rank`, which renders and reports use, so keep the two in sync.
- **Attribute matching**: the DSCA scores alone are weak on gender (roughly chance), colours and carried items, so query constraints are checked against dedicated models.
  - **Models.** OpenVINO `person-attributes-recognition-crossroad-0230` (about 3MB) is auto-downloaded to `checkpoints/attributes/`. It gives 8 probabilities (male, bag, backpack, hat, longsleeves, longpants, longhair, coat). `yolov8n-pose.pt` lives in `checkpoints/`.
  - **Per track** (`SearchEngine._attributes`, run over the 5 preview crops):
    - The attribute probabilities are averaged into `Track.attrs`.
    - Pose keypoints locate the top garment (shoulders → hips) and the bottom garment (hips → knees). `_garment_regions` returns no bottom region when the hips are not visible.
    - `_color_hist` assigns each region's pixels to the nearest of the `COLOR_PROTOS` in L-down-weighted Lab space, giving `attrs["top_color"]` and `attrs["bottom_color"]` (colour → fraction).
  - **Query side.** `query_attributes(prompt)` parses gender words, colour+garment pairs (`TOP_GARMENTS`/`BOTTOM_GARMENTS`) and `ATTR_PATTERNS` into constraints `{key, value, weight}` (`CONSTRAINT_WEIGHTS`). There is no negation handling; it was removed on purpose.
  - **Scoring.** `search` adds per track `attr_p` (from `constraint_p`), `penalty = Σ weight·(1 − p)` and `attr_min`. Fused = weighted cosines − penalty. The "matching attributes only" filter drops tracks with `attr_min < 0.5`. `SearchEngine.rank` and `app.js::rerank` both implement this, so keep them in sync.
  - **Fallbacks.** If openvino or the attribute model is missing, constraints are empty and ranking is pure DSCA. If the pose model is missing, colours fall back to the attribute model's colour points.
- **Jobs** (`webapp/jobs.py`): prepare, analyse and render run on a single worker thread, and the UI polls `/api/jobs/{id}`. HTTP handlers are plain `def`, so they run in FastAPI's threadpool and never block the event loop. Checkpoint switching is refused while a job is busy. Embeddings are re-encoded lazily when the active checkpoint changes.
- **Storage**: files live in `uploads/videos/<id>/`, which is wiped at every server start because the index is in memory. Videos are addressed only by id; there is no path-based file access.
- **Startup**: `SearchEngine` passes `pretrained_vit=False` when a checkpoint exists, because the ImageNet ViT weights would be overwritten anyway; this skips a 331MB download. If `bert-base-uncased` cannot be loaded, `TextEncoder` silently falls back to an untrained encoder, so checkpoints fail the full-match check in `SearchEngine.load_checkpoint`. The engine prints a warning when that happens.
- **UI**: `webapp/static/` is vanilla ES modules with no build step and no CDN. Strings live in `js/i18n.js` (VI/EN). Theme tokens are in `css/app.css`.

### Gotchas

- `DAMRModel.forward` accepts several legacy call signatures: `(texts, frames)`, a batch dict, the 4-arg `(q_app, q_mot, keyframes, motion_frames)`, or kwargs. Several outputs are aliases: `T` = `T_full` and `V` = `V_video`. Keep these working, because `train.py` and `test.py` call it in different ways.
- `test.py` and `webapp/engine.py` hardcode the model hyperparameters (`embed_dim=768`, `vit_depth=6`, `vit_heads=12`, 16 frames) instead of reading `configs/default.yaml`. If the architecture config changes, update those call sites too, or checkpoints will fail to load.
- `models/motion_difference.py` and `models/visual_appearance.py` (CLIP-based) are legacy and not imported by the current pipeline.
- Checkpoints store `model_state_dict` and `history`. `paths.get_checkpoint_path` falls back to `best_damr.pth`, then to the root `checkpoints/`.
