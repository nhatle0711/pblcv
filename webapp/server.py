"""FastAPI app: video registry, background analysis/render jobs, search, report and static UI."""

import hashlib
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import video_io
from .engine import DEFAULT_WEIGHTS, RANK_COLORS, TRACKER_VERSION, Analysis, SearchEngine
from .jobs import JobManager
from .report import build_report

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_VIDEOS = 20
MAX_ANALYSES = 8


class VideoEntry:
    def __init__(self, vid: str, name: str, source: Path, folder: Path):
        self.id = vid
        self.name = name
        self.source = source
        self.folder = folder
        self.proxy = folder / "proxy.mp4"
        self.info: Optional[Dict[str, Any]] = None  # probe of the proxy
        self.prepare_job: Optional[str] = None

    @property
    def ready(self) -> bool:
        return self.info is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.id,
            "name": self.name,
            "ready": self.ready,
            "info": self.info,
            "prepare_job": None if self.ready else self.prepare_job,
            "stream_url": f"/api/videos/{self.id}/stream" if self.ready else None,
        }


class AnalyzeParams(BaseModel):
    max_frames: int = Field(300, ge=0, le=20000)   # 0 = whole video
    stride: int = Field(1, ge=1, le=4)
    conf: float = Field(0.35, ge=0.1, le=0.9)


class SearchBody(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500)


class RankBody(SearchBody):
    weights: Dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    top_k: int = Field(5, ge=1, le=10)
    lang: str = "vi"
    attr_filter: bool = False
    gender_filter: bool = False   # legacy name of attr_filter

    @property
    def filter_attrs(self) -> bool:
        return self.attr_filter or self.gender_filter


class CheckpointBody(BaseModel):
    name: str


class SampleBody(BaseModel):
    name: str


def create_app(engine: SearchEngine, uploads_dir: Path, samples_dir: Path, max_upload_mb: int = 1024) -> FastAPI:
    app = FastAPI(title="SpaceTime-DSCA Person Search", docs_url="/api/docs", redoc_url=None)

    @app.middleware("http")
    async def revalidate_static(request, call_next):
        # Local demo: always revalidate UI files (cheap 304s) so edits show up without a hard reload
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response
    jobs = JobManager()
    videos: "OrderedDict[str, VideoEntry]" = OrderedDict()
    analyses: "OrderedDict[str, Analysis]" = OrderedDict()
    analysis_keys: Dict[tuple, str] = {}
    registry_lock = threading.Lock()

    store = uploads_dir / "videos"
    shutil.rmtree(store, ignore_errors=True)  # session-scoped: index lives in memory
    store.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ helpers
    def get_video(vid: str) -> VideoEntry:
        v = videos.get(vid)
        if v is None:
            raise HTTPException(404, "Video not found")
        videos.move_to_end(vid)
        return v

    def get_analysis(aid: str) -> Analysis:
        a = analyses.get(aid)
        if a is None:
            raise HTTPException(404, "Analysis not found (expired). Please analyse the video again.")
        analyses.move_to_end(aid)
        return a

    def evict() -> None:
        while len(videos) > MAX_VIDEOS:
            vid, v = videos.popitem(last=False)
            for key, aid in list(analysis_keys.items()):
                if key[0] == vid:
                    analysis_keys.pop(key)
                    analyses.pop(aid, None)
            shutil.rmtree(v.folder, ignore_errors=True)
        while len(analyses) > MAX_ANALYSES:
            aid, _ = analyses.popitem(last=False)
            for key, val in list(analysis_keys.items()):
                if val == aid:
                    analysis_keys.pop(key)

    def prepare(v: VideoEntry, progress) -> None:
        if v.ready:
            return
        if not v.proxy.exists():
            tmp = v.folder / "proxy.tmp.mp4"
            video_io.make_proxy(v.source, tmp, lambda f: progress("prepare", f * 100, 100))
            tmp.replace(v.proxy)
        v.info = video_io.probe(v.proxy)
        progress("prepare", 100, 100)

    def register(vid: str, name: str, source: Path, folder: Path) -> VideoEntry:
        with registry_lock:
            v = videos.get(vid)
            if v is not None:
                videos.move_to_end(vid)
                prev = jobs.get(v.prepare_job) if v.prepare_job else None
                if v.ready or (prev is not None and prev.state in ("queued", "running")):
                    return v
            else:
                v = VideoEntry(vid, name, source, folder)
                videos[vid] = v
                evict()
        job = jobs.submit("prepare", {"prepare": (0, 100)}, lambda j: prepare(v, j.progress))
        v.prepare_job = job.id
        return v

    def sample_files() -> Dict[str, Path]:
        if not samples_dir.is_dir():
            return {}
        return {
            p.name: p for p in sorted(samples_dir.iterdir())
            if p.is_file() and p.suffix.lower() in video_io.VIDEO_EXTENSIONS and p.stat().st_size > 10_000
        }

    def resolve_weights(w: Dict[str, float]) -> Dict[str, float]:
        return {k: float(w.get(k, DEFAULT_WEIGHTS[k])) for k in DEFAULT_WEIGHTS}

    # ------------------------------------------------------------------ status / checkpoints
    @app.get("/api/status")
    def status():
        info = engine.checkpoint_info
        return {
            "device": str(engine.device),
            "detector": engine.detector_backend,
            "checkpoint": info,
            "checkpoints": engine.scan_checkpoints(),
            "ffmpeg": video_io.FFMPEG_EXE is not None,
            "busy": jobs.busy(),
            "palette": [hex_ for _, hex_ in RANK_COLORS],
            "default_weights": DEFAULT_WEIGHTS,
            "max_upload_mb": max_upload_mb,
            "attributes": engine.attr_model is not None,
        }

    @app.post("/api/checkpoint")
    def switch_checkpoint(body: CheckpointBody):
        if jobs.busy():
            raise HTTPException(409, "A job is running. Wait for it to finish before switching checkpoints.")
        try:
            return {"checkpoint": engine.load_checkpoint(body.name)}
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(422, str(e))

    # ------------------------------------------------------------------ videos
    @app.get("/api/samples")
    def list_samples():
        return [
            {"name": n, "size_mb": round(p.stat().st_size / 1e6, 1), "thumb_url": f"/api/samples/{n}/thumb"}
            for n, p in sample_files().items()
        ]

    @app.get("/api/samples/{name}/thumb")
    def sample_thumb(name: str):
        p = sample_files().get(name)
        if p is None:
            raise HTTPException(404, "Sample not found")
        return Response(video_io.first_frame_jpeg(p), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=3600"})

    @app.post("/api/videos/sample")
    def add_sample(body: SampleBody):
        p = sample_files().get(body.name)
        if p is None:
            raise HTTPException(404, "Sample not found")
        st = p.stat()
        vid = hashlib.sha1(f"{p}:{st.st_size}:{st.st_mtime}".encode()).hexdigest()[:16]
        folder = store / vid
        folder.mkdir(exist_ok=True)
        return register(vid, p.name, p, folder).to_dict()

    @app.post("/api/videos")
    def upload_video(file: UploadFile = File(...)):
        ext = Path(file.filename or "").suffix.lower()
        if ext not in video_io.VIDEO_EXTENSIONS:
            raise HTTPException(415, f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(video_io.VIDEO_EXTENSIONS))}")
        tmp = store / f"incoming_{uuid.uuid4().hex}{ext}"
        sha = hashlib.sha1()
        size, limit = 0, max_upload_mb * 1024 * 1024
        try:
            with tmp.open("wb") as out:
                while chunk := file.file.read(1 << 20):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"File larger than {max_upload_mb} MB")
                    sha.update(chunk)
                    out.write(chunk)
            vid = sha.hexdigest()[:16]
            if vid in videos:
                tmp.unlink(missing_ok=True)
                return get_video(vid).to_dict()
            try:
                video_io.probe(tmp)
            except ValueError as e:
                raise HTTPException(422, str(e))
            folder = store / vid
            folder.mkdir(exist_ok=True)
            source = folder / f"original{ext}"
            tmp.replace(source)
        finally:
            tmp.unlink(missing_ok=True)
        return register(vid, Path(file.filename).name, source, folder).to_dict()

    @app.get("/api/videos/{vid}")
    def video_info(vid: str):
        return get_video(vid).to_dict()

    @app.get("/api/videos/{vid}/stream")
    def stream(vid: str):
        v = get_video(vid)
        if not v.ready:
            raise HTTPException(409, "Video is still being prepared")
        return FileResponse(v.proxy, media_type="video/mp4")

    # ------------------------------------------------------------------ analysis
    @app.post("/api/videos/{vid}/analyze")
    def analyze(vid: str, params: AnalyzeParams):
        v = get_video(vid)
        key = (vid, params.max_frames, params.stride, round(params.conf, 2), TRACKER_VERSION)
        if key in analysis_keys and analysis_keys[key] in analyses:
            return {"analysis_id": analysis_keys[key], "cached": True, "job_id": None}

        aid = uuid.uuid4().hex[:12]

        def run(job):
            prepare(v, job.progress)
            a = engine.analyze(v.proxy, vid, aid, {**params.model_dump(), "tracker": TRACKER_VERSION}, job.progress)
            with registry_lock:
                analyses[aid] = a
                analysis_keys[key] = aid
                evict()
            return {"analysis_id": aid}

        job = jobs.submit("analyze", {
            "prepare": (0, 3), "detect": (3, 45), "crops": (45, 50), "encode": (50, 92), "stitch": (92, 100),
        }, run)
        return {"analysis_id": None, "cached": False, "job_id": job.id}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return job.to_dict()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        return {"cancelled": jobs.cancel(job_id)}

    @app.get("/api/analyses/{aid}")
    def analysis_info(aid: str):
        a = get_analysis(aid)
        payload = engine.tracks_payload(a)
        payload["video_name"] = videos[a.video_id].name if a.video_id in videos else ""
        return payload

    @app.post("/api/analyses/{aid}/search")
    def search(aid: str, body: SearchBody):
        a = get_analysis(aid)
        t0 = time.time()
        res = engine.search(a, body.prompt)
        res["elapsed_ms"] = int((time.time() - t0) * 1000)
        res["checkpoint"] = engine.ckpt_key
        return res

    # ------------------------------------------------------------------ outputs
    @app.post("/api/analyses/{aid}/render")
    def render(aid: str, body: RankBody):
        a = get_analysis(aid)
        v = get_video(a.video_id)
        weights = resolve_weights(body.weights)

        def run(job):
            ranked = engine.rank(engine.search(a, body.prompt)["scores"], weights, body.filter_attrs)
            if not ranked:
                raise ValueError("No persons to render")
            tag = hashlib.sha1(f"{body.prompt}|{weights}|{body.top_k}|{engine.ckpt_key}".encode()).hexdigest()[:10]
            out = v.folder / f"render_{a.id}_{tag}.mp4"
            if not out.exists():
                raw = v.folder / f"render_{tag}.raw.mp4"
                engine.render(v.proxy, a, ranked, body.prompt, body.top_k, raw, job.progress)
                video_io.to_h264(raw, out, a.frames_processed / a.fps,
                                 lambda f: job.progress("encode_h264", f * 100, 100))
            return {"url": f"/api/videos/{v.id}/renders/{out.name}", "filename": f"{Path(v.name).stem}_tracked.mp4"}

        job = jobs.submit("render", {"render": (0, 85), "encode_h264": (85, 100)}, run)
        return {"job_id": job.id}

    @app.get("/api/videos/{vid}/renders/{filename}")
    def get_render(vid: str, filename: str, download: Optional[str] = None):
        v = get_video(vid)
        p = (v.folder / filename).resolve()
        if p.parent != v.folder.resolve() or not p.name.startswith("render_") or not p.is_file():
            raise HTTPException(404, "Render not found")
        return FileResponse(p, media_type="video/mp4", filename=download or None)

    @app.post("/api/analyses/{aid}/report")
    def report(aid: str, body: RankBody):
        a = get_analysis(aid)
        weights = resolve_weights(body.weights)
        res = engine.search(a, body.prompt)
        ranked = engine.rank(res["scores"], weights, body.filter_attrs)
        name = videos[a.video_id].name if a.video_id in videos else a.video_id
        data = build_report(a, res, ranked, weights, body.top_k, body.lang, name, engine.ckpt_key)
        fname = f"report_{Path(name).stem}_{time.strftime('%Y%m%d_%H%M%S')}.zip"
        return Response(data, media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{fname}"'})

    # ------------------------------------------------------------------ UI
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
