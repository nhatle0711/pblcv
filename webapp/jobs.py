"""Background job runner with stage-aware progress, ETA and cooperative cancellation."""

import threading
import time
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional, Tuple


class JobCancelled(Exception):
    pass


class Job:
    def __init__(self, kind: str, stages: Dict[str, Tuple[float, float]]):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.stages = stages            # stage -> (start_pct, end_pct) of the overall bar
        self.state = "queued"           # queued | running | done | error | cancelled
        self.stage = ""
        self.current = 0
        self.total = 0
        self.pct = 0.0
        self.error: Optional[str] = None
        self.result: Any = None
        self.created = time.time()
        self.started: Optional[float] = None
        self.finished: Optional[float] = None
        self._cancel = threading.Event()

    def progress(self, stage: str, current: float, total: float) -> None:
        """Called from the worker. Raises JobCancelled when the user asked to stop."""
        if self._cancel.is_set():
            raise JobCancelled()
        self.stage = stage
        self.current = int(current)
        self.total = int(total)
        lo, hi = self.stages.get(stage, (0.0, 100.0))
        frac = (current / total) if total else 0.0
        self.pct = max(self.pct, lo + (hi - lo) * min(1.0, max(0.0, frac)))

    def to_dict(self) -> Dict[str, Any]:
        elapsed = (self.finished or time.time()) - (self.started or time.time())
        eta = None
        if self.state == "running" and 3.0 < self.pct < 100.0:
            eta = round(elapsed * (100.0 - self.pct) / self.pct, 1)
        return {
            "id": self.id,
            "kind": self.kind,
            "state": self.state,
            "stage": self.stage,
            "current": self.current,
            "total": self.total,
            "pct": round(self.pct, 1),
            "elapsed_s": round(elapsed, 1) if self.started else 0.0,
            "eta_s": eta,
            "error": self.error,
            "result": self.result if self.state == "done" else None,
        }


class JobManager:
    """Runs one job at a time (the model and detector are shared) and keeps recent job states."""

    def __init__(self, keep: int = 50):
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="job")
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()
        self._keep = keep
        self._lock = threading.Lock()

    def submit(self, kind: str, stages: Dict[str, Tuple[float, float]], fn: Callable[[Job], Any]) -> Job:
        job = Job(kind, stages)
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > self._keep:
                self._jobs.popitem(last=False)
        self._pool.submit(self._run, job, fn)
        return job

    def _run(self, job: Job, fn: Callable[[Job], Any]) -> None:
        if job._cancel.is_set():
            job.state = "cancelled"
            return
        job.state = "running"
        job.started = time.time()
        try:
            job.result = fn(job)
            job.pct = 100.0
            job.state = "done"
        except JobCancelled:
            job.state = "cancelled"
        except Exception as e:
            traceback.print_exc()
            job.error = str(e) or e.__class__.__name__
            job.state = "error"
        finally:
            job.finished = time.time()

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None or job.state in ("done", "error", "cancelled"):
            return False
        job._cancel.set()
        return True

    def busy(self) -> bool:
        return any(j.state in ("queued", "running") for j in self._jobs.values())
