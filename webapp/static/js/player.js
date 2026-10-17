// Video player with a canvas overlay that draws tracked boxes in sync with playback,
// custom controls, and an appearance timeline (one lane per person).

const fmt = (sec) => {
  if (!isFinite(sec)) sec = 0;
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}`;
};

export class Player {
  constructor(els) {
    Object.assign(this, els);
    this.ctx = this.canvas.getContext("2d");
    this.info = null;        // {fps, frames, duration, width, height}
    this.analysis = null;    // {frames_processed, tracks:[...]}
    this.boxMaps = new Map();
    this.items = [];         // view model: [{id, color, label, sub, rank, dim, hidden, thumb}]
    this.selected = null;
    this.hoverId = null;
    this.onSelect = null;
    this._raf = null;
    this._lastFrame = -1;

    const v = this.video;
    v.addEventListener("play", () => { this.playBtn.classList.add("playing"); this._loop(); });
    v.addEventListener("pause", () => { this.playBtn.classList.remove("playing"); this.draw(); });
    v.addEventListener("ended", () => this.playBtn.classList.remove("playing"));
    ["seeked", "loadeddata", "timeupdate"].forEach((e) => v.addEventListener(e, () => this.draw()));
    v.addEventListener("loadedmetadata", () => this._fitAspect());

    this.playBtn.addEventListener("click", () => this.toggle());
    this.prevBtn.addEventListener("click", () => this.step(-1));
    this.nextBtn.addEventListener("click", () => this.step(1));
    this.speedSel.addEventListener("change", () => { v.playbackRate = Number(this.speedSel.value); });

    new ResizeObserver(() => this.draw(true)).observe(this.canvas);

    this.canvas.addEventListener("mousemove", (e) => this._hover(e));
    this.canvas.addEventListener("mouseleave", () => { this.hoverId = null; this.tooltip.hidden = true; this.draw(true); });
    this.canvas.addEventListener("click", (e) => {
      const hit = this._hit(e);
      if (hit != null) this.onSelect?.(hit, { seek: false });
      else this.toggle();
    });

    this._bindTimelineScrub();
  }

  // ------------------------------------------------------------------ data
  load(src, info) {
    this.info = info;
    this.analysis = null;
    this.boxMaps.clear();
    this.items = [];
    this.selected = null;
    this.video.src = src;
    this.video.load();
    this._fitAspect();
    this.renderTimeline();
    this.draw(true);
  }

  setAnalysis(analysis) {
    this.analysis = analysis;
    this.boxMaps.clear();
    for (const t of analysis?.tracks ?? []) {
      const m = new Map();
      for (const b of t.boxes) m.set(b[0], b);
      this.boxMaps.set(t.id, m);
    }
  }

  setView(items, selected) {
    this.items = items;
    this.selected = selected;
    this.renderTimeline();
    this.draw(true);
  }

  // ------------------------------------------------------------------ transport
  get fps() { return this.info?.fps || 25; }
  get frameCount() { return this.info?.frames || 0; }

  currentFrame() {
    return Math.max(0, Math.min(this.frameCount - 1, Math.floor(this.video.currentTime * this.fps + 1e-4)));
  }

  seekFrame(f) {
    f = Math.max(0, Math.min(this.frameCount - 1, f));
    // Land in the middle of the frame so floor() maps back to the same index
    this.video.currentTime = (f + 0.5) / this.fps;
  }

  step(d) {
    this.video.pause();
    this.seekFrame(this.currentFrame() + d);
  }

  toggle() {
    if (!this.video.src) return;
    if (this.video.paused) this.video.play().catch(() => {});
    else this.video.pause();
  }

  _loop() {
    const v = this.video;
    if (v.requestVideoFrameCallback) {
      const cb = () => { this.draw(); if (!v.paused && !v.ended) v.requestVideoFrameCallback(cb); };
      v.requestVideoFrameCallback(cb);
    } else {
      cancelAnimationFrame(this._raf);
      const cb = () => { this.draw(); if (!v.paused && !v.ended) this._raf = requestAnimationFrame(cb); };
      this._raf = requestAnimationFrame(cb);
    }
  }

  _fitAspect() {
    const w = this.video.videoWidth || this.info?.width;
    const h = this.video.videoHeight || this.info?.height;
    if (w && h) this.wrap.style.aspectRatio = `${w} / ${h}`;
  }

  // ------------------------------------------------------------------ overlay
  _layout() {
    const cw = this.canvas.clientWidth, ch = this.canvas.clientHeight;
    const vw = this.info?.width || this.video.videoWidth || 1;
    const vh = this.info?.height || this.video.videoHeight || 1;
    const s = Math.min(cw / vw, ch / vh);
    return { cw, ch, s, ox: (cw - vw * s) / 2, oy: (ch - vh * s) / 2 };
  }

  draw(force = false) {
    const f = this.currentFrame();
    this._updateHud(f);
    if (!force && f === this._lastFrame && !this.video.paused) return;
    this._lastFrame = f;

    const dpr = window.devicePixelRatio || 1;
    const { cw, ch, s, ox, oy } = this._layout();
    if (this.canvas.width !== Math.round(cw * dpr) || this.canvas.height !== Math.round(ch * dpr)) {
      this.canvas.width = Math.round(cw * dpr);
      this.canvas.height = Math.round(ch * dpr);
    }
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);
    if (!this.analysis) return;

    const processed = this.analysis.frames_processed;
    this.outOfRange.hidden = f < processed || !this.video.src;
    this._boxesNow = [];

    // Draw lowest rank first so #1 ends up on top
    const visible = this.items.filter((it) => !it.hidden);
    for (let i = visible.length - 1; i >= 0; i--) {
      const it = visible[i];
      const b = this.boxMaps.get(it.id)?.get(f);
      if (!b) continue;
      const x = ox + b[1] * s, y = oy + b[2] * s, w = (b[3] - b[1]) * s, h = (b[4] - b[2]) * s;
      this._boxesNow.push({ id: it.id, x, y, w, h });
      const isSel = this.selected === it.id;
      const isHover = this.hoverId === it.id;
      const faded = this.selected != null && !isSel;

      ctx.save();
      ctx.globalAlpha = it.dim ? 0.55 : faded ? 0.6 : 1;
      if (isSel) { ctx.shadowColor = it.color; ctx.shadowBlur = 14; }
      ctx.lineWidth = isSel ? 3.5 : isHover ? 3 : it.dim ? 1.5 : 2.2;
      ctx.strokeStyle = it.color;
      if (it.dim) ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.roundRect ? ctx.roundRect(x, y, w, h, 4) : ctx.rect(x, y, w, h);
      ctx.stroke();
      ctx.restore();

      // label pill
      ctx.save();
      ctx.globalAlpha = it.dim ? 0.7 : faded ? 0.75 : 1;
      ctx.font = `700 ${Math.max(10, Math.min(13, w / 6))}px ui-sans-serif, system-ui, sans-serif`;
      const text = it.sub ? `${it.label}  ${it.sub}` : it.label;
      const tw = ctx.measureText(text).width + 12;
      const th = 20;
      const ly = y - th - 3 < 0 ? y + 3 : y - th - 3;
      ctx.fillStyle = it.color;
      ctx.beginPath();
      ctx.roundRect ? ctx.roundRect(x, ly, tw, th, 6) : ctx.rect(x, ly, tw, th);
      ctx.fill();
      ctx.fillStyle = it.darkText ? "#0b0f19" : "#ffffff";
      ctx.textBaseline = "middle";
      ctx.fillText(text, x + 6, ly + th / 2 + 0.5);
      ctx.restore();
    }
  }

  _updateHud(f) {
    const dur = this.info?.duration || 0;
    this.timeText.textContent = `${fmt(this.video.currentTime || 0)} / ${fmt(dur)}`;
    this.frameText.textContent = this.frameCount ? `F ${f + 1} / ${this.frameCount}` : "";
    const p = this.frameCount ? Math.min(1, (this.video.currentTime * this.fps) / this.frameCount) : 0;
    this.tlPlayhead.style.left = `calc(var(--lbl) + (100% - var(--lbl)) * ${p})`;
  }

  _hit(e) {
    const r = this.canvas.getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    // _boxesNow is ordered back-to-front; search front-to-back
    const boxes = this._boxesNow || [];
    for (let i = boxes.length - 1; i >= 0; i--) {
      const b = boxes[i];
      if (px >= b.x && px <= b.x + b.w && py >= b.y && py <= b.y + b.h) return b.id;
    }
    return null;
  }

  _hover(e) {
    const id = this._hit(e);
    this.canvas.style.cursor = id != null ? "pointer" : "default";
    if (id !== this.hoverId) { this.hoverId = id; this.draw(true); }
    if (id == null) { this.tooltip.hidden = true; return; }
    const it = this.items.find((x) => x.id === id);
    if (!it) return;
    this.tooltip.innerHTML = `${it.thumb ? `<img src="${it.thumb}" alt="">` : ""}
      <div><b>${it.label}</b>${it.tip ? `<div class="tt-sub">${it.tip}</div>` : ""}</div>`;
    this.tooltip.hidden = false;
    const r = this.wrap.getBoundingClientRect();
    const tw = this.tooltip.offsetWidth, th = this.tooltip.offsetHeight;
    let x = e.clientX - r.left + 14, y = e.clientY - r.top + 14;
    if (x + tw > r.width - 8) x = e.clientX - r.left - tw - 14;
    if (y + th > r.height - 8) y = e.clientY - r.top - th - 14;
    this.tooltip.style.left = `${Math.max(8, x)}px`;
    this.tooltip.style.top = `${Math.max(8, y)}px`;
  }

  // ------------------------------------------------------------------ timeline
  renderTimeline() {
    const total = this.frameCount;
    const dur = this.info?.duration || 0;
    this.tlRuler.innerHTML = "";
    this.tlLanes.innerHTML = "";
    if (!total) return;

    const steps = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
    const step = steps.find((s) => dur / s <= 8) || 1200;
    for (let s = 0; s <= dur + 1e-6; s += step) {
      const el = document.createElement("span");
      el.style.left = `${(s / dur) * 100}%`;
      el.textContent = s >= 60 ? `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}` : `${+s.toFixed(1)}s`;
      this.tlRuler.appendChild(el);
    }

    const processed = this.analysis?.frames_processed ?? 0;
    const unPct = processed && processed < total ? ((total - processed) / total) * 100 : 0;
    const frag = document.createDocumentFragment();
    for (const it of this.items) {
      if (it.hidden && !it.keepLane) continue;
      const t = this.analysis?.tracks.find((x) => x.id === it.id);
      if (!t) continue;
      const lane = document.createElement("div");
      lane.className = "tl-lane" + (this.selected === it.id ? " sel" : "") + (it.dim || it.hidden ? " dim" : "");
      lane.style.setProperty("--c", it.color);
      lane.innerHTML = `<div class="tl-label"><i style="background:${it.color}"></i>${it.label}</div><div class="tl-track"></div>`;
      const track = lane.lastElementChild;
      for (const [a, b] of runs(t.boxes)) {
        const seg = document.createElement("div");
        seg.className = "tl-seg";
        seg.style.left = `${(a / total) * 100}%`;
        seg.style.width = `${((b - a + 1) / total) * 100}%`;
        seg.dataset.id = it.id;
        seg.dataset.start = a;
        track.appendChild(seg);
      }
      if (unPct) {
        const un = document.createElement("div");
        un.className = "tl-unanalyzed";
        un.style.width = `${unPct}%`;
        track.appendChild(un);
      }
      lane.querySelector(".tl-label").addEventListener("click", () => this.onSelect?.(it.id, { seek: true }));
      frag.appendChild(lane);
    }
    this.tlLanes.appendChild(frag);
  }

  _bindTimelineScrub() {
    let dragging = false;
    const frameAt = (clientX) => {
      const area = this.tlRuler.getBoundingClientRect();
      const p = Math.max(0, Math.min(1, (clientX - area.left) / area.width));
      return Math.floor(p * this.frameCount);
    };
    const onDown = (e) => {
      if (!this.frameCount || e.target.closest(".tl-label")) return;
      const seg = e.target.closest(".tl-seg");
      if (seg) {
        this.onSelect?.(Number(seg.dataset.id), { seek: false });
      }
      dragging = true;
      this.video.pause();
      this.seekFrame(frameAt(e.clientX));
      e.preventDefault();
    };
    this.tlBody.addEventListener("pointerdown", onDown);
    window.addEventListener("pointermove", (e) => { if (dragging) this.seekFrame(frameAt(e.clientX)); });
    window.addEventListener("pointerup", () => { dragging = false; });
  }
}

/** Contiguous frame runs [start, end] from a track's sorted box list. */
function runs(boxes) {
  const out = [];
  let a = null, prev = null;
  for (const b of boxes) {
    const f = b[0];
    if (a === null) { a = prev = f; continue; }
    if (f === prev + 1) { prev = f; continue; }
    out.push([a, prev]);
    a = prev = f;
  }
  if (a !== null) out.push([a, prev]);
  return out;
}
