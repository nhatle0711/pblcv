import { t, getLang, setLang, onLangChange, applyI18n, EXAMPLE_CHIPS } from "./i18n.js";
import { get, post, postBlob, upload, pollJob, downloadBlob } from "./api.js";
import { Player } from "./player.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const SOFTMAX_TAU = 0.07;
const NEUTRAL = "#94a3b8";
const DARK_TEXT_RANKS = new Set([1, 2, 3, 7, 8, 9]);
const STAGE_ORDER = ["prepare", "detect", "crops", "encode"];

const store = {
  get(k, d) { try { const v = localStorage.getItem("ps." + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem("ps." + k, JSON.stringify(v)); } catch (e) {} },
};

const S = {
  status: null,
  palette: [],
  defaults: { joint: 0.5, app: 0.35, mot: 0.15 },
  video: null,
  analysis: null,
  search: null,
  ranked: [],
  weights: store.get("weights", null),
  topK: store.get("topK", 5),
  showAll: store.get("showAll", false),
  stride: store.get("stride", 1),
  selected: null,
  hidden: new Set(),
  jobs: { analyze: null, render: null },
  searchSeq: 0,
  videoThumb: "",
  stepState: {},
};

const player = new Player({
  video: $("video"), canvas: $("overlay"), wrap: $("player"), tooltip: $("tooltip"), outOfRange: $("out-of-range"),
  playBtn: $("play-btn"), prevBtn: $("prev-frame"), nextBtn: $("next-frame"), timeText: $("time-text"),
  frameText: $("frame-text"), speedSel: $("speed"),
  tlBody: $("tl-body"), tlRuler: $("tl-ruler"), tlLanes: $("tl-lanes"), tlPlayhead: $("tl-playhead"),
});
player.onSelect = (id, opts) => select(id, opts);

// ======================================================================== utils
function toast(kind, title, msg, ms = 4200) {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.innerHTML = `<div><b>${esc(title)}</b>${msg ? `<p>${esc(msg)}</p>` : ""}</div><button aria-label="close"><svg><use href="#i-x"/></svg></button>`;
  const close = () => { el.classList.add("out"); setTimeout(() => el.remove(), 200); };
  el.querySelector("button").onclick = close;
  $("toasts").appendChild(el);
  setTimeout(close, ms);
}
const fail = (e) => {
  if (e?.cancelled) return toast("warn", t("toast_cancelled"));
  toast("error", t("toast_error"), e?.status === 0 ? t("toast_offline") : e?.message, 7000);
};
const fmtSec = (s) => (s >= 60 ? `${Math.floor(s / 60)}m ${String(Math.round(s % 60)).padStart(2, "0")}s` : `${s.toFixed(s < 10 ? 1 : 0)}s`);
const fmtClock = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${(s % 60).toFixed(1).padStart(4, "0")}`;
const pct = (p) => `${(p * 100).toFixed(p >= 0.995 || p < 0.001 ? 0 : 1)}%`;
const trackById = (id) => S.analysis?.tracks.find((x) => x.id === id);

function setPressed(container, attr, value) {
  container.querySelectorAll(`[${attr}]`).forEach((b) => b.setAttribute("aria-pressed", String(b.getAttribute(attr) === String(value))));
}

// ======================================================================== ranking (mirrors SearchEngine.rank)
function normWeights() {
  const w = S.weights;
  const sum = Math.max(0, w.joint) + Math.max(0, w.app) + Math.max(0, w.mot) || 1;
  return { joint: Math.max(0, w.joint) / sum, app: Math.max(0, w.app) / sum, mot: Math.max(0, w.mot) / sum };
}

function rerank() {
  if (!S.search?.scores?.length) { S.ranked = []; refresh(); return; }
  const w = normWeights();
  const ranked = S.search.scores.map((s) => ({ ...s, fused: w.joint * s.joint + w.app * s.app + w.mot * s.mot }));
  ranked.sort((a, b) => b.fused - a.fused);
  const m = ranked[0].fused;
  const exps = ranked.map((s) => Math.exp((s.fused - m) / SOFTMAX_TAU));
  const z = exps.reduce((a, b) => a + b, 0);
  ranked.forEach((s, i) => { s.rank = i + 1; s.prob = exps[i] / z; });
  S.ranked = ranked;
  refresh();
}

// ======================================================================== status & checkpoints
async function loadStatus() {
  const st = await get("/api/status");
  S.status = st;
  S.palette = st.palette;
  S.defaults = st.default_weights;
  if (!S.weights) S.weights = { ...S.defaults };
  renderWeights();

  const sel = $("ckpt-select");
  sel.innerHTML = st.checkpoints.map((c) => `<option value="${esc(c.name)}">${esc(c.name)}</option>`).join("")
    || `<option value="">${esc("— no checkpoint —")}</option>`;
  sel.value = st.checkpoint?.name || "";
  renderCkptMetrics();
  const det = { yolo: "YOLOv8n", torchvision: "Faster R-CNN", opencv: "HOG" }[st.detector] || st.detector;
  $("sys-text").textContent = `${st.device.toUpperCase()} · ${det}`;
}

function renderCkptMetrics() {
  const c = S.status?.checkpoint;
  const m = c?.metrics || {};
  const parts = [];
  if (m["R@1"] != null) parts.push(`R@1 ${m["R@1"].toFixed(1)}%`);
  if (m.mAP != null) parts.push(`mAP ${m.mAP.toFixed(1)}%`);
  if (c?.sub_dataset) parts.push(c.sub_dataset.replace("TVPReid-", ""));
  $("ckpt-metrics").textContent = parts.join(" · ");
}

$("ckpt-select").addEventListener("change", async (e) => {
  const prev = S.status?.checkpoint?.name;
  try {
    const res = await post("/api/checkpoint", { name: e.target.value });
    S.status.checkpoint = res.checkpoint;
    renderCkptMetrics();
    toast("success", t("toast_ckpt"), res.checkpoint.name);
    if (S.search) runSearch();
  } catch (err) {
    e.target.value = prev || "";
    fail(err);
  }
});

// ======================================================================== step 1: video
async function loadSamples() {
  const list = await get("/api/samples").catch(() => []);
  $("samples-wrap").hidden = !list.length;
  $("samples").innerHTML = list.map((s) => `
    <button type="button" class="sample" data-name="${esc(s.name)}" title="${esc(s.name)} · ${s.size_mb} MB">
      <img src="${esc(s.thumb_url)}" alt="" loading="lazy"><span>${esc(s.name)}</span>
    </button>`).join("");
}

$("samples").addEventListener("click", async (e) => {
  const btn = e.target.closest(".sample");
  if (!btn) return;
  document.querySelectorAll(".sample").forEach((b) => b.classList.toggle("active", b === btn));
  try {
    S.videoThumb = btn.querySelector("img")?.src || "";
    setVideo(await post("/api/videos/sample", { name: btn.dataset.name }));
  } catch (err) { fail(err); }
});

const dz = $("dropzone");
["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
dz.addEventListener("drop", (e) => { const f = e.dataTransfer?.files?.[0]; if (f) uploadFile(f); });
$("file-input").addEventListener("change", (e) => { const f = e.target.files?.[0]; if (f) uploadFile(f); e.target.value = ""; });

async function uploadFile(file) {
  const limit = S.status?.max_upload_mb || 1024;
  if (file.size > limit * 1024 * 1024) return toast("error", t("toast_too_big", { mb: limit }));
  const prog = dz.querySelector(".dz-progress");
  const bar = prog.querySelector("i");
  prog.hidden = false;
  dz.querySelector(".dz-title").textContent = t("dz_uploading");
  dz.querySelector(".dz-sub").textContent = file.name;
  try {
    const v = await upload("/api/videos", file, (p) => { bar.style.width = `${p * 100}%`; prog.querySelector(".dz-pct").textContent = `${Math.round(p * 100)}%`; });
    document.querySelectorAll(".sample").forEach((b) => b.classList.remove("active"));
    S.videoThumb = "";
    toast("success", t("toast_uploaded"), file.name);
    setVideo(v);
  } catch (err) {
    fail(err);
  } finally {
    prog.hidden = true;
    bar.style.width = "0";
    applyI18n(dz);
  }
}

async function setVideo(v) {
  S.video = v;
  S.analysis = null;
  S.search = null;
  S.ranked = [];
  S.selected = null;
  S.hidden.clear();
  $("dsca").hidden = true;
  $("empty-state").hidden = true;
  $("player-card").hidden = false;
  $("no-person").hidden = true;
  $("results").hidden = true;
  renderVideoMeta();
  updateSteps();

  if (!v.ready) {
    const busy = $("player-busy");
    busy.hidden = false;
    $("player-busy-text").textContent = t("preparing");
    try {
      await pollJob(v.prepare_job, (j) => { $("player-busy-bar").style.width = `${j.pct}%`; });
      v = await get(`/api/videos/${v.video_id}`);
      if (S.video?.video_id !== v.video_id) return;
      S.video = v;
    } catch (err) {
      fail(err);
      return;
    } finally {
      busy.hidden = true;
    }
  }
  player.load(v.stream_url, v.info);
  player.setView([], null);
  if (!S.videoThumb) {
    // Uploads have no server thumbnail: grab the first decoded frame from the player
    player.video.addEventListener("loadeddata", () => {
      try {
        const c = document.createElement("canvas");
        c.width = 160;
        c.height = Math.round(160 * (v.info.height / v.info.width));
        c.getContext("2d").drawImage(player.video, 0, 0, c.width, c.height);
        S.videoThumb = c.toDataURL("image/jpeg", 0.7);
        renderVideoMeta();
      } catch (e) {}
    }, { once: true });
  }
  renderVideoMeta();
  updateFramesNote();
  updateSteps();
}

function renderVideoMeta() {
  const v = S.video;
  if (!v) return;
  $("vs-thumb").src = S.videoThumb || "";
  $("vs-name").textContent = v.name;
  const i = v.info;
  $("vs-sub").textContent = i ? `${fmtClock(i.duration)} · ${i.width}×${i.height} · ${+i.fps.toFixed(2)} fps` : t("preparing");
  $("vs-name").title = v.name;
}

// ======================================================================== step 2: analyse
function analyzeParams() {
  return { max_frames: Number($("max-frames").value), stride: S.stride, conf: Number($("conf").value) };
}

function updateFramesNote() {
  const note = $("frames-note");
  const info = S.video?.info;
  if (!info) { note.hidden = true; return; }
  const n = Number($("max-frames").value);
  note.hidden = false;
  const partial = n && n < info.frames;
  note.classList.toggle("warn", !!partial);
  note.textContent = partial
    ? t("frames_note_part", { n, total: info.frames, sec: fmtClock(n / info.fps), dur: fmtClock(info.duration) })
    : t("frames_note_all", { total: info.frames, dur: fmtClock(info.duration) });
}

$("stride-seg").addEventListener("click", (e) => {
  const b = e.target.closest("[data-stride]");
  if (!b) return;
  S.stride = Number(b.dataset.stride);
  store.set("stride", S.stride);
  setPressed($("stride-seg"), "data-stride", S.stride);
});
$("conf").addEventListener("input", (e) => { $("conf-val").textContent = Number(e.target.value).toFixed(2); });
$("max-frames").addEventListener("change", updateFramesNote);

$("analyze-btn").addEventListener("click", analyze);
$("analyze-cancel").addEventListener("click", () => { if (S.jobs.analyze) post(`/api/jobs/${S.jobs.analyze}/cancel`).catch(() => {}); });

async function analyze() {
  const v = S.video;
  if (!v?.ready || S.jobs.analyze) return;
  const btn = $("analyze-btn");
  btn.disabled = true;
  btn.classList.add("loading");
  const box = $("analyze-progress");
  try {
    const res = await post(`/api/videos/${v.video_id}/analyze`, analyzeParams());
    let aid = res.analysis_id;
    if (res.cached) {
      toast("info", t("toast_cached"));
    } else {
      S.jobs.analyze = res.job_id;
      box.hidden = false;
      setStage(null, 0);
      const out = await pollJob(res.job_id, (j) => {
        setStage(j.stage, j.pct);
        $("analyze-eta").textContent = j.eta_s != null ? t("eta", { s: fmtSec(j.eta_s) }) : "";
      });
      aid = out.analysis_id;
    }
    if (S.video?.video_id !== v.video_id) return;
    await loadAnalysis(aid, !res.cached);
    setCollapsed("step-analyze", true);
  } catch (err) {
    fail(err);
  } finally {
    S.jobs.analyze = null;
    box.hidden = true;
    btn.disabled = false;
    btn.classList.remove("loading");
    updateSteps();
  }
}

function setStage(stage, p) {
  const idx = STAGE_ORDER.indexOf(stage);
  $("stages").querySelectorAll("li").forEach((li, i) => {
    li.classList.toggle("done", idx > i);
    li.classList.toggle("active", idx === i);
  });
  $("analyze-bar").style.width = `${p}%`;
  $("analyze-pct").textContent = `${Math.round(p)}%`;
}

async function loadAnalysis(aid, fresh) {
  const a = await get(`/api/analyses/${aid}`);
  S.analysis = a;
  S.search = null;
  S.ranked = [];
  S.selected = null;
  S.hidden.clear();
  player.setAnalysis(a);
  $("no-person").hidden = a.tracks.length > 0;
  if (fresh) {
    const job = S.jobs.analyze ? await get(`/api/jobs/${S.jobs.analyze}`).catch(() => null) : null;
    toast("success", t("toast_analyzed"), t("toast_analyzed_sub", { n: a.tracks.length, sec: job ? job.elapsed_s : "–" }));
  }
  updateSteps();
  refresh();
  if (a.tracks.length && $("prompt").value.trim()) runSearch();
  else if (a.tracks.length) $("prompt").focus();
}

// ======================================================================== step 3: search
function renderExamples() {
  $("examples").innerHTML = Object.entries(EXAMPLE_CHIPS)
    .map(([k, label]) => `<button type="button" data-ex="${k}" title="${esc(t(k))}">${esc(label)}</button>`).join("");
}

function autoGrow() {
  const el = $("prompt");
  el.style.height = "auto";
  el.style.height = `${Math.min(el.scrollHeight + 2, 140)}px`;
}
$("examples").addEventListener("click", (e) => {
  const b = e.target.closest("[data-ex]");
  if (!b) return;
  $("prompt").value = t(b.dataset.ex);
  store.set("prompt", $("prompt").value);
  autoGrow();
  $("prompt").focus();
  if (S.analysis?.tracks.length) runSearch();
});
$("prompt").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("search-form").requestSubmit(); }
});
$("prompt").addEventListener("input", () => { store.set("prompt", $("prompt").value); autoGrow(); });
$("search-form").addEventListener("submit", (e) => { e.preventDefault(); runSearch(); });

async function runSearch() {
  const prompt = $("prompt").value.trim();
  if (!prompt || !S.analysis?.tracks.length) return;
  const seq = ++S.searchSeq;
  const btn = $("search-btn");
  btn.classList.add("loading");
  try {
    const res = await post(`/api/analyses/${S.analysis.analysis_id}/search`, { prompt });
    if (seq !== S.searchSeq) return;
    S.search = res;
    renderDsca();
    rerank();
    if (S.ranked[0] && S.selected == null) {
      // Jump to the best match the first time
      player.seekFrame(trackById(S.ranked[0].track_id)?.start ?? 0);
    }
  } catch (err) {
    fail(err);
  } finally {
    btn.classList.remove("loading");
  }
}

function renderDsca() {
  const s = S.search;
  $("dsca").hidden = !s;
  if (!s) return;
  $("q-app").innerHTML = markKeywords(s.q_app, s.app_keywords);
  $("q-mot").innerHTML = markKeywords(s.q_mot, s.mot_keywords);
}

/** Highlights each extracted keyword inside its sub-prompt (whole words, case-insensitive). */
function markKeywords(text, keywords) {
  let html = esc(text);
  const kws = [...new Set(keywords)].sort((a, b) => b.length - a.length);
  if (!kws.length) return html;
  const re = new RegExp(`\\b(${kws.map((k) => esc(k).replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})\\b`, "gi");
  return html.replace(re, "<mark>$1</mark>");
}

// ======================================================================== tuning
function renderWeights() {
  const w = S.weights;
  const n = normWeights();
  document.querySelectorAll("[data-w]").forEach((inp) => { inp.value = w[inp.dataset.w]; });
  document.querySelectorAll("[data-wv]").forEach((o) => { o.textContent = `${Math.round(n[o.dataset.wv] * 100)}%`; });
  document.querySelectorAll("[data-wb]").forEach((i) => { i.style.width = `${n[i.dataset.wb] * 100}%`; });
  $("wtext").textContent = ["joint", "app", "mot"].map((k) => Math.round(n[k] * 100)).join("/");
  setPressed($("topk-seg"), "data-k", S.topK);
  $("show-all").checked = S.showAll;
}

// ---- weights popover
const pop = $("weights-pop");
const popBtn = $("weights-btn");
function setPopover(open) {
  pop.hidden = !open;
  popBtn.setAttribute("aria-expanded", String(open));
  if (open) pop.querySelector("input")?.focus({ preventScroll: true });
}
popBtn.addEventListener("click", () => setPopover(pop.hidden));
$("weights-close").addEventListener("click", () => { setPopover(false); popBtn.focus(); });
document.addEventListener("pointerdown", (e) => {
  if (!pop.hidden && !pop.contains(e.target) && !popBtn.contains(e.target)) setPopover(false);
});
document.querySelectorAll("[data-w]").forEach((inp) => inp.addEventListener("input", () => {
  S.weights[inp.dataset.w] = Number(inp.value);
  store.set("weights", S.weights);
  renderWeights();
  rerank();
}));
$("topk-seg").addEventListener("click", (e) => {
  const b = e.target.closest("[data-k]");
  if (!b) return;
  S.topK = Number(b.dataset.k);
  store.set("topK", S.topK);
  renderWeights();
  refresh();
});
$("show-all").addEventListener("change", (e) => { S.showAll = e.target.checked; store.set("showAll", S.showAll); refresh(); });
$("weights-reset").addEventListener("click", () => {
  S.weights = { ...S.defaults };
  store.set("weights", S.weights);
  renderWeights();
  rerank();
});

// ======================================================================== selection
function select(id, { seek = true } = {}) {
  S.selected = S.selected === id ? null : id;
  if (S.selected != null && seek) {
    const tr = trackById(id);
    if (tr) { player.video.pause(); player.seekFrame(tr.start); }
  }
  refresh();
  if (S.selected != null) {
    document.querySelector(`.cand[data-id="${id}"]`)?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
}

// ======================================================================== view model
function viewItems() {
  const a = S.analysis;
  if (!a) return [];
  if (!S.ranked.length) {
    return a.tracks.map((tr) => ({
      id: tr.id, color: S.palette[(tr.id - 1) % S.palette.length] || NEUTRAL, darkText: true,
      label: `P${tr.id}`, sub: "", rank: null, dim: false, hidden: S.hidden.has(tr.id), keepLane: true,
      thumb: tr.thumb, tip: `${fmtClock(tr.start / a.fps)} → ${fmtClock(tr.end / a.fps)}`,
    }));
  }
  return S.ranked.map((s) => {
    const inTop = s.rank <= S.topK;
    const tr = trackById(s.track_id);
    return {
      id: s.track_id,
      rank: s.rank,
      color: inTop ? S.palette[s.rank - 1] : NEUTRAL,
      darkText: inTop ? DARK_TEXT_RANKS.has(s.rank) : false,
      label: `#${s.rank} P${s.track_id}`,
      sub: inTop ? pct(s.prob) : "",
      dim: !inTop,
      hidden: S.hidden.has(s.track_id) || (!inTop && !S.showAll),
      keepLane: inTop,
      thumb: tr?.thumb,
      tip: `${t("score")} ${s.fused.toFixed(3)} · ${pct(s.prob)}`,
    };
  });
}

function refresh() {
  player.setView(viewItems(), S.selected);
  renderResults();
}

// ======================================================================== results
function renderResults() {
  const a = S.analysis;
  const box = $("results");
  if (!a || !a.tracks.length) { box.hidden = true; return; }
  box.hidden = false;
  const items = viewItems();
  const searched = S.ranked.length > 0;
  $("render-btn").disabled = !searched || !!S.jobs.render;
  $("report-btn").disabled = !searched;
  $("rank-toolbar").classList.toggle("disabled", !searched);
  if (!searched) setPopover(false);

  // ---- summary
  const sum = $("summary");
  if (searched) {
    const top = S.ranked[0];
    const tr = trackById(top.track_id);
    sum.style.setProperty("--c1", S.palette[0]);
    sum.innerHTML = `
      ${tr?.thumb ? `<img src="${tr.thumb}" alt="">` : ""}
      <div>
        <div class="sum-kicker">${esc(t("best_match"))}</div>
        <div class="sum-title">${esc(t("person"))} ${top.track_id}</div>
        <div class="sum-sub"><q>${esc(S.search.prompt)}</q></div>
      </div>
      <div class="sum-stats">
        <div class="stat"><b>${pct(top.prob)}</b><span>${esc(t("match_prob"))}</span></div>
        <div class="stat"><b>${a.tracks.length}</b><span>${esc(t("persons_found"))}</span></div>
        <div class="stat"><b>${S.search.elapsed_ms}<small style="font-size:12px">ms</small></b><span>${esc(t("query_ms"))}</span></div>
      </div>`;
  } else {
    sum.style.setProperty("--c1", NEUTRAL);
    sum.innerHTML = `<div class="sum-sub" style="grid-column:1/-1">${esc(t("no_search_yet", { n: a.tracks.length }))}</div>`;
  }

  // ---- candidates (FLIP animation on re-rank)
  const list = $("cands");
  const before = new Map([...list.children].map((el) => [el.dataset.id, el.getBoundingClientRect()]));
  const shown = searched ? items.filter((it) => it.rank <= S.topK || S.showAll) : items;
  $("cand-count").textContent = searched ? `${Math.min(S.topK, S.ranked.length)} / ${S.ranked.length}` : `${items.length}`;

  const ranges = {};
  if (searched) for (const k of ["joint", "app", "mot"]) {
    const vals = S.ranked.map((s) => s[k]);
    ranges[k] = [Math.min(...vals), Math.max(...vals)];
  }
  const bar = (k, v) => {
    const [lo, hi] = ranges[k];
    const w = hi > lo ? 12 + 88 * (v - lo) / (hi - lo) : 100;
    return `<div class="sb" style="--sb:var(--c-${k})"><span>${k}</span><div class="bar"><i style="width:${w}%"></i></div><span class="mono">${v.toFixed(3)}</span></div>`;
  };

  list.innerHTML = shown.map((it) => {
    const tr = trackById(it.id);
    const s = searched ? S.ranked.find((x) => x.track_id === it.id) : null;
    const off = S.hidden.has(it.id);
    const cls = ["cand", S.selected === it.id ? "sel" : "", off ? "off" : "", it.dim ? "outside" : ""].join(" ");
    const time = `${fmtClock(tr.start / a.fps)} → ${fmtClock(tr.end / a.fps)}`;
    return `
    <article class="${cls}" data-id="${it.id}" style="--c:${it.color};--on-c:${it.darkText ? "#0b0f19" : "#fff"}" title="${esc(t("rank_hint"))}">
      <div class="thumb">
        <img src="${tr.thumb}" alt="" loading="lazy">
        <span class="rank-badge">${s ? `#${s.rank}` : `P${it.id}`}</span>
      </div>
      <div class="cand-body">
        <div class="cand-top">
          <h3>${esc(t("person"))} ${it.id}</h3>
          ${s ? `<span class="prob">${pct(s.prob)}<small>${esc(t("match_prob"))}</small></span>` : ""}
        </div>
        <div class="cand-meta">
          ${s ? `<span>${esc(t("score"))} <span class="mono">${s.fused.toFixed(4)}</span></span>` : ""}
          <span>${esc(t("appears"))} <span class="mono">${time}</span></span>
        </div>
        ${s ? `<div class="scorebars">${bar("joint", s.joint)}${bar("app", s.app)}${bar("mot", s.mot)}</div>` : ""}
        <div class="film">${tr.film.map((f) => `<img src="${f}" alt="" loading="lazy">`).join("")}</div>
      </div>
      <button type="button" class="vis" data-vis="${it.id}" title="${esc(t("show_hide"))}"><svg><use href="#${off ? "i-eye-off" : "i-eye"}"/></svg></button>
    </article>`;
  }).join("");

  for (const el of list.children) {
    const old = before.get(el.dataset.id);
    if (!old) continue;
    const now = el.getBoundingClientRect();
    const dx = old.left - now.left, dy = old.top - now.top;
    if (!dx && !dy) continue;
    el.style.transform = `translate(${dx}px, ${dy}px)`;
    el.style.transition = "none";
    requestAnimationFrame(() => {
      el.style.transition = "transform .35s cubic-bezier(.2,.8,.2,1)";
      el.style.transform = "";
    });
  }
}

$("cands").addEventListener("click", (e) => {
  const vis = e.target.closest("[data-vis]");
  if (vis) {
    const id = Number(vis.dataset.vis);
    S.hidden.has(id) ? S.hidden.delete(id) : S.hidden.add(id);
    refresh();
    return;
  }
  const card = e.target.closest(".cand");
  if (card) select(Number(card.dataset.id));
});

// ======================================================================== outputs
function rankBody() {
  return { prompt: S.search.prompt, weights: normWeights(), top_k: S.topK, lang: getLang() };
}

$("render-btn").addEventListener("click", async () => {
  if (!S.search || S.jobs.render) return;
  const btn = $("render-btn");
  const prog = btn.querySelector(".btn-progress");
  btn.disabled = true;
  btn.classList.add("loading");
  prog.hidden = false;
  try {
    const { job_id } = await post(`/api/analyses/${S.analysis.analysis_id}/render`, rankBody());
    S.jobs.render = job_id;
    const out = await pollJob(job_id, (j) => { prog.firstElementChild.style.width = `${j.pct}%`; });
    toast("success", t("toast_render_done"));
    const a = Object.assign(document.createElement("a"), { href: `${out.url}?download=${encodeURIComponent(out.filename)}`, download: out.filename });
    document.body.appendChild(a);
    a.click();
    a.remove();
  } catch (err) {
    fail(err);
  } finally {
    S.jobs.render = null;
    btn.classList.remove("loading");
    prog.hidden = true;
    prog.firstElementChild.style.width = "0";
    btn.disabled = false;
  }
});

$("report-btn").addEventListener("click", async () => {
  if (!S.search) return;
  const btn = $("report-btn");
  btn.disabled = true;
  btn.classList.add("loading");
  try {
    const { blob, filename } = await postBlob(`/api/analyses/${S.analysis.analysis_id}/report`, rankBody());
    downloadBlob(blob, filename);
    toast("success", t("toast_report_done"), filename);
  } catch (err) {
    fail(err);
  } finally {
    btn.disabled = false;
    btn.classList.remove("loading");
  }
});

// ======================================================================== steps, theme, language, keys
function setCollapsed(id, collapsed) {
  const el = $(id);
  el.dataset.collapsed = String(collapsed);
  el.querySelector(".step-toggle")?.setAttribute("aria-expanded", String(!collapsed));
}

// Expand / collapse via the summary row, the "Change" button and the chevron
document.querySelectorAll("[data-expand]").forEach((b) => b.addEventListener("click", () => setCollapsed(b.dataset.expand, false)));
document.querySelectorAll("[data-toggle]").forEach((b) => b.addEventListener("click", () => {
  const id = b.dataset.toggle;
  setCollapsed(id, $(id).dataset.collapsed !== "true");
}));

function updateSteps() {
  const ready = !!S.video?.ready;
  const analysed = !!S.analysis;
  const hasTracks = !!S.analysis?.tracks.length;
  const states = {
    "step-video": S.video ? "done" : "active",
    "step-analyze": !ready ? "locked" : analysed ? "done" : "active",
    "step-search": hasTracks ? "active" : "locked",
  };
  for (const [id, state] of Object.entries(states)) {
    $(id).dataset.state = state;
    // Auto-collapse on entering "done", auto-expand otherwise; manual toggles stick until the next transition
    if (S.stepState[id] !== state && id !== "step-search") setCollapsed(id, state === "done");
    S.stepState[id] = state;
  }
  if (S.jobs.analyze) setCollapsed("step-analyze", false);

  if (analysed) {
    const p = S.analysis.params;
    $("as-main").textContent = t("as_main", {
      mode: t(`speed_${p.stride}`), n: S.analysis.frames_processed, conf: Number(p.conf).toFixed(2),
    });
    $("as-sub").textContent = t("as_sub", { persons: S.analysis.tracks.length });
  }
  $("analyze-btn").disabled = !ready || !!S.jobs.analyze;
  $("analyze-btn").querySelector("span").textContent = t(analysed ? "reanalyze" : "analyze");
  $("search-btn").disabled = !hasTracks;
}

$("theme-btn").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("ps.theme", root.dataset.theme); } catch (e) {}
});
document.querySelectorAll("[data-lang]").forEach((b) => b.addEventListener("click", () => setLang(b.dataset.lang)));
onLangChange(() => {
  renderExamples();
  renderVideoMeta();
  updateFramesNote();
  updateSteps();
  renderDsca();
  refresh();
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !pop.hidden) { setPopover(false); popBtn.focus(); return; }
  if (e.target.closest("input, textarea, select") || e.ctrlKey || e.metaKey || e.altKey) return;
  if (!S.video?.ready) return;
  if (e.code === "Space") { e.preventDefault(); player.toggle(); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); player.step(-1); }
  else if (e.key === "ArrowRight") { e.preventDefault(); player.step(1); }
  else if (/^[1-9]$/.test(e.key) && S.ranked.length) {
    const s = S.ranked[Number(e.key) - 1];
    if (s) select(s.track_id);
  }
});

// ======================================================================== boot
(async function init() {
  applyI18n();
  renderExamples();
  setPressed($("stride-seg"), "data-stride", S.stride);
  $("prompt").value = store.get("prompt", "");
  autoGrow();
  try {
    await loadStatus();
  } catch (err) {
    fail(err);
  }
  renderWeights();
  updateSteps();
  loadSamples();
})();
