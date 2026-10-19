"""Builds a downloadable ZIP report (self-contained HTML + JSON + CSV + crops) for one search."""

import csv
import html
import io
import json
import time
import zipfile
from typing import Any, Dict, List

from .engine import ATTR_NAMES, Analysis, _data_url, rank_color

TEXT = {
    "vi": {
        "title": "Báo cáo tìm kiếm người trong video",
        "query": "Câu mô tả",
        "q_app": "Q_app (ngoại hình)",
        "q_mot": "Q_mot (chuyển động)",
        "video": "Video",
        "model": "Checkpoint",
        "weights": "Trọng số",
        "processed": "Số frame đã phân tích",
        "persons": "Số người phát hiện",
        "created": "Thời gian tạo",
        "rank": "Hạng",
        "person": "Người",
        "fused": "Điểm tổng hợp",
        "prob": "Xác suất khớp",
        "time": "Xuất hiện",
        "frames": "Khung hình",
        "note": "Điểm là độ tương đồng cosine giữa văn bản và video. Xác suất khớp = softmax(điểm / 0.07) trên các người trong video.",
    },
    "en": {
        "title": "Video person search report",
        "query": "Query",
        "q_app": "Q_app (appearance)",
        "q_mot": "Q_mot (motion)",
        "video": "Video",
        "model": "Checkpoint",
        "weights": "Weights",
        "processed": "Frames analysed",
        "persons": "Persons detected",
        "created": "Created",
        "rank": "Rank",
        "person": "Person",
        "fused": "Fused score",
        "prob": "Match probability",
        "time": "Appears",
        "frames": "Frames",
        "note": "Scores are text-video cosine similarities. Match probability = softmax(score / 0.07) over the persons in this video.",
    },
}


def _fmt_time(sec: float) -> str:
    return f"{int(sec // 60):02d}:{sec % 60:05.2f}"


def build_report(analysis: Analysis, search: Dict[str, Any], ranked: List[Dict[str, Any]],
                 weights: Dict[str, float], top_k: int, lang: str, video_name: str, ckpt: str) -> bytes:
    tx = TEXT.get(lang, TEXT["en"])
    top = ranked[:top_k]
    fps = analysis.fps
    rows = []
    for s in top:
        t = analysis.tracks[s["track_id"]]
        rows.append({
            "rank": s["rank"],
            "person_id": s["track_id"],
            "fused": round(s["fused"], 5),
            "prob": round(s["prob"], 5),
            "joint": s["joint"],
            "app": s["app"],
            "mot": s["mot"],
            "clip": s.get("clip"),
            "clip_raw": s.get("clip_raw"),
            **{f"{k}_prob": round(t.attrs[k], 3) if k in t.attrs else None for k in ATTR_NAMES},
            "top_color": max(t.attrs["top_color"], key=t.attrs["top_color"].get) if t.attrs.get("top_color") else None,
            "bottom_color": max(t.attrs["bottom_color"], key=t.attrs["bottom_color"].get) if t.attrs.get("bottom_color") else None,
            "attribute_penalty": s.get("penalty", 0.0),
            "start_frame": t.start,
            "end_frame": t.end,
            "start_s": round(t.start / fps, 3),
            "end_s": round(t.end / fps, 3),
        })

    meta = {
        "query": search["prompt"],
        "q_app": search["q_app"],
        "q_mot": search["q_mot"],
        "video": video_name,
        "checkpoint": ckpt,
        "weights": weights,
        "top_k": top_k,
        "frames_processed": analysis.frames_processed,
        "frames_total": analysis.frames_total,
        "fps": fps,
        "persons_detected": len(analysis.tracks),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("results.json", json.dumps({"meta": meta, "results": rows}, ensure_ascii=False, indent=2))

        csv_buf = io.StringIO()
        if rows:
            wr = csv.DictWriter(csv_buf, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        zf.writestr("results.csv", "﻿" + csv_buf.getvalue())

        for r in rows:
            t = analysis.tracks[r["person_id"]]
            prefix = f"crops/rank{r['rank']:02d}_person{r['person_id']}"
            if t.thumb:
                zf.writestr(f"{prefix}_main.jpg", t.thumb)
            for i, j in enumerate(t.film):
                zf.writestr(f"{prefix}_{i + 1}.jpg", j)

        zf.writestr("report.html", _html(analysis, meta, rows, tx, lang))
    return buf.getvalue()


def _html(analysis: Analysis, meta: Dict[str, Any], rows: List[Dict[str, Any]], tx: Dict[str, str], lang: str) -> str:
    e = html.escape
    w = meta["weights"]
    cards = []
    for r in rows:
        t = analysis.tracks[r["person_id"]]
        color = rank_color(r["rank"])[1]
        film = "".join(f'<img src="{_data_url(j)}" alt="">' for j in t.film)
        main = f'<img class="main" src="{_data_url(t.thumb)}" alt="">' if t.thumb else ""
        cards.append(f"""
<article class="card" style="--c:{color}">
  <div class="badge">#{r['rank']}</div>
  {main}
  <div class="body">
    <h3>{tx['person']} {r['person_id']}</h3>
    <p class="score"><b>{r['fused']:.4f}</b> · {tx['prob']}: <b>{r['prob'] * 100:.1f}%</b></p>
    <p class="sub">joint {r['joint']:.4f} · app {r['app']:.4f} · mot {r['mot']:.4f}{f" · clip {r['clip']:.4f}" if r.get('clip') is not None else ""}</p>
    <p class="sub">{tx['time']}: {_fmt_time(r['start_s'])} → {_fmt_time(r['end_s'])} · {tx['frames']} {r['start_frame']}–{r['end_frame']}</p>
    <div class="film">{film}</div>
  </div>
</article>""")

    return f"""<!doctype html>
<html lang="{lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(tx['title'])}</title>
<style>
:root{{--bg:#f6f7fb;--fg:#0f172a;--mut:#64748b;--card:#fff;--line:#e2e8f0}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0b1020;--fg:#e2e8f0;--mut:#94a3b8;--card:#131a2e;--line:#243049}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:32px 16px}}
h1{{margin:0 0 4px;font-size:26px}}
.meta{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:8px 24px;background:var(--card);
border:1px solid var(--line);border-radius:14px;padding:16px 20px;margin:20px 0}}
.meta div span{{display:block;color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.04em}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}}
.card{{position:relative;display:flex;gap:14px;background:var(--card);border:1px solid var(--line);border-left:5px solid var(--c);
border-radius:14px;padding:14px}}
.card .main{{width:92px;height:150px;object-fit:cover;border-radius:10px;background:#0002}}
.badge{{position:absolute;top:10px;right:12px;background:var(--c);color:#000;font-weight:700;border-radius:999px;padding:2px 10px}}
.body h3{{margin:0 0 4px;font-size:17px}} .body p{{margin:2px 0}} .sub{{color:var(--mut);font-size:13px}}
.film{{display:flex;gap:4px;margin-top:8px}} .film img{{width:40px;height:64px;object-fit:cover;border-radius:6px}}
.note{{color:var(--mut);font-size:13px;margin-top:20px}}
</style></head><body><main>
<h1>{e(tx['title'])}</h1>
<div class="meta">
  <div><span>{tx['query']}</span>{e(meta['query'])}</div>
  <div><span>{tx['q_app']}</span>{e(meta['q_app'])}</div>
  <div><span>{tx['q_mot']}</span>{e(meta['q_mot'])}</div>
  <div><span>{tx['video']}</span>{e(meta['video'])}</div>
  <div><span>{tx['model']}</span>{e(str(meta['checkpoint']))}</div>
  <div><span>{tx['weights']}</span>{" / ".join(f"{k} {v:.2f}" for k, v in w.items())}</div>
  <div><span>{tx['processed']}</span>{meta['frames_processed']} / {meta['frames_total']}</div>
  <div><span>{tx['persons']}</span>{meta['persons_detected']}</div>
  <div><span>{tx['created']}</span>{meta['created']}</div>
</div>
<section class="grid">{''.join(cards)}</section>
<p class="note">{e(tx['note'])}</p>
</main></body></html>"""
