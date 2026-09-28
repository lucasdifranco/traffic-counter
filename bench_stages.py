#!/usr/bin/env python3
"""Per-stage processing time of the pipeline, one row per stage.

Mirrors what stream_process.FrameProcessor does per frame, but times each step
separately: read, detect+track, count, draw overlay, HUD.

Usage:
  python bench_stages.py                       # 300 frames of samples/clip.mp4
  python bench_stages.py -u samples/clip.mp4 -n 600
  python bench_stages.py --device cpu
"""

from __future__ import annotations

import argparse
import statistics
import time

import cv2

from crossing import LineCounter, load_lines
from stream_process import CLASS_NAMES, TOLERANCES, TRACKER_CONFIG, classes, draw_hud, model


def draw_table(frame, counters):
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    pad, row_h = 6, 18
    h, w = frame.shape[:2]
    y = 10
    for counter in counters:
        pos, neg = counter.directions
        rows = [["class", pos, neg]]
        rows += [[n, str(v[pos]), str(v[neg])] for n, v in sorted(counter.counts.items())]
        total = counter.totals()
        rows.append(["TOTAL", str(total[pos]), str(total[neg])])
        widths = [0, 0, 0]
        for row in rows:
            for i, cell in enumerate(row):
                (tw, _), _ = cv2.getTextSize(cell, font, scale, thick)
                widths[i] = max(widths[i], tw)
        table_w = sum(widths) + pad * (len(widths) + 1)
        table_h = row_h * (len(rows) + 1) + pad * 2
        x0 = w - 10 - table_w
        cv2.rectangle(frame, (x0, y), (x0 + table_w, y + table_h), (0, 0, 0), -1)
        cv2.putText(frame, counter.line.name, (x0 + pad, y + pad + 12), font, scale, (0, 255, 255), thick)
        ty = y + pad + 12 + row_h
        for row in rows:
            tx = x0 + pad
            for i, cell in enumerate(row):
                cv2.putText(frame, cell, (tx, ty), font, scale, (255, 255, 255), thick)
                tx += widths[i] + pad
            ty += row_h
        y += table_h + 8


def stats(vals):
    s = sorted(vals)
    return statistics.mean(vals), statistics.median(vals), s[min(len(s) - 1, int(0.95 * len(s)))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-u", "--url", default="samples/clip.mp4")
    ap.add_argument("-n", "--frames", type=int, default=300)
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--tracker", default=TRACKER_CONFIG, help="tracker yaml to use")
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    lines, _ = load_lines("line.json")
    counters = [LineCounter(ln, classes=CLASS_NAMES, **TOLERANCES) for ln in lines]

    cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.url}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    stages = {k: [] for k in ("read", "track", "count", "draw", "hud")}
    frames_ms = []
    idx = 0
    xyxy, ids, cids = [], [], []
    while idx < args.frames + args.warmup:
        t_frame = time.perf_counter()

        t = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        t_read = time.perf_counter() - t
        idx += 1

        t = time.perf_counter()
        r = model.track(frame, classes=classes, persist=True, tracker=args.tracker,
                        device=args.device, quantize=16, verbose=False)[0]
        t_track = time.perf_counter() - t

        t = time.perf_counter()
        seen = []
        b = r.boxes
        xyxy, ids, cids = [], [], []
        if b.id is not None:
            xyxy = b.xyxy.cpu().tolist()
            ids = b.id.int().cpu().tolist()
            cids = b.cls.int().cpu().tolist()
            for (x1, y1, x2, y2), tid, cid in zip(xyxy, ids, cids):
                seen.append(tid)
                for c in counters:
                    c.update(tid, (x1 + x2) / 2, (y1 + y2) / 2, idx / src_fps, model.names[cid])
        for c in counters:
            c.evict_missing(set(seen))
        t_count = time.perf_counter() - t

        t = time.perf_counter()
        for ln in lines:
            cv2.line(frame, (round(ln.p1[0]), round(ln.p1[1])), (round(ln.p2[0]), round(ln.p2[1])), (0, 0, 255), 3)
        for (x1, y1, x2, y2), tid, cid in zip(xyxy, ids, cids):
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{model.names[cid]} ID:{tid}", (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        draw_table(frame, counters)
        t_draw = time.perf_counter() - t

        t = time.perf_counter()
        draw_hud(frame, 0.0, 0.0, 0, idx)
        t_hud = time.perf_counter() - t

        if idx > args.warmup:
            stages["read"].append(t_read)
            stages["track"].append(t_track)
            stages["count"].append(t_count)
            stages["draw"].append(t_draw)
            stages["hud"].append(t_hud)
            frames_ms.append(time.perf_counter() - t_frame)

    cap.release()
    if not frames_ms:
        raise SystemExit("no frames processed")

    budget = 1000.0 / src_fps
    print(f"source   {args.url}  {src_fps:.0f} fps  device={args.device}  tracker={args.tracker}  frames={len(frames_ms)}")
    print(f"budget   {budget:.1f} ms/frame for real time\n")
    print(f"{'stage':10s} {'mean':>8s} {'median':>8s} {'p95':>8s} {'% budget':>9s}")
    total = 0.0
    for name in ("read", "track", "count", "draw", "hud"):
        mean, median, p95 = stats(stages[name])
        total += mean
        print(f"{name:10s} {1000*mean:8.2f} {1000*median:8.2f} {1000*p95:8.2f} {100*1000*mean/budget:8.1f}%")
    mean, median, p95 = stats(frames_ms)
    print(f"{'TOTAL':10s} {1000*mean:8.2f} {1000*median:8.2f} {1000*p95:8.2f} {100*1000*mean/budget:8.1f}%")
    print(f"\nthroughput {len(frames_ms)/sum(frames_ms):.1f} fps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
