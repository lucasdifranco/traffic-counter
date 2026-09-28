#!/usr/bin/env python3
"""Real-time frame processing of a YouTube video/livestream.

Resolves a direct media URL with yt-dlp, opens it with OpenCV (FFMPEG backend)
and runs a per-frame processing hook in real time.

Usage:
  python stream_process.py                 # default URL below
  python stream_process.py -u <youtube-url> -q 720
  python stream_process.py -u clip.mp4     # local file, no yt-dlp
  python stream_process.py --no-display --save out.mp4

In the window: Space pauses/resumes, q or Esc quits.
"""

from __future__ import annotations
from ultralytics import YOLO
from collections import defaultdict

import argparse
import json
import os
import signal
import sys
import time
from collections import deque

import cv2
import numpy as np

from crossing import LineCounter, load_lines

DEFAULT_URL = "https://www.youtube.com/watch?v=D2tqr5ekXO8"
LINE_CONFIG = "line.json"
TRACKER_CONFIG = "trackers/traffic.yaml"  # TrackTrack + lost_match/ReID for occlusion
EVENTS_PATH = "events.jsonl"
SUMMARY_PATH = "summary.json"
# crossing tolerances (pixels / frames); calibrate against the hand-count
TOLERANCES = {"band": 4.0, "min_run": 3, "min_disp": 8.0, "min_hits": 3, "evict_after": 15}

_running = True

model = YOLO("models/yolo26m.pt")  # load a pretrained model (recommended for training)
classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
CLASS_NAMES = [model.names[c] for c in classes]

def _stop(signum, frame):  # noqa: ARG001
    global _running
    _running = False


def resolve_stream_url(url: str, max_height: int = 720) -> tuple[str, dict]:
    """Return a direct, ffmpeg-readable media URL for a YouTube page URL."""
    from yt_dlp import YoutubeDL

    opts = {
        "quiet": True,
        "no_warnings": True,
        # Single stream only - OpenCV cannot mux a separate video+audio pair.
        # Video-only is preferred: we never need the audio track, and YouTube
        # often offers no progressive (muxed) format at all.
        "format": (
            f"bestvideo[height<={max_height}][vcodec^=avc1]/"
            f"bestvideo[height<={max_height}]/"
            f"best[height<={max_height}]/bestvideo/best"
        ),
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if "url" not in info:  # some extractors nest the chosen format
            info = info["requested_formats"][0]
    return info["url"], info


class FrameProcessor:
    """Put your computer-vision logic here.

    `process` is called once per frame and must return the frame to display.
    Keep it fast: anything slower than the source frame interval causes the
    reader to fall behind (frames are then dropped by --skip-late).
    """

    def __init__(self) -> None:
        self.frame_idx = 0
        if not os.path.exists(LINE_CONFIG):
            raise FileNotFoundError(
                f"{LINE_CONFIG} not found - run 'python pick_line.py' to create it"
            )
        self._raw_lines, self._size = load_lines(LINE_CONFIG)
        self._lines = []
        self._counters: list[LineCounter] = []
        self._ready = False
        open(EVENTS_PATH, "w").close()  # fresh event log per run

    def _build_counters(self, frame: np.ndarray) -> None:
        """Rescale the picked lines to the actual frame and create the counters."""
        h, w = frame.shape[:2]
        sx, sy = 1.0, 1.0
        if self._size:
            sx, sy = w / self._size[0], h / self._size[1]
        self._lines = [ln.scaled(sx, sy) for ln in self._raw_lines]
        self._counters = [LineCounter(ln, classes=CLASS_NAMES, **TOLERANCES) for ln in self._lines]
        self._ready = True

    def _draw_lines(self, frame: np.ndarray) -> None:
        for ln in self._lines:
            p1 = (round(ln.p1[0]), round(ln.p1[1]))
            p2 = (round(ln.p2[0]), round(ln.p2[1]))
            cv2.line(frame, p1, p2, (0, 0, 255), 3)

            # A and B name the two sides of the line (left/right when vertical,
            # top/bottom when horizontal), not the endpoints.
            nx, ny = ln.normal()
            mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            plus_name, minus_name = ln.side_names()
            for name, sign in ((plus_name, 1.0), (minus_name, -1.0)):
                lx = round(mx + sign * nx * 22)
                ly = round(my + sign * ny * 22)
                cv2.putText(frame, name, (lx - 6, ly + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # line number/name at the top end
            top = p1 if p1[1] <= p2[1] else p2
            cv2.putText(frame, ln.name, (top[0] + 8, top[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _draw_counts(self, frame: np.ndarray) -> None:
        """Per-class counting table in the top-right corner."""
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        pad, row_h = 6, 18
        h, w = frame.shape[:2]
        y = 10
        for counter in self._counters:
            pos, neg = counter.directions
            rows = [["class", pos, neg]]
            rows += [
                [name, str(v[pos]), str(v[neg])] for name, v in sorted(counter.counts.items())
            ]
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
            cv2.putText(
                frame, counter.line.name, (x0 + pad, y + pad + 12), font, scale, (0, 255, 255), thick
            )
            ty = y + pad + 12 + row_h
            for row in rows:
                tx = x0 + pad
                for i, cell in enumerate(row):
                    cv2.putText(frame, cell, (tx, ty), font, scale, (255, 255, 255), thick)
                    tx += widths[i] + pad
                ty += row_h
            y += table_h + 8

    @staticmethod
    def _log_event(event: dict) -> None:
        with open(EVENTS_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")

    def summary(self) -> dict:
        return {c.line.name: dict(c.counts) for c in self._counters}

    def write_summary(self, path: str = SUMMARY_PATH) -> dict:
        """Save the final totals, per class and per direction, to a JSON file."""
        data = {
            "by_class": self.summary(),
            "totals": {c.line.name: c.totals() for c in self._counters},
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        return data

    def process(self, frame: np.ndarray, t: float) -> np.ndarray:
        """`t` is seconds since the first frame.

        Returns the frame to display.
        """
        self.frame_idx += 1

        # use YOLO model to detect and track objects in the frame
        results = model.track(frame, classes=classes, persist=True, tracker=TRACKER_CONFIG, device="cuda", quantize=16, verbose=False)

        if not self._ready:
            self._build_counters(frame)
        self._draw_lines(frame)

        seen: set[int] = set()
        boxes = results[0].boxes

        # draw bounding boxes and labels; feed the anchor to the line counters
        if boxes.id is not None:
            xyxy = boxes.xyxy.cpu().tolist()
            track_ids = boxes.id.int().cpu().tolist()
            class_idxs = boxes.cls.int().cpu().tolist()
            confidences = boxes.conf.cpu().tolist()

            for box, track_id, class_idx, conf in zip(xyxy, track_ids, class_idxs, confidences):
                x1, y1, x2, y2 = map(int, box)
                seen.add(track_id)
                label = f"{model.names[class_idx]} {conf:.2f} ID:{track_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2  # anchor = bbox center
                for counter in self._counters:
                    event = counter.update(track_id, cx, cy, t, model.names[class_idx])
                    if event is not None:
                        self._log_event(event)

        for counter in self._counters:
            counter.evict_missing(seen)

        self._draw_counts(frame)
        return frame


def draw_hud(
    frame: np.ndarray,
    fps: float,
    proc_ms: float,
    dropped: int,
    idx: int,
) -> np.ndarray:
    text = f"{fps:5.1f} fps | {proc_ms:4.1f} ms/frame | frame {idx} | dropped {dropped}"
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (8, 8), (14 + tw, 20 + th + base), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (14, 14 + th),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    return frame


def open_capture(media_url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(media_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open the resolved stream URL")
    # Small buffer keeps us close to live for streams that honour it.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "-u",
        "--url",
        default=DEFAULT_URL,
        help="YouTube URL or local video file",
    )
    p.add_argument(
        "-q",
        "--quality",
        type=int,
        default=720,
        help="max height, e.g. 480/720/1080",
    )
    p.add_argument(
        "--no-display",
        action="store_true",
        help="headless (no cv2 window)",
    )
    p.add_argument(
        "--save",
        metavar="PATH",
        help="write processed frames to this mp4",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="stop after N frames (0 = unlimited)",
    )
    p.add_argument(
        "--skip-late",
        action="store_true",
        help="drop frames when processing falls behind (keeps output live)",
    )
    args = p.parse_args()

    signal.signal(signal.SIGINT, _stop)

    if os.path.exists(args.url):
        # Local file: skip yt-dlp entirely. Recording a clip once and iterating
        # against it is much faster than hitting the live stream every run.
        media_url, is_live = args.url, False
        print(f"[*] local file {args.url}", file=sys.stderr)
    else:
        print(f"[*] resolving {args.url} ...", file=sys.stderr)
        media_url, info = resolve_stream_url(args.url, args.quality)
        is_live = bool(info.get("is_live"))
        print(
            f"[*] {info.get('title', '?')} | {info.get('width')}x{info.get('height')}"
            f" | live={is_live}",
            file=sys.stderr,
        )

    cap = open_capture(media_url)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_interval = 1.0 / src_fps

    writer = None
    if args.save:
        writer = cv2.VideoWriter(
            args.save, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (width, height)
        )

    proc = FrameProcessor()
    shown: deque[float] = deque(maxlen=60)  # wall-clock time each frame was emitted
    costs: deque[float] = deque(maxlen=60)  # per-frame read+process cost, seconds
    dropped = 0
    idx = 0
    t0 = time.monotonic()
    paused = False
    pause_start = 0.0
    out = None  # last processed frame, kept while paused

    try:
        while _running:
            if paused:
                # frozen: keep the window alive and wait for space / quit
                if not args.no_display and out is not None:
                    cv2.imshow("stream", out)
                    key = cv2.waitKey(30) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord(" "):
                        paused = False
                        t0 += time.monotonic() - pause_start  # keep the schedule intact
                else:
                    time.sleep(0.05)
                continue

            loop_start = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                if is_live:
                    # transient gap on a livestream: reconnect
                    print("[!] stream stalled, reconnecting ...", file=sys.stderr)
                    cap.release()
                    media_url, _ = resolve_stream_url(args.url, args.quality)
                    cap = open_capture(media_url)
                    # Rebase the clock, or every frame after the gap looks late
                    # and --skip-late dumps a batch to "catch up".
                    t0 = time.monotonic() - idx * frame_interval
                    continue
                break

            idx += 1
            behind = (time.monotonic() - t0) - idx * frame_interval
            if args.skip_late and behind > frame_interval:
                dropped += 1
                continue

            out = proc.process(frame, loop_start - t0)

            costs.append(time.monotonic() - loop_start)
            shown.append(time.monotonic())
            # Throughput is the wall-clock spacing of emitted frames. Timing
            # read+process alone measures how fast we can drain the decode
            # buffer, which on a buffered HLS source reads in the hundreds.
            span = shown[-1] - shown[0]
            fps = (len(shown) - 1) / span if len(shown) > 1 and span > 0 else 0.0
            proc_ms = 1000.0 * sum(costs) / len(costs)
            out = draw_hud(out, fps, proc_ms, dropped, idx)

            if writer is not None:
                writer.write(out)
            if not args.no_display:
                cv2.imshow("stream", out)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    paused = True
                    pause_start = time.monotonic()

            if args.max_frames and idx >= args.max_frames:
                break

            # Pace against an absolute schedule, live sources included: OpenCV
            # hands us a buffered HLS segment as fast as we can decode it and
            # then blocks on the network for the next one. Sleeping a fixed
            # interval per iteration would also accumulate drift.
            lag = (t0 + idx * frame_interval) - time.monotonic()
            if lag > 0:
                time.sleep(lag)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                # OpenCV built without GUI support
                pass

    print(f"[*] counts: {proc.summary()}", file=sys.stderr)
    print(f"[*] summary saved to {SUMMARY_PATH}", file=sys.stderr)
    print(f"[*] processed {idx} frames, dropped {dropped}", file=sys.stderr)
    proc.write_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
