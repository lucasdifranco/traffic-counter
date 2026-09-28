#!/usr/bin/env python3
"""Pick the counting line: click two points, press Enter to save the config.

Usage:
  python pick_line.py                          # frame 120 of samples/clip.mp4
  python pick_line.py samples/frame.png        # any still image
  python pick_line.py samples/clip.mp4 --frame 300 -o line.json

Keys: left-click sets each endpoint (a third click starts over),
      r resets, Enter saves, Esc/q quits without saving.
"""

from __future__ import annotations

import argparse
import json
import sys

import cv2

WIN = "pick line: click 2 points, Enter to save, r reset, Esc quit"
points: list[tuple[int, int]] = []
base = None
shown = None


def redraw() -> None:
    global shown
    shown = base.copy()
    for p in points:
        cv2.circle(shown, p, 6, (0, 255, 0), -1)
    if len(points) == 2:
        cv2.line(shown, points[0], points[1], (0, 0, 255), 2)
    cv2.putText(
        shown,
        f"points: {points}  |  Enter=save  r=reset  Esc=quit",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.imshow(WIN, shown)


def on_mouse(event, x, y, flags, param):  # noqa: ARG001
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if len(points) == 2:
        points.clear()
    points.append((x, y))
    redraw()


def load_frame(source: str, frame_idx: int):
    img = cv2.imread(source)
    if img is not None:
        return img
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"could not read '{source}'")
    return img


def main() -> int:
    global base
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("source", nargs="?", default="samples/clip.mp4")
    p.add_argument("--frame", type=int, default=120, help="frame index if source is a video")
    p.add_argument("-o", "--out", default="line.json", help="config path to write")
    args = p.parse_args()

    base = load_frame(args.source, args.frame)
    h, w = base.shape[:2]
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)
    redraw()

    saved = False
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10):  # Enter
            if len(points) == 2:
                cfg = {
                    "p1": list(points[0]),
                    "p2": list(points[1]),
                    "size": [w, h],  # rescale if the stream resolution differs
                }
                with open(args.out, "w", encoding="utf-8") as fh:
                    json.dump(cfg, fh, indent=2)
                print(f"saved {args.out}: {cfg}")
                saved = True
                break
        elif k in (27, ord("q")):  # Esc / q
            break
        elif k == ord("r"):
            points.clear()
            redraw()

    cv2.destroyAllWindows()
    if not saved:
        print("nothing saved", file=sys.stderr)
    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())
