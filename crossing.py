#!/usr/bin/env python3
"""Pure line-crossing counting: no OpenCV, no model, unit-testable.

Direction convention:
  side = sign( cross(B-A, P-A) ), with A = p1 and B = p2. A and B label the
  two *sides* of the line, not its endpoints: A is the left side of a vertical
  line and the top side of a horizontal one, so "A->B" reads left-to-right /
  top-to-bottom on screen. Which side is '+' follows the sign of the cross
  product; swapping p1 and p2 swaps the A/B sides and the directions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Line:
    name: str
    p1: tuple[float, float]
    p2: tuple[float, float]

    def geometry(self, x: float, y: float) -> tuple[float, float]:
        """Return (proj, dist): proj is 0 at A and 1 at B, dist is the signed
        perpendicular distance in pixels."""
        (ax, ay), (bx, by) = self.p1, self.p2
        dx, dy = bx - ax, by - ay
        length2 = dx * dx + dy * dy
        if length2 == 0:
            raise ValueError(f"line {self.name!r} is degenerate (p1 == p2)")
        proj = ((x - ax) * dx + (y - ay) * dy) / length2
        dist = (dx * (y - ay) - dy * (x - ax)) / math.sqrt(length2)
        return proj, dist

    def orientation(self) -> str:
        """'horizontal' if the line is mostly left-right, else 'vertical'."""
        dx = abs(self.p2[0] - self.p1[0])
        dy = abs(self.p2[1] - self.p1[1])
        return "horizontal" if dx >= dy else "vertical"

    def normal(self) -> tuple[float, float]:
        """Unit normal pointing to the '+' side (left of A->B)."""
        dx = self.p2[0] - self.p1[0]
        dy = self.p2[1] - self.p1[1]
        n = math.hypot(dx, dy)
        return (-dy / n, dx / n)

    def side_names(self) -> tuple[str, str]:
        """Labels for the '+' and the '-' side, in that order.

        A is the left side of a vertical line and the top side of a horizontal
        one, so `A->B` reads left-to-right / top-to-bottom on screen; B is the
        opposite side. A and B name sides, not endpoints.
        """
        nx, ny = self.normal()
        plus_is_a = (nx < 0) if self.orientation() == "vertical" else (ny < 0)
        return ("A", "B") if plus_is_a else ("B", "A")

    def scaled(self, sx: float, sy: float) -> "Line":
        return Line(
            self.name,
            (self.p1[0] * sx, self.p1[1] * sy),
            (self.p2[0] * sx, self.p2[1] * sy),
        )


def load_lines(path: str) -> tuple[list[Line], list[int] | None]:
    """Read the config. Accepts both the single-line form and the list form:

    {"p1": [x, y], "p2": [x, y], "size": [w, h]}
    {"size": [w, h], "lines": [{"name": "via1", "p1": [x, y], "p2": [x, y]}, ...]}
    """
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    size = cfg.get("size")
    if "lines" in cfg:
        raw = cfg["lines"]
    else:
        raw = [{"name": "line1", "p1": cfg["p1"], "p2": cfg["p2"]}]
    lines = [
        Line(str(r.get("name", f"line{i + 1}")), tuple(r["p1"]), tuple(r["p2"]))
        for i, r in enumerate(raw)
    ]
    return lines, size


@dataclass
class _Track:
    side: int | None = None  # confirmed side, None until first sighting
    run: int = 0  # consecutive observations on the opposite side
    last: tuple[float, float] | None = None  # anchor at last commit
    hits: int = 0


class LineCounter:
    """Count one line's crossings per track_id, exactly one event per crossing.

    Guards, in order: the anchor must project inside the drawn segment, must
    leave the dead band, must hold the new side for `min_run` frames, the track
    must have `min_hits` observations, and must have moved `min_disp` pixels
    since its last commit (rejects a box oscillating in place on the line).
    """

    def __init__(
        self,
        line: Line,
        band: float = 4.0,
        min_run: int = 3,
        min_disp: float = 8.0,
        min_hits: int = 3,
        evict_after: int = 15,
        classes: list[str] | None = None,
    ) -> None:
        self.line = line
        self.band = band
        self.min_run = min_run
        self.min_disp = min_disp
        self.min_hits = min_hits
        self.evict_after = evict_after
        # A/B name the line's two sides (see Line.side_names), so a crossing
        # from the '+' side to the '-' side is named `_plus_dir`. Prefilled with
        # `classes` so the table shows every monitored class even when zero.
        plus_name, minus_name = line.side_names()
        self._plus_dir = f"{plus_name}->{minus_name}"
        self._minus_dir = f"{minus_name}->{plus_name}"
        self.directions = ("A->B", "B->A")  # fixed column order for the table
        self.counts: dict[str, dict[str, int]] = (
            {name: {self._plus_dir: 0, self._minus_dir: 0} for name in classes} if classes else {}
        )
        self._tracks: dict[int, _Track] = {}
        self._absent: dict[int, int] = {}

    def totals(self) -> dict[str, int]:
        """Direction totals summed over every class."""
        total = {self._plus_dir: 0, self._minus_dir: 0}
        for bucket in self.counts.values():
            total[self._plus_dir] += bucket[self._plus_dir]
            total[self._minus_dir] += bucket[self._minus_dir]
        return total

    def update(
        self, track_id: int, x: float, y: float, t: float, class_name: str
    ) -> dict | None:
        """Feed one observation. Returns an event dict on a crossing, else None."""
        st = self._tracks.setdefault(track_id, _Track())
        self._absent.pop(track_id, None)
        st.hits += 1

        proj, dist = self.line.geometry(x, y)
        if not 0.0 <= proj <= 1.0:  # outside the drawn segment: ignore
            return None
        if abs(dist) < self.band:  # dead band: ignore, state untouched
            return None
        side = 1 if dist > 0 else -1

        if st.side is None:  # first sighting seeds the reference side
            st.side, st.last = side, (x, y)
            return None
        if side == st.side:  # still on the reference side
            st.run = 0
            return None

        st.run += 1
        if st.run < self.min_run or st.hits < self.min_hits:
            return None
        lx, ly = st.last
        if math.hypot(x - lx, y - ly) < self.min_disp:
            return None

        direction = self._plus_dir if st.side > 0 else self._minus_dir
        st.side, st.run, st.last = side, 0, (x, y)
        self.counts.setdefault(class_name, {self._plus_dir: 0, self._minus_dir: 0})[direction] += 1
        return {
            "ts": round(t, 3),
            "line": self.line.name,
            "track_id": int(track_id),
            "class": class_name,
            "direction": direction,
        }

    def evict_missing(self, seen_ids: set[int]) -> None:
        """Age tracks not seen this frame; drop the ones gone too long."""
        for tid in list(self._tracks):
            if tid in seen_ids:
                self._absent.pop(tid, None)
                continue
            n = self._absent.get(tid, 0) + 1
            if n >= self.evict_after:
                del self._tracks[tid]
                self._absent.pop(tid, None)
            else:
                self._absent[tid] = n
