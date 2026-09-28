#!/usr/bin/env python3
"""One runnable check for the crossing logic: python test_crossing.py"""

import json
import os
import tempfile

from crossing import Line, LineCounter, load_lines

# A=(0,0), B=(0,100): x>0 is the right side (side<0), x<0 the left (side>0).
LINE = Line("t", (0.0, 0.0), (0.0, 100.0))


def feed(counter, xs, ys=None, tid=1, dt=1 / 30, cls="car"):
    ys = ys or [50.0] * len(xs)
    events = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        ev = counter.update(tid, x, y, i * dt, cls)
        if ev:
            events.append(ev)
    return events


def test_one_event_per_crossing():
    c = LineCounter(LINE)
    ev = feed(c, [-50, -30, -10, 10, 30, 50])  # left -> right
    assert len(ev) == 1, ev
    assert ev[0]["direction"] == "A->B", ev  # vertical: A left, B right
    assert c.counts == {"car": {"A->B": 1, "B->A": 0}}, c.counts
    # stays on the far side: no second event
    assert feed(c, [60, 70, 80]) == []
    assert c.counts["car"]["A->B"] == 1
    # and the reverse crossing counts once as B->A
    ev2 = feed(c, [50, 0, -50, -70, -90])
    assert [e["direction"] for e in ev2] == ["B->A"], ev2
    assert c.counts == {"car": {"A->B": 1, "B->A": 1}}, c.counts
    assert c.totals() == {"A->B": 1, "B->A": 1}, c.totals()
    # a different class is bucketed separately
    c2 = LineCounter(LINE)
    assert len(feed(c2, [-50, -30, -10, 10, 30, 50], tid=7, cls="truck")) == 1
    assert c2.counts == {"truck": {"A->B": 1, "B->A": 0}}, c2.counts


def test_side_names_follow_orientation():
    # vertical: A is the left side, B the right
    assert LINE.orientation() == "vertical"
    assert LINE.side_names() == ("A", "B")
    # horizontal A->B: '+' is below, so A is the top side and B the bottom
    h = Line("h", (0.0, 0.0), (100.0, 0.0))
    assert h.orientation() == "horizontal"
    assert h.side_names() == ("B", "A")

    c = LineCounter(h)
    xs = [10, 26, 42, 58, 74, 90]
    # moving up (below -> above) crosses B -> A
    up = feed(c, xs, ys=[50, 30, 10, -10, -30, -50])
    assert [e["direction"] for e in up] == ["B->A"], up
    # moving down (above -> below) crosses A -> B
    down = feed(c, xs, ys=[-50, -30, -10, 10, 30, 50], tid=2)
    assert [e["direction"] for e in down] == ["A->B"], down
    assert c.totals() == {"A->B": 1, "B->A": 1}, c.totals()


def test_dead_band_kills_oscillation():
    c = LineCounter(LINE, band=4.0)
    assert feed(c, [2, -2, 2, -2, 2]) == []  # all |dist| < 4
    assert c.counts == {}, c.counts


def test_outside_segment_ignored():
    c = LineCounter(LINE)
    assert feed(c, [-50, 50, -50], ys=[200.0, 200.0, 200.0]) == []  # proj = 2
    assert c.counts == {}, c.counts


def test_min_run_needs_sustained_side():
    c = LineCounter(LINE, min_run=3)
    # only two frames on the far side, then it flaps back: no event
    assert feed(c, [-50, 10, 30, -10, -30]) == []
    assert c.counts == {}, c.counts


def test_config_accepts_list_and_single():
    with tempfile.TemporaryDirectory() as d:
        single = os.path.join(d, "single.json")
        with open(single, "w", encoding="utf-8") as fh:
            json.dump({"p1": [1, 2], "p2": [3, 4], "size": [100, 50]}, fh)
        lines, size = load_lines(single)
        assert [ln.name for ln in lines] == ["line1"] and size == [100, 50], (lines, size)

        multi = os.path.join(d, "multi.json")
        with open(multi, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "size": [100, 50],
                    "lines": [
                        {"name": "via1", "p1": [1, 2], "p2": [3, 4]},
                        {"name": "via2", "p1": [5, 6], "p2": [7, 8]},
                    ],
                },
                fh,
            )
        lines, size = load_lines(multi)
        assert [ln.name for ln in lines] == ["via1", "via2"], lines


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("[*] all crossing tests passed")
