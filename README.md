# Vehicle counting by direction on a traffic camera

This work aims to count the vehicles that cross a virtual line in a traffic camera stream,
separating them by direction and logging each crossing. The pipeline combines the YOLO26m
detector, the TrackTrack tracker with re-identification, and a per-line counter configured
in a file; the decisions, the trade-offs and the limitations are documented in
`SOLUTION.md`.

## Running

Requires Python 3.12 or newer. The detector and PyTorch are installed separately, and the
detector weights are downloaded on the first run if they are missing.

```bash
pip install -r requirements.txt
pip install ultralytics
# GPU (optional but recommended):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

```bash
python test_crossing.py                          # line-crossing logic tests
python pick_line.py                              # picks the line, writes line.json
python stream_process.py                          # live stream
python stream_process.py -u samples/clip.mp4      # local file
```

In the window, the space bar pauses and resumes the run, and the keys `q` or `Esc` stop
it. The output is the `events.jsonl`, with one event per line, and the `summary.json`, with
the totals per class and per direction.

## Files

| path | content |
|---|---|
| `stream_process.py` | pipeline: detection, tracking, counting and overlay |
| `crossing.py` | line-crossing logic, with no OpenCV dependency |
| `test_crossing.py` | tests for the crossing logic |
| `pick_line.py` | tool that draws the line and writes `line.json` |
| `bench_stages.py` | per-stage processing time measurement |
| `trackers/traffic.yaml` | the TrackTrack configuration adopted |
| `line.json` | the virtual line marked on the scene |
| `samples/evidence_30s.mp4` | 32 s clip with the pipeline result |
| `samples/flowchart.png` | pipeline flow |
| `SOLUTION.md` | decisions, trade-offs, limitations and numbers |

## Results

The count was evaluated against a manual count of 60 s on `samples/clip.mp4`, with 1,800
frames. The per-direction totals matched the manual reference, and one class-label error
remained on a single vehicle. The detailed accuracy and the per-stage processing time are
in `SOLUTION.md`.
