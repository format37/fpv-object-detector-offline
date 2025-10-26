# FPV Object Detector (Offline)

Detect and track garbage in drone FPV video footage using YOLO11 with ByteTrack tracking.

## Features
- Detects bottles and paper objects
- Tracks unique objects across frames
- Outputs annotated video with real-time counter
- GPU/CPU support with auto-detection

## Installation

```bash
git clone https://github.com/format37/fpv-object-detector-offline.git
cd fpv-object-detector-offline
pip install ultralytics opencv-python torch
```

## Quick Start

```bash
./run.sh
```

Or manually:
```bash
python detect_garbage.py input.mp4 --output output.mp4 --model yolo11m.pt --confidence 0.25 -d 0
```

## Configuration

### Command Line Options
- `--model` - Model size: `yolo11n.pt` (fast), `yolo11s.pt`, `yolo11m.pt` (accurate)
- `--confidence` - Detection threshold: `0.25` (default), lower = more detections
- `--device` - `0` (GPU), `cpu` (CPU), or auto-detect
- `--output` - Output video path

### Target Classes (edit in `detect_garbage.py`)
```python
TARGET_CLASSES = [39, 73]  # 39: bottle, 73: paper
CLASS_NAMES = {39: 'Bottle', 73: 'Paper'}
```

### ByteTrack Tracking
ByteTrack is configured via `tracker="bytetrack.yaml"` in the code. Uses Ultralytics default config:
- Tracks objects across frames with unique IDs
- Handles occlusions and re-identification
- No additional configuration needed

To customize, create `bytetrack.yaml` in project root with [Ultralytics tracker parameters](https://docs.ultralytics.com/modes/track/).
