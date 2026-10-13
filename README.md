# Real-Time Road Accident Detection System

AI-powered road accident detection system using YOLOv8 with a **multi-frame temporal confirmation engine** to prevent false alerts.

## Key Features

- **Multi-frame accident confirmation** — requires 5+ consecutive detection frames before alerting (no single-frame false triggers)
- **State machine architecture** — `MONITORING → CANDIDATE → CONFIRMED → COOLDOWN` with configurable thresholds
- **ByteTrack object tracking** — tracks detection regions across frames for persistence evidence
- **Professional monitoring dashboard** — dark-themed Tkinter UI with live feed, state display, performance metrics, and event log
- **Evidence capture** — auto-saves timestamped screenshots of confirmed accidents
- **Alert cooldown** — prevents duplicate alerts for the same event (30s default)
- **Structured logging** — session log files with severity levels
- **Flexible input** — webcam, video file, or RTSP stream

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with webcam (default)
python app.py

# Run with video file
python app.py testing1.mp4

# Run with specific camera index
python app.py 1
```

## Architecture

```
project/
├── app.py                          # Entry point
├── config.py                       # All configurable parameters
├── requirements.txt
│
├── detection/
│   ├── detector.py                 # YOLO inference + tracking wrapper
│   ├── tracker.py                  # Track persistence analyzer
│   └── accident_engine.py          # State machine (core logic)
│
├── camera/
│   └── camera_manager.py           # Video capture + FPS measurement
│
├── alerts/
│   └── alert_manager.py            # Sound + evidence capture
│
├── ui/
│   └── dashboard.py                # Professional monitoring UI
│
├── utils/
│   └── logger.py                   # Structured logging
│
├── tests/
│   └── test_accident_engine.py     # 10 state machine tests
│
├── evidence/                       # Auto-saved accident screenshots
├── logs/                           # Session log files
├── data/                           # Training dataset
└── runs/detect/train/              # Training metrics
```

## How the Detection Pipeline Works

```
CAMERA FRAME
     ↓
YOLO INFERENCE  (with ByteTrack tracking)
     ↓
CONFIDENCE FILTERING  (threshold: 0.40)
     ↓
ACCIDENT ENGINE STATE MACHINE
     ↓
  ┌─ MONITORING ──→ detection? ──→ CANDIDATE
  │                                    ↓
  │                          5+ consecutive frames?
  │                                    ↓
  │                               CONFIRMED ──→ ALERT + EVIDENCE
  │                                    ↓
  │                               COOLDOWN (30s)
  └────────────────────────────────────┘
```

### False Positive Prevention

| Scenario | Result |
|---|---|
| Single frame detection | ❌ No alert (needs 5+ frames) |
| 3 frames then gap | ❌ Candidate rejected |
| Normal traffic | ❌ No detections above threshold |
| 5+ consecutive frames | ✅ Accident confirmed → Alert |
| Same accident continues | ❌ Cooldown suppresses duplicates |

## Configuration

All parameters are in [`config.py`](config.py):

| Parameter | Default | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | 0.40 | Minimum YOLO detection confidence |
| `CONFIRMATION_FRAMES` | 5 | Consecutive frames needed to confirm |
| `MAX_GAP_FRAMES` | 2 | Allowed detection gap within candidate |
| `ALERT_COOLDOWN_SEC` | 30.0 | Cooldown after confirmed accident |
| `ENABLE_TRACKING` | True | Use ByteTrack for detection tracking |
| `SAVE_EVIDENCE` | True | Auto-save accident screenshots |
| `ENABLE_SOUND` | True | Play alarm on confirmed accident |

## Model Information

| Property | Value |
|---|---|
| Architecture | YOLOv8s |
| Classes | 1 (`Accident`) |
| Training images | 947 |
| Best mAP50 | 0.586 |
| Input size | 640×640 |

## Running Tests

```bash
python -m unittest tests.test_accident_engine -v
```

## Dependencies

- Python ≥ 3.8
- ultralytics (YOLOv8)
- opencv-python
- pillow
- pygame-ce
- numpy
- PyYAML

## License

MIT License
