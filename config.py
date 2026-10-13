"""
Central configuration for Road Accident Detection System.

All adjustable parameters are defined here. Modify these values
to tune detection sensitivity, performance, and behavior.
"""
from pathlib import Path

# ── Base Paths ───────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "best.pt")
ALARM_SOUND_PATH = str(BASE_DIR / "alarm.wav")
EVIDENCE_DIR = str(BASE_DIR / "evidence")
LOG_DIR = str(BASE_DIR / "logs")

# ── Camera / Video Source ────────────────────────────────────────
# Integer for webcam index, string for video file path or RTSP URL
# Examples:  0,  "testing1.mp4",  "rtsp://192.168.1.100:554/stream"
CAMERA_SOURCE = 0

# ── YOLO Model Settings ─────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.40   # Minimum detection confidence
IOU_THRESHOLD = 0.50          # Non-max suppression IoU threshold
INPUT_SIZE = 640              # Model input resolution (pixels)
DEVICE = ""                   # "" = auto-detect, "cpu", "cuda:0"

# ── Object Tracking ─────────────────────────────────────────────
ENABLE_TRACKING = True
TRACKER_TYPE = "bytetrack.yaml"   # "bytetrack.yaml" or "botsort.yaml"

# ── Accident Detection State Machine ────────────────────────────
#
# State flow:
#   MONITORING  →  CANDIDATE  →  CONFIRMED  →  COOLDOWN  →  MONITORING
#
# MONITORING:  No evidence.  Waiting for detections.
# CANDIDATE:   Detection started.  Accumulating frame-level evidence.
# CONFIRMED:   Enough consecutive evidence.  Alert triggered.
# COOLDOWN:    Suppressing duplicate alerts for the same event.
#
CONFIRMATION_FRAMES = 5       # Consecutive detection frames to confirm accident
MAX_GAP_FRAMES = 2            # Max non-detection frames before candidate resets
CONFIRMED_DISPLAY_SEC = 3.0   # Minimum seconds to hold CONFIRMED state
ALERT_COOLDOWN_SEC = 30.0     # Seconds before allowing a new alert

# ── Alert System ─────────────────────────────────────────────────
ENABLE_SOUND = True
SOUND_DURATION_SEC = 5.0      # How long alarm plays per confirmed event

# ── Evidence Capture ─────────────────────────────────────────────
SAVE_EVIDENCE = True
MAX_EVIDENCE_FILES = 200      # Auto-delete oldest files when exceeded

# ── UI ───────────────────────────────────────────────────────────
WINDOW_TITLE = "AI Road Safety Monitor"
WINDOW_WIDTH = 1300
WINDOW_HEIGHT = 800
UI_UPDATE_MS = 50             # UI refresh interval (~20 FPS display)

# ── Logging ──────────────────────────────────────────────────────
LOG_LEVEL = "INFO"            # DEBUG, INFO, WARNING, ERROR
LOG_TO_FILE = True
