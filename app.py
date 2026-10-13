"""
Road Accident Detection System — Application Entry Point.

Usage:
    python app.py                     # webcam (default)
    python app.py testing1.mp4        # video file
    python app.py 1                   # webcam index 1

The system uses a multi-frame state machine to confirm accidents
before alerting, preventing false positives from single-frame glitches.
"""
from __future__ import annotations

import sys
import tkinter as tk

import config
from utils.logger import setup_logger

logger = setup_logger("app")


def main() -> None:
    """Launch the accident detection dashboard."""

    # Allow command-line source override
    if len(sys.argv) > 1:
        src = sys.argv[1]
        if src.isdigit():
            config.CAMERA_SOURCE = int(src)
        else:
            config.CAMERA_SOURCE = src
        logger.info("Source override from CLI: %s", config.CAMERA_SOURCE)

    # ── Tk root ──────────────────────────────────────────────────
    root = tk.Tk()

    # Centre on screen
    root.update_idletasks()
    w, h = config.WINDOW_WIDTH, config.WINDOW_HEIGHT
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = (sw - w) // 2
    y = max((sh - h) // 2 - 30, 0)
    root.geometry(f"{w}x{h}+{x}+{y}")

    # Bring to front briefly
    root.lift()
    root.attributes("-topmost", True)
    root.after(600, lambda: root.attributes("-topmost", False))

    # ── Dashboard ────────────────────────────────────────────────
    from ui.dashboard import Dashboard  # deferred to keep startup fast

    app = Dashboard(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop(), root.destroy()))

    logger.info("Application started")
    root.mainloop()
    logger.info("Application closed")


if __name__ == "__main__":
    main()
