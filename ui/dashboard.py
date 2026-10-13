"""
Professional dark-themed monitoring dashboard for the Accident Detection System.

Layout
------
::

    ┌──────────────────────────────────────────────────────────────┐
    │  AI ROAD SAFETY MONITOR          [YOLOv8]  System ● ONLINE  │
    ├────────────────────────────────────────┬─────────────────────┤
    │                                        │  Detection Status   │
    │        LIVE CAMERA FEED                │  ● MONITORING       │
    │                                        │  Confidence ▓░░ 0%  │
    │                                        ├─────────────────────┤
    │                                        │  Performance        │
    │                                        │  FPS  ·  Latency    │
    │                                        ├─────────────────────┤
    │                                        │  Session Stats      │
    ├────────────────────────────────────────│  Frames · Accidents │
    │  [▶ Start]  [■ Stop]  Source: [0    ] │  Suppressed · Uptime│
    │                                        ├─────────────────────┤
    │                                        │  Event Log          │
    └────────────────────────────────────────┴─────────────────────┘
"""
from __future__ import annotations

import os
import queue
import threading
import time
import tkinter as tk
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

import config
from alerts.alert_manager import AlertManager
from camera.camera_manager import CameraManager
from detection.accident_engine import AccidentEngine, AccidentState
from detection.detector import AccidentDetector
from detection.tracker import TrackAnalyzer
from utils.logger import setup_logger

logger = setup_logger("dashboard")

# ── Colour palette ───────────────────────────────────────────────
BG = "#080c14"
PANEL = "#0f1420"
CARD = "#161d2e"
ACCENT = "#3b82f6"
ACCENT2 = "#8b5cf6"
SAFE = "#10b981"
DANGER = "#ef4444"
WARNING = "#f59e0b"
TEXT = "#f1f5f9"
SUBTEXT = "#64748b"
BORDER = "#1e293b"
DARK_BG = "#0a0d15"


class Dashboard:
    """The main application: builds the UI, loads the model, and
    orchestrates the detection loop in a background thread."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(config.WINDOW_TITLE)
        self.root.configure(bg=BG)
        self.root.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")
        self.root.minsize(960, 640)

        # ── modules ──────────────────────────────────────────────
        self.detector = AccidentDetector()
        self.engine = AccidentEngine()
        self.tracker = TrackAnalyzer()
        self.alert_mgr = AlertManager()
        self.camera: Optional[CameraManager] = None

        # ── state ────────────────────────────────────────────────
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)

        # ── build ────────────────────────────────────────────────
        self._build_ui()
        self._schedule_clock()
        self._schedule_frame_poll()
        self._load_model_async()

    # ═════════════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ═════════════════════════════════════════════════════════════

    def _build_ui(self) -> None:
        self._build_header()

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=(0, 10))

        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        self._build_video_panel(left)
        self._build_controls(left)

        right = tk.Frame(body, bg=BG, width=310)
        right.pack(side="right", fill="y", padx=(12, 0))
        right.pack_propagate(False)
        self._build_status_card(right)
        self._build_perf_card(right)
        self._build_stats_card(right)
        self._build_log_card(right)

    # ── header ───────────────────────────────────────────────────

    def _build_header(self) -> None:
        hdr = tk.Frame(self.root, bg=BG, pady=10)
        hdr.pack(fill="x", padx=20)

        tk.Label(
            hdr, text="AI Road Safety Monitor",
            font=("Segoe UI", 17, "bold"), bg=BG, fg=TEXT,
        ).pack(side="left")

        tk.Label(
            hdr, text=" YOLOv8 ", font=("Segoe UI", 8, "bold"),
            bg=ACCENT2, fg="white", padx=6, pady=2,
        ).pack(side="left", padx=(10, 0))

        tk.Label(
            hdr, text=" ACCIDENT DETECTION ",
            font=("Segoe UI", 8, "bold"),
            bg=ACCENT, fg="white", padx=6, pady=2,
        ).pack(side="left", padx=4)

        self._clock_label = tk.Label(
            hdr, text="", font=("Segoe UI", 9), bg=BG, fg=SUBTEXT,
        )
        self._clock_label.pack(side="right", padx=(0, 4))

        self._model_label = tk.Label(
            hdr, text="Loading model …",
            font=("Segoe UI", 9), bg=BG, fg=SUBTEXT,
        )
        self._model_label.pack(side="right", padx=(0, 16))

    # ── video panel ──────────────────────────────────────────────

    def _build_video_panel(self, parent: tk.Frame) -> None:
        wrap = tk.Frame(
            parent, bg=PANEL,
            highlightbackground=BORDER, highlightthickness=1,
        )
        wrap.pack(fill="both", expand=True)

        bar = tk.Frame(wrap, bg=CARD, pady=6)
        bar.pack(fill="x")
        tk.Label(
            bar, text="  Live Feed", font=("Segoe UI", 10, "bold"),
            bg=CARD, fg=TEXT,
        ).pack(side="left")
        self._feed_badge = tk.Label(
            bar, text=" OFFLINE ",
            font=("Segoe UI", 8, "bold"),
            bg=CARD, fg=SUBTEXT, padx=8,
        )
        self._feed_badge.pack(side="right", padx=8)

        self.canvas = tk.Canvas(wrap, bg=DARK_BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _: self._draw_placeholder())
        self._draw_placeholder()

    def _draw_placeholder(self) -> None:
        self.canvas.update_idletasks()
        w = self.canvas.winfo_width() or 640
        h = self.canvas.winfo_height() or 400
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, w, h, fill=DARK_BG, outline="")
        self.canvas.create_text(
            w // 2, h // 2 - 12,
            text="Camera feed will appear here",
            font=("Segoe UI", 14), fill="#1e2a3e",
        )
        self.canvas.create_text(
            w // 2, h // 2 + 16,
            text="Click  Start Detection  to begin",
            font=("Segoe UI", 10), fill="#151d2d",
        )

    # ── controls ─────────────────────────────────────────────────

    def _build_controls(self, parent: tk.Frame) -> None:
        bar = tk.Frame(parent, bg=BG, pady=8)
        bar.pack(fill="x")

        self._start_btn = tk.Button(
            bar, text="▶  Start Detection",
            font=("Segoe UI", 10, "bold"),
            bg=ACCENT, fg="white", relief="flat", cursor="hand2",
            activebackground="#2563eb", activeforeground="white",
            padx=16, pady=6, command=self._start,
        )
        self._start_btn.pack(side="left", padx=(0, 6))

        self._stop_btn = tk.Button(
            bar, text="■  Stop",
            font=("Segoe UI", 10, "bold"),
            bg=CARD, fg=SUBTEXT, relief="flat", cursor="hand2",
            activebackground=DANGER, activeforeground="white",
            padx=16, pady=6, state="disabled", command=self._stop,
        )
        self._stop_btn.pack(side="left", padx=(0, 16))

        tk.Label(
            bar, text="Source:", font=("Segoe UI", 9), bg=BG, fg=SUBTEXT,
        ).pack(side="left", padx=(0, 4))
        self._source_var = tk.StringVar(value=str(config.CAMERA_SOURCE))
        tk.Entry(
            bar, textvariable=self._source_var, width=18,
            font=("Segoe UI", 10), bg=CARD, fg=TEXT,
            insertbackground=TEXT, relief="flat", bd=4,
        ).pack(side="left")

    # ── status card ──────────────────────────────────────────────

    def _build_status_card(self, parent: tk.Frame) -> None:
        card = tk.Frame(parent, bg=CARD,
                        highlightbackground=BORDER, highlightthickness=1)
        card.pack(fill="x", pady=(0, 8))

        tk.Label(card, text="Detection Status",
                 font=("Segoe UI", 9, "bold"),
                 bg=CARD, fg=SUBTEXT, pady=6, padx=10).pack(anchor="w")

        row = tk.Frame(card, bg=CARD, padx=10, pady=4)
        row.pack(fill="x")
        self._state_dot = tk.Canvas(row, width=14, height=14,
                                    bg=CARD, highlightthickness=0)
        self._state_dot.pack(side="left", padx=(0, 8))
        self._dot_id = self._state_dot.create_oval(1, 1, 13, 13,
                                                    fill=SUBTEXT, outline="")
        self._state_label = tk.Label(row, text="STANDBY",
                                     font=("Segoe UI", 15, "bold"),
                                     bg=CARD, fg=SUBTEXT)
        self._state_label.pack(side="left")

        # Evidence counter (only visible during candidate)
        self._evidence_label = tk.Label(
            card, text="", font=("Segoe UI", 8),
            bg=CARD, fg=SUBTEXT, padx=10,
        )
        self._evidence_label.pack(anchor="w")

        # Confidence bar
        tk.Label(card, text="Confidence",
                 font=("Segoe UI", 8), bg=CARD, fg=SUBTEXT,
                 padx=10).pack(anchor="w", pady=(4, 0))
        self._conf_canvas = tk.Canvas(card, bg=DARK_BG, height=14,
                                      highlightthickness=0)
        self._conf_canvas.pack(fill="x", padx=10, pady=(2, 10))
        self._draw_conf_bar(0.0)

    def _draw_conf_bar(self, value: float) -> None:
        c = self._conf_canvas
        c.update_idletasks()
        W = max(c.winfo_width(), 10)
        c.delete("all")
        c.create_rectangle(0, 0, W, 14, fill=BORDER, outline="")
        colour = DANGER if value >= 0.5 else SAFE
        fill_w = int(W * min(value, 1.0))
        if fill_w > 0:
            c.create_rectangle(0, 0, fill_w, 14, fill=colour, outline="")
        c.create_text(W // 2, 7, text=f"{value * 100:.1f} %",
                      font=("Segoe UI", 7, "bold"), fill="white")

    # ── performance card ─────────────────────────────────────────

    def _build_perf_card(self, parent: tk.Frame) -> None:
        card = tk.Frame(parent, bg=CARD,
                        highlightbackground=BORDER, highlightthickness=1)
        card.pack(fill="x", pady=(0, 8))

        tk.Label(card, text="Performance",
                 font=("Segoe UI", 9, "bold"),
                 bg=CARD, fg=SUBTEXT, pady=6, padx=10).pack(anchor="w")

        grid = tk.Frame(card, bg=CARD, padx=10)
        grid.pack(fill="x", pady=(0, 10))
        self._fps_val = self._metric(grid, "FPS", "—", 0, 0)
        self._lat_val = self._metric(grid, "Latency", "—", 0, 1)

    def _metric(self, parent, label, value, row, col):
        f = tk.Frame(parent, bg=BORDER, padx=8, pady=6)
        f.grid(row=row, column=col, padx=(0 if col == 0 else 4, 0),
               sticky="ew")
        parent.columnconfigure(col, weight=1)
        v = tk.Label(f, text=value, font=("Segoe UI", 16, "bold"),
                     bg=BORDER, fg=TEXT)
        v.pack()
        tk.Label(f, text=label, font=("Segoe UI", 7),
                 bg=BORDER, fg=SUBTEXT).pack()
        return v

    # ── stats card ───────────────────────────────────────────────

    def _build_stats_card(self, parent: tk.Frame) -> None:
        card = tk.Frame(parent, bg=CARD,
                        highlightbackground=BORDER, highlightthickness=1)
        card.pack(fill="x", pady=(0, 8))

        tk.Label(card, text="Session Statistics",
                 font=("Segoe UI", 9, "bold"),
                 bg=CARD, fg=SUBTEXT, pady=6, padx=10).pack(anchor="w")

        grid = tk.Frame(card, bg=CARD, padx=10)
        grid.pack(fill="x", pady=(0, 10))
        self._s_frames = self._metric(grid, "Frames", "0", 0, 0)
        self._s_accidents = self._metric(grid, "Accidents", "0", 0, 1)
        self._s_suppressed = self._metric(grid, "Suppressed", "0", 1, 0)
        self._s_uptime = self._metric(grid, "Uptime", "00:00", 1, 1)

    # ── log card ─────────────────────────────────────────────────

    def _build_log_card(self, parent: tk.Frame) -> None:
        card = tk.Frame(parent, bg=CARD,
                        highlightbackground=BORDER, highlightthickness=1)
        card.pack(fill="both", expand=True)

        hdr = tk.Frame(card, bg=CARD)
        hdr.pack(fill="x", pady=(6, 0), padx=10)
        tk.Label(hdr, text="Event Log", font=("Segoe UI", 9, "bold"),
                 bg=CARD, fg=SUBTEXT).pack(side="left")
        tk.Button(hdr, text="Clear", font=("Segoe UI", 7),
                  bg=CARD, fg=SUBTEXT, relief="flat", cursor="hand2",
                  command=self._clear_log).pack(side="right")

        container = tk.Frame(card, bg=CARD)
        container.pack(fill="both", expand=True, padx=4, pady=(2, 6))

        sb = tk.Scrollbar(container)
        sb.pack(side="right", fill="y")

        self._log_text = tk.Text(
            container, bg=DARK_BG, fg=TEXT,
            font=("Consolas", 8), relief="flat",
            state="disabled", wrap="word",
            selectbackground=ACCENT2, bd=0,
            padx=6, pady=4, yscrollcommand=sb.set,
        )
        self._log_text.pack(fill="both", expand=True)
        sb.config(command=self._log_text.yview)

        self._log_text.tag_config("danger", foreground=DANGER)
        self._log_text.tag_config("safe", foreground=SAFE)
        self._log_text.tag_config("warning", foreground=WARNING)
        self._log_text.tag_config("time", foreground=SUBTEXT)
        self._log_text.tag_config("info", foreground=ACCENT)

    # ═════════════════════════════════════════════════════════════
    # LOGGING HELPERS
    # ═════════════════════════════════════════════════════════════

    def _log(self, msg: str, tag: str = "safe") -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_text.config(state="normal")
        self._log_text.insert("end", f"[{ts}] ", "time")
        self._log_text.insert("end", msg + "\n", tag)
        self._log_text.see("end")
        self._log_text.config(state="disabled")

    def _clear_log(self) -> None:
        self._log_text.config(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.config(state="disabled")

    # ═════════════════════════════════════════════════════════════
    # TIMERS
    # ═════════════════════════════════════════════════════════════

    def _schedule_clock(self) -> None:
        now = datetime.now().strftime("%H:%M:%S")
        self._clock_label.config(text=now)

        if self._start_time is not None:
            elapsed = timedelta(seconds=int(time.time() - self._start_time))
            self._s_uptime.config(text=str(elapsed))

        self.root.after(1000, self._schedule_clock)

    def _schedule_frame_poll(self) -> None:
        """Poll the frame queue and push the latest frame to the UI."""
        latest: Optional[Dict[str, Any]] = None
        try:
            while True:
                latest = self._frame_queue.get_nowait()
        except queue.Empty:
            pass

        if latest is not None:
            self._apply_frame_data(latest)

        self.root.after(config.UI_UPDATE_MS, self._schedule_frame_poll)

    # ═════════════════════════════════════════════════════════════
    # MODEL LOADING
    # ═════════════════════════════════════════════════════════════

    def _load_model_async(self) -> None:
        def _work():
            ok = self.detector.load()
            self.root.after(0, lambda: self._on_model_result(ok))

        threading.Thread(target=_work, daemon=True).start()

    def _on_model_result(self, success: bool) -> None:
        if success:
            names = self.detector.class_names
            self._model_label.config(text=f"Model ready  — {names}", fg=SAFE)
            self._log(f"Model loaded — classes: {names}", "info")
        else:
            self._model_label.config(text="Model load FAILED", fg=DANGER)
            self._log("Model failed to load!", "danger")

    # ═════════════════════════════════════════════════════════════
    # START / STOP
    # ═════════════════════════════════════════════════════════════

    def _start(self) -> None:
        if not self.detector.is_loaded:
            self._log("Model not ready yet!", "danger")
            return

        # Determine source
        src_str = self._source_var.get().strip()
        source: Any = int(src_str) if src_str.isdigit() else src_str

        self.camera = CameraManager(source)
        if not self.camera.open():
            self._log(f"Cannot open source: {source}", "danger")
            return

        self.engine.reset()
        self.tracker.reset()
        self.detector.reset_tracker()
        self.running = True
        self._start_time = time.time()

        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal", bg=DANGER, fg="white")
        self._feed_badge.config(text=" LIVE ", fg=SAFE)
        self._log(f"Detection started — source: {source}", "info")

        self._thread = threading.Thread(
            target=self._detection_loop, daemon=True,
        )
        self._thread.start()

    def _stop(self) -> None:
        self.running = False
        self.alert_mgr.stop_sound()

        if self.camera:
            self.camera.release()

        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled", bg=CARD, fg=SUBTEXT)
        self._feed_badge.config(text=" OFFLINE ", fg=SUBTEXT)
        self._update_state_display(AccidentState.MONITORING, active=False)
        self.root.after(300, self._draw_placeholder)
        self._log("Detection stopped", "info")

    # ═════════════════════════════════════════════════════════════
    # DETECTION LOOP  (runs in background thread)
    # ═════════════════════════════════════════════════════════════

    def _detection_loop(self) -> None:
        logger.info("Detection loop started")

        while self.running and self.camera and self.camera.is_open:
            frame, cam_fps = self.camera.read()
            if frame is None:
                self.root.after(0, lambda: self._log(
                    "Camera feed ended", "warning"))
                break

            # ── inference ────────────────────────────────────────
            try:
                result = self.detector.detect(frame)
            except Exception as exc:
                logger.error("Inference error: %s", exc)
                continue

            now = time.time()

            # ── tracking ─────────────────────────────────────────
            self.tracker.update(result.detections, now)

            # ── accident engine ──────────────────────────────────
            event = self.engine.update(
                result.has_accident, result.max_confidence, now,
            )

            # ── alert on confirmation ────────────────────────────
            if event == "confirmed":
                duration = now - self.engine.candidate_start
                self.alert_mgr.trigger_alert(
                    result.annotated_frame,
                    self.engine.peak_confidence,
                    duration,
                )

            # ── annotate frame ───────────────────────────────────
            display = self._annotate_frame(
                result.annotated_frame, result, cam_fps,
            )

            # ── push to UI queue (drop if full) ──────────────────
            data: Dict[str, Any] = {
                "frame": display,
                "state": self.engine.state,
                "confidence": result.max_confidence,
                "peak": self.engine.peak_confidence,
                "fps": cam_fps,
                "latency": result.inference_time_ms,
                "frames": self.camera.frame_count,
                "accidents": self.engine.total_confirmed,
                "suppressed": self.engine.total_candidates_rejected,
                "evidence_count": self.engine.evidence_count,
                "event": event,
                "cooldown": self.engine.cooldown_remaining,
            }
            try:
                self._frame_queue.put_nowait(data)
            except queue.Full:
                pass  # drop frame — UI can't keep up

        logger.info("Detection loop ended")

    # ── frame annotation ─────────────────────────────────────────

    def _annotate_frame(
        self, frame: np.ndarray, result, fps: float,
    ) -> np.ndarray:
        """Draw status banner, FPS, and confidence bar on the frame."""
        h, w = frame.shape[:2]
        state = self.engine.state

        # ── status banner ────────────────────────────────────────
        if state == AccidentState.CONFIRMED:
            color = (0, 0, 220)
            text = "ACCIDENT DETECTED"
        elif state == AccidentState.CANDIDATE:
            color = (0, 150, 240)
            ev = self.engine.evidence_count
            req = self.engine.confirmation_frames
            text = f"CANDIDATE  {ev}/{req}"
        elif state == AccidentState.COOLDOWN:
            color = (200, 120, 40)
            remaining = self.engine.cooldown_remaining
            text = f"COOLDOWN  {remaining:.0f}s"
        else:
            color = (16, 160, 80)
            text = "MONITORING"

        cv2.rectangle(frame, (0, 0), (w, 34), color, -1)
        cv2.putText(
            frame, text, (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
            cv2.LINE_AA,
        )

        # ── FPS + latency ────────────────────────────────────────
        info = f"FPS: {fps:.1f}  |  {result.inference_time_ms:.0f} ms"
        tw = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
        cv2.putText(
            frame, info, (w - tw - 10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
            cv2.LINE_AA,
        )

        # ── confidence bar at bottom ─────────────────────────────
        if result.max_confidence > 0:
            bar_w = int(w * min(result.max_confidence, 1.0))
            bar_clr = (0, 0, 220) if result.max_confidence >= 0.5 else (16, 185, 129)
            cv2.rectangle(frame, (0, h - 6), (bar_w, h), bar_clr, -1)

        return frame

    # ═════════════════════════════════════════════════════════════
    # UI UPDATES  (called on main thread via _schedule_frame_poll)
    # ═════════════════════════════════════════════════════════════

    def _apply_frame_data(self, data: Dict[str, Any]) -> None:
        self._update_canvas(data["frame"])
        self._update_state_display(data["state"], active=True)
        self._draw_conf_bar(data.get("peak", data["confidence"]))

        # Evidence counter
        if data["state"] == AccidentState.CANDIDATE:
            self._evidence_label.config(
                text=f"Evidence: {data['evidence_count']}/"
                     f"{self.engine.confirmation_frames} frames",
            )
        elif data["state"] == AccidentState.COOLDOWN:
            self._evidence_label.config(
                text=f"Cooldown: {data['cooldown']:.0f}s remaining",
            )
        else:
            self._evidence_label.config(text="")

        # Performance
        self._fps_val.config(text=f"{data['fps']:.1f}")
        self._lat_val.config(text=f"{data['latency']:.0f}ms")

        # Stats
        self._s_frames.config(text=f"{data['frames']:,}")
        self._s_accidents.config(
            text=str(data["accidents"]),
            fg=DANGER if data["accidents"] > 0 else TEXT,
        )
        self._s_suppressed.config(text=str(data["suppressed"]))

        # Log events
        if data["event"]:
            self._log_event(data["event"], data)

    def _update_canvas(self, frame_bgr: np.ndarray) -> None:
        self.canvas.update_idletasks()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail((cw, ch), Image.LANCZOS)
        x = (cw - img.width) // 2
        y = (ch - img.height) // 2

        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(x, y, anchor="nw", image=self._photo)

    def _update_state_display(
        self, state: AccidentState, active: bool = True,
    ) -> None:
        mapping = {
            AccidentState.MONITORING: ("MONITORING", SAFE),
            AccidentState.CANDIDATE: ("CANDIDATE", WARNING),
            AccidentState.CONFIRMED: ("ACCIDENT!", DANGER),
            AccidentState.COOLDOWN: ("COOLDOWN", ACCENT),
        }
        label, colour = mapping.get(state, ("STANDBY", SUBTEXT))

        if not active:
            label, colour = "STANDBY", SUBTEXT

        self._state_dot.itemconfig(self._dot_id, fill=colour)
        self._state_label.config(text=label, fg=colour)

        if state == AccidentState.CONFIRMED:
            self.root.title(f"*** ACCIDENT *** — {config.WINDOW_TITLE}")
        else:
            self.root.title(config.WINDOW_TITLE)

    def _log_event(self, event: str, data: Dict[str, Any]) -> None:
        messages = {
            "candidate_start": (
                f"Accident candidate  (conf={data['confidence']:.2f})",
                "warning",
            ),
            "candidate_rejected": (
                "Candidate rejected — false alarm suppressed",
                "safe",
            ),
            "confirmed": (
                f"ACCIDENT CONFIRMED  conf={data['peak']:.2f}",
                "danger",
            ),
            "cooldown_start": (
                f"Cooldown started ({config.ALERT_COOLDOWN_SEC:.0f}s)",
                "info",
            ),
            "cooldown_end": (
                "Cooldown expired — monitoring resumed",
                "info",
            ),
        }
        msg, tag = messages.get(event, (event, "info"))
        self._log(msg, tag)

    # ═════════════════════════════════════════════════════════════
    # CLEANUP
    # ═════════════════════════════════════════════════════════════

    def stop(self) -> None:
        """Graceful shutdown."""
        self._stop()
        self.alert_mgr.cleanup()
