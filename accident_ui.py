import os
import threading
import tkinter as tk
from datetime import datetime
from collections import deque

import cv2
import pygame
from PIL import Image, ImageTk
from ultralytics import YOLO

# ─────────────────────────── COLOUR PALETTE ────────────────────────────────
BG        = "#0d0f14"
PANEL     = "#13161e"
CARD      = "#1a1e2a"
ACCENT    = "#4f8ef7"
ACCENT2   = "#7b5ef8"
SAFE_CLR  = "#22c55e"
DANGER_CLR= "#ef4444"
TEXT      = "#e2e8f0"
SUBTEXT   = "#94a3b8"
BORDER    = "#252a38"

# ───────────────────────────── APP CLASS ───────────────────────────────────
class AccidentDetectionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Accident Detection System")
        self.root.configure(bg=BG)
        self.root.geometry("1180x720")
        self.root.minsize(900, 600)
        self.root.resizable(True, True)

        self.running       = False
        self.cap           = None
        self.model         = None
        self.alarm_playing = False
        self.total_alerts  = 0
        self.frames_seen   = 0
        self.alarm_sound   = None
        self._thread       = None
        self._current_photo= None

        self._build_ui()
        self._load_model_async()

    # ─────────────────────────── LAYOUT ────────────────────────────────────
    def _build_ui(self):
        # ── top bar ──────────────────────────────────────────────────────
        top = tk.Frame(self.root, bg=BG, pady=12)
        top.pack(fill="x", padx=20)

        tk.Label(top, text="Accident Detection System",
                 font=("Segoe UI", 18, "bold"), bg=BG, fg=TEXT).pack(side="left")
        tk.Label(top, text="  YOLOv8 Live", font=("Segoe UI", 9),
                 bg=ACCENT2, fg="white", padx=8, pady=2).pack(side="left", padx=12)

        self.model_label = tk.Label(top, text="Loading model...",
                                    font=("Segoe UI", 9), bg=BG, fg=SUBTEXT)
        self.model_label.pack(side="right")

        # ── main body ────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        self._build_video_panel(left)
        self._build_controls(left)

        right = tk.Frame(body, bg=BG, width=300)
        right.pack(side="right", fill="y", padx=(12, 0))
        right.pack_propagate(False)
        self._build_status_card(right)
        self._build_stats_card(right)
        self._build_log_card(right)

    def _build_video_panel(self, parent):
        frame = tk.Frame(parent, bg=PANEL, highlightbackground=BORDER,
                         highlightthickness=1)
        frame.pack(fill="both", expand=True)

        hdr = tk.Frame(frame, bg=CARD, pady=8)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Live Feed", font=("Segoe UI", 10, "bold"),
                 bg=CARD, fg=TEXT, padx=12).pack(side="left")
        self.feed_status = tk.Label(hdr, text="OFFLINE",
                                    font=("Segoe UI", 9, "bold"),
                                    bg=CARD, fg=SUBTEXT, padx=12)
        self.feed_status.pack(side="right")

        self.canvas = tk.Canvas(frame, bg="#080a10", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._draw_placeholder())
        self._draw_placeholder()

    def _draw_placeholder(self):
        self.canvas.update_idletasks()
        w = self.canvas.winfo_width() or 640
        h = self.canvas.winfo_height() or 400
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, w, h, fill="#080a10", outline="")
        self.canvas.create_text(w//2, h//2 - 10,
                                text="Camera feed will appear here",
                                font=("Segoe UI", 13), fill="#2a3045")
        self.canvas.create_text(w//2, h//2 + 18,
                                text="Click  Start Detection  to begin",
                                font=("Segoe UI", 10), fill="#1e2535")

    def _build_controls(self, parent):
        bar = tk.Frame(parent, bg=BG, pady=10)
        bar.pack(fill="x")

        self.start_btn = tk.Button(
            bar, text="Start Detection", font=("Segoe UI", 11, "bold"),
            bg=ACCENT, fg="white", relief="flat", cursor="hand2",
            activebackground="#3a7aee", activeforeground="white",
            padx=20, pady=8, command=self._start)
        self.start_btn.pack(side="left", padx=(0, 8))

        self.stop_btn = tk.Button(
            bar, text="Stop", font=("Segoe UI", 11, "bold"),
            bg=CARD, fg=SUBTEXT, relief="flat", cursor="hand2",
            activebackground=DANGER_CLR, activeforeground="white",
            padx=20, pady=8, state="disabled", command=self._stop)
        self.stop_btn.pack(side="left", padx=(0, 8))

        tk.Label(bar, text="Camera index:", font=("Segoe UI", 9),
                 bg=BG, fg=SUBTEXT).pack(side="left", padx=(16, 4))
        self.cam_var = tk.StringVar(value="0")
        tk.Entry(bar, textvariable=self.cam_var, width=4,
                 font=("Segoe UI", 10), bg=CARD, fg=TEXT,
                 insertbackground=TEXT, relief="flat", bd=4).pack(side="left")

    def _build_status_card(self, parent):
        card = tk.Frame(parent, bg=CARD, highlightbackground=BORDER,
                        highlightthickness=1)
        card.pack(fill="x", pady=(0, 10))

        tk.Label(card, text="Detection Status",
                 font=("Segoe UI", 9, "bold"),
                 bg=CARD, fg=SUBTEXT, pady=8, padx=12).pack(anchor="w")

        badge = tk.Frame(card, bg=CARD, pady=10)
        badge.pack(fill="x", padx=12, pady=(0, 4))

        self.status_dot = tk.Canvas(badge, width=16, height=16,
                                    bg=CARD, highlightthickness=0)
        self.status_dot.pack(side="left", padx=(0, 8))
        self._dot_oval = self.status_dot.create_oval(2, 2, 14, 14,
                                                     fill=SUBTEXT, outline="")

        self.status_text = tk.Label(badge, text="STANDBY",
                                    font=("Segoe UI", 14, "bold"),
                                    bg=CARD, fg=SUBTEXT)
        self.status_text.pack(side="left")

        tk.Label(card, text="Confidence", font=("Segoe UI", 9),
                 bg=CARD, fg=SUBTEXT, padx=12).pack(anchor="w", pady=(4, 0))

        self._conf_canvas = tk.Canvas(card, bg="#0d0f14", height=16,
                                      highlightthickness=0)
        self._conf_canvas.pack(fill="x", padx=12, pady=(4, 12))
        self._draw_conf_bar(0.0)

    def _draw_conf_bar(self, value: float):
        c = self._conf_canvas
        c.update_idletasks()
        W = c.winfo_width() or 260
        c.delete("all")
        c.create_rectangle(0, 0, W, 16, fill="#1a1e2a", outline="")
        colour = DANGER_CLR if value >= 0.5 else SAFE_CLR
        fill_w = int(W * min(value, 1.0))
        if fill_w > 0:
            c.create_rectangle(0, 0, fill_w, 16, fill=colour, outline="")
        c.create_text(W // 2, 8, text=f"{value * 100:.1f}%",
                      font=("Segoe UI", 8, "bold"), fill="white")

    def _build_stats_card(self, parent):
        card = tk.Frame(parent, bg=CARD, highlightbackground=BORDER,
                        highlightthickness=1)
        card.pack(fill="x", pady=(0, 10))

        tk.Label(card, text="Session Stats",
                 font=("Segoe UI", 9, "bold"),
                 bg=CARD, fg=SUBTEXT, pady=8, padx=12).pack(anchor="w")

        grid = tk.Frame(card, bg=CARD)
        grid.pack(fill="x", padx=12, pady=(0, 12))
        self.stat_frames = self._stat_item(grid, "Frames", "0", 0)
        self.stat_alerts = self._stat_item(grid, "Alerts", "0", 1)

    def _stat_item(self, parent, label, value, col):
        f = tk.Frame(parent, bg=BORDER)
        f.grid(row=0, column=col, padx=(0 if col == 0 else 6, 0), pady=8, sticky="ew", ipadx=10)
        parent.columnconfigure(col, weight=1)
        lbl = tk.Label(f, text=value, font=("Segoe UI", 20, "bold"),
                       bg=BORDER, fg=TEXT)
        lbl.pack()
        tk.Label(f, text=label, font=("Segoe UI", 8), bg=BORDER,
                 fg=SUBTEXT).pack()
        return lbl

    def _build_log_card(self, parent):
        card = tk.Frame(parent, bg=CARD, highlightbackground=BORDER,
                        highlightthickness=1)
        card.pack(fill="both", expand=True)

        hdr = tk.Frame(card, bg=CARD)
        hdr.pack(fill="x", pady=(8, 0), padx=12)
        tk.Label(hdr, text="Detection Log",
                 font=("Segoe UI", 9, "bold"),
                 bg=CARD, fg=SUBTEXT).pack(side="left")
        tk.Button(hdr, text="Clear", font=("Segoe UI", 8),
                  bg=CARD, fg=SUBTEXT, relief="flat",
                  cursor="hand2", activebackground=CARD,
                  command=self._clear_log).pack(side="right")

        container = tk.Frame(card, bg=CARD)
        container.pack(fill="both", expand=True, padx=4, pady=(4, 8))

        sb = tk.Scrollbar(container)
        sb.pack(side="right", fill="y")

        self.log_text = tk.Text(container, bg="#0d0f14", fg=TEXT,
                                font=("Consolas", 8), relief="flat",
                                state="disabled", wrap="word",
                                selectbackground=ACCENT2, bd=0,
                                padx=8, pady=6, yscrollcommand=sb.set)
        self.log_text.pack(fill="both", expand=True)
        sb.config(command=self.log_text.yview)

        self.log_text.tag_config("danger", foreground=DANGER_CLR)
        self.log_text.tag_config("safe",   foreground=SAFE_CLR)
        self.log_text.tag_config("time",   foreground=SUBTEXT)
        self.log_text.tag_config("info",   foreground=ACCENT)

    # ──────────────────────────── LOGGING ──────────────────────────────────
    def _log(self, msg: str, tag: str = "safe"):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.config(state="normal")
        self.log_text.insert("end", f"[{ts}] ", "time")
        self.log_text.insert("end", msg + "\n", tag)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    # ──────────────────────── MODEL LOADING ────────────────────────────────
    def _load_model_async(self):
        def _load():
            try:
                self.model_label.config(text="Loading model...", fg=SUBTEXT)
                self.model = YOLO("best.pt")
                pygame.mixer.init()
                self.alarm_sound = pygame.mixer.Sound("alarm.wav")
                self.root.after(0, lambda: self.model_label.config(
                    text="Model ready", fg=SAFE_CLR))
                self.root.after(0, lambda: self._log("Model loaded (best.pt)", "info"))
            except Exception as e:
                self.root.after(0, lambda err=e: self.model_label.config(
                    text=f"Error: {err}", fg=DANGER_CLR))
        threading.Thread(target=_load, daemon=True).start()

    # ───────────────────────── START / STOP ────────────────────────────────
    def _start(self):
        if self.model is None:
            self._log("Model not ready yet!", "danger")
            return

        cam_src = int(self.cam_var.get()) if self.cam_var.get().isdigit() else 0
        self.cap = cv2.VideoCapture(cam_src)
        if not self.cap.isOpened():
            self._log(f"Cannot open camera {cam_src}", "danger")
            return

        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal", bg=DANGER_CLR, fg="white")
        self.feed_status.config(text="LIVE", fg=SAFE_CLR)
        self._log(f"Started — camera: {cam_src}", "info")

        self._thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._thread.start()

    def _stop(self):
        self.running = False
        if self.alarm_sound and self.alarm_playing:
            self.alarm_sound.stop()
            self.alarm_playing = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled", bg=CARD, fg=SUBTEXT)
        self.feed_status.config(text="OFFLINE", fg=SUBTEXT)
        self._update_status(False, 0.0)
        self.root.after(300, self._draw_placeholder)
        self._log("Detection stopped.", "info")

    # ───────────────────────── DETECTION LOOP ──────────────────────────────
    def _detection_loop(self):
        output_folder = os.path.join(os.getcwd(), "runs", "detect", "outputs")
        os.makedirs(output_folder, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fw = int(self.cap.get(3))
        fh = int(self.cap.get(4))
        out_path = os.path.join(output_folder, "output_video.avi")
        writer = cv2.VideoWriter(out_path, fourcc, 20.0, (fw, fh))

        while self.running and self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                break

            results   = self.model(frame, verbose=False)
            annotated = results[0].plot()
            boxes     = results[0].boxes.cpu().numpy()

            detected = False
            max_conf = 0.0
            for box in boxes:
                c = float(box.conf[0])
                if c > max_conf:
                    max_conf = c
                if c > 0.5:
                    detected = True

            self.frames_seen += 1
            if detected:
                self.total_alerts += 1
                if self.alarm_sound and not self.alarm_playing:
                    self.alarm_sound.play(-1)
                    self.alarm_playing = True
                self.root.after(0, lambda conf=max_conf: self._log(
                    f"ACCIDENT detected  conf={conf:.2f}", "danger"))
            else:
                if self.alarm_sound and self.alarm_playing:
                    self.alarm_sound.stop()
                    self.alarm_playing = False

            writer.write(annotated)

            frame_copy = annotated.copy()
            self.root.after(0, lambda f=frame_copy, c=max_conf, d=detected,
                            fs=self.frames_seen, ta=self.total_alerts:
                            self._update_ui(f, c, d, fs, ta))

        writer.release()
        if self.cap:
            self.cap.release()
        self.root.after(0, lambda: self._log(f"Saved output to {out_path}", "info"))

    # ──────────────────────── UI UPDATES ───────────────────────────────────
    def _update_ui(self, frame, conf, detected, frames, alerts):
        self._update_canvas(frame)
        self._update_status(detected, conf)
        self.stat_frames.config(text=str(frames))
        self.stat_alerts.config(text=str(alerts),
                                fg=DANGER_CLR if alerts > 0 else TEXT)

    def _update_canvas(self, frame_bgr):
        self.canvas.update_idletasks()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.thumbnail((cw, ch), Image.LANCZOS)
        x = (cw - img.width)  // 2
        y = (ch - img.height) // 2
        photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(x, y, anchor="nw", image=photo)
        self._current_photo = photo

    def _update_status(self, detected: bool, conf: float):
        if detected:
            self.status_dot.itemconfig(self._dot_oval, fill=DANGER_CLR)
            self.status_text.config(text="ACCIDENT!", fg=DANGER_CLR)
            self.root.title("*** ACCIDENT DETECTED *** — Accident Detection System")
        else:
            colour = SAFE_CLR if self.running else SUBTEXT
            label  = "MONITORING" if self.running else "STANDBY"
            self.status_dot.itemconfig(self._dot_oval, fill=colour)
            self.status_text.config(text=label, fg=colour)
            self.root.title("Accident Detection System")
        self._draw_conf_bar(conf)


# ─────────────────────────────── MAIN ──────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()

    # Centre window on screen and force to front
    root.update_idletasks()
    w, h = 1180, 720
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = (sw - w) // 2
    y = (sh - h) // 2
    root.geometry(f"{w}x{h}+{x}+{y}")
    root.lift()
    root.attributes("-topmost", True)
    root.after(500, lambda: root.attributes("-topmost", False))  # release topmost after 0.5s

    app  = AccidentDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app._stop(), root.destroy()))
    root.mainloop()
