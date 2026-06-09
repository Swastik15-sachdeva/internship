"""
dashboard.py — Recruiter live monitoring dashboard (CustomTkinter)
Run from main.py, not directly.
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import threading
import time
from datetime import datetime
from database.database import get_risk_score
from modules.report_generator import generate_report

ctk.set_appearance_mode("system")
ctk.set_default_color_theme("blue")

RISK_COLORS = {
    "safe":    ("#3B6D11", "#EAF3DE"),
    "review":  ("#854F0B", "#FAEEDA"),
    "high":    ("#A32D2D", "#FCEBEB"),
}

def risk_band(score: int) -> str:
    if score <= 20:   return "safe"
    elif score <= 50: return "review"
    else:             return "high"

def risk_label(score: int) -> str:
    b = risk_band(score)
    return {"safe": "Safe", "review": "Needs Review", "high": "High Risk"}[b]


class Dashboard(ctk.CTk):
    def __init__(self, candidate_name: str, monitor_thread=None):
        super().__init__()
        self.candidate = candidate_name
        self.monitor_thread = monitor_thread
        self._running = True

        self.title(f"AI Recruiter — {candidate_name}")
        self.geometry("820x680")
        self.resizable(True, True)

        self._build_ui()
        self._start_refresh()

    # ── UI construction ──────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header bar
        header = ctk.CTkFrame(self, corner_radius=0, height=56)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            header, text="🎯  AI Recruiter",
            font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, padx=20, pady=10)

        self.candidate_label = ctk.CTkLabel(
            header, text=f"Candidate: {self.candidate}",
            font=ctk.CTkFont(size=13)
        )
        self.candidate_label.grid(row=0, column=1, padx=10)

        self.time_label = ctk.CTkLabel(
            header, text="",
            font=ctk.CTkFont(size=12)
        )
        self.time_label.grid(row=0, column=2, padx=20)

        # Main content area
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=1, column=0, sticky="nsew", padx=12, pady=8)
        main.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(1, weight=1)
        main.grid_rowconfigure(2, weight=1)

        # ── Metric cards ─────────────────────────────────────
        metrics_frame = ctk.CTkFrame(main, fg_color="transparent")
        metrics_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        for i in range(5):
            metrics_frame.grid_columnconfigure(i, weight=1)

        self.metric_vars = {}
        metric_defs = [
            ("look_away", "Looking Away", "👀"),
            ("talking",   "Talking",      "🗣️"),
            ("phone",     "Phone",        "📱"),
            ("no_face",   "No Face",      "❌"),
            ("multi",     "Multi-Person", "👥"),
        ]
        self._metric_cards = {}
        for col, (key, label, icon) in enumerate(metric_defs):
            card = ctk.CTkFrame(metrics_frame, corner_radius=10)
            card.grid(row=0, column=col, padx=4, sticky="ew")
            var = ctk.StringVar(value="0")
            self.metric_vars[key] = var
            ctk.CTkLabel(card, text=icon, font=ctk.CTkFont(size=22)).pack(pady=(10, 2))
            ctk.CTkLabel(card, textvariable=var,
                         font=ctk.CTkFont(size=26, weight="bold")).pack()
            ctk.CTkLabel(card, text=label,
                         font=ctk.CTkFont(size=11)).pack(pady=(2, 10))
            self._metric_cards[key] = card

        # ── Identity + Risk score ─────────────────────────────
        status_frame = ctk.CTkFrame(main, corner_radius=10)
        status_frame.grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(0, 8))
        status_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(status_frame, text="Identity",
                     font=ctk.CTkFont(size=12)).grid(row=0, column=0, padx=14, pady=(12,2))
        self.identity_label = ctk.CTkLabel(
            status_frame, text="Checking…",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.identity_label.grid(row=1, column=0, padx=14, pady=(0, 12))

        ctk.CTkLabel(status_frame, text="Risk Score",
                     font=ctk.CTkFont(size=12)).grid(row=0, column=1, padx=14, pady=(12,2))
        self.score_label = ctk.CTkLabel(
            status_frame, text="0",
            font=ctk.CTkFont(size=26, weight="bold")
        )
        self.score_label.grid(row=1, column=1, padx=14, pady=(0, 12))

        self.risk_band_label = ctk.CTkLabel(
            status_frame, text="Safe",
            font=ctk.CTkFont(size=12)
        )
        self.risk_band_label.grid(row=1, column=2, padx=14, pady=(0, 12))

        self.risk_bar = ctk.CTkProgressBar(status_frame, width=160)
        self.risk_bar.set(0)
        self.risk_bar.grid(row=0, column=2, rowspan=2, padx=14, pady=12)

        # ── Current status ────────────────────────────────────
        cur_frame = ctk.CTkFrame(main, corner_radius=10)
        cur_frame.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(0, 8))
        ctk.CTkLabel(cur_frame, text="Current Detection",
                     font=ctk.CTkFont(size=12)).pack(anchor="w", padx=14, pady=(10,2))
        self.current_status = ctk.CTkLabel(
            cur_frame, text="Waiting for session to start…",
            font=ctk.CTkFont(size=12), wraplength=260, justify="left"
        )
        self.current_status.pack(anchor="w", padx=14, pady=(0, 12))

        # ── Alert log ─────────────────────────────────────────
        log_frame = ctk.CTkFrame(main, corner_radius=10)
        log_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(0, 8))
        log_frame.grid_rowconfigure(1, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(log_frame, text="Alert Timeline",
                     font=ctk.CTkFont(size=13, weight="bold")).grid(
                         row=0, column=0, sticky="w", padx=14, pady=(10, 4))

        self.log_box = ctk.CTkTextbox(log_frame, font=ctk.CTkFont(family="Courier", size=12))
        self.log_box.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.log_box.configure(state="disabled")

        # ── Buttons ───────────────────────────────────────────
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 10))

        ctk.CTkButton(
            btn_frame, text="📄  Generate Report",
            command=self._generate_report, width=180
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            btn_frame, text="🔴  End Session",
            command=self._end_session, fg_color="#A32D2D",
            hover_color="#791F1F", width=140
        ).pack(side="right", padx=4)

    # ── Update methods ───────────────────────────────────────

    def update_status(self, status_dict: dict):
        """Called from monitoring thread via after(). status_dict keys:
        identity, face_count, direction, talking, phone, violations, score
        """
        # Identity
        identity = status_dict.get("identity", "checking")
        colors_map = {
            "match":    ("✓ Verified",     "#3B6D11"),
            "mismatch": ("✗ Mismatch",     "#A32D2D"),
            "no_face":  ("No face",        "#854F0B"),
            "unknown":  ("Unknown",        "#888780"),
            "checking": ("Checking…",      "#888780"),
        }
        id_text, id_color = colors_map.get(identity, ("Unknown", "#888780"))
        self.identity_label.configure(text=id_text, text_color=id_color)

        # Risk score
        score = status_dict.get("score", 0)
        self.score_label.configure(text=str(score))
        band = risk_band(score)
        self.risk_band_label.configure(text=risk_label(score))
        self.risk_bar.set(min(score / 100, 1.0))

        # Current detection summary
        direction = status_dict.get("direction", "forward")
        talking   = status_dict.get("talking", False)
        phone     = status_dict.get("phone", False)
        faces     = status_dict.get("face_count", 0)
        lines = []
        if direction != "forward":
            lines.append(f"👀 Looking {direction}")
        if talking:
            lines.append("🗣️ Talking detected")
        if phone:
            lines.append("📱 Phone in frame!")
        if faces == 0:
            lines.append("❌ No face detected")
        elif faces > 1:
            lines.append(f"👥 {faces} persons in frame")
        if not lines:
            lines.append("✓ All clear")
        self.current_status.configure(text="\n".join(lines))

        # Metric counts
        violations = status_dict.get("violations", [])
        counts = {k: 0 for k in self.metric_vars}
        event_map = {
            "Looking Away":   "look_away",
            "Talking":        "talking",
            "Phone Detected": "phone",
            "No Face":        "no_face",
            "Multiple Faces": "multi",
        }
        for v in violations:
            key = event_map.get(v["event"])
            if key:
                counts[key] += 1
        for k, var in self.metric_vars.items():
            var.set(str(counts[k]))

    def add_log_entry(self, text: str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _start_refresh(self):
        """Refresh clock every second."""
        def tick():
            while self._running:
                now = datetime.now().strftime("%H:%M:%S")
                self.time_label.configure(text=now)
                time.sleep(1)
        t = threading.Thread(target=tick, daemon=True)
        t.start()

    def _generate_report(self):
        try:
            path = generate_report(self.candidate)
            messagebox.showinfo("Report saved", f"PDF report saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not generate report:\n{e}")

    def _end_session(self):
        if messagebox.askyesno("End session", "End monitoring and generate report?"):
            self._running = False
            try:
                path = generate_report(self.candidate)
                messagebox.showinfo("Session ended", f"Report saved:\n{path}")
            except Exception as e:
                print(f"[Dashboard] Report error: {e}")
            self.destroy()
