"""
sop_gui.py - Interactive Tkinter GUI for SOP Sequence Detection.

Usage:
    python sop_gui.py
"""

import os
import sys
import threading
import queue
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk

# Setup data path before importing sop_inference
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "sop_inference" / "_data"
if not DATA_DIR.exists():
    DATA_DIR = HERE / "_data"
os.environ["SOP_MONITOR_HOME"] = str(DATA_DIR)

sys.path.insert(0, str(HERE))
from sop_inference import SOPMonitor

MODEL_NAME = "Assembling_box_Good1_train_18Nov_1719"


class SOPApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SOP Sequence Detection Monitor")
        self.root.geometry("1200x720")
        self.root.configure(bg="#1e1e2e")

        self.mon = None
        self.cap = None
        self.running = False
        self.cycle_active = False
        self.thread = None
        self.frame_queue = queue.Queue(maxsize=2)

        self._build_ui()
        self._load_model_async()

    def _build_ui(self):
        # Top Header Bar
        header = tk.Frame(self.root, bg="#181825", height=50)
        header.pack(fill=tk.X, side=tk.TOP)
        self.lbl_title = tk.Label(header, text="SOP Sequence Monitor", fg="#cdd6f4", bg="#181825",
                                  font=("Segoe UI", 16, "bold"))
        self.lbl_title.pack(side=tk.LEFT, padx=16, pady=8)

        self.lbl_model_status = tk.Label(header, text="Loading model...", fg="#f9e2af", bg="#181825",
                                         font=("Segoe UI", 11))
        self.lbl_model_status.pack(side=tk.RIGHT, padx=16, pady=8)

        # Main Layout: Left = Video, Right = Steps Panel
        content = tk.Frame(self.root, bg="#1e1e2e")
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Video Frame
        left_panel = tk.Frame(content, bg="#11111b", bd=2, relief=tk.GROOVE)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.video_label = tk.Label(left_panel, bg="#11111b", text="Initializing Camera...", fg="#6c7086",
                                    font=("Segoe UI", 14))
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Right Panel: Sequence & Controls
        right_panel = tk.Frame(content, bg="#181825", width=420)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)

        # Status Badge
        self.lbl_state = tk.Label(right_panel, text="STANDBY", bg="#313244", fg="#cdd6f4",
                                  font=("Segoe UI", 13, "bold"), pady=6)
        self.lbl_state.pack(fill=tk.X, padx=12, pady=(12, 6))

        # Steps Tree / Checklist
        tk.Label(right_panel, text="SOP Sequence Checklist:", bg="#181825", fg="#a6adc8",
                 font=("Segoe UI", 11, "bold"), anchor="w").pack(fill=tk.X, padx=12, pady=(6, 2))

        self.steps_container = tk.Frame(right_panel, bg="#181825")
        self.steps_container.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        self.step_widgets = {}

        # Verdict / Anomaly Info
        self.lbl_verdict = tk.Label(right_panel, text="Verdict: -", bg="#181825", fg="#cdd6f4",
                                    font=("Segoe UI", 12, "bold"), anchor="w")
        self.lbl_verdict.pack(fill=tk.X, padx=12, pady=(4, 0))

        self.txt_anomalies = tk.Label(right_panel, text="", bg="#181825", fg="#f38ba8",
                                      font=("Segoe UI", 9), anchor="nw", justify=tk.LEFT, wraplength=380)
        self.txt_anomalies.pack(fill=tk.X, padx=12, pady=(2, 8))

        # Bottom Controls
        controls = tk.Frame(right_panel, bg="#181825")
        controls.pack(fill=tk.X, padx=12, pady=12, side=tk.BOTTOM)

        self.btn_start = tk.Button(controls, text="▶ Start Cycle", font=("Segoe UI", 11, "bold"),
                                   bg="#a6e3a1", fg="#11111b", command=self.start_cycle, state=tk.DISABLED,
                                   relief=tk.FLAT, pady=6)
        self.btn_start.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        self.btn_stop = tk.Button(controls, text="⏹ Stop / Check", font=("Segoe UI", 11, "bold"),
                                  bg="#f38ba8", fg="#11111b", command=self.stop_cycle, state=tk.DISABLED,
                                  relief=tk.FLAT, pady=6)
        self.btn_stop.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(6, 0))

    def _load_model_async(self):
        def worker():
            try:
                mon = SOPMonitor(model=MODEL_NAME)
                self.root.after(0, self._on_model_loaded, mon)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load model:\n{e}"))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _on_model_loaded(self, mon):
        self.mon = mon
        self.lbl_model_status.config(text=f"Model: {MODEL_NAME[:28]} (Ready)", fg="#a6e3a1")
        self._populate_steps(mon.steps)
        self._start_camera_stream()

    def _populate_steps(self, steps):
        for widget in self.steps_container.winfo_children():
            widget.destroy()
        self.step_widgets.clear()

        for s in steps:
            sid = s["step"]
            row = tk.Frame(self.steps_container, bg="#313244", pady=4, padx=6)
            row.pack(fill=tk.X, pady=3)

            icon = tk.Label(row, text="⚪", bg="#313244", fg="#6c7086", font=("Segoe UI", 12))
            icon.pack(side=tk.LEFT, padx=(0, 6))

            desc = tk.Label(row, text=f"{sid}. {s['label']}", bg="#313244", fg="#cdd6f4",
                            font=("Segoe UI", 10), anchor="w")
            desc.pack(side=tk.LEFT, fill=tk.X, expand=True)

            status = tk.Label(row, text="Pending", bg="#313244", fg="#a6adc8", font=("Segoe UI", 9))
            status.pack(side=tk.RIGHT, padx=4)

            self.step_widgets[sid] = {"row": row, "icon": icon, "status": status, "label": s["label"]}

    def _start_camera_stream(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.btn_start.config(state=tk.NORMAL)
        self.thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.thread.start()
        self._poll_ui()

    def _capture_worker(self):
        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    continue

                status = None
                if self.cycle_active and self.mon:
                    try:
                        status = self.mon.process_frame(frame)
                    except Exception as e:
                        print("process_frame error:", e)

                # Draw HUD if active
                if self.cycle_active and self.mon and status:
                    display_bgr = self.mon.draw(frame, status)
                else:
                    display_bgr = frame

                # Put in UI queue
                if not self.frame_queue.full():
                    self.frame_queue.put((display_bgr, status))

    def _poll_ui(self):
        try:
            while not self.frame_queue.empty():
                frame_bgr, status = self.frame_queue.get_nowait()
                # Convert BGR to RGB for Tkinter
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                h, w, _ = rgb.shape
                # Scale to fit window while keeping aspect ratio
                max_w, max_h = 760, 560
                scale = min(max_w / w, max_h / h)
                nw, nh = int(w * scale), int(h * scale)
                img = Image.fromarray(cv2.resize(rgb, (nw, nh)))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

                if status:
                    self._update_status_ui(status)
        except Exception:
            pass

        if self.running:
            self.root.after(30, self._poll_ui)

    def _update_status_ui(self, status):
        # Update badge
        self.lbl_state.config(text=f"{status.state}  ({status.fps:.1f} FPS)")
        if status.state == "OUT OF ORDER":
            self.lbl_state.config(bg="#f38ba8", fg="#11111b")
        elif status.state == "STEP IN PROGRESS":
            self.lbl_state.config(bg="#f9e2af", fg="#11111b")
        else:
            self.lbl_state.config(bg="#89b4fa", fg="#11111b")

        # Update step list
        done_ids = {d["step"]: d for d in status.steps_done}
        doing_ids = set(status.doing)

        for sid, widgets in self.step_widgets.items():
            if sid in done_ids:
                info = done_ids[sid]
                in_order = info.get("in_order", True)
                if in_order:
                    widgets["icon"].config(text="🟢", fg="#a6e3a1")
                    widgets["status"].config(text=f"Done ({info['t']:.1f}s)", fg="#a6e3a1")
                    widgets["row"].config(bg="#1e3a2f")
                else:
                    widgets["icon"].config(text="🔴", fg="#f38ba8")
                    widgets["status"].config(text="OUT OF ORDER", fg="#f38ba8")
                    widgets["row"].config(bg="#3b1e28")
            elif sid in doing_ids:
                widgets["icon"].config(text="🟡", fg="#f9e2af")
                widgets["status"].config(text="In Progress...", fg="#f9e2af")
                widgets["row"].config(bg="#3a3620")
            else:
                widgets["icon"].config(text="⚪", fg="#6c7086")
                widgets["status"].config(text="Pending", fg="#a6adc8")
                widgets["row"].config(bg="#313244")

    def start_cycle(self):
        if not self.mon:
            return
        self.mon.start("cycle")
        self.cycle_active = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.lbl_verdict.config(text="Verdict: Inspecting...", fg="#89b4fa")
        self.txt_anomalies.config(text="")
        for sid, widgets in self.step_widgets.items():
            widgets["icon"].config(text="⚪", fg="#6c7086")
            widgets["status"].config(text="Pending", fg="#a6adc8")
            widgets["row"].config(bg="#313244")

    def stop_cycle(self):
        if not self.mon or not self.cycle_active:
            return
        self.cycle_active = False
        res = self.mon.stop()
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

        # Update Verdict Display
        if res.verdict == "NORMAL":
            self.lbl_verdict.config(text="Verdict: NORMAL (PASS)", fg="#a6e3a1")
            self.lbl_state.config(text="CYCLE PASSED", bg="#a6e3a1", fg="#11111b")
        elif res.verdict == "ANOMALY":
            self.lbl_verdict.config(text="Verdict: ANOMALY (FAIL)", fg="#f38ba8")
            self.lbl_state.config(text="ANOMALY DETECTED", bg="#f38ba8", fg="#11111b")
        else:
            self.lbl_verdict.config(text=f"Verdict: {res.verdict}", fg="#f9e2af")
            self.lbl_state.config(text="INSPECTION FINISHED", bg="#f9e2af", fg="#11111b")

        if res.anomalies:
            self.txt_anomalies.config(text="\n".join(f"• {a}" for a in res.anomalies))
        else:
            self.txt_anomalies.config(text="All steps completed successfully in sequence.")

    def close(self):
        self.running = False
        self.cycle_active = False
        if self.cap:
            self.cap.release()
        if self.mon:
            self.mon.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SOPApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()
