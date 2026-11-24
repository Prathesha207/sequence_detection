"""
sequence_inference.py - Lean ML Orchestrator for SOP Sequence Detection.
Coordinates GPU model lifecycle, session mutual exclusion, and frame dispatch to sop_inference.
Returns ONLY what the ML models (SOPMonitor / TCN / Action Matcher) actually produce.
"""
from __future__ import annotations

import contextlib
import dataclasses
import importlib
import json
import logging
import math
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from app.core.logger import setup_logger
    logger = setup_logger("sequence-inference")
except Exception:
    logger = logging.getLogger("sequence-inference")

__all__ = [
    "SequenceInferenceError", "SequenceModelError", "SequenceBusyError",
    "SequenceConfig", "configure", "get_config", "prepare_environment",
    "list_models", "resolve_model_name", "preflight", "model_steps",
    "SequenceEngine", "SequenceSession", "FrameResult", "FinalResult",
    "get_engine", "set_engine", "preload", "shutdown", "merge_final", "blank_stats",
    "draw_final",
]

class SequenceInferenceError(RuntimeError): pass
class SequenceModelError(SequenceInferenceError): pass
class SequenceBusyError(SequenceInferenceError): pass

@dataclass
class SequenceConfig:
    model_name: Optional[str] = None
    data_home: Optional[str] = None
    save_results: bool = False
    out_dir: Optional[str] = None
    check_every_s: float = 0.5
    stale_session_s: float = 300.0
    stream_annotated: bool = os.environ.get("SOP_STREAM_ANNOTATED", "0") == "1"

_config = SequenceConfig()

def configure(**kw) -> SequenceConfig:
    for k, v in kw.items():
        if hasattr(_config, k): setattr(_config, k, v)
    return _config

def get_config() -> SequenceConfig: return _config

_env_lock, _env_info, _infer_mod = threading.Lock(), None, None

# Set default SOP_MONITOR_HOME immediately if not set
if not os.environ.get("SOP_MONITOR_HOME"):
    for cand in [
        Path("d:/sequence_detection/ml/sop_inference/_data"),
        Path(__file__).resolve().parent / "sop_inference" / "_data",
        Path(__file__).resolve().parents[3] / "ml" / "sop_inference" / "_data",
    ]:
        if (cand / "models.json").is_file():
            os.environ["SOP_MONITOR_HOME"] = str(cand.resolve())
            break

def prepare_environment() -> Dict[str, Any]:
    global _env_info
    with _env_lock:
        if _env_info is not None: return _env_info
        explicit = _config.data_home or os.environ.get("SOP_MONITOR_HOME")
        if explicit:
            home = Path(explicit).expanduser().resolve()
        else:
            candidates = [
                Path("d:/sequence_detection/ml/sop_inference/_data"),
                Path(__file__).resolve().parent / "sop_inference" / "_data",
                Path(__file__).resolve().parents[3] / "ml" / "sop_inference" / "_data",
                (Path.home() / "SOP_Inference").resolve(),
            ]
            home = next((c for c in candidates if (c / "models.json").is_file()), candidates[0])
        try:
            home.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        os.environ["SOP_MONITOR_HOME"] = str(home)
        _env_info = {"home": str(home)}
        logger.info(f"[SEQ] Data folder: {home}")
        return _env_info

def import_infer():
    global _infer_mod
    if _infer_mod is not None: return _infer_mod
    prepare_environment()
    # Ensure local ML source trees are in sys.path as resilient fallback
    for p in [
        str(Path(__file__).resolve().parent),
        str(Path(__file__).resolve().parents[3] / "ml"),
        "d:/sequence_detection/ml",
    ]:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    for name in [os.environ.get("SOP_INFER_MODULE"), "sop_inference.infer", "sop_inference", "infer"]:
        if not name: continue
        try:
            mod = importlib.import_module(name)
            if hasattr(mod, "SOPMonitor"):
                _infer_mod = mod
                logger.info(f"[SEQ] ML module loaded: {name}")
                return mod
        except Exception as err:
            logger.warning(f"[SEQ] Failed importing {name}: {err}")
            continue
    raise SequenceModelError("Cannot import SOPMonitor. Ensure sop_inference is installed.")

def _torch():
    try:
        import torch; return torch
    except ImportError: return None

def _home() -> Path: return Path(prepare_environment()["home"])

def list_models(home: Optional[Path] = None) -> List[Dict[str, Any]]:
    home = Path(home) if home else _home()
    d = home / "trained"
    act_file = d / "active_model.txt"
    active = act_file.read_text(encoding="utf-8").strip() if act_file.is_file() else None
    out = []
    if d.is_dir():
        for p in sorted(d.iterdir()):
            f = p / "model.json"
            if not f.is_file(): continue
            try:
                m = json.loads(f.read_text(encoding="utf-8"))
                out.append({"name": p.name, "trained_at": m.get("trained_at", ""),
                            "steps": len((m.get("spec") or {}).get("steps") or []),
                            "fps": m.get("fps"), "detectors": m.get("detectors") or [],
                            "has_action_model": (p / "action_matcher.pt").is_file(),
                            "active": p.name == active})
            except Exception as e:
                out.append({"name": p.name, "error": str(e), "active": False})
    out.sort(key=lambda m: m.get("trained_at", ""), reverse=True)
    return out

def resolve_model_name(requested: Optional[str] = None, home: Optional[Path] = None) -> str:
    home = Path(home) if home else _home()
    models = [m for m in list_models(home) if "error" not in m]
    if not models: raise SequenceModelError(f"No trained models in {home / 'trained'}")
    names = [m["name"] for m in models]
    want = (requested or os.environ.get("SOP_MODEL_NAME") or _config.model_name or "").strip()
    if want:
        if want in names: return want
        low = {n.lower(): n for n in names}
        if want.lower() in low: return low[want.lower()]
        pref = [n for n in names if n.lower().startswith(want.lower())]
        if pref: return pref[0]
        raise SequenceModelError(f"Model '{want}' not found. Available: {', '.join(names)}")
    act_file = home / "trained" / "active_model.txt"
    if act_file.is_file():
        act = act_file.read_text(encoding="utf-8").strip()
        if act in names: return act
    return names[0]

def model_steps(name: Optional[str] = None, home: Optional[Path] = None) -> List[Dict[str, Any]]:
    home = Path(home) if home else _home()
    name = resolve_model_name(name, home)
    m = json.loads((home / "trained" / name / "model.json").read_text(encoding="utf-8"))
    return [{"step": s["id"], "label": s.get("action_query") or s.get("text") or f"step {s['id']}"}
            for s in (m.get("spec") or {}).get("steps") or []]

def preflight(name: Optional[str] = None, home: Optional[Path] = None) -> Dict[str, Any]:
    home = Path(home) if home else _home()
    name = resolve_model_name(name, home)
    d = home / "trained" / name
    problems = [f"{name}/{f} is missing" for f in ("model.json", "temporal.pt") if not (d / f).is_file()]
    return {"ok": not problems, "model": name, "problems": problems}

def _clean(o: Any) -> Any:
    if isinstance(o, dict): return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)): return [_clean(v) for v in o]
    if isinstance(o, np.generic): return _clean(o.item())
    if isinstance(o, np.ndarray): return _clean(o.tolist())
    if isinstance(o, float): return o if math.isfinite(o) else None
    if dataclasses.is_dataclass(o) and not isinstance(o, type): return _clean(dataclasses.asdict(o))
    return o

@dataclass
class FrameResult:
    stats: Dict[str, Any]
    status: Any
    frame_size: Tuple[int, int] = (0, 0)

@dataclass
class FinalResult:
    stats: Dict[str, Any]
    raw: Any = None

def blank_stats(session_id: str = "", filename: Optional[str] = None, kind: str = "video",
                model_name: Optional[str] = None, steps: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Clean initial state before first frame: only true ML sequence keys."""
    steps = steps or []
    total = len(steps)
    return {
        "session_id": session_id, "filename": filename, "kind": kind, "model_name": model_name,
        "state": "IDLE", "status": "queued", "frame": 0, "frames_processed": 0, "t": 0.0,
        "sequence": [], "steps_done": [], "doing": [], "taken_back": [],
        "steps": [{"step": s["step"], "label": s.get("label", f"Step {s['step']}"), "status": "waiting",
                   "t": None, "detected": None, "in_order": None} for s in steps],
        "total_steps": total, "steps_done_count": 0, "steps_progress": 0.0,
        "is_anomaly_frame": False, "record": False,
        "events": [], "detections": [],
        "fps": 0.0, "latency_ms": 0.0, "stage_ms": {}, "model_frame": False, "final": False, "verdict": None,
    }

def merge_final(stats: Dict[str, Any], final: FinalResult) -> Dict[str, Any]:
    """Attaches final inspection verdict directly from the ML."""
    stats.update(final.stats)
    stats["final"] = True
    stats["verdict"] = final.stats.get("verdict")
    stats["status"] = final.stats.get("verdict") or stats.get("status")
    return _clean(stats)

def draw_final(frame_bgr: np.ndarray, final: FinalResult) -> np.ndarray:
    """Renders a clean verdict banner on the final video frame."""
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    verdict = str(final.stats.get("verdict") or "UNKNOWN").upper()
    seq_str = " -> ".join(str(s) for s in (final.stats.get("sequence") or []))
    dur = final.stats.get("seconds", 0.0)

    if verdict == "NORMAL":
        banner_color = (30, 160, 60)   # Green
    elif verdict == "ANOMALY":
        banner_color = (40, 40, 220)   # Red
    else:
        banner_color = (0, 165, 255)   # Amber / Uncertain

    overlay = img.copy()
    box_h = max(90, int(h * 0.2))
    cv2.rectangle(overlay, (0, h - box_h), (w, h), banner_color, -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

    scale_title = max(0.7, min(1.6, w / 750.0))
    scale_sub = max(0.45, min(0.95, w / 1000.0))
    title_text = f"VERDICT: {verdict}"
    sub_text = f"Sequence: [{seq_str}]  |  Duration: {dur}s"

    cv2.putText(img, title_text, (int(w * 0.04), h - int(box_h * 0.52)),
                cv2.FONT_HERSHEY_DUPLEX, scale_title, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, sub_text, (int(w * 0.04), h - int(box_h * 0.18)),
                cv2.FONT_HERSHEY_SIMPLEX, scale_sub, (240, 240, 240), 1, cv2.LINE_AA)
    return img

class SequenceSession:
    def __init__(self, engine: SequenceEngine, mon: Any, infer_mod: Any, session_id: str, kind: str,
                 source: Optional[str], model_name: str):
        self.engine, self.mon, self._infer = engine, mon, infer_mod
        self.session_id, self.kind, self.source, self.model_name = session_id, kind, source, model_name
        self.steps = [dict(s) for s in mon.steps]
        self.lock = threading.RLock()
        self.closed, self.finished, self.final = False, False, None
        self.frames, self.last_activity, self.last_stats, self._t0 = 0, time.time(), None, None

    def process(self, frame_bgr: np.ndarray, t: Optional[float] = None) -> FrameResult:
        with self.lock:
            if self.closed: raise SequenceInferenceError(f"Session {self.session_id} is already finished")
            if frame_bgr is None or getattr(frame_bgr, "ndim", 0) != 3:
                raise ValueError("process() requires a BGR image (H, W, 3)")
            if t is None:
                now = time.perf_counter()
                self._t0 = now if self._t0 is None else self._t0
                t = now - self._t0
            torch = _torch()
            ctx = torch.inference_mode() if torch is not None else contextlib.nullcontext()
            with ctx:
                st = self.mon.process_frame(frame_bgr, float(t))
            self.frames += 1
            self.last_activity = time.time()
            stats = self._frame_stats(st, frame_bgr)
            self.last_stats = stats
            return FrameResult(stats=stats, status=st, frame_size=(frame_bgr.shape[1], frame_bgr.shape[0]))

    def draw(self, frame_bgr: np.ndarray, fr: Optional[FrameResult] = None, boxes: bool = True, hud: bool = True) -> np.ndarray:
        img = frame_bgr.copy()
        st = fr.status if fr is not None else None
        if boxes and st is not None:
            H, W = img.shape[:2]
            fw, fh = st.frame_size if st.frame_size and st.frame_size[0] else (W, H)
            sx, sy = W / float(fw), H / float(fh)
            for d in st.detections:
                x1, y1, x2, y2 = (int(round(float(v))) for v in d["box"])
                cv2.rectangle(img, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)), (80, 200, 120), 2)
                cv2.putText(img, f"{d['name']} {float(d['conf']):.2f}", (int(x1 * sx), max(14, int(y1 * sy) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 120), 1, cv2.LINE_AA)
        if hud:
            try: return self.mon.draw(img, st)
            except Exception as e: logger.debug(f"[SEQ] HUD error: {e}")
        return img

    def finish(self) -> FinalResult:
        with self.lock:
            if self.finished and self.final is not None: return self.final
            try:
                res = self.mon.stop()
                stats = {
                    "verdict": res.verdict,
                    "sequence": list(res.sequence),
                    "steps": [dataclasses.asdict(s) for s in res.steps],
                    "anomalies": list(res.anomalies),
                    "report": res.report,
                    "seconds": round(float(res.seconds), 2),
                    "fps": round(float(res.fps), 1),
                    "frames_read": res.frames_read,
                    "frames_to_model": res.frames_to_model,
                    "frame_ms": res.frame_ms,
                    "out_dir": res.out_dir,
                    "stopped": res.stopped,
                }
                self.final = FinalResult(stats=_clean(stats), raw=res)
            except RuntimeError as e:
                self.final = FinalResult(stats={"verdict": None, "sequence": [], "anomalies": [str(e)]})
            self.finished = self.closed = True
            self.engine._release(self)
            return self.final

    def abort(self, reason: str = "") -> None:
        with self.lock:
            if self.closed: return
            self.closed = True
            try: self.mon.start("idle")
            except Exception as e: logger.debug(f"[SEQ] abort: {e}")
            self.engine._release(self)
            logger.info(f"[SEQ] {self.session_id}: inspection dropped ({reason})")

    def idle_s(self) -> float: return time.time() - self.last_activity

    def _frame_stats(self, st: Any, frame: np.ndarray) -> Dict[str, Any]:
        """Returns the real measurements and sequence progression produced by the ML."""
        H, W = frame.shape[:2]
        fw, fh = st.frame_size if st.frame_size and st.frame_size[0] else (W, H)
        sx, sy = W / float(fw), H / float(fh)
        dets = [
            {"id": i, "name": d["name"], "conf": round(float(d["conf"]), 3),
             "box": [round(float(d["box"][0]) * sx, 1), round(float(d["box"][1]) * sy, 1),
                     round(float(d["box"][2]) * sx, 1), round(float(d["box"][3]) * sy, 1)]}
            for i, d in enumerate(st.detections)
        ]
        events = [
            {"kind": e.kind, "step": e.step, "label": e.label, "t": round(float(e.t), 2),
             "in_order": e.in_order, "detected": e.detected, "note": e.note}
            for e in st.events
        ]

        # Map model steps against real-time ML state
        done_map = {d["step"]: d for d in st.steps_done}
        doing_set = set(st.doing)
        taken_back_set = set(st.taken_back)
        steps_out = []
        for s in self.steps:
            sid = s["step"]
            if sid in taken_back_set:
                st_val = "taken_back"
            elif sid in done_map:
                st_val = "done" if done_map[sid].get("in_order", True) else "out_of_order"
            elif sid in doing_set:
                st_val = "doing"
            else:
                st_val = "waiting"
            d_info = done_map.get(sid, {})
            steps_out.append({
                "step": sid,
                "label": s.get("label", f"Step {sid}"),
                "status": st_val,
                "t": d_info.get("t"),
                "detected": d_info.get("detected"),
                "in_order": d_info.get("in_order"),
            })

        has_anomaly = any(e.in_order is False for e in st.events) or bool(st.taken_back)
        if st.state == "WARMING":
            live_status = "WARMING"
        elif has_anomaly:
            live_status = "ANOMALY"
        elif st.state == "INSPECTING":
            live_status = "NORMAL"
        else:
            live_status = str(st.state)

        total = len(self.steps)
        done_count = len(st.steps_done)
        progress = round((done_count / total) * 100, 1) if total > 0 else 0.0

        return _clean({
            "frame": int(st.index),
            "frames_processed": int(st.index),
            "t": round(float(st.t), 2),
            "state": st.state,
            "status": live_status,
            "steps_done": st.steps_done,
            "doing": list(st.doing),
            "taken_back": list(st.taken_back),
            "sequence": [d["step"] for d in st.steps_done],
            "steps": steps_out,
            "total_steps": total,
            "steps_done_count": done_count,
            "steps_progress": progress,
            "is_anomaly_frame": has_anomaly,
            "record": st.state in ("INSPECTING", "WARMING"),
            "events": events,
            "detections": dets,
            "fps": round(float(st.fps), 1),
            "latency_ms": round(float(st.frame_ms), 1),
            "stage_ms": {k: round(float(v), 1) for k, v in (st.stage_ms or {}).items()},
            "model_frame": bool(st.model_frame),
        })

class SequenceEngine:
    def __init__(self):
        self._lock = threading.RLock()
        self._mon, self._infer, self._active = None, None, None
        self.model_name, self.info = None, {}

    @property
    def loaded(self) -> bool: return self._mon is not None

    @property
    def active_session_id(self) -> Optional[str]:
        return self._active.session_id if self._active is not None else None

    def load(self, model_name: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
        with self._lock:
            if self._active is not None:
                raise SequenceBusyError(f"Cannot reload model while inspection '{self._active.session_id}' is running")
            env = prepare_environment()
            name = resolve_model_name(model_name, Path(env["home"]))
            if self.loaded and name == self.model_name and not force: return self.info
            pre = preflight(name, Path(env["home"]))
            if not pre["ok"]:
                raise SequenceModelError(f"Model '{name}' cannot be loaded: " + ", ".join(pre["problems"]))
            infer = import_infer()
            self.close()
            t0 = time.time()
            logger.info(f"[SEQ] Loading model '{name}'...")
            mon = infer.SOPMonitor(name, out_dir=_config.out_dir, save_results=_config.save_results,
                                   check_every_s=_config.check_every_s, log=lambda s: logger.debug(f"[ML] {s}"))
            self._mon, self._infer, self.model_name = mon, infer, name
            self._cudnn_benchmark_off()
            torch = _torch()
            self.info = _clean({
                "model": name, "home": env["home"], "detectors": list(mon.detectors),
                "steps": list(mon.steps), "cuda": bool(torch is not None and torch.cuda.is_available()),
                "load_seconds": round(time.time() - t0, 1)
            })
            logger.info(f"[SEQ] Model '{name}' ready in {self.info['load_seconds']}s")
            return self.info

    def close(self) -> None:
        with self._lock:
            if self._active is not None: self._active.abort("Engine closed")
            if self._mon is not None:
                try: self._mon.close()
                except Exception as e: logger.warning(f"[SEQ] Closing model: {e}")
            self._mon, self._infer, self.model_name, self.info = None, None, None, {}

    @staticmethod
    def _cudnn_benchmark_off() -> None:
        torch = _torch()
        if torch is not None:
            try: torch.backends.cudnn.benchmark = False
            except Exception: pass

    def open_session(self, session_id: str, kind: str = "video", source: Optional[str] = None,
                     model_name: Optional[str] = None) -> SequenceSession:
        with self._lock:
            a = self._active
            if a is not None:
                if a.session_id == session_id: a.abort("Replaced by new run of same session")
                elif a.idle_s() > _config.stale_session_s: a.abort(f"Idle {a.idle_s():.0f}s - taken over by '{session_id}'")
                else: raise SequenceBusyError(f"Inspection '{a.session_id}' is currently running on the GPU")
            want = resolve_model_name(model_name) if model_name else None
            if not self.loaded or (want is not None and want != self.model_name): self.load(want)
            self._cudnn_benchmark_off()
            self._mon.start(session_id)
            sess = SequenceSession(self, self._mon, self._infer, session_id, kind, source, self.model_name)
            self._active = sess
            logger.info(f"[SEQ] Session '{session_id}' ({kind}) started on model '{self.model_name}'")
            return sess

    def _release(self, sess: SequenceSession) -> None:
        with self._lock:
            if self._active is sess: self._active = None

    def steps(self) -> List[Dict[str, Any]]: return list(self.info.get("steps") or [])

_engine: Optional[SequenceEngine] = None
_engine_lock = threading.Lock()

def get_engine() -> SequenceEngine:
    global _engine
    with _engine_lock:
        if _engine is None: _engine = SequenceEngine()
        return _engine

def set_engine(engine: SequenceEngine) -> None:
    global _engine
    with _engine_lock: _engine = engine

def preload(model_name: Optional[str] = None) -> Dict[str, Any]:
    return get_engine().load(model_name)

def shutdown() -> None:
    global _engine
    with _engine_lock:
        e, _engine = _engine, None
    if e is not None: e.close()
