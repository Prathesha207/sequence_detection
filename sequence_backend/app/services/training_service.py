"""
training_service.py - Real SOP sequence training pipeline runner using the installed sop_sequence wheel.
Spawns `sop_sequence.train` in a dedicated subprocess, parses live stage / [progress] / loss stdout
telemetry, persists job history to SQLite, and deploys trained packages into active inference.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.app_paths import APP_DIR
from app.core.database import Base, SessionLocal, engine
from app.core.logger import setup_logger
from app.ml import app_state
from app.ml import sequence_inference as seqinf
from app.models.training_model import TrainingSession, TrainingSetting

logger = setup_logger("training-service")


def _get_packages_dir() -> Path:
    """Return writable directory for trained model output packages."""
    preferred = APP_DIR / "training_packages"
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "sequence_training_packages"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

_RE_STAGE_START = re.compile(r"^\[(\d+)\s+([^\]]+)\]\s+\.\.\.")
_RE_STAGE_DONE = re.compile(r"^\[(\d+)\s+([^\]]+)\]\s+done in\s+([\d.]+)\s*s")
_RE_STAGE2_STEP = re.compile(r"^\s*(\d+)\.\s+(.+?)\s+\[object:\s*(.+?)\]\s*$")
_RE_PROMPT_STEP = re.compile(r"^\s*(\d+)[.)\]:-]\s*(.+?)\s*$")
_RE_PROGRESS = re.compile(r"^\[progress\]\s+(.+?)(?:\s+\(~(\d+)s left\))?\s*(?:\((\d+)%\))?$")
_RE_ITER_LOSS = re.compile(r"^\s*(TCN|LSTR|Action)\s+iter\s+(\d+)\s+loss\s+([\d.]+)", re.IGNORECASE)
_RE_FINAL_ACTION_LOSS = re.compile(r"final loss\s+([\d.]+)", re.IGNORECASE)


_STAGE_META: Dict[str, Tuple[str, str, int]] = {
    "inputs": ("Validating Inputs", "Checking video file, prompt & wheel config", 1),
    "detectors": ("Loading Detectors", "Loading YOLO object detector weights & classes", 2),
    "spec": ("Parsing SOP Steps", "Matching numbered SOP steps to detector objects", 1),
    "places": ("Place Learning", "Sampling frames & learning workspace object places", 30),
    "save": ("Saving Model Spec", "Writing model configuration & learned places", 1),
    "step model": ("Step Model Training", "Extracting features & training sequence step model", 40),
    "references": ("Visual References", "Building visual reference prototypes per step", 15),
    "action model": ("Action Model & Self-Check", "Training action matcher & verifying step sequence", 30),
    "finish": ("Finalizing Model", "Saving active model package & writing result.json", 2),
}


def _build_stage_list(wheel_cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[int, Tuple[int, int]]]:
    """Build ordered stage descriptors and proportional progress bands from ML's @stage definitions."""
    raw_stages: List[Tuple[int, str]] = wheel_cfg.get("stages") or [
        (0, "inputs"), (1, "detectors"), (2, "spec"), (3, "places"),
        (4, "save"), (5, "step model"), (6, "references"), (7, "action model"), (8, "finish"),
    ]
    total_est = sum(_STAGE_META.get(k, (k.title(), "", 10))[2] for _, k in raw_stages) or 1
    stages: List[Dict[str, Any]] = []
    bands: Dict[int, Tuple[int, int]] = {}
    cursor = 0.0
    for idx, key in raw_stages:
        title, desc, est_s = _STAGE_META.get(key, (key.replace("_", " ").title(), f"Running {key}", 10))
        share = (est_s / total_est) * 99.0
        b_start = int(round(cursor))
        cursor += share
        b_end = min(99, max(b_start + 1, int(round(cursor))))
        bands[idx] = (b_start, b_end)
        stages.append(
            {
                "id": idx,
                "key": key,
                "title": title,
                "description": desc,
                "status": "pending",
                "duration_seconds": None,
                "estimated_seconds": est_s,
                "sub_progress": None,
            }
        )
    return stages, bands


def _read_wheel_config() -> Dict[str, Any]:
    """Read training defaults (kind, fps, total_stages, stages) directly from the installed sop_sequence wheel."""
    info: Dict[str, Any] = {"kind": "", "fps": 0.0, "total_stages": 9, "stages": []}
    try:
        import sop_sequence
        pkg_dir = Path(sop_sequence.data_folder)
        cfg_file = pkg_dir / "training" / "config.json"
        if cfg_file.is_file():
            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            info["kind"] = str(cfg.get("kind") or "").upper()
            info["fps"] = float(cfg.get("fps") or 0.0)
        ts_file = pkg_dir / "training" / "train_sequence.py"
        if ts_file.is_file():
            matches = re.findall(r'@stage\(\s*(\d+)\s*,\s*"([^"]+)"\s*\)', ts_file.read_text(encoding="utf-8"))
            if matches:
                parsed = sorted(((int(n), title) for n, title in matches), key=lambda x: x[0])
                info["stages"] = parsed
                info["total_stages"] = max(n for n, _ in parsed) + 1
    except Exception as e:
        logger.debug(f"Could not read wheel training config: {e}")
    return info


def _detect_runtime_device() -> str:
    """Detect whether PyTorch CUDA is available at runtime instead of assuming a device."""
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def _parse_prompt_steps(prompt: str) -> List[Dict[str, Any]]:
    """Extract initial numbered step entries from the user's SOP prompt (before ML Stage 2 refines them)."""
    parsed: List[Dict[str, Any]] = []
    for raw_line in (prompt or "").splitlines():
        m = _RE_PROMPT_STEP.match(raw_line)
        if m:
            parsed.append(
                {
                    "id": int(m.group(1)),
                    "action_query": m.group(2).strip(),
                    "part": "",
                    "count": 1,
                }
            )
    return parsed


def _format_ts_prefix() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _resolve_workspace_dir(raw_root: Optional[str]) -> Path:
    """Resolve '<raw_root>/sequence_train' cleanly (or default APP_DIR/sequence_train)."""
    cleaned = (raw_root or "").strip()
    if cleaned:
        base = Path(cleaned).expanduser().resolve()
    else:
        base = _get_packages_dir().parent
    if base.name.lower() in {"sequence_train", "sequence_inference"}:
        return base.parent / "sequence_train"
    return base / "sequence_train"


def _safe_model_folder_name(model_name: str, fallback: str) -> str:
    """Convert UI model_name into a clean, safe folder name avoiding inference system dirs."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (model_name or "").strip()).strip("._-")
    if not cleaned or cleaned.lower() in {"trained", "weights", "outputs", "specs"}:
        return fallback
    return cleaned


def _resolve_model_output_dir(workspace: Path, model_name: str, fallback_id: str) -> Path:
    """Resolve '<workspace>/<Model_Name>', versioning cleanly only if a completed package exists."""
    folder_name = _safe_model_folder_name(model_name, fallback_id)
    cand = workspace / folder_name
    if not cand.exists():
        return cand
    res_file = cand / "result.json"
    if res_file.is_file():
        try:
            prev = json.loads(res_file.read_text(encoding="utf-8"))
            if prev.get("status") != "ok":
                return cand
        except Exception:
            pass
    idx = 2
    while (workspace / f"{folder_name}_{idx}").exists():
        idx += 1
    return workspace / f"{folder_name}_{idx}"


def _ensure_inference_workspace(workspace: Path) -> None:
    """Ensure <workspace> has trained/, weights/, and models.json so inference can run directly from it."""
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "trained").mkdir(parents=True, exist_ok=True)
    (workspace / "weights" / "detectors").mkdir(parents=True, exist_ok=True)
    try:
        import sop_sequence
        pkg_dir = Path(sop_sequence.data_folder)
        if not (workspace / "models.json").is_file() and (pkg_dir / "models.json").is_file():
            shutil.copy2(pkg_dir / "models.json", workspace / "models.json")
        src_weights = pkg_dir / "weights"
        if src_weights.is_dir():
            for item in src_weights.iterdir():
                if item.is_file():
                    dst = workspace / "weights" / item.name
                    if not dst.exists():
                        shutil.copy2(item, dst)
            src_det = src_weights / "detectors"
            if src_det.is_dir():
                for pt in src_det.glob("*.pt"):
                    dst_pt = workspace / "weights" / "detectors" / pt.name
                    if not dst_pt.exists():
                        shutil.copy2(pt, dst_pt)
    except Exception as e:
        logger.warning(f"Could not seed inference assets into {workspace}: {e}")


class RealTrainingService:
    """Runs sop_sequence wheel training via subprocess.Popen with live stdout telemetry."""

    def __init__(self):
        self._lock = threading.Lock()
        self._current_job: Optional[Dict[str, Any]] = None
        self._proc: Optional[subprocess.Popen] = None
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._job_output_dirs: Dict[str, Path] = {}
        self._job_start_ts: Dict[str, float] = {}
        self._ensure_table()

    def get_default_output_dir(self) -> str:
        return str(_get_packages_dir().parent)

    def get_saved_output_root(self) -> str:
        db = SessionLocal()
        try:
            row = db.query(TrainingSetting).filter(TrainingSetting.key == "output_root").first()
            return (row.value or "").strip() if row else ""
        except Exception:
            return ""
        finally:
            db.close()

    def get_effective_workspace(self, raw_root: Optional[str] = None) -> Path:
        root = (raw_root if raw_root is not None else self.get_saved_output_root()).strip()
        return _resolve_workspace_dir(root)

    def set_saved_output_root(self, raw_root: str) -> Dict[str, Any]:
        cleaned = (raw_root or "").strip()
        try:
            ws = _resolve_workspace_dir(cleaned)
            _ensure_inference_workspace(ws)
            db = SessionLocal()
            try:
                row = db.query(TrainingSetting).filter(TrainingSetting.key == "output_root").first()
                if row is None:
                    row = TrainingSetting(key="output_root", value=cleaned)
                    db.add(row)
                else:
                    row.value = cleaned
                db.commit()
            finally:
                db.close()
            if cleaned:
                seqinf.set_data_home(str(ws))
            else:
                seqinf.set_data_home(None)
            from app.core.app_paths import get_inference_base_dir
            inf_dir = get_inference_base_dir()
            return {
                "ok": True,
                "saved_output_dir": cleaned,
                "workspace_dir": str(ws),
                "inference_dir": str(inf_dir),
            }
        except Exception as e:
            logger.error(f"Failed setting output folder '{raw_root}': {e}")
            return {"ok": False, "error": f"Cannot prepare output folder: {e}"}

    def get_wheel_config(self) -> Dict[str, Any]:
        return _read_wheel_config()

    def _ensure_table(self) -> None:
        try:
            Base.metadata.create_all(bind=engine)
            with engine.connect() as conn:
                cols = {r[1] for r in conn.exec_driver_sql("PRAGMA table_info(training_sessions)").fetchall()}
                if "output_dir" not in cols:
                    conn.exec_driver_sql("ALTER TABLE training_sessions ADD COLUMN output_dir TEXT DEFAULT ''")
                    conn.commit()
        except Exception as e:
            logger.warning(f"Failed to ensure training_sessions table: {e}")

    def _upsert_session(self, job: Dict[str, Any]) -> None:
        db = SessionLocal()
        try:
            row = db.query(TrainingSession).filter(TrainingSession.job_id == job["job_id"]).first()
            if row is None:
                row = TrainingSession(
                    job_id=job["job_id"],
                    model_name=job.get("model_name", ""),
                    prompt=job.get("prompt", ""),
                    dataset_video_name=job.get("selected_video"),
                    base_detector=job.get("base_detector", ""),
                    epochs=int(job.get("total_epochs") or 0),
                    batch_size=int(job.get("batch_size") or 0),
                    learning_rate=float(job.get("learning_rate") or 0.0),
                    device=job.get("device") or "",
                )
                db.add(row)

            row.model_name = job.get("model_name", row.model_name)
            row.status = job.get("status", "idle")
            row.current_epoch = int(job.get("current_epoch") or 0)
            row.progress_percent = int(job.get("progress_percent") or 0)
            row.train_loss = float(job.get("train_loss") or 0.0)
            row.val_loss = float(job.get("val_loss") or 0.0)
            row.accuracy = float(job.get("accuracy") or 0.0)
            row.eta_seconds = int(job.get("eta_seconds", 0))
            row.elapsed_seconds = int(job.get("elapsed_seconds", 0))
            row.steps_json = json.dumps(job.get("steps", []))
            row.logs_json = json.dumps(job.get("logs", [])[-300:])
            row.metrics_json = json.dumps(job.get("metrics_history", []))
            row.output_dir = str(job.get("output_dir") or "")
            if job.get("completed_at"):
                try:
                    row.completed_at = datetime.fromisoformat(str(job["completed_at"]).replace("Z", ""))
                except Exception:
                    row.completed_at = datetime.utcnow()

            db.commit()
        except Exception as e:
            logger.error(f"DB error saving training session {job.get('job_id')}: {e}")
            db.rollback()
        finally:
            db.close()

    def _row_to_job_dict(self, row: TrainingSession) -> Dict[str, Any]:
        return {
            "job_id": row.job_id,
            "status": row.status,
            "current_epoch": row.current_epoch,
            "total_epochs": row.epochs,
            "progress_percent": row.progress_percent,
            "train_loss": row.train_loss,
            "val_loss": row.val_loss if row.val_loss else None,
            "accuracy": row.accuracy if (row.status == "completed" or row.accuracy) else None,
            "eta_seconds": row.eta_seconds,
            "elapsed_seconds": row.elapsed_seconds,
            "logs": json.loads(row.logs_json or "[]"),
            "metrics_history": json.loads(row.metrics_json or "[]"),
            "steps": json.loads(row.steps_json or "[]"),
            "selected_video": row.dataset_video_name,
            "model_name": row.model_name,
            "prompt": row.prompt,
            "output_dir": getattr(row, "output_dir", "") or "",
            "created_at": row.created_at.isoformat() if row.created_at else datetime.utcnow().isoformat(),
            "completed_at": row.completed_at.isoformat() if row.completed_at else None,
        }

    def get_current_job(self, job_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if self._current_job is not None and (not job_id or self._current_job.get("job_id") == job_id):
                snap = dict(self._current_job)
                snap.pop("_stage_bands", None)
                if snap.get("status") in ("preparing", "training", "evaluating"):
                    st_ts = self._job_start_ts.get(snap["job_id"])
                    if st_ts:
                        live_elapsed = max(0, int(time.time() - st_ts))
                        snap["elapsed_seconds"] = max(snap.get("elapsed_seconds", 0), live_elapsed)
                if snap.get("stages"):
                    snap["stages"] = [dict(s) for s in snap["stages"]]
                return snap

        if job_id:
            db = SessionLocal()
            try:
                row = db.query(TrainingSession).filter(TrainingSession.job_id == job_id).first()
                if row:
                    return self._row_to_job_dict(row)
            finally:
                db.close()

        wheel_cfg = _read_wheel_config()
        return {
            "job_id": "",
            "status": "idle",
            "current_epoch": 0,
            "total_epochs": wheel_cfg["total_stages"],
            "progress_percent": 0,
            "train_loss": 0.0,
            "val_loss": None,
            "accuracy": None,
            "eta_seconds": 0,
            "elapsed_seconds": 0,
            "logs": ["[System] Training engine idle. Record or upload a demonstration video to start."],
            "metrics_history": [],
            "steps": [],
            "selected_video": None,
            "model_name": "",
            "model_kind": wheel_cfg["kind"],
            "fps": wheel_cfg["fps"],
            "created_at": datetime.utcnow().isoformat(),
        }

    def reset_current_job(self) -> Dict[str, Any]:
        """Clear a finished/failed/stopped job from active status so UI can return to setup."""
        with self._lock:
            if self._worker_thread and self._worker_thread.is_alive():
                return {"error": "Cannot reset while training is actively running", "busy": True}
            self._current_job = None
        return self.get_current_job()

    def get_history(self) -> List[Dict[str, Any]]:
        db = SessionLocal()
        try:
            rows = (
                db.query(TrainingSession)
                .order_by(TrainingSession.created_at.desc())
                .limit(50)
                .all()
            )
            results: List[Dict[str, Any]] = []
            for row in rows:
                steps = json.loads(row.steps_json or "[]")
                results.append(
                    {
                        "job_id": row.job_id,
                        "model_name": row.model_name,
                        "prompt": row.prompt,
                        "dataset_video_name": row.dataset_video_name,
                        "epochs": row.epochs,
                        "final_accuracy": row.accuracy,
                        "final_loss": row.train_loss,
                        "status": row.status,
                        "created_at": row.created_at.isoformat() if row.created_at else "",
                        "duration_seconds": row.elapsed_seconds,
                        "steps_count": len(steps),
                    }
                )
            return results
        finally:
            db.close()

    def _resolve_wheel_train_script(self) -> Path:
        try:
            import sop_sequence.train as whl_train
            p = Path(whl_train.__file__).resolve()
            if p.is_file():
                return p
        except Exception as e:
            raise RuntimeError(f"sop_sequence wheel is not installed in Python environment: {e}") from e
        raise RuntimeError("sop_sequence.train module file missing in site-packages")

    def _resolve_detector_weights(self, base_detector: str) -> Optional[Path]:
        """Resolve a user-selected YOLO .pt weights file from the sop_sequence wheel.
        Returns None when base_detector is empty ('Auto') so train.py uses wheel models.json directly.
        """
        want = (base_detector or "").strip()
        if not want:
            return None
        if os.path.isfile(want):
            return Path(want).resolve()
        try:
            import sop_sequence
            pkg_dir = Path(sop_sequence.data_folder)
            models_file = pkg_dir / "models.json"
            if models_file.is_file():
                entries = json.loads(models_file.read_text(encoding="utf-8"))
                want_low = want.lower()
                for e in entries:
                    name = str(e.get("name", "")).lower()
                    stem = Path(str(e.get("path", ""))).stem.lower()
                    if want_low == name or want_low in name or (stem and stem in want_low):
                        cand = pkg_dir / str(e["path"])
                        if cand.is_file():
                            return cand.resolve()
        except Exception as e:
            logger.warning(f"Could not resolve detector weights from sop_sequence wheel: {e}")
        return None

    def _resolve_video_path(self, payload: Dict[str, Any]) -> Optional[str]:
        explicit = (payload.get("video_path") or "").strip()
        if explicit and os.path.isfile(explicit):
            return os.path.abspath(explicit)

        # Fallback: look up active video session by filename if uploaded via video_router
        vid_name = (payload.get("dataset_video_name") or "").strip()
        if vid_name:
            try:
                from app.ml.video_inference_service import video_inference_service
                for sess in reversed(list(video_inference_service.sessions.values())):
                    if sess.get("original_filename") == vid_name:
                        cand = sess.get("inference_video_path") or sess.get("temp_file")
                        if cand and os.path.isfile(cand):
                            return os.path.abspath(cand)
            except Exception:
                pass
        return None

    def start_training(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if self._worker_thread and self._worker_thread.is_alive():
                return {"error": "A training job is already running", "busy": True, "job": self._current_job}

            job_id = f"train_{int(time.time())}"
            model_name = (payload.get("model_name") or "").strip() or f"SOP_Model_{int(time.time())}"
            prompt = (payload.get("prompt") or "").strip()
            base_detector = (payload.get("base_detector") or "").strip()
            steps = _parse_prompt_steps(prompt) or (payload.get("steps") or [])
            video_name = payload.get("dataset_video_name") or ""

            if not prompt:
                return {"error": "Training Prompt is required"}

            video_path = self._resolve_video_path(payload)
            if not video_path:
                return {"error": "Training video file not found on disk. Please upload or record a video first."}

            try:
                train_script = self._resolve_wheel_train_script()
            except RuntimeError as e:
                return {"error": str(e)}

            weights_path = self._resolve_detector_weights(base_detector)

            if not app_state.try_enter_training(job_id):
                active_kind = app_state.get_active_inference_kind()
                return {
                    "error": f"GPU is currently in use by {active_kind or 'another process'} -- stop inference before starting training",
                    "busy": True,
                }

            custom_out = (payload.get("output_dir") or "").strip()
            saved_out = self.get_saved_output_root()
            chosen_root = custom_out or saved_out

            if custom_out and custom_out != saved_out:
                save_res = self.set_saved_output_root(custom_out)
                if not save_res.get("ok"):
                    app_state.exit_training()
                    return {"error": save_res.get("error", f"Invalid output directory '{custom_out}'")}

            try:
                workspace_dir = _resolve_workspace_dir(chosen_root)
                _ensure_inference_workspace(workspace_dir)
                output_dir = _resolve_model_output_dir(workspace_dir, model_name, job_id)
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                app_state.exit_training()
                return {"error": f"Cannot prepare training folder: {e}"}

            self._job_output_dirs[job_id] = output_dir

            init_logs = [
                f"[{_format_ts_prefix()}] Launching sop_sequence wheel training for '{model_name}'...",
                f"[{_format_ts_prefix()}] Source video: {Path(video_path).name}",
                f"[{_format_ts_prefix()}] Workspace (train & infer): {workspace_dir}",
                f"[{_format_ts_prefix()}] Model output folder: {output_dir}",
            ]
            if weights_path:
                init_logs.append(f"[{_format_ts_prefix()}] Detector weights (selected): {weights_path.name}")
            else:
                init_logs.append(f"[{_format_ts_prefix()}] Detector weights: Auto (wheel models.json)")

            wheel_cfg = _read_wheel_config()
            stages_list, stage_bands = _build_stage_list(wheel_cfg)
            total_stages = max(1, len(stages_list))
            self._job_start_ts[job_id] = time.time()
            total_est_eta = sum(int(s.get("estimated_seconds") or 0) for s in stages_list)
            initial_job = {
                "job_id": job_id,
                "status": "preparing",
                "current_epoch": 0,
                "total_epochs": total_stages,
                "progress_percent": 0,
                "train_loss": 0.0,
                "val_loss": None,
                "accuracy": None,
                "eta_seconds": total_est_eta,
                "stage_eta_seconds": None,
                "stage_name": stages_list[0]["title"] if stages_list else "Preparing",
                "stage_detail": "Starting ML training pipeline...",
                "elapsed_seconds": 0,
                "logs": init_logs,
                "metrics_history": [],
                "stages": stages_list,
                "_stage_bands": stage_bands,
                "selected_video": video_name or Path(video_path).name,
                "model_name": model_name,
                "model_kind": wheel_cfg["kind"],
                "fps": wheel_cfg["fps"],
                "prompt": prompt,
                "base_detector": weights_path.name if weights_path else base_detector,
                "batch_size": 0,
                "learning_rate": 0.0,
                "device": _detect_runtime_device(),
                "output_dir": str(output_dir),
                "created_at": datetime.utcnow().isoformat(),
                "steps": steps,
            }

            self._current_job = initial_job
            self._upsert_session(initial_job)
            self._stop_event.clear()

            self._worker_thread = threading.Thread(
                target=self._run_training_subprocess,
                args=(job_id, train_script, video_path, output_dir, weights_path, prompt, model_name),
                daemon=True,
            )
            self._worker_thread.start()

            return {"job": dict(self._current_job)}

    def stop_training(self, job_id: str) -> Dict[str, Any]:
        proc_to_kill = None
        snapshot = None
        with self._lock:
            if not self._current_job:
                return {"status": "not_running"}
            self._stop_event.set()
            proc_to_kill = self._proc
            self._current_job["status"] = "stopped"
            self._current_job["logs"].append(f"[{_format_ts_prefix()}] Training stop requested by user.")
            snapshot = dict(self._current_job)

        if proc_to_kill is not None and proc_to_kill.poll() is None:
            try:
                proc_to_kill.terminate()
                try:
                    proc_to_kill.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    proc_to_kill.kill()
            except Exception as e:
                logger.warning(f"Error terminating training subprocess: {e}")

        if snapshot:
            self._upsert_session(snapshot)
        return {"job": snapshot}

    def _run_training_subprocess(
        self,
        job_id: str,
        train_script: Path,
        video_path: str,
        output_dir: Path,
        weights_path: Optional[Path],
        prompt: str,
        model_name: str,
    ) -> None:
        start_time = self._job_start_ts.get(job_id) or time.time()
        self._job_start_ts[job_id] = start_time
        cmd = [
            sys.executable,
            "-u",
            str(train_script),
            "--input",
            video_path,
            "--output_dir",
            str(output_dir),
            "--prompt",
            prompt,
            "--name",
            model_name,
        ]
        if weights_path and weights_path.is_file():
            cmd += ["--weights", str(weights_path)]

        logger.info(f"[TRAIN] Spawning subprocess for {job_id}: {' '.join(cmd)}")
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        proc: Optional[subprocess.Popen] = None
        current_stage = 0
        metric_step_idx = 0
        last_persist = 0.0

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creation_flags,
            )
            with self._lock:
                self._proc = proc
            app_state.set_active_pid(proc.pid)

            if proc.stdout is not None:
                for raw_line in proc.stdout:
                    if self._stop_event.is_set():
                        break
                    line = raw_line.rstrip("\r\n")
                    if not line.strip():
                        continue

                    now = time.time()
                    elapsed = int(now - start_time)
                    snapshot = None

                    with self._lock:
                        if not self._current_job or self._current_job["job_id"] != job_id:
                            break
                        job = self._current_job
                        job["elapsed_seconds"] = elapsed
                        job["logs"].append(line)
                        if len(job["logs"]) > 400:
                            job["logs"] = job["logs"][-400:]

                        # 1. Parse live stage, progress, ETA, and iteration telemetry directly from ML stdout
                        stages_arr = job.get("stages") or []
                        stage_bands: Dict[int, Tuple[int, int]] = job.get("_stage_bands") or {}
                        total_stages = max(1, len(stages_arr) or int(job.get("total_epochs") or 9))
                        m_done = _RE_STAGE_DONE.match(line.strip())
                        m_start = _RE_STAGE_START.match(line.strip())
                        m_stage2_step = _RE_STAGE2_STEP.match(line)
                        m_prog = _RE_PROGRESS.match(line.strip())
                        m_iter = _RE_ITER_LOSS.match(line)
                        m_final_loss = _RE_FINAL_ACTION_LOSS.search(line)

                        def _compute_eta(cur_st: int, sub_frac: float = 0.0, ml_stage_eta: Optional[int] = None) -> int:
                            rem = 0
                            for s in stages_arr:
                                sid = s["id"]
                                if sid > cur_st and s["status"] == "pending":
                                    rem += int(s.get("estimated_seconds") or 0)
                                elif sid == cur_st:
                                    if ml_stage_eta is not None:
                                        rem += ml_stage_eta
                                    else:
                                        est = int(s.get("estimated_seconds") or 0)
                                        rem += max(0, int(round(est * (1.0 - sub_frac))))
                            return max(1, rem) if rem > 0 else 0

                        if m_stage2_step and current_stage == 2:
                            s_id = int(m_stage2_step.group(1))
                            s_item = {
                                "id": s_id,
                                "action_query": m_stage2_step.group(2).strip(),
                                "part": m_stage2_step.group(3).strip(),
                                "count": 1,
                            }
                            existing = [s for s in job.get("steps", []) if s.get("id") != s_id]
                            existing.append(s_item)
                            existing.sort(key=lambda x: x["id"])
                            job["steps"] = existing

                        if m_start:
                            st_num = int(m_start.group(1))
                            current_stage = st_num
                            if st_num == 2:
                                job["steps"] = []
                            for s in stages_arr:
                                if s["id"] < st_num and s["status"] != "done":
                                    s["status"] = "done"
                                elif s["id"] == st_num:
                                    s["status"] = "active"
                                    s["sub_progress"] = 0
                            st_obj = next((s for s in stages_arr if s["id"] == st_num), None)
                            job["current_epoch"] = st_num
                            job["stage_name"] = st_obj["title"] if st_obj else m_start.group(2).strip()
                            job["stage_detail"] = st_obj["description"] if st_obj else f"Running {m_start.group(2).strip()}..."
                            job["stage_eta_seconds"] = None
                            b_start = stage_bands.get(st_num, (int((st_num / total_stages) * 100), 99))[0]
                            job["progress_percent"] = max(job["progress_percent"], min(99, b_start))
                            job["eta_seconds"] = _compute_eta(st_num, 0.0)
                            if st_num >= 3:
                                job["status"] = "evaluating" if st_num == 7 else "training"
                        elif m_done:
                            st_num = int(m_done.group(1))
                            dur_s = round(float(m_done.group(3)), 1)
                            for s in stages_arr:
                                if s["id"] <= st_num:
                                    s["status"] = "done"
                                    if s["id"] == st_num:
                                        s["duration_seconds"] = dur_s
                                        s["sub_progress"] = 100
                            next_st = min(total_stages - 1, st_num + 1)
                            next_obj = next((s for s in stages_arr if s["id"] == next_st), None)
                            job["current_epoch"] = min(total_stages, st_num + 1)
                            if next_obj and st_num + 1 < total_stages:
                                job["stage_name"] = next_obj["title"]
                                job["stage_detail"] = f"Completed stage {st_num + 1}/{total_stages} in {dur_s:.1f}s"
                            else:
                                job["stage_name"] = m_done.group(2).strip()
                            b_end = stage_bands.get(st_num, (0, int(((st_num + 1) / total_stages) * 100)))[1]
                            job["progress_percent"] = max(job["progress_percent"], min(99, b_end))
                            job["stage_eta_seconds"] = None
                            job["eta_seconds"] = _compute_eta(st_num + 1, 0.0)
                        elif m_prog:
                            full_task = m_prog.group(1).strip()
                            sub_eta = int(m_prog.group(2)) if m_prog.group(2) is not None else None
                            sub_pct = float(m_prog.group(3)) if m_prog.group(3) is not None else None
                            if full_task:
                                pretty_task = full_task[0].upper() + full_task[1:]
                                if sub_eta is not None:
                                    pretty_task = f"{pretty_task} (~{sub_eta}s left)"
                                if sub_pct is not None:
                                    pretty_task = f"{pretty_task} ({int(sub_pct)}%)"
                                job["stage_detail"] = pretty_task
                            if sub_eta is not None:
                                job["stage_eta_seconds"] = sub_eta
                            if sub_pct is not None:
                                frac = max(0.0, min(1.0, sub_pct / 100.0))
                                st_obj = next((s for s in stages_arr if s["id"] == current_stage), None)
                                if st_obj:
                                    st_obj["sub_progress"] = int(round(sub_pct))
                                b_start, b_end = stage_bands.get(
                                    current_stage,
                                    (int((current_stage / total_stages) * 100), int(((current_stage + 1) / total_stages) * 100)),
                                )
                                interp = int(round(b_start + frac * (b_end - b_start)))
                                job["progress_percent"] = max(job["progress_percent"], min(99, interp))
                                job["eta_seconds"] = _compute_eta(current_stage, frac, sub_eta)
                            elif sub_eta is not None:
                                job["eta_seconds"] = _compute_eta(current_stage, 0.0, sub_eta)
                        elif m_iter:
                            kind_tag = m_iter.group(1).upper()
                            it_num = int(m_iter.group(2))
                            if kind_tag in ("TCN", "LSTR"):
                                job["model_kind"] = kind_tag
                            loss_val = round(float(m_iter.group(3)), 4)
                            job["train_loss"] = loss_val
                            job["stage_detail"] = f"{kind_tag} iteration {it_num} · loss {loss_val:.3f}"
                            b_start, b_end = stage_bands.get(current_stage, (34, 66))
                            iter_frac = min(0.95, (it_num + 1) / 600.0)
                            interp = int(round(b_start + iter_frac * (b_end - b_start)))
                            job["progress_percent"] = max(job["progress_percent"], min(b_end, interp))
                            st_obj = next((s for s in stages_arr if s["id"] == current_stage), None)
                            if st_obj:
                                st_obj["sub_progress"] = max(st_obj.get("sub_progress") or 0, int(round(iter_frac * 100)))
                            job["eta_seconds"] = _compute_eta(current_stage, iter_frac)
                            metric_step_idx += 1
                            job["metrics_history"].append(
                                {
                                    "epoch": metric_step_idx,
                                    "iter": it_num,
                                    "label": f"{kind_tag} iter {it_num}",
                                    "train_loss": loss_val,
                                    "val_loss": None,
                                    "accuracy": None,
                                    "step_accuracy": None,
                                    "timestamp": int(now),
                                }
                            )
                        elif m_final_loss:
                            loss_val = round(float(m_final_loss.group(1)), 4)
                            job["train_loss"] = loss_val
                            metric_step_idx += 1
                            job["metrics_history"].append(
                                {
                                    "epoch": metric_step_idx,
                                    "label": "final loss",
                                    "train_loss": loss_val,
                                    "val_loss": None,
                                    "accuracy": None,
                                    "step_accuracy": None,
                                    "timestamp": int(now),
                                }
                            )

                        if now - last_persist >= 1.5:
                            snapshot = dict(job)
                            last_persist = now

                    if snapshot:
                        self._upsert_session(snapshot)

            return_code = proc.wait()
            if self._stop_event.is_set():
                return

            elapsed_total = round(time.time() - start_time, 1)
            result_file = output_dir / "result.json"
            result_data: Dict[str, Any] = {}
            if result_file.is_file():
                try:
                    result_data = json.loads(result_file.read_text(encoding="utf-8"))
                except Exception as e:
                    logger.warning(f"Could not parse result.json: {e}")

            final_snapshot = None
            with self._lock:
                if self._current_job and self._current_job["job_id"] == job_id:
                    job = self._current_job
                    if return_code == 0 and result_data.get("status") == "ok":
                        step_checks = result_data.get("action_self_check_steps") or {}
                        total_s = len(step_checks)
                        ok_s = sum(1 for v in step_checks.values() if isinstance(v, dict) and v.get("status") == "COMPLETED")
                        if total_s > 0:
                            acc: Optional[float] = round((ok_s / total_s) * 100.0, 1)
                        elif result_data.get("self_check") == "NORMAL":
                            acc = 100.0
                        else:
                            acc = None

                        total_stages = int(job.get("total_epochs") or 9)
                        stage_secs = result_data.get("stage_seconds") or {}
                        for s in job.get("stages") or []:
                            s["status"] = "done"
                            s["sub_progress"] = 100
                            sec_val = stage_secs.get(f"{s['id']} {s['key']}")
                            if sec_val is not None:
                                s["duration_seconds"] = round(float(sec_val), 1)
                        job["status"] = "completed"
                        job["current_epoch"] = total_stages
                        job["progress_percent"] = 100
                        job["eta_seconds"] = 0
                        job["stage_name"] = "Training Completed"
                        job["stage_detail"] = f"Self-check: {result_data.get('self_check') or 'OK'}"
                        job["elapsed_seconds"] = int(round(result_data.get("elapsed_seconds") or elapsed_total))
                        job["accuracy"] = acc
                        job["model_name"] = result_data.get("model_name") or model_name
                        job["self_check"] = result_data.get("self_check")
                        if result_data.get("steps"):
                            job["steps"] = [
                                {
                                    "id": s.get("id", idx + 1),
                                    "action_query": s.get("label") or f"Step {idx + 1}",
                                    "part": s.get("object") or "",
                                    "count": 1,
                                }
                                for idx, s in enumerate(result_data["steps"])
                            ]
                        job["completed_at"] = datetime.utcnow().isoformat()
                        job["logs"].append(
                            f"[{_format_ts_prefix()}] Training completed! Model: '{job['model_name']}' | Self-check: {job.get('self_check')}"
                        )
                        try:
                            if output_dir.parent.name.lower() == "sequence_train":
                                ws = output_dir.parent
                                _ensure_inference_workspace(ws)
                                src_m = output_dir / "trained" / job["model_name"]
                                if src_m.is_dir():
                                    shutil.copytree(src_m, ws / "trained" / job["model_name"], dirs_exist_ok=True)
                                    (ws / "trained" / "active_model.txt").write_text(job["model_name"], encoding="utf-8")
                                    seqinf.set_data_home(str(ws))
                        except Exception as sync_err:
                            logger.warning(f"Auto-sync to sequence_train workspace failed: {sync_err}")
                    else:
                        err_msg = result_data.get("error") or f"Training failed with exit code {return_code}"
                        for s in job.get("stages") or []:
                            if s["id"] == current_stage:
                                s["status"] = "failed"
                        job["status"] = "failed"
                        job["eta_seconds"] = 0
                        job["elapsed_seconds"] = int(round(elapsed_total))
                        job["error"] = err_msg
                        job["stage_detail"] = err_msg
                        job["logs"].append(f"[{_format_ts_prefix()}] ERROR: {err_msg}")
                    final_snapshot = dict(job)

            if final_snapshot:
                self._upsert_session(final_snapshot)

        except Exception as e:
            logger.error(f"Training subprocess error in {job_id}: {e}", exc_info=True)
            err_snapshot = None
            with self._lock:
                if self._current_job and self._current_job["job_id"] == job_id:
                    self._current_job["status"] = "failed"
                    self._current_job["error"] = str(e)
                    self._current_job["logs"].append(f"[{_format_ts_prefix()}] Subprocess error: {e}")
                    err_snapshot = dict(self._current_job)
            if err_snapshot:
                self._upsert_session(err_snapshot)
        finally:
            with self._lock:
                self._proc = None
            app_state.exit_training()

    def deploy_model(self, job_id: str) -> Dict[str, Any]:
        """Copies trained model package into active SOP_MONITOR_HOME and loads it for inference."""
        output_dir = None
        model_name = "trained_model"

        with self._lock:
            if self._current_job and self._current_job.get("job_id") == job_id:
                model_name = self._current_job.get("model_name", model_name)
                if self._current_job.get("output_dir"):
                    output_dir = Path(self._current_job["output_dir"])

        if output_dir is None and job_id:
            output_dir = self._job_output_dirs.get(job_id)

        db = SessionLocal()
        try:
            row = db.query(TrainingSession).filter(TrainingSession.job_id == job_id).first() if job_id else None
            if row:
                if model_name == "trained_model" and row.model_name:
                    model_name = row.model_name
                if output_dir is None and getattr(row, "output_dir", ""):
                    output_dir = Path(row.output_dir)
        finally:
            db.close()

        if output_dir is None and job_id:
            output_dir = _get_packages_dir() / job_id

        # Check result.json inside the trained package directory
        if output_dir and (output_dir / "result.json").is_file():
            try:
                res = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
                model_name = res.get("model_name") or model_name
            except Exception:
                pass

        # Copy trained/<model_name> + weights + models.json into unified workspace SOP_MONITOR_HOME
        if output_dir and output_dir.is_dir():
            src_model_dir = output_dir / "trained" / model_name
            if not src_model_dir.is_dir():
                return {"success": False, "message": f"Trained model folder not found: {src_model_dir}"}

            try:
                # If package lives inside a sequence_train workspace, make that workspace active inference home
                if output_dir.parent.name.lower() == "sequence_train":
                    home = output_dir.parent
                    _ensure_inference_workspace(home)
                    seqinf.set_data_home(str(home))
                else:
                    home = Path(seqinf.prepare_environment()["home"])
                (home / "trained").mkdir(parents=True, exist_ok=True)
                (home / "weights" / "detectors").mkdir(parents=True, exist_ok=True)

                # Copy trained model directory
                dst_model_dir = home / "trained" / model_name
                shutil.copytree(src_model_dir, dst_model_dir, dirs_exist_ok=True)

                # Copy detector weights
                src_det_dir = output_dir / "weights" / "detectors"
                if src_det_dir.is_dir():
                    for pt_file in src_det_dir.glob("*.pt"):
                        dst_pt = home / "weights" / "detectors" / pt_file.name
                        if not dst_pt.exists():
                            shutil.copy2(pt_file, dst_pt)

                # Copy any shared weight/backbone assets dynamically if missing in active home
                src_weights_dir = output_dir / "weights"
                if src_weights_dir.is_dir():
                    for wfile in src_weights_dir.iterdir():
                        if wfile.is_file():
                            dst_w = home / "weights" / wfile.name
                            if not dst_w.exists():
                                shutil.copy2(wfile, dst_w)

                # Merge models.json entries without duplicating existing names
                src_mj = output_dir / "models.json"
                dst_mj = home / "models.json"
                if src_mj.is_file():
                    src_entries = json.loads(src_mj.read_text(encoding="utf-8"))
                    dst_entries = json.loads(dst_mj.read_text(encoding="utf-8")) if dst_mj.is_file() else []
                    existing_names = {e.get("name") for e in dst_entries}
                    for entry in src_entries:
                        if entry.get("name") not in existing_names:
                            dst_entries.append(entry)
                            existing_names.add(entry.get("name"))
                    dst_mj.write_text(json.dumps(dst_entries, indent=1), encoding="utf-8")

                # Set as active model
                (home / "trained" / "active_model.txt").write_text(model_name, encoding="utf-8")

                # Hot-load into inference engine if GPU isn't currently locked by another session
                try:
                    eng = seqinf.get_engine()
                    if eng.active_session_id is None:
                        eng.load(model_name, force=True)
                except Exception as e:
                    logger.warning(f"Deferred engine reload during deploy: {e}")

                logger.info(f"[DEPLOY] Deployed trained model '{model_name}' to {home}")
            except Exception as e:
                logger.error(f"Failed deploying model '{model_name}': {e}", exc_info=True)
                return {"success": False, "message": f"Failed to deploy model: {e}"}

        return {
            "success": True,
            "message": f"Model '{model_name}' successfully deployed as active SOP model.",
            "model_name": model_name,
        }


training_service = RealTrainingService()
