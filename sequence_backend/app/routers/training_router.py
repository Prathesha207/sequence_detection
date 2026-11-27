"""
training_router.py - Endpoints for managing and monitoring the SOP model training pipeline.
"""

import os
import re
import string
import uuid
import shutil
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query, Body, UploadFile, File
from app.core.app_paths import APP_DIR
from app.services.training_service import training_service

router = APIRouter(prefix="/training", tags=["Training Pipeline"])


def _get_uploads_dir() -> Path:
    """Return writable directory for uploaded training videos."""
    preferred = APP_DIR / "training_uploads"
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "sequence_training_uploads"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


@router.post("/upload-video")
async def upload_training_video(file: UploadFile = File(...)):
    """Save an uploaded training video to disk and return its path for train.py."""
    try:
        orig_name = Path(file.filename or f"training_{uuid.uuid4().hex[:8]}.mp4").name
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", orig_name) or "training.mp4"
        dest_path = _get_uploads_dir() / f"{uuid.uuid4().hex[:8]}_{safe_name}"

        with open(dest_path, "wb") as out_f:
            shutil.copyfileobj(file.file, out_f)

        file_size = os.path.getsize(dest_path) if dest_path.exists() else 0
        if file_size == 0:
            dest_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Uploaded video file is empty")

        return {
            "ok": True,
            "video_path": str(dest_path),
            "filename": orig_name,
            "file_size": file_size,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save training video: {e}")


@router.get("/detectors")
def list_training_detectors():
    """List available YOLO detectors packaged inside the installed sop_sequence wheel."""
    try:
        import sop_sequence
        models_file = Path(sop_sequence.data_folder) / "models.json"
        if not models_file.is_file():
            return []
        return json.loads(models_file.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
def get_training_config():
    """Return wheel detector options, DB-saved output folder, workspace path, and wheel config."""
    try:
        saved_out = training_service.get_saved_output_root()
        workspace_dir = str(training_service.get_effective_workspace(saved_out))
        return {
            "ok": True,
            "saved_output_dir": saved_out,
            "workspace_dir": workspace_dir,
            "default_output_dir": training_service.get_default_output_dir(),
            "detectors": list_training_detectors(),
            "wheel_config": training_service.get_wheel_config(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/output-folder")
def save_training_output_folder(payload: Dict[str, Any] = Body(...)):
    """Save output folder in SQLite DB and configure sequence_train as unified training + inference home."""
    try:
        raw_out = (payload.get("output_dir") or "").strip()
        res = training_service.set_saved_output_root(raw_out)
        if not res.get("ok"):
            raise HTTPException(status_code=400, detail=res.get("error", "Invalid output folder"))
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
def get_training_status(job_id: Optional[str] = Query(None)):
    """Retrieve the current training job status, telemetry metrics, and logs."""
    try:
        job = training_service.get_current_job(job_id)
        return {"ok": True, "job": job}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
def start_training_job(payload: Dict[str, Any] = Body(...)):
    """Launch the real sop_sequence wheel training pipeline in a subprocess."""
    try:
        res = training_service.start_training(payload)
        if "error" in res:
            status_code = 409 if res.get("busy") else 400
            raise HTTPException(status_code=status_code, detail=res["error"])
        return {"ok": True, "job": res.get("job")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
def stop_training_job(payload: Dict[str, Any] = Body(...)):
    """Stop an active training job and terminate its subprocess."""
    try:
        job_id = payload.get("job_id", "")
        res = training_service.stop_training(job_id)
        return {"ok": True, "job": res.get("job")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=List[Dict[str, Any]])
def get_training_history():
    """List completed and previous training runs with their metrics."""
    try:
        return training_service.get_history()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deploy")
def deploy_trained_model(payload: Dict[str, Any] = Body(...)):
    """Deploy a trained model package from training_packages into active inference."""
    try:
        job_id = payload.get("job_id", "")
        res = training_service.deploy_model(job_id)
        if not res.get("success"):
            raise HTTPException(status_code=400, detail=res.get("message", "Deploy failed"))
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
def reset_training_status():
    """Clear finished/failed job state back to idle while keeping SQLite history."""
    try:
        res = training_service.reset_current_job()
        if "error" in res:
            raise HTTPException(status_code=409, detail=res["error"])
        return {"ok": True, "job": res}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _list_os_roots() -> List[str]:
    if os.name == "nt":
        return [f"{letter}:\\" for letter in string.ascii_uppercase if os.path.exists(f"{letter}:\\")]
    return ["/"]


@router.get("/browse-folders")
def browse_training_folders(path: Optional[str] = Query(None)):
    """Cross-platform directory browser (Windows drives & Linux/POSIX roots) for output_dir selection."""
    roots = _list_os_roots()
    home_dir = str(Path.home())
    raw = (path or "").strip()

    if raw == "__ROOTS__":
        folders = [{"name": r, "path": r} for r in roots]
        return {
            "ok": True,
            "current_path": "",
            "parent_path": None,
            "roots": roots,
            "home_path": home_dir,
            "folders": folders,
        }

    target = Path(raw).expanduser() if raw else Path(training_service.get_default_output_dir())
    if not target.exists() or not target.is_dir():
        target = Path.home()
    try:
        resolved = target.resolve()
    except Exception:
        resolved = target

    if resolved.parent != resolved:
        parent_path: Optional[str] = str(resolved.parent)
    elif os.name == "nt":
        parent_path = "__ROOTS__"
    else:
        parent_path = None

    folders: List[Dict[str, str]] = []
    try:
        with os.scandir(resolved) as it:
            entries = sorted(
                (e for e in it if e.is_dir(follow_symlinks=False) and not e.name.startswith(".") and not e.name.startswith("$")),
                key=lambda e: e.name.lower(),
            )
            for e in entries:
                folders.append({"name": e.name, "path": e.path})
    except PermissionError:
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot list folder '{resolved}': {e}")

    return {
        "ok": True,
        "current_path": str(resolved),
        "parent_path": parent_path,
        "roots": roots,
        "home_path": home_dir,
        "folders": folders,
    }


@router.post("/mkdir")
def create_training_folder(payload: Dict[str, Any] = Body(...)):
    """Create a subfolder cleanly inside the cross-platform folder browser."""
    parent = (payload.get("parent_path") or "").strip()
    name = (payload.get("folder_name") or "").strip()
    if not parent or not name:
        raise HTTPException(status_code=400, detail="parent_path and folder_name are required")
    if any(sep in name for sep in ("/", "\\", "..")) or name in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid folder name")
    try:
        dest = Path(parent).expanduser().resolve() / name
        dest.mkdir(parents=False, exist_ok=True)
        return {"ok": True, "path": str(dest)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not create folder: {e}")
