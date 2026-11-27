"""
model_router.py - Endpoints for inspecting, selecting, and deleting SOP sequence models.
"""
from typing import List, Dict, Any, Optional
import gc
import json
import re
import shutil
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.ml import sequence_inference as seqinf
from app.ml import app_state

router = APIRouter(prefix="/models", tags=["Models"])


class SelectModelRequest(BaseModel):
    model_name: Optional[str] = None
    name: Optional[str] = None


@router.get("", response_model=List[Dict[str, Any]])
def list_models():
    """List all available trained sequence models."""
    try:
        return seqinf.list_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {e}")


@router.get("/active")
def get_active_model():
    """Get the currently configured active sequence model and its steps."""
    try:
        active = seqinf.resolve_model_name()
        return {
            "model": active,
            "steps": seqinf.model_steps(active),
            "preflight": seqinf.preflight(active)
        }
    except seqinf.SequenceInferenceError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{name}/steps")
def model_steps(name: str):
    """Get step definitions for a specific trained model."""
    try:
        return seqinf.model_steps(name)
    except seqinf.SequenceInferenceError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{name}/preflight")
def preflight_model(name: str):
    """Check if model weights and spec files exist and are valid."""
    try:
        return seqinf.preflight(name)
    except seqinf.SequenceInferenceError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/select")
def select_model(req: SelectModelRequest):
    """Set the active workspace model and load it into the inference engine when idle."""
    requested = (req.model_name or req.name or "").strip()
    if not requested:
        raise HTTPException(status_code=400, detail="model_name is required")
    try:
        env = seqinf.prepare_environment()
        resolved = seqinf.resolve_model_name(requested)
        pre = seqinf.preflight(resolved)
        if not pre.get("ok"):
            problems = ", ".join(pre.get("problems") or ["invalid model files"])
            raise HTTPException(status_code=400, detail=f"Model '{resolved}' cannot be loaded: {problems}")

        eng = seqinf.get_engine()
        if eng.active_session_id is not None:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot switch model while inspection '{eng.active_session_id}' is running",
            )

        trained_dir = Path(env["home"]) / "trained"
        trained_dir.mkdir(parents=True, exist_ok=True)
        (trained_dir / "active_model.txt").write_text(resolved, encoding="utf-8")

        loaded = False
        if app_state.get_mode() != "TRAINING":
            eng.load(resolved)
            loaded = True

        return {
            "ok": True,
            "model": resolved,
            "loaded": loaded,
            "steps": seqinf.model_steps(resolved),
            "preflight": pre,
        }
    except HTTPException:
        raise
    except seqinf.SequenceBusyError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except seqinf.SequenceInferenceError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{name}")
def delete_model(name: str):
    """Delete a user-trained workspace model (pre-trained .whl models cannot be deleted)."""
    target = (name or "").strip()
    if (
        not target
        or target in {".", ".."}
        or "/" in target
        or "\\" in target
        or target.lower() in {"trained", "weights", "outputs", "specs"}
    ):
        raise HTTPException(status_code=400, detail="Invalid model name")

    try:
        env = seqinf.prepare_environment()
        ws_root = Path(env["home"]).resolve()
        trained_root = (ws_root / "trained").resolve()
        ws_model_dir = (trained_root / target).resolve()

        if ws_model_dir.parent != trained_root:
            raise HTTPException(status_code=400, detail="Invalid model path")

        if not ws_model_dir.is_dir():
            wheel_models = [m for m in seqinf.list_models() if m.get("name") == target and m.get("source") == "wheel"]
            if wheel_models:
                raise HTTPException(status_code=403, detail="Pre-trained wheel models cannot be deleted")
            raise HTTPException(status_code=404, detail=f"Trained model '{target}' not found in workspace")

        if app_state.get_mode() == "TRAINING":
            raise HTTPException(status_code=409, detail="Cannot delete a model while training is in progress")

        eng = seqinf.get_engine()
        if eng.active_session_id is not None:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot delete model while inspection '{eng.active_session_id}' is running",
            )

        if eng.loaded and eng.model_name == target:
            eng.close()
            gc.collect()

        shutil.rmtree(ws_model_dir)

        # Clean up matching training package directory (e.g. <workspace>/<sanitized_name>) if present
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", target).strip("._-")
        for child in list(ws_root.iterdir()):
            if not child.is_dir():
                continue
            if child.name.lower() in {"trained", "weights", "outputs", "specs"}:
                continue
            res_file = child / "result.json"
            belongs_to_target = False
            if res_file.is_file():
                try:
                    meta = json.loads(res_file.read_text(encoding="utf-8", errors="ignore"))
                    if meta.get("model_name") == target:
                        belongs_to_target = True
                except Exception:
                    pass
            elif child.name == target or (safe_name and child.name == safe_name and (child / "sop_prompt.txt").is_file()):
                belongs_to_target = True

            if belongs_to_target:
                shutil.rmtree(child, ignore_errors=True)

        # Update active_model.txt if it pointed to the deleted model
        act_file = trained_root / "active_model.txt"
        remaining = [m for m in seqinf.list_models() if "error" not in m]
        next_active: Optional[str] = remaining[0]["name"] if remaining else None
        if act_file.is_file():
            cur_act = act_file.read_text(encoding="utf-8").strip()
            if cur_act == target:
                if next_active:
                    act_file.write_text(next_active, encoding="utf-8")
                else:
                    act_file.unlink(missing_ok=True)
        elif next_active:
            act_file.write_text(next_active, encoding="utf-8")

        return {
            "ok": True,
            "deleted": target,
            "next_active_model": next_active,
        }
    except HTTPException:
        raise
    except PermissionError as e:
        raise HTTPException(status_code=423, detail=f"Model files are in use and could not be deleted: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model '{target}': {e}")
