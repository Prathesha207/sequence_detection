"""
model_router.py - Endpoints for inspecting trained SOP sequence models and steps.
"""
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from app.ml import sequence_inference as seqinf

router = APIRouter(prefix="/models", tags=["Models"])


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
