"""
debug_routes.py - reports installed ML modules and trained sequence models.
    GET /debug/ml-module-info
"""

import os
import sys
import hashlib
import importlib.metadata as md

from fastapi import APIRouter

router = APIRouter(prefix="/debug", tags=["debug"])


def _file_hash(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:12]
    except Exception:
        return "unreadable"


@router.get("/ml-module-info")
def ml_module_info():
    """
    Reports installed sop_inference / sequence_inference modules,
    wheel version, and trained model availability.
    """
    info = {"import_succeeded": False}
    try:
        from app.ml import sequence_inference as seqinf
        seqinf.prepare_environment()
        infer_mod = seqinf.import_infer()
        file_path = getattr(infer_mod, "__file__", None)
        info.update({
            "module_name": "sop_inference",
            "import_succeeded": True,
            "file_path": file_path,
            "is_installed_wheel": bool(
                file_path and ("site-packages" in file_path or "dist-packages" in file_path)
            ),
            "last_modified": os.path.getmtime(file_path) if file_path and os.path.exists(file_path) else None,
            "sha256_short": _file_hash(file_path) if file_path else None,
            "version": getattr(infer_mod, "__version__", "1.0.2"),
        })
        try:
            info["installed_wheel_version"] = md.version("sop_inference")
        except md.PackageNotFoundError:
            info["installed_wheel_version"] = None
    except Exception as e:
        info["error"] = str(e)

    # Active sequence models
    try:
        from app.ml import sequence_inference as seqinf
        info["models"] = seqinf.list_models()
        info["active_model"] = seqinf.resolve_model_name()
    except Exception as e:
        info["models_error"] = str(e)

    return info