"""
camera_inference_service.py
===========================
Live, real-time per-frame SOP SEQUENCE inference (camera feeds, OAK streams, RTSP).

When to use this file:
- A caller (e.g. `oak_camera_service.py`) grabs ONE frame at a time and needs the live sequence state back at once:
  which steps are done / doing / out of order, the detections, the timing.

When NOT to use this file:
- A whole uploaded / recorded video file -> `video_inference_service.py`.

Everything the ML does goes through the common layer `sequence_inference.py` (the same one the video service uses),
so a camera frame and a video frame return the SAME stats dict.

Main functions:
- `run_inference(frame, session_id, video_name, model_name)` -> (stats, annotated_frame)
- `finish_session(session_id)`      end of a product cycle -> stats incl. the VERDICT (NORMAL / ANOMALY / UNCERTAIN)
- `get_session_status(session_id)`
- `clear_session(session_id)`       drop the session (no verdict)
- `reset_session_for_next_video(session_id)`
- `cleanup_stale_sessions()`

One camera inspection = one product cycle:  frames ... -> finish_session() -> clear_session() -> next product.
(The ML keeps every frame's features of an inspection in memory, so do not let one inspection run for hours.)
"""

import time
import threading
from typing import Dict, Any, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None

from app.ml import app_state  # shared GPU mutual-exclusion flag (video vs camera vs training)
from app.core.logger import setup_logger

try:
    from app.ml import sequence_inference as seqinf
except ImportError:                                   # stand-alone use
    import sequence_inference as seqinf

logger = setup_logger("camera-inference")

_SESSION_IDLE_TIMEOUT_SEC = 300

_sessions: Dict[str, Dict[str, Any]] = {}
_sessions_lock = threading.Lock()


# ── shared engine registration ──
def _set_camera_engine(engine: Any) -> None:
    """register an already loaded SequenceEngine (normally not needed: seqinf.get_engine() is shared by itself)"""
    if isinstance(engine, seqinf.SequenceEngine):
        seqinf.set_engine(engine)
        logger.info("[CAMERA] Shared SequenceEngine registered successfully.")
    else:
        logger.warning("[CAMERA] _set_camera_engine ignored: not a SequenceEngine "
                       "(call sequence_inference.preload() at startup instead)")


_cuda_operational_cached = None


def is_cuda_operational() -> bool:
    """Verifies that PyTorch can actually execute CUDA kernels on GPU 0,
    rather than just checking if the driver library is queryable."""
    global _cuda_operational_cached
    if _cuda_operational_cached is not None:
        return _cuda_operational_cached
    if torch is None:
        return False

    if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
        return False

    for attempt in range(1, 4):
        try:
            if not torch.cuda.is_initialized():
                torch.cuda.init()
            torch.cuda.set_device(0)

            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.allow_tf32 = True
                # NOT benchmark=True: the sequence model's TCN input grows by one frame every frame, so benchmark
                # mode would re-tune the convolutions on every single frame (the ML itself switches it off).
                torch.backends.cudnn.benchmark = False

            t = torch.zeros((1, 1), device="cuda:0")
            _ = t + 1.0
            del t
            torch.cuda.synchronize(0)
            _cuda_operational_cached = True
            return True
        except Exception as e:
            logger.warning(f"CUDA initialization attempt {attempt}/3 encountered: {e}. Retrying...")
            time.sleep(0.3)

    logger.error("CUDA is present but kernel execution failed after 3 attempts; falling back to CPU.")
    _cuda_operational_cached = False
    return False


def _get_or_create_session(session_id: str, original_filename: Optional[str],
                           model_name: Optional[str]) -> Dict[str, Any]:
    # Fast path: session already exists, no need to touch the lock at all.
    existing = _sessions.get(session_id)
    if existing is not None and "seq" in existing:
        return existing

    with _sessions_lock:
        if session_id in _sessions and "seq" in _sessions[session_id]:
            return _sessions[session_id]

        if not app_state.try_enter_inference("camera"):
            active_kind = app_state.get_active_inference_kind()
            if app_state.get_mode() == "TRAINING":
                raise RuntimeError("Training is in progress -- camera inference cannot start")
            raise RuntimeError(f"GPU is currently in use by {active_kind or 'another process'} "
                               "-- try again shortly")

        try:
            # the model is loaded ONCE per process (seqinf.preload() at startup); a first call loads it here
            seq = seqinf.get_engine().open_session(session_id, kind="camera", source=original_filename,
                                                   model_name=model_name)
        except seqinf.SequenceBusyError as e:
            app_state.exit_inference("camera")
            raise RuntimeError(str(e))
        except Exception as e:
            logger.error(f"Failed to start the sequence inspection for camera session {session_id}: {e}",
                         exc_info=True)
            app_state.exit_inference("camera")
            raise RuntimeError(str(e)) if isinstance(e, seqinf.SequenceInferenceError) else e

        session = {
            "seq": seq,
            "frames_processed": 0,
            "original_filename": original_filename,
            "model_name": seq.model_name,
            "last_active": time.time(),
            "inference_claimed": True,
            "inference_lock": threading.Lock(),
            "last_stats": None,
            "final": None,
        }
        _sessions[session_id] = session
        return session


def cleanup_stale_sessions(max_idle_seconds: int = _SESSION_IDLE_TIMEOUT_SEC) -> None:
    """Call this periodically (e.g. from a FastAPI startup background task
    running every few minutes) -- without it, _sessions grows forever for
    any camera client that disconnects without calling clear_session()."""
    now = time.time()
    stale = [sid for sid, s in list(_sessions.items())
             if now - s.get("last_active", now) > max_idle_seconds]
    for sid in stale:
        logger.info(f"Evicting stale camera session {sid}")
        clear_session(sid)


def run_inference(frame, session_id: str,
                  video_name: Optional[str] = None,
                  model_name: Optional[str] = None,
                  **_kw) -> Tuple[Dict[str, Any], Any]:
    """
    Returns (stats, annotated_frame) as a TUPLE -- stats is JSON-safe on its own (no numpy arrays
    inside it), annotated_frame (boxes + status bar + sequence overlay) is ready to encode/stream.

    model_name (optional, first frame of a session only) picks a trained model; default = the configured / active one.

    stats keys: status ("NORMAL" | "ANOMALY" | "WARMING", at finish verdict), state, steps[]
    (step / label / status done|doing|waiting|out_of_order|taken_back / t / detected), sequence[],
    steps_done_count / total_steps / steps_progress, is_anomaly_frame, record, events[], detections[],
    fps, latency_ms, stage_ms.
    """
    if app_state.get_mode() == "TRAINING":
        return {"session_id": session_id, "status": "paused",
                "state": "PAUSED", "is_anomaly_frame": False,
                "frames_processed": 0}, frame

    try:
        session = _get_or_create_session(session_id, video_name, model_name)
    except RuntimeError as e:
        logger.warning(f"Could not start camera session {session_id}: {e}")
        return {"session_id": session_id, "status": "error",
                "state": "ERROR", "is_anomaly_frame": False,
                "error": str(e), "frames_processed": 0}, frame

    session["last_active"] = time.time()

    # If already finished, return cached final stats
    if session.get("final") is not None:
        return dict(session["last_stats"] or {}), frame

    seq = session["seq"]
    try:
        with session["inference_lock"]:
            fr = seq.process(frame)                      # t = None: camera's wall clock
    except Exception as e:
        logger.error(f"Error in sequence inference for session {session_id}: {e}", exc_info=True)
        return {"session_id": session_id, "status": "error", "state": "ERROR", "error": str(e),
                "frames_processed": session["frames_processed"]}, frame

    session["frames_processed"] = fr.stats.get("frames_processed", fr.stats.get("frame", session["frames_processed"] + 1))

    try:
        annotated_frame = seq.draw(frame, fr)
    except Exception as draw_err:
        logger.warning(f"Sequence draw failed on camera frame {session['frames_processed']}: {draw_err}")
        annotated_frame = frame

    stats = dict(fr.stats)
    stats["session_id"] = session_id
    stats["original_filename"] = session.get("original_filename")
    session["last_stats"] = stats
    return stats, annotated_frame


def finish_session(session_id: str) -> Optional[Dict[str, Any]]:
    """End of a product cycle: the verdict of the whole inspection (NORMAL / ANOMALY / UNCERTAIN), the sequence in the
    order it was done, every step's status / start / done time and the findings.  The GPU claim is released here; the
    session stays readable (get_session_status) until clear_session().  -> the final stats, or None for an unknown
    session."""
    session = _sessions.get(session_id)
    if not session:
        return None
    with session["inference_lock"]:
        if session.get("final") is None:
            final = session["seq"].finish()
            base = dict(session.get("last_stats") or seqinf.blank_stats(
                session_id, session.get("original_filename"), "camera", session.get("model_name"),
                session["seq"].steps))
            session["last_stats"] = seqinf.merge_final(base, final)
            session["final"] = final
            if session.get("inference_claimed"):
                session["inference_claimed"] = False
                app_state.exit_inference("camera")
            logger.info(f"Camera session {session_id}: verdict {final.stats.get('verdict')}  "
                        f"sequence {final.stats.get('sequence')}")
    return session["last_stats"]


def get_session_status(session_id: str) -> Optional[Dict[str, Any]]:
    session = _sessions.get(session_id)
    if not session:
        return None
    stats = session.get("last_stats")
    if stats and isinstance(stats, dict):
        return {k: v for k, v in stats.items() if not k.startswith("_")}
    return stats


def clear_session(session_id: str) -> None:
    with _sessions_lock:
        session = _sessions.pop(session_id, None)
    if session:
        try:
            seq = session.get("seq")
            if seq is not None and not seq.closed:
                seq.abort("session cleared")
        finally:
            if session.get("inference_claimed"):
                app_state.exit_inference("camera")
        logger.info(f"Cleared camera session {session_id}")


def reset_session_for_next_video(session_id: str) -> None:
    clear_session(session_id)
