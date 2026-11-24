"""
video_inference_service.py
==========================
Handles batch / offline video file SOP SEQUENCE inference.

When to use this file:
- An MP4/AVI video has been uploaded by the user, OR
- A recorded camera video file is being processed asynchronously.
- Manages long-running video decoding sessions, stream queues (MJPEG), saving annotated videos to disk,
  progress tracking and the final VERDICT.

When NOT to use this file:
- If you are receiving live frames in real-time from an OAK camera or RTSP stream,
  use `camera_inference_service.py` instead!

Everything the ML does goes through the common layer `sequence_inference.py` (the same one the camera service uses),
so a video frame and a camera frame return the SAME stats dict.

Per frame the stats hold the live sequence (steps done / doing / out of order, detections, timing); at the end of the
video they also hold the verdict:  stats["verdict"]  NORMAL | ANOMALY | UNCERTAIN, stats["sequence"],
stats["steps_result"], stats["reasons"], stats["report"].

Main exports:

- `video_inference_service = VideoInferenceService()`
- `ml_inference_service` (backward compatibility alias)
"""

import tempfile
import asyncio
import os
import time
import uuid
from typing import Dict, Any, Optional
import cv2

import concurrent.futures

try:
    import torch
except ImportError:
    torch = None

from app.ml import app_state  # shared GPU mutual-exclusion flag with training_service.py
                               # AND with camera_inference_service.py (video vs camera)
from app.core.logger import setup_logger

try:
    from app.ml import sequence_inference as seqinf
except ImportError:                                   # stand-alone use
    import sequence_inference as seqinf

logger = setup_logger("video-inference")


# ── shared engine registration ──
def _set_shared_engine(engine: Any) -> None:
    """register an already loaded SequenceEngine (normally not needed: seqinf.get_engine() is shared by itself)"""
    if isinstance(engine, seqinf.SequenceEngine):
        seqinf.set_engine(engine)
        logger.info("[VIDEO] Shared SequenceEngine registered successfully.")
    else:
        logger.warning("[VIDEO] _set_shared_engine ignored: not a SequenceEngine "
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
        # Do not log here on every call; setup scripts handle the "no cuda" warning
        return False

    # Robust retry loop to wake up GPU from power-saving / D3 sleep
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


def _infer_frame(seq, frame, t):
    """runs on the single ML worker thread: one frame through the sequence model"""
    return seq.process(frame, t)


class VideoInferenceService:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        # Global lock to ensure only one GPU inference runs at a time
        self._gpu_lock = asyncio.Lock()
        # Single-thread executor for synchronous inference to avoid blocking the asyncio event loop
        self._ml_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        # Non-blocking async background executor for saving anomaly frames to disk without stalling inference
        self._io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    # ── model information (no GPU needed) ──
    def list_models(self):
        """the trained sequence models available (name, trained_at, steps, active ...)"""
        return seqinf.list_models()

    def model_info(self) -> Dict[str, Any]:
        """the loaded model's info (detectors, steps, fps, action model ...); {} before the first load"""
        return dict(seqinf.get_engine().info)

    def stop_all_sessions(self):
        for sid, session in self.sessions.items():
            if not session["stop_event"].is_set():
                session["stop_event"].set()

    def create_session(self, *args, original_filename: Optional[str] = None,
                       session_id: Optional[str] = None, model_name: Optional[str] = None, **_kw) -> str:
        """model_name (optional) picks a trained model for this video; default = the configured / active one."""
        if args:
            if isinstance(args[0], str) and not original_filename:
                original_filename = args[0]
            elif len(args) > 1 and isinstance(args[1], str) and not original_filename:
                original_filename = args[1]

        # Fast, best-effort rejection up front so the caller gets an immediate
        # error instead of a session that will fail later. This is NOT the
        # real claim on the GPU -- that happens in process_video_task, right
        # before frames actually start flowing, and is released when that
        # task ends. A session sitting here queued (created but never
        # uploaded/started) must not hold the lock indefinitely.
        if app_state.get_mode() == "TRAINING":
            raise RuntimeError("Training is in progress -- please wait for it to finish before starting inference")
        if app_state.get_active_inference_kind() == "camera":
            raise RuntimeError("Live camera inference is currently active -- please stop it before starting a video upload")

        # Preemptively stop any currently running jobs so the GPU lock is freed immediately
        self.stop_all_sessions()

        session_id = session_id or str(uuid.uuid4())
        try:                                    # the step list of the model, so the UI can show it before frame 1
            resolved = seqinf.resolve_model_name(model_name)
            steps = seqinf.model_steps(resolved)
        except Exception as e:
            logger.warning(f"Could not read the model's steps for session {session_id}: {e}")
            resolved, steps = model_name, []
        stats = seqinf.blank_stats(session_id, original_filename, "video", resolved, steps)
        stats.update({"total_frames": 0, "progress": 0.0})
        self.sessions[session_id] = {
            "status": "queued",
            "queue": asyncio.Queue(maxsize=2),
            "original_filename": original_filename,
            "model_name": model_name,
            "seq": None,
            "temp_files_to_cleanup": [],
            "stats": stats,
            "stop_event": asyncio.Event(),
            "run_seq": 0,
            "temp_file": None
        }
        return session_id

    def get_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        session = self.sessions.get(session_id)
        if not session:
            return None
        return session["stats"]

    async def get_stream_generator(self, session_id: str):
        session = self.sessions.get(session_id)
        if not session:
            return

        # Immediately yield pre-extracted frame 0 (if available) so the client's
        # <img> tag gets an instant frame without waiting for ML processing
        if session.get("last_frame_bytes"):
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + session["last_frame_bytes"] +
                b"\r\n"
            )

        queue = session["queue"]
        try:
            while True:
                frame_bytes = await queue.get()
                if frame_bytes is None:
                    # Graceful completion sentinel received
                    break

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame_bytes +
                    b"\r\n"
                )
        except asyncio.CancelledError:
            logger.info(f"Stream disconnected for session {session_id}")

    async def stop_session(self, session_id: str, timeout: float = 2.5):
        session = self.sessions.get(session_id)
        if session:
            session["stop_event"].set()
            session["status"] = "stopped"
            session["stats"]["status"] = "stopped"
            last_bytes = session.get("last_frame_bytes")
            session_dir = session.get("session_dir")
            if last_bytes and session_dir and os.path.isdir(session_dir):
                try:
                    with open(os.path.join(session_dir, "last_frame.jpg"), "wb") as f:
                        f.write(last_bytes)
                except Exception as e:
                    logger.warning(f"Could not write last_frame.jpg on stop_session: {e}")

            # Gracefully wait for the active frame processing task to exit and release _gpu_lock
            start_wait = time.time()
            while session.get("is_task_active", False) and (time.time() - start_wait < timeout):
                await asyncio.sleep(0.01)

    def clear_session_files(self, session_id: str):
        """Explicitly called only when the user clears/resets the video session."""
        session = self.sessions.get(session_id)
        if not session:
            return
        temp_files = list(session.get("temp_files_to_cleanup", []))
        for tf in temp_files:
            try:
                if tf and os.path.exists(tf):
                    if "Desktop" in tf and "recordings" in tf:
                        continue
                    os.remove(tf)
                    logger.info(f"Cleaned up temporary upload file on session clear: {tf}")
            except Exception as e:
                logger.warning(f"Could not remove temporary file {tf}: {e}")
        session["temp_files_to_cleanup"] = []

    def start_run(self, session_id: str) -> int:
        """Invalidate any previous task and prepare an entirely fresh stream."""
        session = self.sessions[session_id]
        session["run_seq"] += 1
        session["stop_event"].set()  # tells an older task to stop promptly
        old_queue = session["queue"]
        try:
            if old_queue.full():
                old_queue.get_nowait()
            old_queue.put_nowait(None)  # close old MJPEG consumers
        except (asyncio.QueueEmpty, asyncio.QueueFull):
            pass
        session["queue"] = asyncio.Queue(maxsize=2)
        session["stop_event"] = asyncio.Event()
        session["status"] = "processing"
        self._reset_session_stats(session_id)
        return session["run_seq"]

    def _is_current_run(self, session_id: str, run_seq: int) -> bool:
        session = self.sessions.get(session_id)
        return bool(session and session.get("run_seq") == run_seq)

    def _reset_session_stats(self, session_id: str):
        """Single source of truth for what a 'fresh run' of a session looks
        like. Used at the start of every process_video_task call, so a
        stop -> restart (replay) never leaks stale stats/steps
        from a previous run into the new one. video_router.py should NOT
        hand-reset individual stat fields itself -- call this instead, so
        the two never drift out of sync."""
        session = self.sessions.get(session_id)
        if not session:
            return
        old = session["stats"]
        steps = [{"step": r["step"], "label": r["label"]} for r in old.get("steps") or []]
        fresh = seqinf.blank_stats(session_id, session.get("original_filename"), "video",
                                   old.get("model_name"), steps)
        fresh.update({
            "status": "processing",
            "total_frames": old.get("total_frames", 0),
            "progress": 0.0,
            "original_filename": session.get("original_filename") or old.get("original_filename"),
        })
        old.clear()
        old.update(fresh)

    async def process_video_task(self, session_id: str, temp_file_path: str,
                                 original_filename: Optional[str] = None,
                                 run_seq: Optional[int] = None):
        session = self.sessions.get(session_id)
        if not session or (run_seq is not None and not self._is_current_run(session_id, run_seq)):
            return

        # Direct callers without a router generation still get a generation.
        if run_seq is None:
            run_seq = session.get("run_seq", 0)

        session["temp_file"] = temp_file_path
        if original_filename:
            session["original_filename"] = original_filename
            session["stats"]["original_filename"] = original_filename

        # start_run has already reset the stats.  Do not let a superseded
        # task reset statistics belonging to its replacement.
        if not self._is_current_run(session_id, run_seq):
            return

        # Tracks whether THIS task successfully claimed the cross-kind
        # inference lock, so the finally block releases it exactly once,
        # and only if it was actually acquired.
        claimed_inference_lock = False
        session_dir = output_path = video_filename = results_json_path = None

        async with self._gpu_lock:
            if not self._is_current_run(session_id, run_seq):
                return
            # Closes the race where an already-queued session could start
            # running just as a training job claims the GPU. create_session()
            # blocks NEW sessions, this blocks ones that were queued just before
            # training set the flag.
            if app_state.get_mode() == "TRAINING":
                session["status"] = "error"
                session["stats"]["status"] = "error"
                session["stats"]["reasons"] = ["Training is in progress -- try again after it finishes"]
                await self._cleanup_session(session_id)
                return

            if session["stop_event"].is_set():
                session["status"] = "stopped"
                session["stats"]["status"] = "stopped"
                await self._cleanup_session(session_id)
                return

            # Real claim on the GPU, right before we actually start loading a
            # model / running frames. Rejects if a camera session is active
            # (or training raced in between the check above and here).
            if not app_state.try_enter_inference("video"):
                session["status"] = "error"
                session["stats"]["status"] = "error"
                session["stats"]["reasons"] = [
                    "GPU is currently in use by another inference session (camera or training) -- try again shortly"
                ]
                await self._cleanup_session(session_id)
                return
            claimed_inference_lock = True

            session["status"] = "processing"
            session["stats"]["status"] = "processing"
            session["is_task_active"] = True

            logger.info(f"Starting inference for session {session_id}")

            cap = None
            out_writer = None
            seq = None
            pending = None
            final = None
            eof = False
            frame_idx = 0
            try:
                # Session output directory
                try:
                    from app.core.app_paths import get_ml_output_dir
                    base_output_dir = str(get_ml_output_dir())
                except Exception:
                    base_output_dir = os.path.join(tempfile.gettempdir(), "sequence_detection_output")
                session_dir = os.path.join(base_output_dir, session_id)
                os.makedirs(session_dir, exist_ok=True)
                session["session_dir"] = session_dir

                # Annotated video output filename
                orig_name = original_filename or session.get("original_filename")
                if orig_name:
                    base_name = os.path.splitext(os.path.basename(orig_name))[0]
                    video_filename = f"{base_name}.mp4" if base_name.startswith("annotated_") else f"annotated_{base_name}.mp4"
                else:
                    video_filename = f"annotated_{session_id}.mp4"

                output_path = os.path.join(session_dir, video_filename)
                results_json_path = os.path.join(session_dir, "results.json")

                # Force FFmpeg backend. The default MSMF backend on Windows often deadlocks in PyInstaller.
                cap = cv2.VideoCapture(temp_file_path, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    # Some browser-uploaded codecs are not readable by the
                    # OpenCV build, but their browser-safe H.264 copy is.
                    fallback_path = session.get("browser_video_path")
                    if fallback_path and fallback_path != temp_file_path:
                        cap.release()
                        cap = cv2.VideoCapture(fallback_path, cv2.CAP_FFMPEG)
                        if cap.isOpened():
                            logger.warning("OpenCV could not open raw upload; using browser transcode fallback")
                    if not cap.isOpened():
                        raise ValueError(f"Could not open video file: {temp_file_path}")

                raw_fps = cap.get(cv2.CAP_PROP_FPS)
                video_fps = raw_fps if (raw_fps and 10.0 <= raw_fps <= 120.0) else 30.0

                # ── the ML: loaded once per process (seqinf.preload()); this call only starts one inspection ──
                engine = seqinf.get_engine()
                loop = asyncio.get_running_loop()
                try:
                    seq = await loop.run_in_executor(
                        self._ml_executor,
                        lambda: engine.open_session(session_id, kind="video", source=orig_name,
                                                    model_name=session.get("model_name")))
                except seqinf.SequenceInferenceError as e:
                    raise RuntimeError(str(e)) from e
                logger.info(f"Sequence inspection started for video session {session_id} "
                            f"(model {seq.model_name}, fps={video_fps})")
                session["seq"] = seq
                session["stats"]["model_name"] = seq.model_name
                session["stats"]["output_dir"] = session_dir
                session["stats"]["results_json_path"] = results_json_path

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if total_frames <= 0:
                    cached_total = session.get("stats", {}).get("total_frames", 0)
                    if cached_total and cached_total > 0:
                        total_frames = cached_total
                    else:
                        try:
                            import imageio_ffmpeg
                            nframes, _ = imageio_ffmpeg.count_frames_and_secs(temp_file_path)
                            if nframes and nframes > 0:
                                total_frames = int(nframes)
                        except Exception:
                            pass
                session["stats"]["total_frames"] = max(1, total_frames)

                # Setup VideoWriter to save annotated output
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
                session["stats"]["output_file"] = output_path

                stream_annotated = bool(seqinf.get_config().stream_annotated)
                start_time = time.time()
                consecutive_failures = 0
                last_annotated = None
                # Abort only after 30 consecutive frame failures (not 10) so transient CUDA
                # errors on fresh user installs (driver warm-up, OOM spikes) don't kill the
                # session prematurely.
                max_consecutive_failures = 30
                # Hard ceiling on a single frame's inference time. Without this, a frame that
                # stalls inside the ML (no exception raised, no return -- just stuck) hangs the
                # whole session forever. A timeout flows into the SAME frame_err handling below
                # (failure counting, eventual clean abort) instead of freezing indefinitely.
                per_frame_timeout_s = 20.0

                while not session["stop_event"].is_set() and self._is_current_run(session_id, run_seq):
                    ret, frame = cap.read()
                    # A Stop -> Start may replace this task while OpenCV was
                    # decoding. Never publish even one old frame/stat update
                    # into the replacement run.
                    if not self._is_current_run(session_id, run_seq):
                        break

                    if not ret or frame is None:
                        # Transient decode drop recovery: try reading up to 5 times
                        # before declaring genuine EOF to prevent premature cutoff
                        recovered_frame = None
                        for _ in range(5):
                            r_retry, f_retry = cap.read()
                            if r_retry and f_retry is not None:
                                recovered_frame = f_retry
                                break
                        if recovered_frame is not None:
                            frame = recovered_frame
                        else:
                            # Genuine End of Video
                            logger.info(f"Session {session_id}: reached true EOF at frame {frame_idx} (estimated total: {total_frames})")
                            eof = True
                            break

                    height, width = frame.shape[:2]
                    frame_idx += 1
                    if frame_idx > session["stats"]["total_frames"]:
                        session["stats"]["total_frames"] = frame_idx

                    if frame_idx % 50 == 0:
                        logger.info(
                            f"Session {session_id}: heartbeat -- frame {frame_idx}/{total_frames} "
                            f"({round(time.time() - start_time, 1)}s elapsed)"
                        )

                    # The video's own clock: frame number / fps (frames are fed faster than real time)
                    t_video = (frame_idx - 1) / video_fps

                    fr = None
                    try:
                        pending = loop.run_in_executor(self._ml_executor, _infer_frame, seq, frame, t_video)
                        fr = await asyncio.wait_for(pending, timeout=per_frame_timeout_s)
                        if not self._is_current_run(session_id, run_seq):
                            break
                        consecutive_failures = 0
                    except asyncio.TimeoutError as frame_err:
                        consecutive_failures += 1
                        logger.warning(
                            f"Session {session_id}: frame {frame_idx} inference TIMED OUT after "
                            f"{per_frame_timeout_s}s ({consecutive_failures}/{max_consecutive_failures}) "
                            "-- treating as a failed frame; processing continues with the next frame."
                        )
                        if consecutive_failures >= max_consecutive_failures:
                            raise RuntimeError(
                                f"Aborting session: {max_consecutive_failures} consecutive frame failures "
                                f"(last: timeout after {per_frame_timeout_s}s on frame {frame_idx})"
                            ) from frame_err
                    except Exception as frame_err:
                        consecutive_failures += 1
                        logger.warning(
                            f"Session {session_id}: frame {frame_idx} inference failed "
                            f"({consecutive_failures}/{max_consecutive_failures}) "
                            f"[{type(frame_err).__name__}]: {frame_err}",
                            exc_info=True
                        )
                        if consecutive_failures >= max_consecutive_failures:
                            raise RuntimeError(
                                f"Aborting session: {max_consecutive_failures} consecutive frame failures (last error: {frame_err})"
                            ) from frame_err

                    if not self._is_current_run(session_id, run_seq):
                        break

                    # The annotated frame (boxes + status bar + the steps so far) -> the MP4 (and the stream, if asked)
                    annotated_frame = frame
                    if fr is not None:
                        try:
                            annotated_frame = seq.draw(frame, fr)
                        except Exception as draw_err:
                            logger.warning(f"Sequence draw failed on video frame {frame_idx}: {draw_err}")
                    last_annotated = annotated_frame
                    if out_writer:
                        out_writer.write(annotated_frame)

                    # Stream: the raw frame (the frontend renders the boxes + steps from the stats) or, with
                    # SOP_STREAM_ANNOTATED=1, the annotated one (any plain <img> viewer then shows everything)
                    success, buffer = cv2.imencode(".jpg", annotated_frame if stream_annotated else frame,
                                                   [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                    if success:
                        frame_bytes = buffer.tobytes()
                        session["last_frame_bytes"] = frame_bytes
                        if frame_idx % 30 == 1 and session_dir and os.path.isdir(session_dir):
                            try:
                                with open(os.path.join(session_dir, "last_frame.jpg"), "wb") as lf:
                                    lf.write(frame_bytes)
                            except Exception:
                                pass
                        try:
                            if session["queue"].full():
                                session["queue"].get_nowait()
                            session["queue"].put_nowait(frame_bytes)
                        except asyncio.QueueFull:
                            pass

                    elapsed = time.time() - start_time
                    fps = frame_idx / elapsed if elapsed > 0 else 0
                    if total_frames > 0 and frame_idx < total_frames:
                        progress = min(99.0, round((frame_idx / total_frames) * 100, 1))
                    elif total_frames > 0 and frame_idx >= total_frames:
                        progress = 99.0
                    else:
                        progress = 0.0

                    # Publish the frame's stats (the same dict the camera service returns).  The lifecycle
                    # `status` follows the live status (NORMAL / ANOMALY / WARMING) while processing, as before;
                    # "completed" + the verdict come at the end.
                    if fr is not None:
                        keep = {k: session["stats"].get(k) for k in
                                ("total_frames", "output_dir", "results_json_path", "output_file")}
                        session["stats"].update(fr.stats)
                        session["stats"].update({k: v for k, v in keep.items() if v is not None})
                        session["stats"]["original_filename"] = session.get("original_filename")
                    session["stats"].update({
                        "frames_processed": frame_idx,
                        "progress": round(progress, 1),
                        "fps": round(fps, 1),                       # wall-clock throughput of the whole run
                        "video_width": width,
                        "video_height": height,
                    })

                    # Yield control back to loop so FastAPI can serve other requests and check for stop signals
                    if session["stop_event"].is_set():
                        break
                    await asyncio.sleep(0.001)

                current = self._is_current_run(session_id, run_seq)

                # ── the end of the video: the VERDICT (the action model judges the whole record) ──
                if eof and current and frame_idx > 0:
                    session["stats"]["status"] = "processing"
                    final = await loop.run_in_executor(self._ml_executor, seq.finish)
                    seqinf.merge_final(session["stats"], final)
                    session["stats"]["status"] = "processing"       # "completed" is set below, after the files
                    logger.info(f"Session {session_id}: verdict {final.stats.get('verdict')}  "
                                f"sequence {final.stats.get('sequence')}")
                    # a 2-second card with the verdict at the end of the annotated video
                    if out_writer and last_annotated is not None:
                        try:
                            card = seqinf.draw_final(last_annotated, final)
                            for _ in range(int(2 * video_fps)):
                                out_writer.write(card)
                            if stream_annotated:
                                ok_, buf_ = cv2.imencode(".jpg", card, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                                if ok_:
                                    session["last_frame_bytes"] = buf_.tobytes()
                                    try:
                                        if session["queue"].full():
                                            session["queue"].get_nowait()
                                        session["queue"].put_nowait(session["last_frame_bytes"])
                                    except asyncio.QueueFull:
                                        pass
                        except Exception as card_err:
                            logger.warning(f"Could not append the verdict card: {card_err}")

                if eof and current:
                    session["status"] = "completed"
                    session["stats"]["status"] = "completed"       # the verdict is in stats["verdict"]
                    session["stats"]["progress"] = 100.0
                    if frame_idx > 0:
                        session["stats"]["total_frames"] = frame_idx
                        session["stats"]["frames_processed"] = frame_idx
                elif session["stop_event"].is_set() and current:
                    session["status"] = "stopped"
                    session["stats"]["status"] = "stopped"
                elif current:
                    session["status"] = "completed"
                    session["stats"]["status"] = "completed"
                    session["stats"]["progress"] = 100.0
                    if frame_idx > 0:
                        session["stats"]["total_frames"] = frame_idx
                        session["stats"]["frames_processed"] = frame_idx

                # Persist the final frame on disk so get_last_frame can serve it even across reloads
                last_bytes = session.get("last_frame_bytes")
                if last_bytes and session_dir and os.path.isdir(session_dir):
                    try:
                        with open(os.path.join(session_dir, "last_frame.jpg"), "wb") as f:
                            f.write(last_bytes)
                    except Exception as e:
                        logger.warning(f"Could not persist last_frame.jpg for session {session_id}: {e}")

                # Write the final session summary (stats incl. verdict, sequence, steps_result, report).
                try:
                    import json
                    with open(results_json_path, "w", encoding="utf-8") as f:
                        json.dump(session["stats"], f, indent=2, default=str)
                except Exception as e:
                    logger.warning(f"Could not write results.json for session {session_id}: {e}")

            except Exception as e:
                logger.error(f"Inference error in session {session_id}: {e}", exc_info=True)
                session["status"] = "error"
                session["stats"]["status"] = "error"
                cause = getattr(e, "__cause__", None)
                if cause and str(cause) not in str(e):
                    session["stats"]["reasons"] = [f"{e} (Root cause: {cause})"]
                else:
                    session["stats"]["reasons"] = [str(e)]
            finally:
                session["is_task_active"] = False
                if out_writer:
                    out_writer.release()
                if cap:
                    cap.release()

                # NOTE: Source video files (temp_file_path / inference_video_path)
                # are preserved so the user can pause, stop, and restart inference
                # at any time without receiving "Video file not found".
                # Cleanup is only performed when the session is explicitly cleared.

                try:
                    if pending:
                        # Give the previous frame a hard limit to finish before declaring the thread deadlocked
                        await asyncio.wait_for(pending, timeout=2.0)
                except asyncio.TimeoutError:
                    logger.error("FATAL: Previous ML frame is STILL running after timeout! The ML thread is deadlocked.")
                    raise RuntimeError("ML thread deadlocked. Session aborted to prevent application freeze.")
                except Exception:
                    pass
                finally:
                    # Release the ML for the next run: a finished inspection is already closed; a stopped / failed
                    # one is dropped without a verdict.  The model weights stay in GPU memory.
                    if seq is not None:
                        try:
                            seq.abort("video task ended")
                        except Exception as e:
                            logger.warning(f"Could not release the sequence session: {e}")
                    session["seq"] = None

                # Release the cross-kind GPU claim exactly once, only if this
                # task actually acquired it.
                if claimed_inference_lock:
                    app_state.exit_inference("video")

                # Never close the replacement stream or schedule expiry for a
                # newer run of this same session.
                if self._is_current_run(session_id, run_seq):
                    await self._cleanup_session(session_id)
                logger.info(f"Finished inference for session {session_id} with status {session['status']}")

        # ── End of async with self._gpu_lock ──
        # Offload archiving to background I/O executor OUTSIDE of _gpu_lock,
        # and ONLY when inference genuinely finished/completed (never stall a restart on stop/pause).
        if session.get("status") == "completed" and session_dir:
            self._archive_completed_session(
                session_id=session_id,
                session_dir=session_dir,
                output_path=output_path,
                video_filename=video_filename,
                results_json_path=results_json_path,
            )

    def _archive_completed_session(
        self,
        session_id: str,
        session_dir: str,
        output_path: str,
        video_filename: str,
        results_json_path: str,
    ):
        """Asynchronously archives final video and artifacts in the background
        without blocking the GPU lock or delaying new inference runs."""
        def _do_archive():
            try:
                from datetime import datetime
                import shutil
                from app.core.app_paths import get_desktop_dir

                desktop = get_desktop_dir()
                today_str = datetime.now().strftime("%Y-%m-%d")
                archive_dir = os.path.join(str(desktop), "inference_results", today_str, session_id)
                os.makedirs(archive_dir, exist_ok=True)

                if os.path.abspath(session_dir) != os.path.abspath(archive_dir):
                    if output_path and os.path.exists(output_path):
                        dest_video = os.path.join(archive_dir, video_filename)
                        shutil.copy2(output_path, dest_video)
                        session = self.sessions.get(session_id)
                        if session and "stats" in session:
                            session["stats"]["output_file"] = dest_video
                        logger.info(f"[ARCHIVE] Saved annotated video copy to Desktop: {dest_video}")

                    if results_json_path and os.path.exists(results_json_path):
                        shutil.copy2(results_json_path, archive_dir)

                    last_frame_file = os.path.join(session_dir, "last_frame.jpg")
                    if os.path.exists(last_frame_file):
                        shutil.copy2(last_frame_file, archive_dir)
                else:
                    session = self.sessions.get(session_id)
                    if session and "stats" in session:
                        session["stats"]["output_file"] = output_path
                    logger.info(f"[ARCHIVE] Video and artifacts saved directly on Desktop: {session_dir}")

                session = self.sessions.get(session_id)
                if session and "stats" in session:
                    session["stats"]["archived_results_dir"] = archive_dir
                logger.info(f"[ARCHIVE] Successfully finalized inference results in: {archive_dir}")
            except Exception as arch_err:
                logger.warning(f"[ARCHIVE] Could not finalize inference results to Desktop: {arch_err}")

        self._io_executor.submit(_do_archive)

    async def _cleanup_session(self, session_id: str):
        session = self.sessions.get(session_id)
        if not session:
            return

        # Send graceful termination to the stream generator.
        # Drain a slot if full to make room, then always put None sentinel in sequence.
        try:
            q = session.get("queue")
            if q is not None:
                if q.full():
                    try:
                        q.get_nowait()
                    except (asyncio.QueueEmpty, ValueError):
                        pass
                q.put_nowait(None)
        except Exception:
            pass


# Primary export for video inference
video_inference_service = VideoInferenceService()

# Backward compatibility alias
ml_inference_service = video_inference_service
