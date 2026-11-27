"""
sequence_registry.py - how an application starts training and inference (registry style, NO outside imports)
===============================================================================================================
The application owns the paths and the job's inputs; this file turns them into the command lines of train.py and
infer.py. Model-internal settings (TCN / LSTR, fps, rules) stay in training/config.json - never passed from here.

    from sequence_registry import SEQUENCE_MODEL, build_train_args, build_infer_args
    cmd = [python, SEQUENCE_MODEL["train_script"]] + build_train_args(job)        # run it, one process per job
    cmd = [python, SEQUENCE_MODEL["infer_script"]] + build_infer_args(job)        # exit 0 NORMAL, 3 ANOMALY
    folder = build_job_folders(job["results_path"])

job (a dict the application fills):
    video_path        the GOOD video (training) or the video to check (inference)
    results_path      training: where the model package is written / inference: the package to use
    data_path         the job's working folder (optional; sop_prompt.txt may be there)
    detector_weights  the trained detector .pt (training; e.g. the object-tracking job's best.pt)
    sop_prompt        the SOP text, numbered steps (training; or sop_prompt.txt in data_path)
    infer_output      inference: where result.json goes (default <results_path>/inference_results)
"""
import os
from typing import Any, Dict, List

_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# CLI argument builders
# ---------------------------------------------------------------------------

def build_train_args(job: Dict[str, Any]) -> List[str]:
    """train.py arguments: only what the application owns (paths, the video, the detector, the SOP)"""
    args = [
        "--input",      job["video_path"],
        "--output_dir", job["results_path"],
        "--data_path",  job.get("data_path", ""),
    ]
    weights = job.get("detector_weights")
    for w in ([weights] if isinstance(weights, str) else (weights or [])):
        args += ["--weights", w]
    if job.get("sop_prompt"):
        args += ["--prompt", job["sop_prompt"]]
    return args


def build_infer_args(job: Dict[str, Any]) -> List[str]:
    """infer.py arguments: the trained package + the video(s) to check"""
    videos = job["video_path"]
    args = ["--model_dir", job["results_path"], "--input"] + ([videos] if isinstance(videos, str) else list(videos))
    if job.get("infer_output"):
        args += ["--output_dir", job["infer_output"]]
    return args + ["--no_preview"]


# ---------------------------------------------------------------------------
# The model entry
# ---------------------------------------------------------------------------
# Keys:
#   display_name       for UI / logs
#   train_script       the training entry (one process per job, one GPU, no torchrun)
#   infer_script       the inference entry
#   build_train_args   Callable(job) -> list[str]
#   build_infer_args   Callable(job) -> list[str]
#   output             what results_path holds after training (the model package)
#   result_file        the JSON the scripts write (status / verdicts)
#   exit_codes         what each script's exit code means

SEQUENCE_MODEL: Dict[str, Any] = {
    "display_name":     "SOP Sequence Model",
    "train_script":     os.path.join(_DIR, "train.py"),
    "infer_script":     os.path.join(_DIR, "infer.py"),
    "build_train_args": build_train_args,
    "build_infer_args": build_infer_args,
    "uses_torchrun":    False,
    "output":           ["models.json", "weights/", "trained/<name>/", "trained/active_model.txt", "sop_prompt.txt",
                         "result.json"],
    "result_file":      "result.json",
    "exit_codes": {
        "train": {0: "trained", 2: "training failed (result.json: failed_step, error)", 1: "unexpected error"},
        "infer": {0: "every video NORMAL", 3: "an ANOMALY / UNCERTAIN", 2: "bad package or input",
                  1: "unexpected error"},
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_job_folders(results_path: str) -> str:
    """the output folder made before training starts (train.py fills it)"""
    os.makedirs(results_path, exist_ok=True)
    return results_path


def train_command(job: Dict[str, Any], python: str = "python") -> List[str]:
    """the whole training command line"""
    return [python, SEQUENCE_MODEL["train_script"]] + build_train_args(job)


def infer_command(job: Dict[str, Any], python: str = "python") -> List[str]:
    """the whole inference command line"""
    return [python, SEQUENCE_MODEL["infer_script"]] + build_infer_args(job)
