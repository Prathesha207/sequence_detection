"""
train.py - SEQUENCE MODEL TRAINING, the application's entry point
========================================================================================
The platform passes only what IT owns (paths, the input video, the detector from the object-tracking job);
model-internal settings (TCN / LSTR, fps, rules) stay in training/config.json.

    python train.py --input good.mp4 --output_dir <trained_models/.../sequential_models>
                    --data_path <training/.../sequential_models>
                    [--weights <object_tracking .../best.pt>] [--prompt "1. ... 2. ..." | --prompt_file sop.txt]
                    [--video_to_frames True] [--stream_mode False]

INPUTS
  --input        the GOOD video: the SOP done once, correctly, every step in order
  --output_dir   where the trained model goes. It becomes a SELF-CONTAINED MODEL PACKAGE that infer.py runs
                 directly (python infer.py --model_dir <output_dir> --input test.mp4)
  --data_path    the session's working folder. The SOP is read from here when no --prompt / --prompt_file is
                 given: sop_prompt.txt | sop.txt | prompt.txt | job.json {"prompt": ...}
  --weights      the trained detector(s) (.pt, YOLO detect / segment / obb), e.g. the objectTracking job's best.pt;
                 repeatable. Without it: the enabled detectors of ./models.json
  --prompt / --prompt_file   the SOP as NUMBERED steps; each step must name one of the detector's classes
  --video_to_frames / --stream_mode   accepted for the platform's common flags; the video is read directly
                 (every frame, 30 fps), no frame folder is needed

STEPS  (one function each, called in order by main(); result.json "failed_step" says where it stopped)
  A  step_a_sop         the SOP text -> sop_prompt.txt
  B  step_b_detectors   the detector(s) copied in -> models.json
  C  step_c_shared      DINOv2 + MediaPipe hand models copied in
  D  step_d_train       the sequence model: training/train_sequence.py stages 0-8
                        (0 inputs, 1 detectors, 2 spec, 3 places, 4 save, 5 step model, 6 references,
                         7 action model + self-check, 8 finish) - an error names its stage
  E  step_e_clean       the work folder (feature caches) removed

OUTPUT  (<output_dir>/)
  models.json                  the detector entries (paths relative to this folder)
  weights/detectors/*.pt       the detector(s) used for training - inference uses the SAME ones
  weights/dinov2_vits14.safetensors, weights/hand_landmarker.task   shared feature / hand models
  trained/<name>/              the sequence model (model.json, places.json, temporal.pt, action_matcher.pt, ...)
  trained/active_model.txt     <name>
  sop_prompt.txt               the SOP it was trained on
  result.json                  {"status": "ok" | "failed", "model_name", "self_check", "error", ...}
EXIT CODE  0 trained · 2 training failed (result.json says why) · 1 unexpected error
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

PKG = Path(__file__).resolve().parent                  # this folder: training/, inference/, weights/, models.json
SHARED = ["dinov2_vits14.safetensors", "hand_landmarker.task"]
SOP_FILES = ["sop_prompt.txt", "sop.txt", "prompt.txt"]


def log(s):
    print(s, flush=True)


def read_prompt(a) -> str:
    if a.prompt:
        return a.prompt
    if a.prompt_file:
        return Path(a.prompt_file).read_text(encoding="utf-8")
    d = Path(a.data_path) if a.data_path else None
    if d and d.is_dir():
        for n in SOP_FILES:
            if (d / n).is_file():
                return (d / n).read_text(encoding="utf-8")
        if (d / "job.json").is_file():
            j = json.loads((d / "job.json").read_text(encoding="utf-8"))
            if j.get("prompt"):
                return str(j["prompt"])
    return ""


def detector_entries(weights: list, out: Path) -> list:
    """the detector(s) -> models.json entries in the package; the .pt files copied into it"""
    dst = out / "weights" / "detectors"
    dst.mkdir(parents=True, exist_ok=True)
    entries = []
    if weights:
        from ultralytics import YOLO
        used = set()
        for w in weights:
            p = Path(w)
            if not p.is_file():
                raise FileNotFoundError(f"detector weights not found: {p}")
            stem = p.stem if p.stem not in ("best", "last") else f"{p.parent.parent.name or 'detector'}_{p.stem}"
            while stem in used:
                stem += "_2"
            used.add(stem)
            m = YOLO(str(p))
            task = getattr(m, "task", "detect") or "detect"
            names = m.names if isinstance(m.names, dict) else dict(enumerate(m.names))
            shutil.copy2(p, dst / f"{stem}.pt")
            entries.append({"type": "yolo", "name": f"{stem} ({task})", "path": f"weights/detectors/{stem}.pt",
                            "conf": 0.25, "imgsz": 0, "enabled": True, "task": task,
                            "classes": [str(names[k]) for k in sorted(names)], "use_for": "trained with this model"})
            del m
    else:
        for e in json.loads((PKG / "models.json").read_text(encoding="utf-8")):
            if not e.get("enabled", True):
                continue
            src = Path(e["path"]) if Path(e["path"]).is_absolute() else PKG / e["path"]
            if not src.is_file():
                raise FileNotFoundError(f"detector weights not found: {src} ({e['name']})")
            shutil.copy2(src, dst / src.name)
            entries.append(dict(e, path=f"weights/detectors/{src.name}"))
    if not entries:
        raise PlatformError("no detector: give --weights or enable one in models.json")
    return entries


# ══════════════════════════════════════════════════════════════════════════════
#  THE PLATFORM STEPS  (main() calls them in order; each fails with a clear message in result.json)
# ══════════════════════════════════════════════════════════════════════════════

def step_a_sop(a, out: Path) -> str:
    # Step A: the SOP text (--prompt / --prompt_file / a file in --data_path), saved into the package
    prompt = read_prompt(a)
    if not prompt.strip():
        raise PlatformError("no SOP: give --prompt / --prompt_file or put sop_prompt.txt in --data_path")
    (out / "sop_prompt.txt").write_text(prompt.strip(), encoding="utf-8")
    log(f"[A sop] {' '.join(prompt.split())[:160]}")
    return prompt


def step_b_detectors(a, out: Path) -> list:
    # Step B: the detector(s) copied into the package + models.json written (inference uses the SAME files)
    entries = detector_entries(a.weights or [], out)
    (out / "models.json").write_text(json.dumps(entries, indent=1), encoding="utf-8")
    for e in entries:
        log(f"[B detector] {e['name']}: {', '.join(e['classes'])}")
    return entries


def step_c_shared(out: Path) -> None:
    # Step C: the shared models (DINOv2 features, MediaPipe hands) copied into the package
    (out / "weights").mkdir(exist_ok=True)
    (out / "trained").mkdir(exist_ok=True)
    for n in SHARED:
        src = PKG / "weights" / n
        if not src.is_file():
            raise PlatformError(f"shared model missing in this folder: {src}")
        if not (out / "weights" / n).is_file():
            shutil.copy2(src, out / "weights" / n)
    log(f"[C shared] {', '.join(SHARED)}")


def step_d_train(a, out: Path, prompt: str, entries: list) -> dict:
    # Step D: the sequence model trained INTO the package (training/train_sequence.py, stages 0-8)
    os.environ["SOP_MONITOR_HOME"] = str(out)           # the engine's data folder = the package (read on import)
    TS = _load_training()                                # training/train_sequence.py (+ its sop_engine.py)
    TS.DATA = out
    try:
        return TS.train(a.input, prompt, [e["name"] for e in entries], a.kind, a.name, a.fps)
    except TS.TrainingError as e:
        raise PlatformError(str(e)) from e


def _load_training():
    """training/train_sequence.py loaded from its FILE (it sits in training/, not on the import path; loading by
    path also keeps editors from flagging an unresolved import). Its own 'import sop_engine' finds the file next
    to it because training/ is put on sys.path first."""
    import importlib.util
    d = PKG / "training"
    sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location("train_sequence", d / "train_sequence.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["train_sequence"] = m
    spec.loader.exec_module(m)
    return m


def step_e_clean(a, out: Path) -> None:
    # Step E: the training caches (outputs/: features, detections, boundaries) removed - only what inference needs
    if not a.keep_work:
        shutil.rmtree(out / "outputs", ignore_errors=True)


class PlatformError(RuntimeError):
    """an expected failure (bad input, training could not learn): exit code 2, the message in result.json"""


def train(input, output_dir, weights=None, prompt=None, prompt_file=None, data_path="", name=None, kind=None,
          fps=None, python=None) -> dict:
    """Python API: one training in its OWN process (the engine's data folder is fixed per process) -> result.json
    as a dict ({"status": "ok" | "failed", "model_name", "self_check", "error", ...}). The log goes to stdout."""
    import subprocess
    cmd = [python or sys.executable, str(Path(__file__).resolve()), "--input", str(input), "--output_dir",
           str(output_dir), "--data_path", str(data_path or "")]
    for w in ([weights] if isinstance(weights, (str, Path)) else (weights or [])):
        cmd += ["--weights", str(w)]
    for k, v in (("--prompt", prompt), ("--prompt_file", prompt_file), ("--name", name), ("--kind", kind),
                 ("--fps", fps)):
        if v:
            cmd += [k, str(v)]
    subprocess.run(cmd, check=False)
    p = Path(output_dir) / "result.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {"status": "failed",
                                                                          "error": "train.py wrote no result.json"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="the GOOD video")
    ap.add_argument("--output_dir", required=True, help="the model package to write")
    ap.add_argument("--data_path", default="", help="session working folder (the SOP file may be here)")
    ap.add_argument("--weights", action="append", help="trained detector .pt (repeatable)")
    ap.add_argument("--prompt", help="the SOP: numbered steps")
    ap.add_argument("--prompt_file", "--prompt-file", dest="prompt_file", help="a text file with the SOP")
    ap.add_argument("--name", help="model name (default: <video>_<date>_<time>)")
    ap.add_argument("--kind", choices=["TCN", "LSTR", "tcn", "lstr"], help="default: training/config.json")
    ap.add_argument("--fps", type=float, help="default: training/config.json (30 = every frame)")
    ap.add_argument("--keep_work", action="store_true", help="keep outputs/_training (feature caches)")
    ap.add_argument("--video_to_frames", default="True", help=argparse.SUPPRESS)
    ap.add_argument("--stream_mode", default="False", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    res = {"status": "failed", "input": str(Path(a.input).resolve()), "output_dir": str(out), "started": time.strftime(
        "%Y-%m-%d %H:%M:%S")}

    t_start = time.time()

    def finish(code):
        res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        res["elapsed_seconds"] = round(time.time() - t_start, 1)              # the whole run (steps A-E)
        (out / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        log(f"[train] result -> {out / 'result.json'}  ({res['status']})")
        sys.stdout.flush()
        return code

    log(f"[train] package: {out}")
    step = "start"
    try:
        step = "A sop"
        prompt = step_a_sop(a, out)                                          # A  the SOP
        res["prompt"] = " ".join(prompt.split())
        step = "B detectors"
        entries = step_b_detectors(a, out)                                   # B  detector(s) + models.json
        res["detectors"] = [{"name": e["name"], "classes": e["classes"]} for e in entries]
        step = "C shared"
        step_c_shared(out)                                                   # C  DINOv2 + hands
        step = "D train"
        r = step_d_train(a, out, prompt, entries)                            # D  the sequence model (stages 0-8)
        step = "E clean"
        step_e_clean(a, out)                                                 # E  work folder removed
        res.update(status="ok", model_name=r["name"], model_folder=r["folder"], self_check=r["self_check"],
                   action_self_check_steps=r.get("action_self_check_steps", {}), steps=r.get("steps", []),
                   train_seconds=r.get("elapsed_seconds"), stage_seconds=r.get("stage_seconds", {}),
                   inference=f'python infer.py --model_dir "{out}" --input <video | camera index>')
        return finish(0)
    except (PlatformError, FileNotFoundError) as e:                          # expected: bad input / not learnable
        res.update(failed_step=step, error=str(e))
        log(f"[train] FAILED in step {step}: {e}")
        return finish(2)
    except Exception as e:                                                   # unexpected: a bug - full traceback
        res.update(failed_step=step, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        log(res["traceback"])
        return finish(1)


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.exit(code)
