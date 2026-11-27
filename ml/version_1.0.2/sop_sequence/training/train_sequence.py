"""
train_sequence.py - TRAIN A SEQUENCE MODEL  (the desktop app's "Train model", without the window)
====================================================================================================
SOP prompt (numbered steps) + ONE good video (the SOP done correctly, every step in order) + the detector(s)
    ->  trained/<name>/   (model.json, places.json, temporal.pt, action_matcher.pt, ...)
    ->  trained/active_model.txt = <name>, so ../inference/infer.py uses it straight away

    python train_sequence.py --video good.mp4 --prompt "An operator ... 1. ... 2. ... 3. ..."
    python train_sequence.py --video good.mp4 --prompt-file sop.txt --detector "packing_box_auto (detect)"
    python train_sequence.py --video good.mp4 --prompt-file sop.txt --kind LSTR --name my_model

STAGES (the app's order; ONE FUNCTION EACH below, called in order by train(); each logs "[n name] ... done in x s"
and an error names its stage: "stage 5 (step model) failed: ...")
  0. INPUTS       video exists, SOP not empty, settings from config.json                   check_inputs()
  1. DETECTORS    models.json (next to this folder) -> the registry the engine uses        load_detectors()
  2. SPEC         numbered SOP -> steps, each step's object = a detector class (rules)     make_spec()
  3. PLACES       where each step's object lies at the start of the good video             learn_places()  places.json
  4. SAVE         prompt, spec, detectors                                                  save_model()    model.json
  5. STEP MODEL   per-frame features -> pickup / place events (TCN or LSTR)                train_step_model() temporal.pt
  6. REFERENCES   reference actions per step (optional, the model works without)           build_references()
  7. ACTION MODEL each step's start / end on the good video from the detector (completion  train_action_model()
                  landmarks, self-correcting timing), the action matcher trained on them    action_matcher.pt
                  (prefix training + detection dropout), the step lifecycle, + the self-check
  8. FINISH       the self-check read back, this model made the ACTIVE one                  finish()
The whole engine is ONE file next to this one: sop_engine.py (built from the app's sopmon/ - see ../tools).
Settings: config.json (fps 30 = every frame, TCN / LSTR). No VLM: the SOP must be NUMBERED and every step must
name one of the detector's classes (write the class in brackets when the words differ: "4. close the lid (close_box)").
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent                                     # models.json, weights/, trained/, specs/, outputs/
os.environ.setdefault("SOP_MONITOR_HOME", str(DATA))  # the engine's data folder (read when sop_engine is imported)
sys.path.insert(0, str(HERE))
import sop_engine as E                                 # noqa: E402  (the training engine, one file)

CFG = {"fps": 30.0, "kind": "TCN"}
if (HERE / "config.json").is_file():
    CFG.update(json.loads((HERE / "config.json").read_text(encoding="utf-8")))


def log(s: str) -> None:
    print(s, flush=True)


class TrainingError(RuntimeError):
    """the sequence model could not be trained (the message says why)"""


# ══════════════════════════════════════════════════════════════════════════════
#  1. DETECTORS
# ══════════════════════════════════════════════════════════════════════════════

def detectors(names=None) -> list:
    """models.json -> the registry: the named detectors (default: every enabled one), paths made absolute"""
    reg = json.loads((DATA / "models.json").read_text(encoding="utf-8"))
    out = []
    for e in reg:
        if (names and e["name"] not in names) or (not names and not e.get("enabled", True)):
            continue
        e = dict(e, enabled=True)
        p = Path(e["path"])
        e["path"] = str(p if p.is_absolute() else (DATA / p).resolve())
        if not Path(e["path"]).is_file():
            raise TrainingError(f"detector weights not found: {e['path']} ({e['name']})")
        out.append(e)
    if names and len(out) != len(names):
        have = [e["name"] for e in reg]
        raise TrainingError(f"detector(s) not in models.json: {sorted(set(names) - {e['name'] for e in out})}; "
                            f"there: {have}")
    if not out:
        raise TrainingError("no detector enabled in models.json")
    return out


def class_names(registry: list) -> list:
    return sorted({str(x).replace("_", " ").strip() for x in E.models.own_labels(registry) if str(x).strip()})


# ══════════════════════════════════════════════════════════════════════════════
#  2. SPEC
# ══════════════════════════════════════════════════════════════════════════════

def build_spec(prompt: str, registry: list, work: Path) -> dict:
    """numbered SOP -> the spec (rules, no VLM); each step's object = the most specific class its text names"""
    labels = class_names(registry)
    if E.pipeline.rule_spec(prompt, labels) is None:
        raise TrainingError("the SOP must be NUMBERED steps (1. ... 2. ...) and every step must name one of the "
                            f"detector's classes: {', '.join(labels)}  (write it in brackets if the words differ)")
    pl = E.pipeline.Pipeline(None, registry, work / "_spec", spec_dir=DATA / "specs", algorithm_only=True,
                             log=lambda s: None)
    spec = pl.compile_spec(prompt, labels)
    if not spec.get("steps"):
        raise TrainingError("no steps found in the SOP")
    return E.models.specific_parts(spec, sorted({c for e in registry for c in (e.get("classes") or [])}), log=log)


# ══════════════════════════════════════════════════════════════════════════════
#  3 - 8. TRAINING STAGES  (one function each; train() below calls them in order)
# ══════════════════════════════════════════════════════════════════════════════

STAGE_SECONDS = {}                                         # "3 places": seconds - filled by stage(), in result.json


def stage(n: int, title: str):
    """runs one stage: logs its start and time; any error is re-raised as TrainingError naming the stage"""
    def wrap(fn):
        def run(*args, **kw):
            t = time.time()
            log(f"[{n} {title}] ...")
            try:
                out = fn(*args, **kw)
            except E.Cancelled:
                raise
            except TrainingError as e:
                msg = str(e)
                raise (e if msg.startswith("stage ") else TrainingError(f"stage {n} ({title}): {msg}")) from None
            except Exception as e:
                raise TrainingError(f"stage {n} ({title}) failed: {type(e).__name__}: {e}") from e
            STAGE_SECONDS[f"{n} {title}"] = round(time.time() - t, 1)
            log(f"[{n} {title}] done in {STAGE_SECONDS[f'{n} {title}']:.1f} s")
            return out
        run.__name__ = fn.__name__
        return run
    return wrap


@stage(0, "inputs")
def check_inputs(video: str, prompt: str, kind, fps, name) -> dict:
    # Stage 0: the good video exists, the SOP is text, the settings are filled from config.json
    if not Path(video).is_file():
        raise TrainingError(f"good video not found: {video}")
    if not str(prompt).strip():
        raise TrainingError("the SOP prompt is empty")
    name = name or E.trained.default_name(video)
    return {"video": str(video), "prompt": " ".join(str(prompt).split()), "kind": (kind or CFG["kind"]).lower(),
            "fps": float(fps or CFG["fps"]), "name": name, "folder": E.trained.model_dir(name),
            "work": DATA / "outputs" / "_training" / name}


@stage(1, "detectors")
def load_detectors(det_names) -> list:
    # Stage 1: the detector(s) from models.json - their weights must exist; their classes are the step objects
    registry = detectors(det_names)
    for e in registry:
        log(f"    {e['name']}: {', '.join(e.get('classes') or [])}")
    return registry


@stage(2, "spec")
def make_spec(job: dict, registry: list) -> dict:
    # Stage 2: numbered SOP -> steps; each step's object = the most specific detector class its words name
    spec = build_spec(job["prompt"], registry, job["work"])
    for s in spec["steps"]:
        log(f"    {s['id']}. {s.get('action_query')}   [object: {s.get('part')}]")
    return spec


class Progress:
    """the engine's progress callback -> '[progress] detecting 120/720 frames (~18s left) (17%)' lines on stdout
    (a UI can parse them). Throttled: a line when the task changes, every 5 %, or at least every 2 s."""

    def __init__(self, every_frac=0.05, every_s=2.0):
        self.every_frac, self.every_s = every_frac, every_s
        self.task, self.frac, self.t = None, -1.0, 0.0

    def __call__(self, stage, frac=None):
        task = "".join(ch for ch in str(stage) if not ch.isdigit()).split("(~")[0].strip()   # without counts / ETA
        now = time.time()
        new = task != self.task
        if not new and frac is not None and frac < 1.0 and frac - self.frac < self.every_frac \
                and now - self.t < self.every_s:
            return
        if not new and frac is None and now - self.t < self.every_s:
            return
        self.task, self.t = task, now
        self.frac = frac if frac is not None else self.frac
        log(f"[progress] {stage}" + (f" ({frac * 100:.0f}%)" if frac is not None else ""))


def make_pipeline(job: dict, registry: list):
    # (helper) the engine's video pipeline: reads the good video, runs the detector, holds the places file;
    # its progress (detecting k/n frames, ~ETA) is printed as '[progress] ...' lines
    job["folder"].mkdir(parents=True, exist_ok=True)
    return E.pipeline.Pipeline(None, registry, job["work"], fps=job["fps"], spec_dir=DATA / "specs",
                               algorithm_only=True, log=log, progress=Progress(),
                               stop=threading.Event(), places_file=str(job["folder"] / "places.json"))


@stage(3, "places")
def learn_places(pl, job: dict, spec: dict) -> None:
    # Stage 3: where each step's object lies at the start of the good video (its source place)
    if not pl.learn_positions(job["video"], job["prompt"], spec=spec):
        raise TrainingError("nothing learned: no step's object was found in the first seconds of the video")


@stage(4, "save")
def save_model(job: dict, spec: dict, registry: list) -> dict:
    # Stage 4: model.json = prompt + spec + detectors + places (the model folder is usable from here on)
    model = E.trained.save(job["name"], job["prompt"], spec, job["video"], registry, str(job["folder"] / "places.json"))
    log(f"    {job['folder']}")
    return model


@stage(5, "step model")
def train_step_model(pl, job: dict, spec: dict) -> dict:
    # Stage 5: per-frame features -> pickup / place events, the TCN (or LSTR) -> temporal.pt
    info = E.temporal.train_model(pl, [job["video"]], job["prompt"], spec, job["folder"], log=log, kind=job["kind"])
    return E.trained.update(job["name"], temporal=info, train_videos=[Path(job["video"]).name], fps=job["fps"])


@stage(6, "references")
def build_references(pl, job: dict, model: dict, registry: list) -> None:
    # Stage 6: reference actions per step (OPTIONAL - a failure here is logged, the model works without them)
    try:
        for ln in E.completion.train(model, job["video"], registry, log=log, check=pl._check):
            log(ln)
    except E.Cancelled:
        raise
    except Exception as e:
        log(f"    not built ({type(e).__name__}: {e}) - the model works without them")


@stage(7, "action model")
def train_action_model(pl, job: dict, model: dict, registry: list) -> None:
    # Stage 7: each step's start / end on the good video -> the action matcher + step lifecycle + self-check
    for ln in E.label_gui.auto_train(dict(model, name=job["name"]), job["video"], registry, log=log,
                                     check=pl._check, kind=job["kind"]):
        log(ln)


@stage(8, "finish")
def finish(job: dict) -> dict:
    # Stage 8: read back the self-check, make this the ACTIVE model (inference uses it next)
    T = E.trained
    m = dict(T.load(job["name"]), name=job["name"])
    check = m.get("action_self_check") if T.action_model(m) is not None else m.get("self_check")
    (DATA / "trained").mkdir(parents=True, exist_ok=True)
    (DATA / "trained" / "active_model.txt").write_text(job["name"], encoding="utf-8")
    log("\n" + "\n".join(T.sequence_lines(m)))
    return {"name": job["name"], "folder": str(job["folder"]), "self_check": check,
            "action_self_check_steps": m.get("action_self_check_steps") or {},     # per step: status, start, end
            "steps": [{"id": s.get("id"), "label": s.get("action_query"), "object": s.get("part")}
                      for s in (m.get("spec") or {}).get("steps", [])]}


def train(video: str, prompt: str, det_names=None, kind: str = None, name: str = None, fps: float = None) -> dict:
    """the whole training, stage by stage -> {"name", "folder", "self_check"}; the model becomes the ACTIVE one"""
    t0 = time.time()
    job = check_inputs(video, prompt, kind, fps, name)                       # 0 inputs
    log(f"[train] {job['name']}  <-  {Path(video).name}   ({job['kind'].upper()}, {job['fps']:g} fps)")
    registry = load_detectors(det_names)                                     # 1 detectors
    spec = make_spec(job, registry)                                          # 2 SOP -> steps
    pl = make_pipeline(job, registry)
    try:
        learn_places(pl, job, spec)                                          # 3 source places
        model = save_model(job, spec, registry)                              # 4 model.json
        model = train_step_model(pl, job, spec)                              # 5 TCN / LSTR
        build_references(pl, job, model, registry)                           # 6 references (optional)
        train_action_model(pl, job, model, registry)                         # 7 action model + self-check
    except E.Cancelled:
        raise TrainingError("training stopped")
    res = finish(job)                                                        # 8 active model
    res["elapsed_seconds"] = round(time.time() - t0, 1)
    res["stage_seconds"] = dict(STAGE_SECONDS)
    log(f"\n[train] {job['name']} trained in {res['elapsed_seconds']:.0f} s  ·  self-check {res['self_check']}  ·  "
        f"now the ACTIVE model (../inference/infer.py uses it)")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="the GOOD video (every step, in order)")
    ap.add_argument("--prompt", help="the SOP: numbered steps")
    ap.add_argument("--prompt-file", help="a text file with the SOP")
    ap.add_argument("--detector", action="append", help="detector name in models.json (repeatable; default: enabled)")
    ap.add_argument("--kind", choices=["TCN", "LSTR", "tcn", "lstr"], help="step model (default config.json)")
    ap.add_argument("--fps", type=float, help="frames per second for training (default config.json: 30 = every frame)")
    ap.add_argument("--name", help="model name (default: <video>_<date>_<time>)")
    a = ap.parse_args()
    prompt = a.prompt or (Path(a.prompt_file).read_text(encoding="utf-8") if a.prompt_file else "")
    if not prompt.strip():
        sys.exit("give --prompt or --prompt-file")
    try:
        train(a.video, prompt, a.detector, a.kind, a.name, a.fps)
    except TrainingError as e:
        log(f"[train] FAILED: {e}")
        sys.exit(2)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
