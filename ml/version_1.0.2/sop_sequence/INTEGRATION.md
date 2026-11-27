# Integrating SOP sequence training and inference

This package does two things:

1. **Train** a sequence model from one good video, an SOP and a detector.
2. **Check** videos or a live camera against that model: were the steps done, and in the right order?

You only need three functions or two commands. Everything else in the package is internal.

## 1. Install

```bat
install_sop_sequence.bat C:\path\to\python.exe
```

The installer runs these steps in order:

1. Installs CUDA torch.
2. Installs `requirements.txt`.
3. Installs the wheel, `sop_sequence-1.0.0-py3-none-any.whl[all]`.
4. Installs MediaPipe hands with `--no-deps`.

Do the MediaPipe step this way when you install by hand too. A plain `pip install mediapipe` replaces `opencv-python`.

It was tested with Python 3.13, Windows 11, and CUDA 13 on an RTX GPU with 8 GB.

## 2. The idea: train → model package → infer

```text
good video + SOP + detector (.pt)  ──train──►  model package folder  ──infer──►  verdict per video / camera
```

Training writes one folder, the **model package**. Inference takes only that folder, so it always uses the
model and detector that were trained together. Store the folder per training job, for example in
`results_path`, and pass it to inference.

```text
<package>/
  models.json              the detector entries
  weights/detectors/*.pt   the detector used for training (a copy)
  weights/dinov2_vits14.safetensors, weights/hand_landmarker.task   shared models
  trained/<name>/          the sequence model (model.json, places.json, temporal.pt, action_matcher.pt, ...)
  trained/active_model.txt <name>
  sop_prompt.txt           the SOP it was trained on
  result.json              the training result
```

A package is about 170 MB, mostly the copied detector and shared models.

## 3. Training

### Command line (one process per job)

```bat
sop-seq-train --input good.mp4 --output_dir <package> --weights <objectTracking ...\best.pt>
              --prompt "1. Insert the battery into the mouse. 2. Fit the cover onto the mouse. 3. ..."
```

| argument | needed | meaning |
|---|---|---|
| `--input` | yes | The **good** video: the SOP done once, correctly, every step in order. |
| `--output_dir` | yes | The model package to write. |
| `--weights` | recommended | The trained YOLO detector (detect, segment or obb). Repeatable. |
| `--prompt` / `--prompt_file` | yes* | The SOP. *Or put `sop_prompt.txt`, `sop.txt`, `prompt.txt` or `job.json {"prompt": ...}` in `--data_path`. |
| `--data_path` | no | The session folder, where the SOP file can be. |
| `--name` | no | The model name. Default: `<video>_<date>_<time>`. |
| `--kind`, `--fps` | no | Defaults come from `training/config.json` (TCN, 30 fps). |
| `--video_to_frames`, `--stream_mode` | no | Accepted for the platform's common flags. The video is read directly. |

**The SOP rules:**
- The steps must be **numbered**: `1. ... 2. ...`.
- Each step must name one of the detector's classes. If the words differ, write the class in brackets, for example
  `4. close the lid (close_box)`.

**Exit codes:**
- 0: trained.
- 2: training failed, for example no SOP, a missing video, or nothing learned. The message is in `result.json`.
- 1: unexpected error. The full traceback is in `result.json`.

`<package>/result.json` after training:

```json
{"status": "ok", "model_name": "...", "self_check": "NORMAL", "detectors": [{"name": "...", "classes": [...]}],
 "elapsed_seconds": 289.9, "train_seconds": 283.6,
 "stage_seconds": {"3 places": 55.7, "5 step model": 102.8, "6 references": 58.3, "7 action model": 66.7, ...},
 "action_self_check_steps": {"1": {"status": "COMPLETED", "start": 0.7, "end": 3.77}, ...},
 "steps": [{"id": 1, "label": "insert the battery into the mouse", "object": "battery"}, ...],
 "prompt": "...", "started": "...", "finished": "...", "inference": "python infer.py --model_dir ..."}
```

**Live progress for a UI.** Read the training's stdout line by line:

| line | when |
|---|---|
| `[n name] ...` / `[n name] done in x s` | A stage starts / ends (stages 0–8). |
| `[progress] detecting 120/570 frames (~18s left) (21%)` | During frame scans. Printed when the task changes, every 5 %, or at least every 2 s. |
| `    TCN iter  200  loss 0.412` | Stage 5, every 200 iterations. |
| `    Action iter  200  loss 0.386` | Stage 7, every 200 of its 900 iterations. |
| `[train] result -> .../result.json (ok)` | The end. |

On failure the file has `"status": "failed"`, `"failed_step"` (A sop, B detectors, C shared, D train or E clean)
and `"error"`. Inside D, the error names its stage, for example `stage 3 (places): ...`.

`self_check` is the trained model run on its own good video. It should be `NORMAL`; anything else means the
training video or the SOP needs checking.

### Python

```python
import sop_sequence as ss
r = ss.train("good.mp4", "jobs/123/pkg", weights="best.pt", prompt="1. ... 2. ...")   # blocks; ~2-4 min
if r["status"] != "ok":
    print(r["error"])
```

`ss.train` runs training in its own process. The engine fixes its data folder once per process, so this keeps
your application process clean.

### The 9 training stages (for reading logs)

Each stage logs `[n name] ...` when it starts and `done in x s` when it ends.

| # | stage | function | does |
|---|---|---|---|
| 0 | inputs | `check_inputs` | The video exists, the SOP is not empty, settings are read. |
| 1 | detectors | `load_detectors` | Loads the detector(s) and their classes. |
| 2 | spec | `make_spec` | Turns the SOP into steps, each with the detector class it acts on. |
| 3 | places | `learn_places` | Finds where each step's object lies at the start of the video. |
| 4 | save | `save_model` | Writes `model.json`. |
| 5 | step model | `train_step_model` | Trains the TCN (`temporal.pt`). |
| 6 | references | `build_references` | Builds reference actions. Optional: a failure here does not stop training. |
| 7 | action model | `train_action_model` | Finds step start and end times, trains the action matcher, runs the self-check. |
| 8 | finish | `finish` | Makes this the active model in the package. |

They are in `training/train_sequence.py`. The platform steps A to E are in `train.py`.

## 4. Inference

### A) Whole video file(s)

Command line:

```bat
sop-seq-infer --model_dir <package> --input test1.mp4 test2.mp4 --output_dir <results> --no_preview
```

Exit codes:
- 0: every video NORMAL.
- 3: at least one ANOMALY or UNCERTAIN.
- 2: a bad package or input.
- 1: unexpected error.

Python:

```python
res = ss.run("jobs/123/pkg", ["test1.mp4", "test2.mp4"], output_dir="jobs/123/infer")
for v in res["results"]:
    v["verdict"], v["sequence"], v["anomalies"], v["steps"]
```

`<results>/result.json` holds one entry per input:

| field | meaning |
|---|---|
| `verdict` | `NORMAL` / `ANOMALY` / `UNCERTAIN` |
| `sequence` | The step ids in the order they were done, for example `[1, 2, 3, 4]`. |
| `anomalies` | Readable reasons, for example "step 1 missing". |
| `steps[]` | One entry per step: `step`, `label`, `status` (OK / MISSING / OUT OF ORDER / REMOVED / UNCERTAIN), `start_s`, `done_s`, `duration_s`, `shown_at_s` (when it was confirmed live), `detected_pct`. |
| `frames_read`, `fps`, `seconds`, `frame_ms` | Speed. |
| `out_dir` | Where the CSVs and the annotated video were saved. |

### B) Your own camera or frame loop (live)

```python
mon = ss.load_monitor("jobs/123/pkg")          # loads every model ONCE (a few seconds); keep it for the app's life
mon.start("product 0001")                      # one inspection = one product / cycle
while running:
    frame = grab_frame()                       # BGR uint8 numpy array, any size
    st = mon.process_frame(frame, t)           # t = seconds since start(); None = wall clock
    for ev in st.events: ...                   # what changed on THIS frame
    show(mon.draw(frame, st))                  # optional: the HUD drawn on a copy of the frame
res = mon.stop()                               # the verdict: res.verdict, res.sequence, res.steps, res.anomalies
mon.start("product 0002")                      # the next product, same loaded models
...
mon.close()                                    # frees the GPU
```

For a video file you read yourself, pass `t = frame_number / video_fps`.
`mon.run_video("test.mp4" | "0" | "rtsp://...", on_step=callback)` reads a source itself and blocks until it ends.

`FrameStatus`, returned for every frame:

| field | meaning |
|---|---|
| `state` | `INSPECTING` / `STEP IN PROGRESS` / `OUT OF ORDER` |
| `steps_done` | `[{"step", "label", "t", "in_order", "detected"}]` in the order the steps were done. |
| `doing`, `taken`, `pending` | Steps happening now, steps whose part is picked up, and steps just finished that are being confirmed. |
| `warnings` | Early order warnings, which are provisional. |
| `taken_back` | Steps shown as done earlier and then undone. |
| `events` | The `StepEvent`s on this frame. |
| `detections` | `[{"name", "conf", "box", "model"}]`. The box is in the analysed frame, `frame_size`, at most 1280 px. |
| `fps`, `frame_ms`, `stage_ms` | Speed. |

`StepEvent.kind` is one of:
- `done`: a step is confirmed. It also carries `in_order` and `detected`.
- `pending`, `doing`, `taken`, `taken_back`.
- `warning` / `warning_withdrawn`: the early out-of-order warning, and its withdrawal.

Every event has `step`, `label`, `t` and `shown_at`.

## 5. Starting jobs from your application: `sequence_registry.py`

`sequence_registry.py` turns a job dict into the command lines. It has no outside imports.

```python
from sop_sequence.sequence_registry import SEQUENCE_MODEL, train_command, infer_command
job = {"video_path": "good.mp4", "results_path": "jobs/1/pkg", "data_path": "jobs/1",
       "detector_weights": "best.pt", "sop_prompt": "1. ... 2. ..."}
subprocess.run(train_command(job, python=sys.executable))            # exit 0 = trained
subprocess.run(infer_command({"video_path": ["test.mp4"], "results_path": "jobs/1/pkg"}, python=sys.executable))
```

- `build_train_args(job)` / `build_infer_args(job)`: only the arguments.
- `SEQUENCE_MODEL`: the script paths, the output files, and what each exit code means.

You can also call the console scripts `sop-seq-train` / `sop-seq-infer`, or the Python functions in sections 3
and 4.

## 6. Rules that save debugging time

- **One package per process for inference.** `load_monitor()` for a second, different package in the same
  process raises an error. Run one worker process per model, or reuse one monitor.
- **One thread per monitor.** Call `process_frame` from one thread. Use one `SOPMonitor` per camera.
- **GPU:** a monitor holds about 1.5 GB. Training uses about 2 GB for 2–4 minutes on a short video. Do not run
  training next to another large GPU job on an 8 GB card.
- **The good video decides everything.** A step missing or out of order in it is learned as normal. Check that
  the training `self_check` is `NORMAL`.
- **A detector that cannot see a part cannot check that step.** Improve the detector, using the object-tracking
  job, and retrain the sequence model with the new `best.pt`.
- **The shipped models:** `ss.data_folder` (the installed `sop_sequence` folder) is itself a package. Run
  `sop-seq-infer --model_dir <it> --list` to see them.
