"""
infer.py - SEQUENCE INFERENCE with a model package made by train.py (platform entry point)
==========================================================================================
Whatever train.py trained is what this runs: the package holds the sequence model AND the detector(s) it was
trained with, so training and inference can never use different files.

    python infer.py --model_dir <train.py's --output_dir> --input test.mp4 [more.mp4 ...] [--output_dir results/]
    python infer.py --model_dir <package> --input 0                      (camera 0, live window)
    python infer.py --model_dir <package> --list                         (the models in the package)

  --model_dir    the package (models.json, weights/, trained/)
  --input        video file(s), a camera index or a stream URL
  --output_dir   where results go (default <model_dir>/inference_results); result.json = verdict per input
  --model        a model name inside the package (default: its trained/active_model.txt)
  --no_preview / --save_video / --real_time / --no_excel / --fps / --require_gpu   as inference/infer.py
EXIT CODE  0 every input NORMAL · 3 at least one ANOMALY / UNCERTAIN · 2 bad package or input · 1 unexpected error

From Python (a host application with its own frames):
    import infer
    mon = infer.load_monitor("<model_dir>")         # loads once
    mon.start(); st = mon.process_frame(frame_bgr, t); ...; res = mon.stop(); mon.close()
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

PKG = Path(__file__).resolve().parent
_ENGINE = None


def engine(model_dir):
    """inference/infer.py loaded with the package as its data folder (once per process)"""
    global _ENGINE
    md = Path(model_dir).resolve()
    if not (md / "models.json").is_file() or not (md / "trained").is_dir():
        raise FileNotFoundError(f"{md} is not a model package from train.py (models.json / trained/ missing)")
    if _ENGINE is not None:
        if Path(os.environ["SOP_MONITOR_HOME"]).resolve() != md:
            raise RuntimeError("one model package per process (start another process for another package)")
        return _ENGINE
    os.environ["SOP_MONITOR_HOME"] = str(md)
    spec = importlib.util.spec_from_file_location("sop_infer", PKG / "inference" / "infer.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["sop_infer"] = m
    spec.loader.exec_module(m)
    _ENGINE = m
    return m


def load_monitor(model_dir, model=None):
    """the integration API: a loaded SOPMonitor for the package's model"""
    M = engine(model_dir)
    return M.SOPMonitor(model) if model else M.SOPMonitor()


def run(model_dir, inputs, output_dir=None, model=None, show=False) -> dict:
    """Python API: inference on video file(s) with a package -> result.json as a dict ({"results": [{"verdict",
    "sequence", "steps", "anomalies", ...}]}). In this process (load_monitor() for your own frame loop)."""
    argv = ["--model_dir", str(model_dir), "--input", *[str(v) for v in ([inputs] if isinstance(inputs, (str, Path))
                                                                         else inputs)]]
    argv += ["--output_dir", str(output_dir)] if output_dir else []
    argv += ["--model", model] if model else []
    argv += [] if show else ["--no_preview"]
    main(argv)
    p = Path(output_dir or Path(model_dir) / "inference_results") / "result.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {"results": [], "error": "no result.json"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--input", nargs="+")
    ap.add_argument("--output_dir")
    ap.add_argument("--model")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no_preview", action="store_true")
    ap.add_argument("--save_video", action="store_true")
    ap.add_argument("--real_time", action="store_true")
    ap.add_argument("--no_excel", action="store_true")
    ap.add_argument("--fps", type=float)
    ap.add_argument("--require_gpu", action="store_true")
    ap.add_argument("--data_path", default="", help=argparse.SUPPRESS)          # platform common flags
    ap.add_argument("--video_to_frames", default="True", help=argparse.SUPPRESS)
    ap.add_argument("--stream_mode", default="False", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        M = engine(a.model_dir)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[infer] {e}")
        return 2
    if a.list or not a.input:
        for r in M.list_models():
            print(f"  {'*' if r['active'] else ' '} {r['name']:<55} {r['trained_at'] or '':<17} "
                  f"self-check {r['self_check']}")
        return 0
    out = Path(a.output_dir or Path(a.model_dir) / "inference_results").resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = M.PipelineConfig()
    if a.model:
        cfg.paths.model = a.model
    cfg.paths.videos = list(a.input)
    cfg.paths.out_base = str(out)
    cfg.inspection.real_time |= a.real_time
    if a.fps is not None:
        cfg.inspection.fps = a.fps
    cfg.ui.show_preview &= not a.no_preview
    cfg.ui.save_video |= a.save_video
    cfg.ui.excel_log &= not a.no_excel
    cfg.device.require_gpu |= a.require_gpu
    try:
        results = M.run_inference(cfg)
    except FileNotFoundError as e:
        print(f"[infer] {e}")
        return 2
    rows = []
    for r in results:
        d = asdict(r)
        d.pop("report", None)
        rows.append(d)
    (out / "result.json").write_text(json.dumps({"model_dir": str(Path(a.model_dir).resolve()), "results": rows},
                                                indent=1, default=str), encoding="utf-8")
    print(f"[infer] result -> {out / 'result.json'}")
    return 0 if results and all(r.verdict == "NORMAL" for r in results) else 3


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.exit(code)
