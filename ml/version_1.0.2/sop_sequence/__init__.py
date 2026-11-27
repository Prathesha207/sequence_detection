"""
sop_sequence - SOP sequence TRAINING + INFERENCE (installed from the wheel).

    import sop_sequence as ss
    r = ss.train("good.mp4", "out/model_pkg", weights="best.pt", prompt="1. ... 2. ...")   # -> result.json dict
    r["status"], r["model_name"], r["self_check"]
    res = ss.run("out/model_pkg", ["test.mp4"])               # -> {"results": [{"verdict", "sequence", ...}]}
    mon = ss.load_monitor("out/model_pkg")                     # your own frame loop:
    mon.start(); st = mon.process_frame(frame_bgr, t); res = mon.stop(); mon.close()

Command line:  sop-seq-train --input good.mp4 --output_dir pkg --weights best.pt --prompt "1. ... 2. ..."
               sop-seq-infer --model_dir pkg --input test.mp4
The shipped models: sop_sequence.data_folder is itself a model package (sop-seq-infer --model_dir <it> --list).
"""
from pathlib import Path


def train(*args, **kw):
    """sop_sequence.train.train(input, output_dir, weights=None, prompt=None, ...) -> result.json dict"""
    from .train import train as _t
    return _t(*args, **kw)


def run(*args, **kw):
    """sop_sequence.infer.run(model_dir, inputs, output_dir=None, model=None, show=False) -> result.json dict"""
    from .infer import run as _r
    return _r(*args, **kw)


def load_monitor(*args, **kw):
    """sop_sequence.infer.load_monitor(model_dir, model=None) -> a loaded SOPMonitor"""
    from .infer import load_monitor as _l
    return _l(*args, **kw)


__version__ = "1.0.2"
data_folder = Path(__file__).resolve().parent
