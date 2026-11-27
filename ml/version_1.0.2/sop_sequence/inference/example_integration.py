"""
example_integration.py - how an application uses the SOPMonitor of a model PACKAGE (made by train.py).

    python example_integration.py --model_dir <package> --video test.mp4     frames from a file, YOUR loop pushes them
    python example_integration.py --model_dir <package> --camera 0           frames from camera 0

The pattern for a GUI application:
  - ONE SOPMonitor for the life of the application (loading takes a few seconds; it is done once)
  - a WORKER thread grabs frames and calls process_frame(); the UI thread only draws what the worker queues
  - start() / stop() around each inspection (each product / cycle); stop() gives the verdict
"""
import argparse
import importlib.util
import queue
import sys
import threading
from pathlib import Path

import cv2


def load_monitor(model_dir: str):
    """sop_sequence.load_monitor when the wheel is installed, else the infer.py one folder up (this source folder)"""
    if importlib.util.find_spec("sop_sequence") is not None:
        import sop_sequence
        return sop_sequence.load_monitor(model_dir)
    top = Path(__file__).resolve().parents[1] / "infer.py"
    spec = importlib.util.spec_from_file_location("sop_sequence_infer", top)
    m = importlib.util.module_from_spec(spec)
    sys.modules["sop_sequence_infer"] = m
    spec.loader.exec_module(m)
    return m.load_monitor(model_dir)


def inspection_worker(mon, source, is_file: bool, ui: "queue.Queue", stop: threading.Event):
    """the worker thread: frames -> monitor -> UI queue; at the end the verdict"""
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    mon.start(name="example")
    i = 0
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            break
        # a FILE: its own time (frame / fps); a CAMERA: None = the wall clock
        st = mon.process_frame(frame, t=i / fps if is_file else None)
        i += 1
        for ev in st.events:                   # what changed on this frame
            ui.put(("event", ev))
        try:
            ui.put_nowait(("frame", mon.draw(frame, st), st))
        except queue.Full:
            pass                               # the UI is behind: skip this picture, never block the inspection
    cap.release()
    ui.put(("verdict", mon.stop()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="a model package made by train.py")
    ap.add_argument("--video", help="a video file")
    ap.add_argument("--camera", type=int, help="a camera index")
    a = ap.parse_args()
    if a.video is None and a.camera is None:
        ap.error("give --video or --camera")

    # 1. load once (at application start)
    mon = load_monitor(a.model_dir)
    print(f"model {mon.model_name}  detectors {mon.detectors}")
    print("loaded:", ", ".join(f"{k} {v:.2f}s" for k, v in mon.load_seconds.items()))
    print("steps:", "  ->  ".join(f"{s['step']}. {s['label']}" for s in mon.steps))

    # 2. an inspection on a worker thread
    ui: "queue.Queue" = queue.Queue(maxsize=4)
    stop = threading.Event()
    src = a.video if a.video is not None else a.camera
    threading.Thread(target=inspection_worker, args=(mon, src, a.video is not None, ui, stop), daemon=True).start()

    # 3. the UI thread
    while True:
        kind, *data = ui.get()
        if kind == "frame":
            img, st = data
            cv2.imshow("SOP monitor (Q = stop)", img)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                stop.set()
        elif kind == "event":
            ev = data[0]
            if ev.kind == "done":
                print(f"[{ev.t:6.1f}s] step {ev.step} {ev.label}: {'done' if ev.in_order else 'OUT OF ORDER'} "
                      f"(detected {ev.detected}%)")
            elif ev.kind == "taken_back":
                print(f"[{ev.t:6.1f}s] step {ev.step} undone: {ev.note}")
        elif kind == "verdict":
            res = data[0]
            print(f"\nVERDICT {res.verdict}   sequence {res.sequence}   {res.fps} fps   files: {res.out_dir}")
            for f in res.anomalies:
                print("  !", f)
            break
    cv2.destroyAllWindows()

    # 4. at application exit
    mon.close()


if __name__ == "__main__":
    main()
