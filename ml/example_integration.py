"""
example_integration.py - Minimal, ready-to-run SOP sequence inference integration.

Usage:
    python example_integration.py                     # camera 0
    python example_integration.py --video test.mp4    # video file
"""

import os
import sys
import argparse
from pathlib import Path

# 1. Setup the data directory (where models.json, weights/, and trained/ live)
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "sop_inference" / "_data"
if not DATA_DIR.exists():
    DATA_DIR = HERE / "_data"
os.environ["SOP_MONITOR_HOME"] = str(DATA_DIR)

# Add sop_inference to sys.path if not installed as a wheel
sys.path.insert(0, str(HERE))

import cv2
from sop_inference import SOPMonitor, get_active_model


def main():
    parser = argparse.ArgumentParser(description="SOP Sequence Inference Integration")
    parser.add_argument("--model", help="Model name (default: active model)")
    parser.add_argument("--video", default="0", help="Camera index (0) or path to video file")
    args = parser.parse_args()

    # 2. Load model ONCE at startup
    model = args.model or get_active_model()
    print(f"[INIT] Loading SOP model '{model}'...")
    mon = SOPMonitor(model=model)
    print(f"[INIT] Loaded detectors: {mon.detectors}")

    print("\nExpected SOP Sequence Steps:")
    for s in mon.steps:
        print(f"  Step {s['step']}: {s['label']}")

    # 3. Setup video source
    src = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {args.video}")
        mon.close()
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    is_camera = str(args.video).isdigit()

    # 4. Start inspection cycle
    mon.start(name="inspection_01")
    print("\n[START] Inspection cycle started. Press 'q' or 'ESC' in window to finish.\n")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate video time (None uses wall clock for live camera)
            t = None if is_camera else (frame_idx / fps)
            frame_idx += 1

            # 5. Process frame -> returns FrameStatus
            st = mon.process_frame(frame, t=t)

            # Check new events that happened on THIS frame
            for ev in st.events:
                if ev.kind == "done":
                    order = "OK (in order)" if ev.in_order else f"OUT OF ORDER ({ev.note})"
                    print(f"  --> Step {ev.step} '{ev.label}' DONE at {ev.t:.1f}s | {order} (conf: {ev.detected}%)")
                elif ev.kind == "doing":
                    print(f"  ... Step {ev.step} '{ev.label}' started doing at {ev.t:.1f}s")
                elif ev.kind == "taken_back":
                    print(f"  <-- Step {ev.step} '{ev.label}' UNDONE / RETRACTED: {ev.note}")

            # 6. Draw HUD overlay and display
            hud_frame = mon.draw(frame, st)
            cv2.imshow("SOP Sequence Detection", hud_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

    finally:
        # 7. Stop inspection and inspect final sequence results
        res = mon.stop()
        cap.release()
        cv2.destroyAllWindows()

        print("\n" + "=" * 60)
        print("  FINAL INSPECTION REPORT")
        print("=" * 60)
        print(f"Verdict         : {res.verdict}")
        print(f"Actual Sequence : {' -> '.join(map(str, res.sequence)) if res.sequence else 'No steps completed'}")
        print(f"FPS Processed   : {res.fps:.1f} fps")

        print("\nStep Breakdown:")
        for s in res.steps:
            duration = f"{s.duration_s:.1f}s" if s.duration_s is not None else "-"
            print(f"  Step {s.step} [{s.status:<12}] {s.label:<32} (took {duration})")

        if res.anomalies:
            print("\nAnomalies / Failures:")
            for a in res.anomalies:
                print(f"  * {a}")
        print("=" * 60)

        # 8. Clean up engine / GPU memory
        mon.close()


if __name__ == "__main__":
    main()
