"""
sop_inference - SOP sequence INFERENCE with trained models (installed from the wheel).

    import sop_inference as si
    mon = si.SOPMonitor()                    # the active model; every model loaded ONCE
    mon.start()
    st = mon.process_frame(frame_bgr, t)     # per frame of YOUR camera loop -> FrameStatus
    res = mon.stop()                         # the verdict
    res = mon.run_video("test.mp4")          # or a whole file / camera / stream
    mon.close()

Data folder: ~/SOP_Inference (or SOP_MONITOR_HOME) - models.json, weights/, trained/, outputs/.
Command line: sop-infer --list | sop-infer --video test.mp4
"""
from .infer import *  # noqa: F401,F403
from .infer import (APP_ROOT, FrameInfo, FrameStatus, InferenceEngine, InferenceResult, LoadedModels,  # noqa: F401
                    SOPMonitor, StepEvent, StepResult, get_active_model, list_models, load_action_model,
                    load_detectors, load_hand_model, load_models, load_step_tcn, load_trained_model, main, warm_up)

__version__ = "1.0.2"
data_folder = APP_ROOT
