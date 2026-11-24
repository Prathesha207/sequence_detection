# SOP Sequence Detection — ML Data Flow

This document describes how Standard Operating Procedure (SOP) sequence detection data flows from the backend machine learning pipeline to the frontend desktop interface.

---

## 1. Pipeline Overview

The sequence detection system monitors assembly, manufacturing, or procedural tasks in real-time. It verifies that every required step is performed in the correct sequence, identifies in-progress actions, detects out-of-order (OOO) or skipped operations, and flags procedural anomalies.

```
┌────────────────────────┐      ┌───────────────────────────┐      ┌───────────────────────────┐
│ Camera / Video Input   │ ---> │ SOP Inference Engine      │ ---> │ FastAPI Dispatch Layer    │
│ (OAK, Webcam, MP4)     │      │ (SOPMonitor / TCN Model)  │      │ (WebSocket / HTTP Poll)   │
└────────────────────────┘      └───────────────────────────┘      └─────────────┬─────────────┘
                                                                                 │
                                                                                 ▼
┌────────────────────────┐      ┌───────────────────────────┐      ┌───────────────────────────┐
│ DetectionCanvas &      │ <--- │ SOP Steps & BBox Mapper   │ <--- │ React Frontend Store      │
│ SequenceTimeline UI    │      │ (mlDataMapper.ts)         │      │ (inferenceStore.ts)       │
└────────────────────────┘      └───────────────────────────┘      └───────────────────────────┘
```

---

## 2. Model & Inference Architecture

The ML subsystem runs on `sop_inference` wheel package (`SOPMonitor`):

- **Model Specification:** Each trained model defines an ordered list of procedural steps (e.g. Step 1: Pick Base Plate, Step 2: Insert Gasket, Step 3: Fasten Screws).
- **Temporal & Action Recognition:** Uses temporal convolutional networks (TCN) and object detection backbones to identify parts, tools, hand interactions, and action completion states.
- **Backend Orchestration:** `sequence_backend/app/ml/sequence_inference.py` coordinates model lifecycle, frame feeding, session state isolation, and verdict computation.
- **Model Endpoints:**
  - `GET /models`: Lists trained sequence models and their active status.
  - `GET /models/active`: Returns the currently loaded model and its preconfigured steps.
  - `GET /models/{name}/steps`: Returns step IDs and action query labels for a model.
  - `GET /models/{name}/preflight`: Verifies presence of `model.json` and weight files (`temporal.pt`, `action_matcher.pt`).

---

## 3. Data Ingestion Channels

### Channel A: Live Stream (OAK Camera or Webcam)

```text
Camera Frames (30 FPS)
  -> oak_camera_service.py / OpenCV capture worker
  -> camera_inference_service.py
  -> sequence_inference.SequenceEngine.process_frame()
  -> Real-time telemetry dictionary
  -> WebSocket /oak/inference/ws/live
  -> App.tsx WebSocket handler
  -> inferenceStore.setStats(data)
  -> mapDetections() -> DetectionCanvas Bounding Boxes
  -> mapBackendResultToSOPSteps() -> SequenceSidebar & Timeline
```

The WebSocket emits a continuous telemetry update for each evaluated frame.

### Channel B: Uploaded Video File

```text
POST /video/upload
  -> video_inference_service.py creates session
  -> Background worker decodes and processes video frames through SequenceEngine
  -> GET /video/status/{session_id} (polled at target video FPS)
  -> inferenceStore.setStats(data)
  -> mapDetections() & mapBackendResultToSOPSteps()
  -> Real-time canvas overlay & step timeline updates
  -> GET /video/stream/{session_id} provides optional MJPEG annotated video
```

---

## 4. Sequence Telemetry Payload Specification

A typical sequence inference update produced by the ML pipeline:

```json
{
  "session_id": "vid-20261123-143022",
  "filename": "assembly_run_04.mp4",
  "kind": "video",
  "model_name": "sop_assy_model_v1",
  "state": "RUNNING",
  "status": "RUNNING",
  "frame": 145,
  "frames_processed": 145,
  "t": 4.83,
  "sequence": [1, 2],
  "steps_done": [1, 2],
  "doing": 3,
  "taken_back": [],
  "steps": [
    {
      "step": 1,
      "label": "Pick Base Plate",
      "status": "done",
      "t": 1.4,
      "detected": 98,
      "in_order": true
    },
    {
      "step": 2,
      "label": "Insert Gasket",
      "status": "done",
      "t": 3.6,
      "detected": 96,
      "in_order": true
    },
    {
      "step": 3,
      "label": "Mount PCB",
      "status": "doing",
      "t": null,
      "detected": null,
      "in_order": null
    },
    {
      "step": 4,
      "label": "Fasten 4x Screws",
      "status": "waiting",
      "t": null,
      "detected": null,
      "in_order": null
    },
    {
      "step": 5,
      "label": "Attach Outer Cover",
      "status": "waiting",
      "t": null,
      "detected": null,
      "in_order": null
    }
  ],
  "total_steps": 5,
  "steps_done_count": 2,
  "steps_progress": 40.0,
  "is_anomaly_frame": false,
  "verdict": null,
  "final": false,
  "detections": [
    {
      "id": "det-1",
      "class_name": "PCB",
      "confidence": 0.95,
      "isAnomaly": false,
      "bbox": [180, 120, 420, 360]
    }
  ],
  "metrics": {
    "fps": 29.4,
    "latency_ms": 34.0,
    "gpu_pct": 52.0
  }
}
```

### Final Inspection Verdict

When processing completes (end of video or stream inspection stop):

```json
{
  "final": true,
  "status": "NORMAL",
  "verdict": "NORMAL",
  "sequence": [1, 2, 3, 4, 5],
  "total_steps": 5,
  "steps_done_count": 5,
  "steps_progress": 100.0
}
```

If an error occurred (e.g. step 4 was performed before step 3, or a step was skipped):
- `status`: `"ANOMALY"`
- `verdict`: `"ANOMALY"`
- `steps[i].in_order`: `false`
- `steps[i].note`: `"Done before an earlier step"` or `"Skipped required step"`

---

## 5. Bounding Box & Coordinate Normalization

Detection bounding boxes in `detections[].bbox` are pixel corner coordinates:

```text
[x1, y1, x2, y2]
```

Where:
- `(x1, y1)` = Top-left corner in pixel space
- `(x2, y2)` = Bottom-right corner in pixel space

### Percentage Coordinate Conversion

`mapDetections()` converts these absolute pixels into responsive 0–100% canvas coordinates:

```text
x      = (x1 / video_width)  * 100
y      = (y1 / video_height) * 100
width  = ((x2 - x1) / video_width)  * 100
height = ((y2 - y1) / video_height) * 100
```

This guarantees pixel-perfect bounding box overlays across all screen resolutions, aspect ratios, and responsive viewport resizes.

---

## 6. Frontend Data Mapping

- **`mapDetections(data, videoWidth, videoHeight)`:**
  Converts backend detection dictionaries into responsive `DetectionBox` objects for `<DetectionCanvas>`.
- **`mapBackendResultToSOPSteps(data)`:**
  Translates sequence lists and step arrays into frontend `SOPStep` objects with status flags (`waiting`, `doing`, `done`, `ooo`) and completion timestamps for `<SequenceSidebar>` and `<SequenceTimeline>`.
- **`useAnomalyStatus(detections, ...)`:**
  Determines the overall system state (`NORMAL` vs `ANOMALY`), calculating real-time inspection status and audio/visual indicators.

---

## 7. Key Files Reference

| Subsystem | File Path | Responsibility |
| :--- | :--- | :--- |
| **ML Module** | `sequence_backend/app/ml/sop_inference/` | Core neural network weights, TCN, and temporal action detectors |
| **ML Orchestrator** | `sequence_backend/app/ml/sequence_inference.py` | Model lifecycle, session isolation, frame processing, verdict logic |
| **Model Router** | `sequence_backend/app/routers/model_router.py` | Model discovery, active model selection, step inspection endpoints |
| **Live Stream Router** | `sequence_backend/app/routers/oak_camera_router.py` | Camera controls and WebSocket streaming (`/oak/inference/ws/live`) |
| **Video Router** | `sequence_backend/app/routers/video_router.py` | Video upload, background inference session management, status polling |
| **State Store** | `sequence_frontend/src/store/inferenceStore.ts` | Centralized Zustand store for inference stats and telemetry |
| **Data Mapper** | `sequence_frontend/src/utils/mlDataMapper.ts` | Bounding box percentage math and SOP step status translation |
| **Sequence UI** | `sequence_frontend/src/components/sequence/` | SOP step cards, progress bar, timeline, and out-of-order indicators |
| **Main Viewport** | `sequence_frontend/src/components/DetectionCanvas.tsx` | Video feed rendering, HUD overlays, and responsive bounding boxes |
