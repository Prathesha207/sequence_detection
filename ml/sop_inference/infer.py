"""
infer.py - SOP sequence INFERENCE in ONE file (no training code).

    python infer.py --list                          the trained models (* = active)
    python infer.py --video test.mp4                the ACTIVE model; the model + detectors are loaded ONCE
    python infer.py --video a.mp4 b.mp4 0           several videos / camera 0, same loaded model
    python infer.py --model <name> --video x.mp4    a named model

Needs next to it: models.json, weights/ (detectors + hand model), trained/ (the trained models).
Per frame: ms per frame + per stage and FPS (window + frame_times.csv); per step: started / done / took / detected %
(step_times.csv); the verdict, the report, result.json and inspection_log in outputs/.

SECTIONS (search for "MODULE:")
  _home              the data folder: SOP_MONITOR_HOME / this file's folder / ~/SOP_Inference (installed wheel)
  textmatch          word matching (object / class names)
  jsonparse          lenient JSON reading
  models             the YOLO detectors: loading (half precision on CUDA), running, the DetectorBank
  featstore          per-frame region features
  mp_hands           MediaPipe hands (placement evidence: 'let go')
  boundaries         motion signals
  relations          placement: put in / came to rest / taken out / vanished inside
  state_events       lasting object changes (appears / disappears / enters)
  evidence_engine    alignment / motion features
  places_spots       where each part lies at the start (learnt places)
  pipeline           the detection pipeline object that holds the loaded detectors
  trained            reading a trained model folder
  object_actions     the ACTION MODEL: each step judged by its own objects
  temporal           LIVE per-frame inference: read -> detect -> features -> step TCN
  config             paths, settings, the active model
  results            result / event dataclasses
  timing             per-frame and per-step timing, CSV
  detectors          detector registry (models.json), file checks
  step_tracker       live step rules (done / doing / taken back)
  inference          MODEL LOADING (one function per model) + InferenceEngine: load once, run() per video
  __init__           list_models()
  main script        config, GPU check, HUD, Excel log, processing loop, INTEGRATION API, CLI  (main())

MODEL LOADING - each model has its own function (search "def load_"):
  load_trained_model(name)   trained/<name>/model.json + places.json   the steps, its detectors, the places
  load_detectors(model)      weights/detectors/*.pt (models.json)      the YOLO detector(s) on the GPU, half precision
  load_step_tcn(model)       trained/<name>/temporal.pt                the step TCN
  load_action_model(model)   trained/<name>/action_matcher.pt          the action model (None if the model has none)
  load_hand_model()          weights/hand_landmarker.task              MediaPipe hands (None if unavailable)
  warm_up(pl, model, hand)   -                                         first CUDA calls before the first frame
  load_models(name)          all of the above, once -> LoadedModels

USE FROM ANOTHER APPLICATION - the class SOPMonitor (search "class SOPMonitor"):
    import infer
    mon = infer.SOPMonitor()                  # the active model (or SOPMonitor("<model name>")): every model loaded ONCE
    mon.start()                               # one inspection
    st = mon.process_frame(frame_bgr, t)      # per frame from YOUR camera loop -> FrameStatus (state, steps_done, ...)
    res = mon.stop()                          # the verdict: res.verdict, res.sequence, res.steps, res.anomalies
    res = mon.run_video("test.mp4")           # or: a whole file / camera index / stream read by the monitor
    mon.close()
  example_integration.py (next to this file) is a complete example.
"""
from __future__ import annotations


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon._home   (from sopmon\_home.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""the DATA folder (models.json, weights/, trained/ read; specs/, outputs/ written), decided once:
  1. the SOP_MONITOR_HOME environment variable, when set
  2. FOLDER mode: this file's own folder when models.json is next to it (the inference_code_only folder)
  3. INSTALLED (the wheel): ~/SOP_Inference - created on first use and filled with the data packed in the wheel
     (sop_inference/_data: models.json, weights/, trained/). Copy new trained models into its trained/."""
import os
import shutil
from pathlib import Path

PACKED_DATA = Path(__file__).resolve().parent / "_data"     # the wheel's data (models, weights)


def home() -> Path:
    h = os.environ.get("SOP_MONITOR_HOME")
    if h:
        return Path(h).expanduser().resolve()
    here = Path(__file__).resolve().parent
    if (here / "models.json").is_file():
        return here
    return (Path.home() / "SOP_Inference").resolve()


def ensure_home(h: Path) -> Path:
    """the data folder ready: on first use copy the wheel's packed data into it"""
    if (h / "models.json").is_file():
        return h
    if not PACKED_DATA.is_dir():
        raise FileNotFoundError(f"{h} has no models.json (and this install carries no packed data)")
    h.mkdir(parents=True, exist_ok=True)
    shutil.copytree(PACKED_DATA, h, dirs_exist_ok=True)
    print(f"[sop_inference] data folder created: {h}")
    return h


os.environ["SOP_MONITOR_HOME"] = str(ensure_home(home()))   # every section below reads the same folder


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.textmatch   (from sopmon\textmatch.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""textmatch.py — matching spec nouns to detector class names.

The DISTINCTIVE word(s) must match, so 'camera module' never collapses onto 'speaker module', and
plural/singular, underscores and dashes are tolerated ('Back_Cover' == 'back covers').
"""
import difflib
import re

_STOP = {"module", "the", "a", "an", "part", "assembly", "unit", "pack", "of",
         "and", "or", "its", "their", "with",             # joining words: 'bucket and its arm' ~ 'seat and roof'
         "no", "not", "non", "without"}                   # negation words are never the distinctive word
_NEG = re.compile(r"^(?:no|not|non|without|bare|missing)\s+")


def negated(s):
    """'no glove', 'No_plate', 'without helmet' -> True"""
    return bool(_NEG.match(norm(s)))


def same_name(a, b):
    """equal up to case, underscores and plurals: 'Yellow_pads' == 'yellow pad'"""
    return [stem(w) for w in norm(a).split()] == [stem(w) for w in norm(b).split()]


def norm(s):
    return re.sub(r"\s+", " ", str(s).lower().replace("_", " ").replace("-", " ")).strip()


def keywords(s):
    return [w for w in re.findall(r"[a-z0-9]+", norm(s)) if w not in _STOP]


def stem(w):
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def kw_overlap(a, b):
    """shared distinctive word, tolerating plural/singular (plate <-> plates)."""
    return bool({stem(w) for w in a} & {stem(w) for w in b})


def sim(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    ta, tb = set(re.findall(r"\w+", a)), set(re.findall(r"\w+", b))
    j = len(ta & tb) / len(ta | tb) if ta | tb else 0.0
    return max(j, difflib.SequenceMatcher(None, a, b).ratio())


def match(noun, options, thr=0.62):
    """spec noun -> best option (detector class), or None."""
    options = [o for o in options if str(o).strip()]
    if not options:
        return None
    nk = set(keywords(noun))
    scored = []
    for o in options:
        if norm(o) == norm(noun):
            return o
        if negated(o) != negated(noun):
            continue                       # 'no glove' never matches 'glove' or 'No_plate'
        ok = set(keywords(o))
        if nk and ok and not kw_overlap(nk, ok):
            continue                       # no shared distinctive word -> not a match
        kw = len({stem(w) for w in nk} & {stem(w) for w in ok}) / max(1, len(nk | ok))
        scored.append((0.6 * kw + 0.4 * sim(noun, o), o))
    if not scored:
        return None
    exact = [o for _, o in scored if same_name(o, noun)]
    if exact:
        return exact[0]
    best_score, best = max(scored, key=lambda x: x[0])
    if best_score >= thr - 0.2 or kw_overlap(nk, set(keywords(best))):
        return best
    return None




# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.jsonparse   (from sopmon\jsonparse.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""Pull the first JSON object out of a VLM answer (which may have text or code fences around it)."""
import json
import re


def _loads(s):
    for t in (s, re.sub(r",\s*([}\]])", r"\1", s)):          # tolerate trailing commas
        try:
            return json.loads(t)
        except Exception:
            continue
    return None


def first_object(raw):
    if isinstance(raw, dict):
        return raw
    if not raw:
        return None
    lo, hi = raw.find("{"), raw.rfind("}")
    if 0 <= lo < hi:
        v = _loads(raw[lo:hi + 1])
        if isinstance(v, dict):
            return v
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v[0]
    for m in re.finditer(r"\{[^{}]*\}", raw):             # cut off mid-answer: a complete inner object
        v = _loads(m.group(0))
        if isinstance(v, dict):
            return v
    return None


def jparse(t, brace="{"):
    """The outermost {...} (or [...]) in t, or None."""
    if isinstance(t, (dict, list)):
        return t
    if not t:
        return None
    if brace == "{":
        return first_object(t)
    a, b = t.find("["), t.rfind("]")
    return _loads(t[a:b + 1]) if 0 <= a < b else None




# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.models   (from sopmon\models.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""models.py — the detection models: open-vocabulary detectors and your own models.

Model entry (saved in models.json):
    {"type": <TYPES key>, "name": "..", "path": "<weights file, HF id or folder>",
     "conf": 0.25, "imgsz": 0 (0 = the size the model was trained at), "enabled": true,
     "classes": [..own models only..], "task": "detect|segment|obb|pose|classify",
     "crop_of": "<object>"  (classifiers only: classify crops of that object instead of the frame),
     "regions": "<json>"    (classifiers only: classify FIXED regions of the frame - slots, a scan zone ...),
     "region_size": [W, H]  (the frame size the regions are drawn on; they are scaled to the frame),
     "ignore_regions": [i]  (regions this classifier cannot judge from this camera - e.g. a slot it calls empty
                             with a plate in it: counted as not visible, never as missing)}

Classifier convention: class 0 is the 'nothing' state (empty, idle); any other class is reported when its
probability is >= conf, class 0 only when no other class is. Region results carry the region index.

inspect_model(path) works out the type of ANY weights you give it (YOLO detect / segment / OBB /
pose / classify, RT-DETR, YOLO-World, YOLOE) and reads its class names.

Each object the spec names goes to an OWN model whose class name matches it; every other object
goes to the enabled open-vocabulary detectors (they find any text prompt, and are given the visual
phrases from the prompt for it). All boxes come back as one list, de-duplicated across models.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np


os.environ.setdefault("YOLO_AUTOINSTALL", "false")          # never pip-install mid-run
os.environ.setdefault("ULTRALYTICS_AUTOINSTALL", "false")

TYPES = {
    "auto": "Auto-detect from the weights / id",
    "yolo": "Own YOLO / RT-DETR detector, segmenter, OBB, pose (.pt)",
    "yolo_cls": "Own YOLO classifier (.pt)",
    "feat_cls": "Own crop classifier (backbone features + trained head, .pt checkpoint)",
    "yolo_world": "YOLO-World (open vocabulary)",
    "yoloe": "YOLOE (open vocabulary)",
}
OPEN_VOCAB = ("yolo_world", "yoloe")
DEFAULT_PATH = {
    "yolo_world": str(home() / "weights" / "yolov8s-worldv2.pt"),
}
DEFAULT_CONF = {"auto": 0.25, "yolo_world": 0.10, "yoloe": 0.15,
                "yolo": 0.25, "yolo_cls": 0.50, "feat_cls": 0.50}


def _device():
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ── what is this model? ──
def _ultra(path):
    """Load ultralytics weights with the right class (YOLO / RTDETR / YOLOWorld / YOLOE)."""
    import ultralytics as U
    low = os.path.basename(str(path)).lower()
    if "rtdetr" in low or "rt-detr" in low:
        return U.RTDETR(path)
    if "yoloe" in low and hasattr(U, "YOLOE"):
        return U.YOLOE(path)
    if "world" in low:
        return U.YOLOWorld(path)
    m = U.YOLO(path)
    return m


def inspect_model(path):
    """-> {"type", "task", "classes", "imgsz"} for any weights file."""
    p = str(path).strip().strip('"')
    low = p.lower()
    is_file = os.path.isfile(p)
    if not is_file and not low.endswith((".pt", ".pth")):      # (an official name like yolo11n.pt downloads itself)
        raise ValueError(f"not a weights file: {p}")
    try:
        m = _ultra(p)
    except Exception:
        ck = _feat_ckpt(p) if is_file else None    # not ultralytics: a features + head checkpoint?
        if ck is None:
            raise
        return {"type": "feat_cls", "task": "classify", "classes": [str(c) for c in ck["class_names"]], "imgsz": 0}
    kind = type(m).__name__
    task = getattr(m, "task", "detect") or "detect"
    names = [str(n) for n in (m.names or {}).values()] if getattr(m, "names", None) else []
    if kind == "YOLOWorld" or "world" in os.path.basename(low):
        return {"type": "yolo_world", "task": "detect", "classes": [], "imgsz": 0}
    if kind == "YOLOE":
        return {"type": "yoloe", "task": task, "classes": [], "imgsz": 0}
    imgsz = 0
    try:
        a = getattr(m.model, "args", {}) or {}
        imgsz = int(a.get("imgsz") or 0) if not isinstance(a.get("imgsz"), (list, tuple)) else int(max(a["imgsz"]))
    except Exception:
        pass
    return {"type": "yolo_cls" if task == "classify" else "yolo", "task": task, "classes": names, "imgsz": imgsz}


def own_labels(models):
    for m in models:                       # an 'auto' entry must be resolved before its classes count
        if m.get("enabled") and m.get("type") in (None, "", "auto"):
            try:
                resolve(m)
            except Exception as e:
                print(f"[models] could not inspect {m.get('path')}: {e}", flush=True)
    return [c for m in models if m.get("enabled") and m["type"] not in OPEN_VOCAB for c in m.get("classes", [])]


# ── box cleanup ──
def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def nms(dets, iou_thr=0.55, single=()):
    """highest-conf wins; suppress overlaps of the SAME object; drop a near-identical box with a
    different label; objects in `single` keep only their strongest box (one unique part)."""
    dets = sorted(dets, key=lambda d: -d["conf"])
    keep = []
    for d in dets:
        if any((k["object"] == d["object"] and iou(d["box"], k["box"]) > iou_thr) or iou(d["box"], k["box"]) > 0.85
               for k in keep):
            continue
        if d["object"] in single and any(k["object"] == d["object"] for k in keep):
            continue
        keep.append(d)
    return keep


# ── backends: detect(frame_bgr) -> [(object, conf, (x1, y1, x2, y2))] ──
class _Ultra:
    def _predict(self, frame, **kw):
        a = dict(conf=self.conf, verbose=False, device=self.dev)
        if self.imgsz:
            a["imgsz"] = self.imgsz
        if self.dev.startswith("cuda"):
            a["half"] = True
        a.update(kw)
        return self.model.predict(frame, **{k: v for k, v in a.items() if v is not None})[0]

    @staticmethod
    def _boxes(r):
        """(cls, conf, box) for detect / segment / pose / OBB results alike."""
        ob = getattr(r, "obb", None)
        if ob is not None and getattr(ob, "xyxyxyxy", None) is not None and len(ob):
            for poly, c, cf in zip(ob.xyxyxyxy.cpu().numpy(), ob.cls.cpu().numpy().astype(int), ob.conf.cpu().numpy()):
                xs, ys = poly[:, 0], poly[:, 1]
                yield int(c), float(cf), (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        elif getattr(r, "boxes", None) is not None and len(r.boxes):
            for b, c, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int),
                                r.boxes.conf.cpu().numpy()):
                yield int(c), float(cf), tuple(float(v) for v in b)


class _YoloWorld(_Ultra):
    """YOLO-World / YOLOE: the text vocabulary is the prompt's phrases, mapped back to the objects."""
    def __init__(self, m, objects, phrases=None):
        import ultralytics as U
        self.conf, self.imgsz, self.dev = m["conf"], int(m.get("imgsz") or 0), _device()
        self.p2o = _phrase_map(objects, phrases)
        self.phrases = list(self.p2o)
        if m["type"] == "yoloe":
            self.model = U.YOLOE(m["path"])
            self.model.set_classes(self.phrases, self.model.get_text_pe(self.phrases))
        else:
            self.model = U.YOLOWorld(m["path"])
            self.model.set_classes(self.phrases)

    def detect(self, frame):
        r = self._predict(frame, half=False)
        return [(self.p2o[self.phrases[c]], cf, b) for c, cf, b in self._boxes(r) if 0 <= c < len(self.phrases)]


class _OwnYolo(_Ultra):
    """Any ultralytics detector: detect, segment, pose, OBB (rotated boxes -> their bounding box), RT-DETR."""
    def __init__(self, m, objects, phrases=None):
        self.model, self.conf, self.dev = _ultra(m["path"]), m["conf"], _device()
        self.imgsz = int(m.get("imgsz") or 0)
        self.cls_obj = _class_ids(self.model.names, objects, m.get("_class_map"))

    def detect(self, frame):
        r = self._predict(frame, classes=list(self.cls_obj) or None)
        return [(self.cls_obj[c], cf, b) for c, cf, b in self._boxes(r) if c in self.cls_obj]


def load_regions(path, polygon=None):
    """fixed regions of the frame -> [{"box": [x1, y1, x2, y2], "poly": [[x, y], ...] or None, "group": int}].
    Reads a slot config ({"roi": {x, y, w, h}, "slots": [{x_frac, y_frac, w_frac, h_frac, row}]}), a box list
    ({"rois": [[x1, y1, x2, y2], ...]}) or {"regions": [{"box", "group"}]}; `polygon` is one region."""
    if not path:
        if polygon is None:
            return []
        xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
        return [{"box": [min(xs), min(ys), max(xs), max(ys)], "poly": [list(p) for p in polygon], "group": 0}]
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if "slots" in cfg:
        r = cfg["roi"]
        return [{"box": [r["x"] + s["x_frac"] * r["w"], r["y"] + s["y_frac"] * r["h"],
                         r["x"] + (s["x_frac"] + s["w_frac"]) * r["w"], r["y"] + (s["y_frac"] + s["h_frac"]) * r["h"]],
                 "poly": None, "group": int(s.get("row", i))} for i, s in enumerate(cfg["slots"])]
    if "rois" in cfg:
        return [{"box": [float(v) for v in b], "poly": None, "group": i} for i, b in enumerate(cfg["rois"])]
    return [{"box": [float(v) for v in r["box"]], "poly": r.get("poly"), "group": int(r.get("group", i))}
            for i, r in enumerate(cfg["regions"])]


def _scaled(regions, size, frame):
    """the regions in this frame's pixels (they are drawn on a frame of `size` = [W, H])"""
    h, w = frame.shape[:2]
    sx, sy = (w / float(size[0]), h / float(size[1])) if size else (1.0, 1.0)
    out = []
    for r in regions:
        b = r["box"]
        out.append(dict(r, box=(b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy),
                        poly=[[p[0] * sx, p[1] * sy] for p in r["poly"]] if r.get("poly") else None))
    return out


def _crop(frame, r):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in r["box"])
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, max(x1 + 1, x2)), min(h, max(y1 + 1, y2))
    crop = frame[y1:y2, x1:x2].copy()
    if r.get("poly") and crop.size:                       # outside the polygon -> black, as in training
        mask = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.array(r["poly"], np.int32) - np.array([x1, y1])], 255)
        crop = cv2.bitwise_and(crop, crop, mask=mask)
    return crop if crop.size else np.zeros((8, 8, 3), np.uint8)


def _pick(probs, conf):
    """class convention: any class but 0 when its probability >= conf (the strongest), else class 0"""
    k = max(range(1, len(probs)), key=lambda i: probs[i]) if len(probs) > 1 else 0
    return (k, float(probs[k])) if k and probs[k] >= conf else (0, float(probs[0]))


class _OwnYoloCls(_Ultra):
    """Classifier. Whole frame: every matched class with probability >= conf becomes a full-frame
    'detection'. With crop_of: it classifies the crops of that object's boxes instead (e.g. a
    good / defective classifier on every detected item, a glove / bare-hand classifier on hands).
    With regions: it classifies fixed regions of the frame (e.g. empty / filled for every slot of a rack)."""
    def __init__(self, m, objects, phrases=None):
        self.model, self.conf, self.dev = _ultra(m["path"]), m["conf"], _device()
        self.imgsz = int(m.get("imgsz") or 0)
        self.crop_of = str(m.get("crop_of") or "").strip().lower() or None
        self.cls_obj = _class_ids(self.model.names, objects, m.get("_class_map"))
        self.regions, self.region_size = load_regions(m.get("regions")), m.get("region_size")

    def _classify(self, im):
        p = self._predict(im, conf=None).probs
        if p is None:
            return []
        idx = [int(i) for i in (p.top5 or [p.top1])]
        confs = [float(c) for c in (p.top5conf.tolist() if p.top5conf is not None else [p.top1conf])]
        return [(self.cls_obj[i], c) for i, c in zip(idx, confs) if i in self.cls_obj and c >= self.conf]

    def detect(self, frame, boxes=None):
        if self.regions:
            regs = _scaled(self.regions, self.region_size, frame)
            a = dict(verbose=False, device=self.dev)
            if self.imgsz:
                a["imgsz"] = self.imgsz
            res = self.model.predict([_crop(frame, r) for r in regs], **a)
            out = []
            for j, (r, rr) in enumerate(zip(regs, res)):
                k, c = _pick(rr.probs.data.float().cpu().numpy(), self.conf)
                if k in self.cls_obj:
                    out.append((self.cls_obj[k], c, r["box"], j))
            return out
        if self.crop_of:
            out = []
            for b in boxes or []:
                x1, y1, x2, y2 = (int(v) for v in b)
                crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if crop.shape[0] >= 8 and crop.shape[1] >= 8:
                    out += [(o, c, tuple(b)) for o, c in self._classify(crop)]
            return out
        h, w = frame.shape[:2]
        return [(o, c, (0.0, 0.0, float(w - 1), float(h - 1))) for o, c in self._classify(frame)]


def _feat_ckpt(path):
    """a features + head checkpoint ({"class_names", "model_state_dict", "base_cfg", "in_dim"}) or None"""
    try:
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    ok = isinstance(ck, dict) and all(k in ck for k in ("class_names", "model_state_dict", "base_cfg", "in_dim"))
    return ck if ok else None


class _FeatCls:
    """Crop classifier from a features + head checkpoint: an ImageNet backbone (base_cfg backbone_names, the
    deepest of layers_to_extract_from), average-pooled, then the trained MLP head. It classifies its regions
    (the entry's, else the checkpoint's polygon) - e.g. idle / scan on the scan zone."""
    def __init__(self, m, objects, phrases=None):
        import torch
        import torch.nn as nn
        import torchvision
        from torchvision import transforms
        ck = _feat_ckpt(m["path"])
        if ck is None:
            raise ValueError(f"not a features + head checkpoint: {m['path']}")
        self.torch, self.conf = torch, m["conf"]
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        names = {i: str(c) for i, c in enumerate(ck["class_names"])}
        self.cls_obj = _class_ids(names, objects, m.get("_class_map"))
        cfg = ck["base_cfg"]
        arch = (cfg.get("backbone_names") or ["resnet50"])[0]
        arch = {"wideresnet50": "wide_resnet50_2"}.get(arch, arch)
        net = getattr(torchvision.models, arch)(weights="IMAGENET1K_V2" if "resnet50" in arch else "DEFAULT")
        layer = (cfg.get("layers_to_extract_from") or ["layer3"])[-1]
        keep = []
        for name, mod in net.named_children():          # the backbone up to (and with) that layer
            keep.append(mod)
            if name == layer:
                break
        self.backbone = nn.Sequential(*keep).to(self.dev).eval()
        d, n = int(ck["in_dim"]), int(ck["n_classes"])
        self.head = nn.Sequential(nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, n))
        self.head.load_state_dict({k.removeprefix("net."): v for k, v in ck["model_state_dict"].items()})
        self.head.to(self.dev).eval()
        self.tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize(tuple(ck["shared_crop_hw"])),
                                      transforms.ToTensor(),
                                      transforms.Normalize(ck["imagenet_mean"], ck["imagenet_std"])])
        self.roi_rect = ck.get("training_roi_rect")
        self.regions = load_regions(m.get("regions"), ck.get("polygon_scan"))
        self.region_size = m.get("region_size")

    def detect(self, frame, boxes=None):
        torch = self.torch
        regs = _scaled(self.regions, self.region_size, frame)
        x = torch.stack([self.tf(np.ascontiguousarray(_crop(frame, r)[:, :, ::-1])) for r in regs]).to(self.dev)
        with torch.no_grad():
            f = torch.nn.functional.adaptive_avg_pool2d(self.backbone(x), 1).flatten(1)
            p = torch.softmax(self.head(f), 1).cpu().numpy()
        out = []
        for j, (r, pr) in enumerate(zip(regs, p)):
            k, c = _pick(pr, self.conf)
            if k in self.cls_obj:
                out.append((self.cls_obj[k], c, r["box"], j))
        return out


def _class_ids(names, objects, class_map=None):
    """model class id -> prompt object. class_map {class name: object} (from link_objects) is exact;
    without it each class is matched to the closest object."""
    out = {}
    for i, n in names.items():
        o = (class_map or {}).get(str(n)) if class_map else match(n, objects)
        if o is not None:
            out[int(i)] = o
    return out


def _phrase_map(objects, phrases):
    """phrase -> canonical object. `phrases` is {object: [extra phrases]} (visual synonyms)."""
    p2o = {}
    for o in objects:
        p2o.setdefault(o, o)
        for ph in (phrases or {}).get(o, []):
            ph = str(ph).strip().lower()
            if ph:
                p2o.setdefault(ph, o)
    return p2o


BACKEND = {"yolo_world": _YoloWorld, "yoloe": _YoloWorld,
           "yolo": _OwnYolo, "yolo_cls": _OwnYoloCls, "feat_cls": _FeatCls}


def resolve(m):
    """an 'auto' entry -> its concrete type, classes and task (done once, when the model is added)."""
    if m.get("type") in (None, "", "auto"):
        info = inspect_model(m["path"])
        m.update(type=info["type"], task=info["task"])
        if info["classes"]:
            m["classes"] = info["classes"]
    return m


def link_objects(models, objects):
    """Which models detect which prompt object. -> {object: (kind, [(model, class name), ...] or
    [open-vocab models], None)}. kind: "own" (EVERY own model with a matching class - their boxes are merged),
    "open" (open-vocabulary detectors), "none". The list is ordered best first: exact class name, then the
    model covering more of this prompt's objects, then the closer name."""
    on = []
    for m in models:
        if not m.get("enabled"):
            continue
        try:
            on.append(resolve(m))
        except Exception as e:
            print(f"[models] skipping {m.get('name')}: {e}", flush=True)
    own = [m for m in on if m["type"] not in OPEN_VOCAB and not m.get("crop_of")]
    ov = [m for m in on if m["type"] in OPEN_VOCAB]
    cover = {id(m): sum(1 for o in objects if match(o, m.get("classes", [])) is not None) for m in own}
    links = {}
    for o in objects:
        hits = []
        for m in own:
            c = match(o, m.get("classes", []))
            if c is not None:
                hits.append(((same_name(o, c), cover[id(m)], sim(o, c)), m, c))
        if hits:
            hits.sort(key=lambda h: h[0], reverse=True)
            links[o] = ("own", [(m, c) for _, m, c in hits], None)
        elif ov:
            links[o] = ("open", ov, None)
        else:
            links[o] = ("none", None, None)
    return links


class NullBank:
    """Detections turned off: same interface as DetectorBank, finds nothing - the VLM does everything."""
    def __init__(self, objects, log=print):
        self.objects = list(dict.fromkeys(objects))
        self.multi = set()
        self.mapping = {o: "VLM only (detections off)" for o in self.objects}
        self.kind = {o: "vlm" for o in self.objects}
        self.names = {i: o for i, o in enumerate(self.objects)}
        self.backends = []
        log("[models] detections OFF - the VLM alone judges every step")

    def detect(self, frame):
        return []

    def detect_rgb(self, im):
        return []


class DetectorBank:
    """objects : the canonical object names the spec needs
    phrases : {object: [visual phrases]} for the open-vocabulary detectors
    multi   : objects that can appear many times at once (count steps, monitored items);
              every other object keeps only its strongest box per frame"""

    def __init__(self, models, objects, phrases=None, multi=(), log=print):
        self.objects = list(dict.fromkeys(objects))
        self.multi = set(multi)
        links = link_objects(models, self.objects)                  # also resolves 'auto' entries
        on = [m for m in models if m.get("enabled") and m.get("type") not in (None, "", "auto")]
        own = [m for m in on if m["type"] not in OPEN_VOCAB]
        ov = [m for m in on if m["type"] in OPEN_VOCAB]
        assign = {id(m): [] for m in on}
        class_map = {id(m): {} for m in on}                        # model -> {its class name: object}
        self.mapping, self.kind = {}, {}
        self.alias = {}                                             # object -> other objects with the same class
        crop_cls = [m for m in own if m["type"] == "yolo_cls" and m.get("crop_of")]
        for o in self.objects:
            kind, hits, _ = links[o]
            if kind != "own":                                       # a crop classifier may know it
                for m in crop_cls:
                    c = match(o, m.get("classes", []))
                    if c is not None:
                        kind, hits = "own", [(m, c)]
                        break
            if kind == "own":
                for hit, cname in hits:                             # every model that knows it
                    assign[id(hit)].append(o)
                    prev = class_map[id(hit)].get(cname)
                    if prev is not None and prev != o:              # 'outer box' and 'sealed outer box': ONE class
                        self.alias.setdefault(o, []).extend([prev] + self.alias.pop(prev, []))
                    class_map[id(hit)][cname] = o
                self.mapping[o] = " + ".join(f"{h['name']} ('{c}')" for h, c in hits)
                self.kind[o] = "own"
            elif ov:
                for m in ov:
                    assign[id(m)].append(o)
                self.mapping[o], self.kind[o] = " + ".join(m["name"] for m in ov), "open"
            else:
                self.mapping[o], self.kind[o] = "NO MODEL (add an open-vocabulary detector)", "none"
        # a crop classifier needs the object it classifies to be detected too
        for m in crop_cls:
            src = str(m["crop_of"]).strip().lower()
            if assign[id(m)] and src not in self.objects:
                self.objects.append(src)
                for mm in ov:
                    assign[id(mm)].append(src)
                self.mapping[src], self.kind[src] = " + ".join(x["name"] for x in ov) or "NO MODEL", "open"
        self.names = {i: o for i, o in enumerate(self.objects)}
        loaded = [m for m in on if assign[id(m)] and m["type"] not in OPEN_VOCAB]
        for i, a in enumerate(loaded):                  # the same classes twice = the same work twice per frame
            for b in loaded[i + 1:]:
                if a.get("classes") and a.get("classes") == b.get("classes"):
                    log(f"[models] NOTE: '{a['name']}' and '{b['name']}' have identical classes - if they are the "
                        f"same weights, disable one (Enable / disable) to halve the detection time")
        self.backends = []
        for m in on:
            objs = assign[id(m)]
            if not objs:
                log(f"[models] {m['name']}: none of its classes is in this prompt - not loaded")
                continue
            log(f"[models] loading {m['name']} ({TYPES.get(m['type'], m['type'])}) for: {', '.join(objs)}")
            ph = {o: (phrases or {}).get(o, []) for o in objs} if m["type"] in OPEN_VOCAB else None
            mm = dict(m, _class_map=class_map[id(m)]) if m["type"] not in OPEN_VOCAB else m
            self.backends.append((m, BACKEND[m["type"]](mm, objs, ph)))
        # region classifiers: model name -> the group (fixture) of each of its regions
        self.region_groups = {m["name"]: [r["group"] for r in b.regions]
                              for m, b in self.backends if getattr(b, "regions", None)}
        # crop classifiers run last, on the boxes the other models found
        self.backends.sort(key=lambda mb: bool(mb[0].get("crop_of")))
        if not self.backends:
            raise RuntimeError("no enabled detection model covers any object of the spec "
                               "(enable an open-vocabulary detector, or add a model with matching classes)")

    def detect(self, frame):
        """frame: BGR (OpenCV) -> [{"object", "conf", "box", "model"}]"""
        h, w = frame.shape[:2]
        out, fixed = [], []
        for m, b in self.backends:
            try:
                if m.get("crop_of"):
                    src = str(m["crop_of"]).strip().lower()
                    raw = b.detect(frame, [d["box"] for d in out if d["object"] == src])
                else:
                    raw = b.detect(frame)
            except Exception as e:                      # one broken model must not stop the run
                if not getattr(b, "_warned", False):
                    print(f"[models] {m['name']} failed: {type(e).__name__}: {e}", flush=True)
                    b._warned = True
                continue
            for o, c, box, *reg in raw:
                if reg:                                  # a fixed region: one answer each, never suppressed
                    fixed.append({"object": o, "conf": c, "box": tuple(box), "model": m["name"], "region": reg[0]})
                    continue
                if m["type"] in OPEN_VOCAB:              # open vocab: drop specks and whole-frame boxes
                    a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
                    if a < 0.0003 * w * h or a > 0.7 * w * h:
                        continue
                out.append({"object": o, "conf": c, "box": tuple(box), "model": m["name"]})
        # one box only when no object on the same class comes in numbers ('wheel' and 'black wheel' on one class: the
        # 4 black wheels must not be cut to 1 by the 'wheel' name the boxes are found under)
        single = {o for o in self.objects if not ({o} | set(self.alias.get(o, []))) & self.multi}
        out = nms(out, single=single)
        # two prompt objects on one model class (the box, later the sealed box): the detection is each of them
        out += [dict(d, object=a) for d in out for a in self.alias.get(d["object"], [])]
        return out + fixed

    def release(self):
        """unload the models (names / mapping stay) - frees the GPU for the VLM once every frame is detected.
        Returns the GPU memory freed in MB (0 without CUDA)."""
        self.backends = []
        import gc
        gc.collect()
        try:
            import torch
            if not torch.cuda.is_available():
                return 0
            before = torch.cuda.memory_reserved()
            torch.cuda.empty_cache()
            return (before - torch.cuda.memory_reserved()) / 2 ** 20
        except Exception:
            return 0

    def detect_rgb(self, im):
        """im: RGB array -> [{"name", "conf", "box", "model"(, "region")}] (the sequence pipeline's format)."""
        bgr = np.ascontiguousarray(np.asarray(im)[:, :, ::-1])
        return [dict({"name": d["object"], "conf": d["conf"], "box": list(d["box"]), "model": d["model"]},
                     **({"region": d["region"]} if "region" in d else {})) for d in self.detect(bgr)]


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.featstore   (from sopmon\featstore.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""featstore.py - the FEATURE STORE: what the detector already computes, kept per box so the temporal model can learn
from it.

  per frame, per detected box:
    box, class, confidence           (the detection cache, outputs/_detcache)
    bench position                   (SIFT alignment, outputs/_evcache - the box centre on the first frame's bench)
    position on the product          (the box relative to the product's box)
    APPEARANCE VECTOR                (this file: the detector's OWN feature maps - YOLO's neck P3 / P4 / P5, the
                                      input of its Detect head - RoI-aligned at the box, averaged, L2-normalised)

The appearance vectors cost one hook: the detector's forward pass computes the maps anyway. They are saved per
video (outputs/_featstore/<detection cache key>.npz), so a temporal model trained on many videos reads them back
instead of running the detector again. They change when the detector is retrained (the key changes with it).

YOLO's features separate CLASSES, not copies of one class: two wheels look alike to it by design. They tell the
model 'this is still the same thing at its place' and 'it is now on the product'; which copy is which comes from
the learnt places and the box's size and shape.
"""
from pathlib import Path

import numpy as np

ROOT = home()
STORE = ROOT / "outputs" / "_featstore"


class RoiFeatures:
    """per-box appearance vectors from the detector's own feature maps. Hooks the Detect head of every ultralytics
    model in the bank: each detect() leaves that frame's maps behind; vectors(dets, shape) cuts the boxes out."""

    def __init__(self, bank):
        # several detectors give vectors of different lengths (their necks differ): each gets its own SLOT in one
        # vector of fixed length - one detector's vectors compare as before, two detectors' are orthogonal (cos 0)
        self.maps, self.strides, self.hooks = {}, {}, []
        self.slot, self.dim = {}, 0
        for m, b in getattr(bank, "backends", None) or []:
            net = getattr(getattr(b, "model", None), "model", None)        # ultralytics YOLO -> its nn.Module
            layers = getattr(net, "model", None)
            if layers is None or not len(layers):
                continue
            head = layers[-1]
            st = getattr(head, "stride", None)
            if st is None:
                continue
            name = m["name"]
            try:                                            # the neck's channels = the head's input channels
                n = int(sum(seq[0].conv.in_channels for seq in head.cv2))
            except Exception:
                continue
            self.slot[name] = (self.dim, n)
            self.dim += n
            self.strides[name] = [float(s) for s in st]

            def hook(mod, inp, name=name):
                x = inp[0] if inp else None
                if isinstance(x, (list, tuple)) and len(x) == len(self.strides[name]):
                    self.maps[name] = [t.detach() for t in x]
            self.hooks.append(head.register_forward_pre_hook(hook))

    @property
    def ok(self):
        return bool(self.hooks)

    def vectors(self, dets, shape):
        """dets (this frame's, from the bank just run on it), shape (h, w) of that frame -> [vector or None]"""
        import torch
        from torchvision.ops import roi_align
        h, w = shape[:2]
        out = [None] * len(dets)
        by = {}
        for i, d in enumerate(dets):
            if d.get("model") in self.maps:
                by.setdefault(d["model"], []).append(i)
        for name, idx in by.items():
            maps, strides = self.maps[name], self.strides[name]
            Hin, Win = maps[0].shape[-2] * strides[0], maps[0].shape[-1] * strides[0]
            r = min(Hin / h, Win / w)                       # the letterbox the detector put the frame in
            px, py = (Win - w * r) / 2, (Hin - h * r) / 2
            bx = torch.tensor([[d["box"][0] * r + px, d["box"][1] * r + py, d["box"][2] * r + px,
                                d["box"][3] * r + py] for d in (dets[i] for i in idx)],
                              dtype=torch.float32, device=maps[0].device)
            parts = []
            for fm, s in zip(maps, strides):
                v = roi_align(fm.float(), [bx], output_size=(2, 2), spatial_scale=1.0 / s, sampling_ratio=2,
                              aligned=True).mean((2, 3))
                parts.append(torch.nn.functional.normalize(v, dim=1))
            v = torch.nn.functional.normalize(torch.cat(parts, 1), dim=1).cpu().numpy().astype(np.float16)
            a, n = self.slot[name]
            if v.shape[1] != n:                             # not the size the head said: leave this detector out
                continue
            full = np.zeros((len(idx), self.dim), np.float16)
            full[:, a:a + n] = v
            for j, i in enumerate(idx):
                out[i] = full[j]
        return out

    def close(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []




# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.mp_hands   (from sopmon\mp_hands.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""mp_hands.py - HANDS from MediaPipe's hand landmarker (Apache 2.0; weights/hand_landmarker.task), as detector-like
boxes named 'hand' - so sopmon/relations.py can see a hand holding an object and LETTING GO of it (release).

  hands(rgb)              -> [[x1, y1, x2, y2], ...] one box per hand (its 21 landmarks, grown a little: the fingers'
                             tips are the landmarks, the hand's skin goes a bit beyond them)
  for_video(video, L)     -> per record frame the hand boxes (cached), for a video file
  add_to(L, boxes)        -> a copy of the record with ('hand', box) added to each frame's detections
The top camera sees hands from above and partly (an arm from the side): MediaPipe finds most, not all - a hand it
misses is simply 'no hand', so release evidence is used when it is there and never required.
"""
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("GLOG_minloglevel", "2")      # MediaPipe / LiteRT: no INFO lines in the app's console
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

mp_hands_ROOT = home()
MODEL = mp_hands_ROOT / "weights" / "hand_landmarker.task"
CACHE = mp_hands_ROOT / "outputs" / "_hands"
NAME = "hand"
GROW = 0.15          # each box grown by this share of its size (skin beyond the landmarks)
MAX_SIDE = 960       # frames larger than this are scaled down for MediaPipe (speed; hands are large)

_LM = None


def available():
    try:
        import mediapipe  # noqa: F401
        return MODEL.exists()
    except Exception:
        return False


def _landmarker():
    global _LM
    if _LM is None:
        from mediapipe.tasks.python import BaseOptions, vision
        opts = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL)), num_hands=4,
                                            min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3,
                                            running_mode=vision.RunningMode.IMAGE)
        _LM = vision.HandLandmarker.create_from_options(opts)
    return _LM


TIPS = (4, 8, 12)    # thumb, index, middle finger tips: what touches / holds an object


def hands(rgb, out_w=None):
    """one RGB frame -> per hand {"box": [x1, y1, x2, y2], "tips": [[x, y] thumb, index, middle tip]} in its pixels
    (or scaled to a frame out_w wide: the record's W0 - the detector's boxes are in the record's size)"""
    import cv2
    import mediapipe as mp
    h, w = rgb.shape[:2]
    k = (out_w / w) if out_w else 1.0
    s = min(1.0, MAX_SIDE / max(h, w))
    im = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    res = _landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(im)))
    out = []
    for lms in res.hand_landmarks or []:
        xs = np.array([p.x for p in lms]) * w
        ys = np.array([p.y for p in lms]) * h
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        gx, gy = GROW * (x2 - x1), GROW * (y2 - y1)
        out.append({"box": [k * float(max(0, x1 - gx)), k * float(max(0, y1 - gy)), k * float(min(w, x2 + gx)),
                            k * float(min(h, y2 + gy))],
                    "tips": [[k * float(xs[i]), k * float(ys[i])] for i in TIPS]})
    return out


def for_video(video, L, log=print):
    """the hands at each record frame of a video file -> [[box, ...] per frame] (cached)"""
    import cv2
    v = Path(video)
    CACHE.mkdir(parents=True, exist_ok=True)
    c = CACHE / f"{v.stem}_{v.stat().st_size}_{len(L['t'])}_{int(L['W0'])}_v2.json"
    if c.exists():
        return json.loads(c.read_text(encoding="utf-8"))
    cap = cv2.VideoCapture(str(v))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    log(f"  [hands] MediaPipe hands over {v.name} ({len(L['t'])} frames) ...")
    out, fi = [], 0
    for tk in L["t"]:
        want = int(round(float(tk) * fps))
        while fi < want and cap.grab():
            fi += 1
        ok, fr = cap.read()
        fi += 1
        out.append(hands(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), out_w=float(L["W0"])) if ok else [])
    cap.release()
    c.write_text(json.dumps(out), encoding="utf-8")
    return out


def add_to(L, hands_per_frame):
    """the record with the hands in its detections (a copy; the detector's own classes untouched) + their finger
    tips per frame (L['hand_tips']: what 'a hand on the object' is measured with)"""
    L2 = dict(L)
    hp = [hands_per_frame[i] if i < len(hands_per_frame) else [] for i in range(len(L["dets"]))]
    L2["dets"] = [list(ds) + [[NAME, h["box"]] for h in hp[i]] for i, ds in enumerate(L["dets"])]
    L2["hand_tips"] = [[p for h in hp[i] for p in h["tips"]] for i in range(len(hp))]
    return L2


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.boundaries   (from sopmon\boundaries.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""boundaries.py - ACTION BOUNDARIES by change-point detection (a measurable baseline; no learned model).

  YOLO + places   the step's part leaves its learnt place (a confirmed pickup)            -> the action STARTS
  window          from that pickup to the NEXT observed pickup of any other part (the hands go to fetch the next
                  part) - an observed event, not the SOP's order, so a swapped video is judged honestly
  features        per frame (3 fps): motion inside the product's box (frame difference, the work on the product),
                  the product's change against its picture just before the pickup (its state), the part at its place
  boundary        END = the change point where the work on the product goes from high to low: the best two-segment
                  (high, then low) split of the motion inside the window; no work seen at all -> no end (uncertain)

    boundaries(video, L)  -> {step: {"start", "end", "bound", "work", "ok"}}   (L = vlm_install_test.live_pass)
"""
from pathlib import Path

import cv2
import numpy as np

boundaries_ROOT = home()
OUT = boundaries_ROOT / "outputs" / "_boundaries"
SIDE = 320                    # the frames the features are computed on (long side, pixels)
boundaries_GROW = 0.3                    # the product's box grown by this much of its size (hands work at its edge)
LAST_S = 12.0                 # the window of the last pickup (no next pickup): this long after it
MIN_WORK = 0.15               # a window whose motion never reaches this share of the video's 95th percentile: no work


def _vkey(video):
    p = Path(video)
    return f"{p.stem}_{p.stat().st_size}"


def rkey(L):
    """what a per-frame cache of a video also depends on: the record's frames (how many, how far apart - the fps the
    model samples at) and the product box (from the model's detectors) - two models on one video differ in these"""
    t = np.asarray(L["t"], float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    Ps = [p for p in L.get("P") or [] if p is not None]
    pm = np.median(np.array(Ps), 0) if Ps else np.zeros(4)
    return f"{len(t)}f{int(round(1000 * dt))}_p" + "-".join(str(int(round(v / 8))) for v in pm)


def signals(video, L, log=print):
    """per live frame: motion inside the product's box, the product's picture (for its change) -> cached npz"""
    OUT.mkdir(parents=True, exist_ok=True)
    c = OUT / f"{_vkey(video)}_{rkey(L)}.npz"
    if c.exists():
        z = np.load(c)
        return {k: z[k] for k in z.files}
    t = np.array(L["t"])
    Ps = [p for p in L["P"] if p is not None]
    Pm = np.median(np.array(Ps), 0) if Ps else np.array([0, 0, L["W0"], L["H0"]], float)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or L["W0"]
    s_live = SIDE / max(L["W0"], L["H0"])
    log(f"  [boundaries] motion features of {Path(video).name} ({len(t)} frames) ...")
    prev, mot, crops, fi = None, [], [], 0
    for k, tk in enumerate(t):
        want = int(round(tk * fps))
        while fi < want:
            if not cap.grab():
                break
            fi += 1
        ok, fr = cap.read()
        fi += 1
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(fr, None, fx=SIDE / max(fr.shape[:2]), fy=SIDE / max(fr.shape[:2]),
                                    interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        P = np.array(L["P"][k] if L["P"][k] is not None else Pm) * s_live
        w, h = P[2] - P[0], P[3] - P[1]
        x1, y1 = int(max(0, P[0] - boundaries_GROW * w)), int(max(0, P[1] - boundaries_GROW * h))
        x2, y2 = int(min(g.shape[1], P[2] + boundaries_GROW * w)), int(min(g.shape[0], P[3] + boundaries_GROW * h))
        d = np.abs(g - prev) if prev is not None and prev.shape == g.shape else np.zeros_like(g)
        mot.append(float((d[y1:y2, x1:x2] > 12).mean()) if x2 > x1 and y2 > y1 else 0.0)
        Pc = np.array(Pm) * s_live                               # the product's picture in a fixed box
        crops.append(cv2.resize(g[int(Pc[1]):int(Pc[3]), int(Pc[0]):int(Pc[2])], (64, 32)))
        prev = g
    cap.release()
    n = len(mot)
    out = {"t": t[:n], "motion": np.array(mot, np.float32), "crops": np.array(crops, np.float32)}
    np.savez_compressed(c, **out)
    return out




# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.relations   (from sopmon\relations.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""relations.py - OBJECT-DESTINATION RELATIONS over time, for steps that put an object INTO / ONTO another
('put the mouse into the box'): when was it really PLACED - not carried over the box - and is it still there.

PER FRAME (one target, one destination; the largest detection of each)
  inside    the target's centre in the destination's box          near   centres close / boxes overlapping
  motion    the target's centre shift per second / its size       size   the target's box area (a held object's
                                                                         changes as it is lifted / turned)
STATE MACHINE (per frame)
  OUTSIDE -> NEAR -> ENTERING -> PLACED -> OCCLUDED_ASSUMED_PLACED -> (seen outside again for REMOVE_S) -> OUTSIDE
  OCCLUDED_ASSUMED_PLACED keeps the LAST CONFIRMED relation while the target is hidden (the lid closed over it): it
  is not new proof. A target that disappears WITHOUT a confirmed placement is UNSEEN - never 'placed'.

PLACEMENT (one stay of the target inside the destination)
  candidate  = entered the destination (inside >= ENTER_S)  and  its motion dropped (still inside the stay)
               and  the destination consistent (seen around it, not moved away)
  persists   = afterwards the target is NOT seen outside the destination for POST_S (hidden is fine) - an object
               passing over the box comes out again; one placed stays
  confirmed  = candidate and persists and ( release           a hand class leaves it (only with a hand detector)
                                          or stable_unheld    still, the same size AND in view without a gap for
                                                              STABLE_S - a hand holding / pressing it covers it
                                                              (the detector loses it), so a held item fails this
                                          or post_state       the step's learnt completed state seen (the caller) )
  uncertain  = candidate, but the target is hidden too soon after to tell (no POST_S of evidence either way)
  passed_over = inside for a while, then seen outside again within POST_S
  not put in  = first seen ALREADY inside (never outside before): not done during the video
  removed_at  = a placed target later seen outside the destination for REMOVE_S: taken out again (step undone)
Times: completion = when the target came to REST inside (else its entry); confirmation = when the evidence was
complete (live: the earliest moment it could be said).
"""
import numpy as np

ENTER_S = 0.4          # inside at least this long to count as entering (one frame is a box jitter)
STILL = 0.6            # motion below this (centre shift per second / sqrt(area)) = at rest (box video: a held
                       # / carried item 1-5, one lying in the box 0-0.5)
STABLE_S = 1.0         # at rest + same size this long = not held
SIZE_TOL = 0.30        # 'the same size': within 30 % of its median in the stay
POST_S = 2.0           # after placement: not seen outside the destination for this long
REMOVE_S = 1.0         # a placed target seen outside this long = taken out again
SUPERSEDE_S = 6.0      # an object held when last seen that comes back into its destination within this was not placed
GAP_S = 0.8           # a stay continues through hidden gaps up to this long while the destination is seen
VANISH_S = 3.0         # OBJECT PERMANENCE: last seen INSIDE, then not seen anywhere this long while the destination
                       # is in view (covered by the hand / the next item, pushed under the flap) = it is in there


def _largest(ds, c):
    bs = [b for n, b in ds if n == c]
    return max(bs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if bs else None


def zone_xy(a, b):
    """the target's centre in the destination's own coordinates: u = across (0 left .. 1 right edge of its box),
    d = up from the box's BOTTOM edge in units of the box's width (the raised lid makes the box taller at the top
    only, so d does not change with the lid)"""
    cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    w = max(1.0, b[2] - b[0])
    return (cx - b[0]) / w, (b[3] - cy) / w


ZONE_MARGIN = 0.2      # the learnt interior grown by this (box widths) - another operator puts things a bit elsewhere


def in_zone(a, b, zone):
    if not zone:
        return True
    u, d = zone_xy(a, b)
    m = ZONE_MARGIN
    return zone["u"][0] - m <= u <= zone["u"][1] + m and zone["d"][0] - m <= d <= zone["d"][1] + m


def frames(L, target, dest, hand=None, zone=None):
    """per frame: visible, inside, near, motion, area, dest seen, hand on target; zone: the destination's learnt
    interior (inside = in its box AND in that zone)"""
    t = np.array(L["t"], float)
    T = len(t)
    dt = np.r_[np.diff(t), np.median(np.diff(t)) if T > 1 else 1 / 3]
    vis, ins, near, mot, area, dseen, hand_on = (np.zeros(T, bool), np.zeros(T, bool), np.zeros(T, bool),
                                                 np.full(T, np.nan), np.full(T, np.nan), np.zeros(T, bool),
                                                 np.zeros(T, bool))
    ins_rect = np.zeros(T, bool)                     # in the destination's whole rectangle (lid included)
    prev = None
    for i, ds in enumerate(L["dets"][:T]):
        a, b = _largest(ds, target), _largest(ds, dest)
        dseen[i] = b is not None
        if a is None:
            prev = None
            continue
        vis[i] = True
        cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
        ar = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
        area[i] = ar
        if b is not None:
            ins_rect[i] = b[0] <= cx <= b[2] and b[1] <= cy <= b[3]
            ins[i] = ins_rect[i] and in_zone(a, b, zone)
            bw, bh = b[2] - b[0], b[3] - b[1]
            near[i] = ins[i] or (abs(cx - (b[0] + b[2]) / 2) < 0.75 * bw + 0.5 * (a[2] - a[0])
                                 and abs(cy - (b[1] + b[3]) / 2) < 0.75 * bh + 0.5 * (a[3] - a[1]))
        if prev is not None:
            mot[i] = np.hypot(cx - prev[0], cy - prev[1]) / np.sqrt(ar) / max(dt[i - 1], 1e-3)
        prev = (cx, cy)
        if hand:
            tips = (L.get("hand_tips") or [None] * T)[i] if L.get("hand_tips") is not None else None
            if tips is not None:
                # a FINGER TIP on the object (its box grown 5 %) - a hand working next to it is not holding it
                gx, gy = 0.05 * (a[2] - a[0]), 0.05 * (a[3] - a[1])
                hand_on[i] = any(a[0] - gx <= x <= a[2] + gx and a[1] - gy <= y <= a[3] + gy for x, y in tips)
            else:
                h = _largest(ds, hand)
                hand_on[i] = h is not None and not (h[2] < a[0] or h[0] > a[2] or h[3] < a[1] or h[1] > a[3])
    return {"t": t, "vis": vis, "ins": ins, "near": near, "mot": mot, "area": area, "dseen": dseen,
            "hand_on": hand_on, "has_hand": bool(hand), "ins_rect": ins_rect}


def _stays(F):
    """the target's stays inside the destination: [i0, i1] runs of 'inside', bridged over hidden gaps <= GAP_S
    while the destination is seen (a hand over the mouse in the box)"""
    t, vis, ins = F["t"], F["vis"], F["ins"]
    out, i0, last = [], None, None
    for i in range(len(t)):
        if vis[i] and ins[i]:
            if i0 is None:
                i0 = i
            last = i
        elif vis[i] and not ins[i]:
            if i0 is not None:
                out.append((i0, last))
            i0, last = None, None
        elif i0 is not None and t[i] - t[last] > GAP_S and not F["dseen"][i]:
            out.append((i0, last))
            i0, last = None, None
    if i0 is not None:
        out.append((i0, last))
    return out


def placements(L, target, dest, live=False, hand=None, zone=None):
    """-> (frames F, [placement dict per stay inside the destination]) - see the module text"""
    F = frames(L, target, dest, hand, zone)
    t, vis, ins, mot, area = F["t"], F["vis"], F["ins"], F["mot"], F["area"]
    t_end = float(t[-1]) if len(t) else 0.0
    out = []
    for i0, i1 in _stays(F):
        ta, tb = float(t[i0]), float(t[i1])
        # FROM OUTSIDE: was the target seen outside the destination before this stay? An object first seen already
        # inside (the box opened with the mouse in it) was not PUT in during the video
        # 'outside' only where the destination is SEEN too (a closed box no longer detected is not 'outside of it');
        # CLEARLY outside = out of its whole rectangle (for 'passed over' / 'taken out': a detection on the closed
        # lid is in the rectangle but out of the learnt inside - not taken out)
        outside = vis & ~ins & F["dseen"]
        clear_out = vis & ~F["ins_rect"] & F["dseen"]
        from_outside = bool(outside[:i0].any())
        # when it first came over the destination's RECTANGLE just before (held under the raised lid): the start of
        # the before/after comparison of the step's learnt state - not its entry into the inside
        j = i0
        while j > 0 and (F["ins_rect"][j - 1] or not vis[j - 1]) and t[i0] - t[j - 1] <= 3.0:
            j -= 1
        while j < i0 and not (vis[j] and F["ins_rect"][j]):
            j += 1
        entered_rect = float(t[j])
        if tb - ta < ENTER_S and not (i1 == len(t) - 1 or not vis[min(i1 + 1, len(t) - 1)]):
            continue                                         # one frame inside, then out: a jitter
        seg = np.arange(i0, i1 + 1)
        m_in = mot[seg]
        # BEFORE: the approach (its motion, for 'dropped'); REST: the first moment from which it stays still
        pre = mot[max(0, i0 - 4):i0 + 1]
        pre_m = float(np.nanmedian(pre)) if np.isfinite(pre).any() else np.nan
        # REST (= released): from this frame the target lies still AND stays in view for STABLE_S - a hand holding or
        # pressing it covers it (the detector loses it), so a held item, even a still one, is not 'at rest'
        # (hands are NOT required to be off here: from above, fingertips working on the next item lie over a flat
        # item at the bottom of the box - a card would never be 'at rest'; hands only ADD evidence, below)
        rest = None
        for j in seg:
            w = seg[(t[seg] >= t[j]) & (t[seg] <= t[j] + STABLE_S)]
            mw = mot[w][np.isfinite(mot[w])]
            if vis[w].all() and len(mw) and (mw < STILL).all() and t[w[-1]] - t[j] >= STABLE_S - 0.35:
                rest = j
                break
        seen = seg[vis[seg]]
        last_seen = float(t[seen[-1]]) if len(seen) else tb
        # HELD WHEN LAST SEEN: a hand on it at its last sighting in the stay - it went out of view IN the hand
        held_end = bool(F["has_hand"] and len(seen) and F["hand_on"][seen[-1]])
        still_any = bool(np.isfinite(m_in).any() and np.nanmin(np.r_[m_in, np.inf]) < STILL)
        dropped = still_any and (not np.isfinite(pre_m) or pre_m >= STILL or
                                 np.nanmin(np.r_[m_in, np.inf]) <= 0.5 * pre_m)
        # DESTINATION consistent: seen over most of the stay
        dest_ok = float(F["dseen"][seg].mean()) >= 0.7
        candidate = bool(tb - ta >= ENTER_S and dropped and dest_ok) or bool(rest is not None and dest_ok)
        # STABLE + UNHELD: at rest and the same size for STABLE_S
        stable = False
        if rest is not None:
            w = seg[(t[seg] >= t[rest]) & (t[seg] <= t[rest] + STABLE_S)]
            a_ = area[w][np.isfinite(area[w])]
            stable = bool(len(a_) and np.abs(a_ / np.median(a_) - 1).max() <= SIZE_TOL)
        # HIDDEN INTO IT: the last moment the target was seen before a long hidden stretch inside the stay (pushed in
        # under the flap / covered by the next item) - its placement time when no rest is proven (a card held up
        # over the box, steady but tilting, then pushed in)
        t_hide = None
        k = 0
        while k < len(seg):
            if not vis[seg[k]]:
                k2 = k
                while k2 < len(seg) and not vis[seg[k2]]:
                    k2 += 1
                end_t = t[seg[k2]] if k2 < len(seg) else t_end
                if k > 0 and end_t - t[seg[k - 1]] > GAP_S * 2:
                    t_hide = float(t[seg[k - 1]])
                    break
                k = k2
            k += 1
        tc = float(t[rest]) if rest is not None and stable else (t_hide if t_hide is not None else
                                                                 (float(t[rest]) if rest is not None else ta))
        if held_end and rest is None and t_hide is None:
            tc = last_seen                                   # pushed in / hidden while held: when it was last seen
        # RELEASE: a hand class leaves the target while it is inside (only with a hand detector)
        release = None
        if F["has_hand"]:
            # the hand was on it in the stay (or just before), then it lies at rest with no finger tip on it
            pre_on = F["hand_on"][max(0, i0 - 3):i1 + 1].any()
            rw = seg[(t[seg] >= t[rest]) & (t[seg] <= t[rest] + STABLE_S)] if rest is not None else seg[:0]
            release = bool(pre_on and len(rw) and not F["hand_on"][rw].any())
        # PERSISTS: not seen OUTSIDE the destination for POST_S after it came to rest (hidden is fine)
        after = (t > tc) & (t <= tc + POST_S)
        out_seen = after & clear_out
        n_out = int(out_seen.sum())
        seen_after = int((after & vis).sum())
        enough = t_end >= tc + POST_S
        # STILL BEING HANDLED (live): the stay goes on at the newest frames - the target seen there but not yet at rest,
        # or just hidden (a hand over it, not yet long enough to call it pushed in): its time is not known yet
        handling = live and not (rest is not None and stable) and t_end - tb <= 2 * GAP_S
        if n_out >= 2:
            persists = False
        elif enough and not handling:
            persists = True
        else:
            persists = None                                   # live / the video ends: not decided yet
        hidden_after = seen_after == 0 and enough            # hidden all along after it (the lid at once)
        # the moment the evidence was complete
        conf_t = None
        if persists and (stable or release):
            conf_t = round(min(t_end, max(tc + POST_S, tc + (STABLE_S if stable else 0.0))), 2)
        # TAKEN OUT AGAIN: after the stay, seen OUTSIDE the destination for REMOVE_S (in a row) - the placement undone
        # the TAKE-OUT must be SEEN: the run starts next to the destination, or right after the target was last seen in
        # it (a quick lift at a few frames/s) - a detection appearing far away long after (the box was closed, a
        # look-alike on the shelf / a sleeve at the image edge) is another object, not this one
        removed_at = None
        run0 = None
        last_in = tb
        for j in range(i1 + 1, len(t)):
            if vis[j] and F["ins_rect"][j]:
                last_in = float(t[j])
            if clear_out[j]:
                if run0 is None and not F["near"][j] and t[j] - last_in > 2 * GAP_S:
                    continue
                run0 = j if run0 is None else run0
                if t[j] - t[run0] >= REMOVE_S:
                    removed_at = round(float(t[run0]), 2)
                    break
            elif vis[j] and ins[j]:
                run0 = None
        # VANISHED INSIDE (object permanence): last seen in the destination's learnt inside, then NOT SEEN ANYWHERE for
        # VANISH_S while the destination stays in view (>= 70 % of the frames), and never seen outside it afterwards -
        # it went in and did not come out. A carried-away object is seen outside; a hovering one is still seen
        vanished = None
        if len(seen) and ins[seen[-1]] and removed_at is None:
            tl = float(t[seen[-1]])
            w = (t > tl) & (t <= tl + VANISH_S)
            # its next sighting WITH the destination in view (one on top of the closed box, the destination no longer
            # detected, says nothing about in / out) must be inside again
            nxt_vis = np.flatnonzero(vis & F["dseen"] & (t > tl))
            if t_end >= tl + VANISH_S and w.any() and not (vis & w).any() and float(F["dseen"][w].mean()) >= 0.7 \
                    and not (clear_out & (t > tl)).any() and (not len(nxt_vis) or ins[nxt_vis[0]]) \
                    and from_outside and i1 > i0:
                vanished = round(tl, 2)
                candidate = True                             # went in and stayed in: its own proof of entering
                if conf_t is None and persists:
                    conf_t = round(min(t_end, tl + VANISH_S), 2)
        kind = ("already_inside" if not from_outside else "passed_over" if persists is False else
                "candidate" if candidate else "inside_briefly")
        out.append({"target": target, "dest": dest, "entered": round(ta, 2), "left": round(tb, 2),
                    "completion": round(tc, 2), "confirmation": conf_t,
                    "candidate": candidate, "persists": persists, "stable_unheld": stable, "release": release,
                    "hidden_after": hidden_after, "kind": kind, "held_end": held_end,
                    "from_outside": from_outside, "removed_at": removed_at, "entered_rect": round(entered_rect, 2),
                    "vanished_inside": vanished,
                    "last_seen": round(last_seen, 2), "t_end": round(t_end, 2), "next_entered": None,
                    "evidence": {"entered_destination": True, "inside_s": round(tb - ta, 2),
                                 "motion_dropped": bool(dropped), "rest_at": round(tc, 2) if rest is not None else None,
                                 "destination_consistent": dest_ok, "stable_unheld": stable,
                                 "release_observed": release if release is not None else "no hand detector",
                                 "not_seen_outside_after_s": POST_S if persists else
                                 (None if persists is None else 0.0),
                                 "seen_after_frames": seen_after, "seen_outside_after_frames": n_out,
                                 "visible_in_stay": round(float(vis[seg].mean()), 2),
                                 "hand_on_when_last_seen": held_end if F["has_hand"] else "no hands"}})
    for k, a in enumerate(out):
        # the target's next REAL stay in the destination (one frame re-detected while the lid closes is not one)
        nxt = [b for b in out[k + 1:] if b["candidate"] or b["left"] - b["entered"] >= ENTER_S]
        a["next_entered"] = nxt[0]["entered"] if nxt else None
    return F, out


def decide(p, post_state_ok):
    """the placement's verdict: CONFIRMED / UNCERTAIN / NOT_PLACED (+ why), with the caller's learnt post-state"""
    if not p.get("from_outside", True):
        return "NOT_PLACED", "already in the destination when first seen - not put in during the video"
    if not p["candidate"]:
        return "NOT_PLACED", "inside only briefly / never came to rest in it"
    if p["persists"] is False:
        if p["evidence"].get("rest_at") is not None and p.get("removed_at") is not None:
            return "NOT_PLACED", (f"came to rest in it at {p['evidence']['rest_at']:.1f}s, but taken out again at "
                                  f"{p['removed_at']:.1f}s")
        return "NOT_PLACED", "passed over the destination - seen outside it again"
    if p.get("held_end") and not (p["release"] or p["stable_unheld"]) and \
            p.get("next_entered") is not None and p["next_entered"] - p["last_seen"] <= SUPERSEDE_S:
        # it went out of view IN the hand and came back into the destination soon after: it was still being handled
        # (the pen lowered in while held, then arranged) - its later stay is the placement
        return "NOT_PLACED", "held when last seen, then handled again - not yet placed then"
    ev = [k for k, v in (("released", p["release"] is True), ("at rest, not held", p["stable_unheld"]),
                         ("its learnt completed state seen", post_state_ok),
                         (f"vanished inside it at {p.get('vanished_inside') or 0:.1f}s and never came out",
                          p.get("vanished_inside") is not None)) if v]
    if p["persists"] and ev:
        return "CONFIRMED", "placed: entered, came to rest, stayed in for %.0f s; " % POST_S + ", ".join(ev)
    if p["persists"] is None:
        return "UNCERTAIN", "entered and came to rest - not yet seen long enough to confirm"
    return "UNCERTAIN", "entered and came to rest, but no release / stable / completed-state evidence"


def timeline(F, decided):
    """per frame the relation STATE, from the frames and the decided placements [(placement, verdict)]"""
    t, vis, ins, near = F["t"], F["vis"], F["ins"], F["near"]
    conf = sorted((p for p, v in decided if v == "CONFIRMED"), key=lambda p: p["completion"])
    st, placed_from, out_run = [], None, 0.0
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 3
    for i in range(len(t)):
        if placed_from is None and any(abs(p["completion"] - t[i]) < 1e-6 or
                                       (p["completion"] <= t[i] and p["left"] >= t[i]) for p in conf):
            placed_from = t[i]
        if placed_from is not None:
            if vis[i] and not ins[i]:
                out_run += dt
                if out_run >= REMOVE_S:
                    placed_from, out_run = None, 0.0
                    st.append("OUTSIDE")
                    continue
                st.append("PLACED")
                continue
            out_run = 0.0
            st.append("PLACED" if vis[i] else "OCCLUDED_ASSUMED_PLACED")
            continue
        if not vis[i]:
            st.append("UNSEEN")
        elif ins[i]:
            st.append("ENTERING")
        elif near[i]:
            st.append("NEAR")
        else:
            st.append("OUTSIDE")
    return st


def segments(t, states):
    """the per-frame states as [{"state", "from", "to"}] runs (the relation log)"""
    out = []
    for i, s in enumerate(states):
        if out and out[-1]["state"] == s:
            out[-1]["to"] = round(float(t[i]), 2)
        else:
            out.append({"state": s, "from": round(float(t[i]), 2), "to": round(float(t[i]), 2)})
    return out


def hand_class(L):
    """a hand class in this detector, if any (release evidence needs one)"""
    names = {n for ds in L["dets"] for n, _ in ds}
    return next((n for n in sorted(names) if "hand" in n.lower()), None)


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.state_events   (from sopmon\state_events.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""state_events.py - STATE STEPS: steps that are not 'a part leaves its own place' get their own visible change.

Packing a phone: 'place the box open', 'close the lid', 'put the box down' all name the outer box - one learnt place,
so one pickup time for three steps; closing the lid hides the box and the phone in the same frame; the tape strips
never leave their spot. Each such step needs its OWN event.

  EVENTS  from the detector, for every class (no SOP-specific rule): it APPEARS / DISAPPEARS (and stays so for
          HOLD_S), its count goes UP / DOWN, one class ENTERS / LEAVES another's box
  LEARN   (the correct video) the steps whose pickup time cannot be trusted - no own place, a place shared with
          another step, or a time tied with another step's - each get one event, in the SOP's order (the correct
          video follows it), preferring events whose classes the step's text names ('sticker place' ->
          sticker_place) -> each step's SIGNATURE (class, kind[, other class]) -> trained/<model>/state_steps.json
  APPLY   (any video) each state step's signature searched for ANYWHERE in the video (not in the expected order):
          its time replaces the shared / missing pickup time; no such event -> not seen
"""
import json
import re
from pathlib import Path

import numpy as np

HOLD_S = 2.0                 # a change counts once it has held this long
TIE_S = 1.0
KIND_WORDS = {"appears": "appear visible open place", "disappears": "disappear close closed covered hidden",
              "up": "more add", "down": "take taken remove fewer", "enters": "into in put insert inside",
              "leaves": "out remove"}


def _smooth(x, w):
    return np.array([np.median(x[max(0, k - w):k + w + 1]) for k in range(len(x))])


def events(L):
    """-> [{"t", "cls", "kind", "other", "held"}] every sustained change of the detector's classes, by time"""
    t = np.array(L["t"], float)
    T = len(t)
    if T < 4:
        return []
    dt = float(np.median(np.diff(t)))
    w = max(1, int(round(0.7 / dt)))
    hold = max(1, int(round(HOLD_S / dt)))
    classes = sorted({n for ds in L["dets"] for n, _ in ds})
    cnt = {c: np.array([sum(1 for n, _ in ds if n == c) for ds in L["dets"]], float) for c in classes}
    out = []

    def changes(sig, make):
        lvl, k = sig[0], 1
        while k < T:
            if sig[k] != lvl and np.all(sig[k:k + hold] == sig[k]):
                j = k + hold
                while j < T and sig[j] == sig[k]:
                    j += 1
                ev = make(lvl, sig[k])
                if ev:
                    out.append(dict(ev, t=float(t[k]), held=round(float(t[min(j, T - 1)] - t[k]), 1)))
                lvl, k = sig[k], j
            else:
                k += 1
    for c in classes:
        s = _smooth(np.minimum(cnt[c], 3), w)
        changes(s, lambda a, b, c=c: {"cls": c, "kind": "appears" if a == 0 else "disappears" if b == 0 else
                                      ("up" if b > a else "down"), "other": None})
    # one class's box inside another's (its centre in the other's box)
    ctr = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    for a in classes:
        for b in classes:
            if a == b:
                continue
            ins = np.array([float(any(n1 == a and any(n2 == b and bb[0] <= ctr(ba)[0] <= bb[2] and
                                                      bb[1] <= ctr(ba)[1] <= bb[3] for n2, bb in ds)
                                      for n1, ba in ds)) for ds in L["dets"]])
            if ins.sum() < hold:
                continue
            changes(_smooth(ins, w), lambda x, y, a=a, b=b: {"cls": a, "kind": "enters" if y > x else "leaves",
                                                             "other": b})
    return sorted(out, key=lambda e: e["t"])




# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.evidence_engine   (from sopmon\evidence_engine.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""evidence_engine.py - the Spatiotemporal Evidence Engine: actions learnt from CHANGES in the scene, the same code for
every SOP (truck assembly, wrench work, pick and place ...). It sits after [3] detect_all and before the decoder.

  frames + detections
        |
  [E1] registration      every frame aligned to the first (SIFT + RANSAC, similarity), so positions are on the bench
                         and not in the image
  [E2] point motion      class-agnostic motion: a grid of points tracked frame to frame (OpenCV LK, or CoTracker3
                         when installed), minus the camera's own motion -> what moves in each box
  [E3] instance tracks   each detector class split into its copies, followed on the bench (Hungarian, gated)
  [E4] features          per track and frame: bench position, speed, motion in the box, distance to the product,
                         position relative to the product (where on it), size, shape
  [E5] event discovery   change points on those features -> segments (AT SOURCE / MOVED / ON PRODUCT / WORKING)
                         -> PICKUP, INSTALL, REMOVE, WORK; a pickup joined to the install that follows it is one
                         OPERATION: source -> interaction -> destination
  [E6] adaptive timing   each pickup re-timed on the source video at a higher rate around it
  [E7] appearance        a DINOv2 vector of each operation's part (as it lay at its source)

  ReferenceEventLearner   a NORMAL video's operations -> the SOP's steps (in its order) = the REFERENCE MEMORY
                          (places/<prompt key>.events.json; no network is trained)
  ReferenceEventMatcher   a test video's operations <-> the memory (source, destination, look, shape) -> each step's
                          time and confidence -> NORMAL / ANOMALY / UNCERTAIN

SHADOW MODE: the result is shown next to the verdict ('agrees' / 'DISAGREES'), never sets it. Promote it only after
it has been checked on real recordings (see experiments / the test plan).
"""
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

evidence_engine_ROOT = home()
evidence_engine_CACHE = evidence_engine_ROOT / "outputs" / "_evcache"

ALIGN_W = 640              # registration + point motion run on frames this wide
GRID = 16                  # point grid spacing (px at ALIGN_W)
SNAP_INLIERS = 200         # a direct match to the first frame this strong replaces the chained transform
GATE = 2.5                 # a detection joins a track within this x the track's size
TRACK_GAP_S = 3.0          # a track not seen for longer ends
JOIN_GAP_S = 10.0          # a track ending where a later one starts (a hand over it) within this = the same copy
MOVING_V = 0.03            # bench units (frame widths) per second: faster = moving
MOVING_M = 1.0             # residual point motion (px / frame at ALIGN_W) in the box: more = handled
ON_HOLD_S = 1.5            # on the product at least this long = installed
LINK_S = 90.0              # an install joins a pickup at most this much earlier
evidence_engine_TIE_S = 1.0                # steps closer than this keep the SOP order
ACCEPT = 0.30              # a reference <-> test match below this score is no match
SURE = 0.50                # findings resting on a match below this are UNCERTAIN

TOOL_WORDS = ("screwdriver", "wrench", "spanner", "drill", "hammer", "driver", "tool", "hand", "glove")


def _binseg(X, pen, min_len=3):
    """change points of the rows of X (n, d): binary segmentation, L2 cost, split while the gain beats `pen`
    -> sorted segment starts (0 first)"""
    n = len(X)
    if n < 2 * min_len:
        return [0]
    cs = np.vstack([np.zeros(X.shape[1]), np.cumsum(X, 0)])
    cs2 = np.concatenate([[0.0], np.cumsum((X ** 2).sum(1))])

    def cost(a, b):
        m = b - a
        s = cs[b] - cs[a]
        return cs2[b] - cs2[a] - (s ** 2).sum() / m

    cuts, todo = [0, n], [(0, n)]
    while todo:
        a, b = todo.pop()
        if b - a < 2 * min_len:
            continue
        base = cost(a, b)
        ks = np.arange(a + min_len, b - min_len + 1)
        gains = np.array([base - cost(a, k) - cost(k, b) for k in ks])
        j = int(np.argmax(gains))
        if gains[j] > pen:
            k = int(ks[j])
            cuts.append(k)
            todo += [(a, k), (k, b)]
    return sorted(set(cuts))[:-1]


class SpatiotemporalEvidenceEngine:
    """src: the Pipeline (frames, detect(), frgb(), bank, _same_class(), _product(), video_path, log ...).
    extract(src, spec) -> the evidence: tracks, events, operations (see the module text)"""

    def __init__(self, tracker="lk", log=print, progress=None, check=None, appearance=True, refine=True):
        self.tracker, self.log = tracker, log
        self.progress = progress or (lambda m, f=None: None)
        self.check = check or (lambda: None)
        self.use_app, self.use_refine = appearance, refine
        self._dino = None


    def families(self, src, spec):
        """which detector names are one class (look-alike copies), which are steps' parts, the product, the tools"""
        objs = list(getattr(src.bank, "objects", None) or [])
        canon = {o: sorted(src._same_class({o}))[0] for o in objs}
        part_of = {}
        for s in spec.get("steps") or []:
            o = match(s["part"], objs) if objs else None
            if o:
                part_of[str(s["id"])] = canon[o]
        prod = sorted({canon[o] for o in (src._product(spec, objs) or set()) if o in canon} - set(part_of.values()))
        tools = sorted({canon[o] for o in objs if any(w in o.lower() for w in TOOL_WORDS)} - set(part_of.values()))
        return {"canon": canon, "part_of": part_of, "product": prod, "tools": tools}


    def _events(self, tr, t, tool=False):
        S = tr["s"]
        n = len(S)
        p0 = S[0]["ws"]
        size = float(np.median([s["size"] for s in S[:max(3, n // 10)]]))
        on = np.array([1.0 if s["dprod"] == 0.0 else 0.0 for s in S])
        dp = np.array([1.0 if s["dprod"] is None else min(s["dprod"], 1.0) for s in S])
        sp = np.array([s["speed"] for s in S])
        sp = np.array([np.median(sp[max(0, i - 1):i + 2]) for i in range(n)])
        mv = ((sp > MOVING_V) | (np.array([s["motion"] for s in S]) > MOVING_M)).astype(float)
        away = np.array([min(np.linalg.norm(s["ws"] - p0) / max(size, 1e-3), 3.0) / 3.0 for s in S])
        X = np.stack([on, dp, mv, away], 1)
        cuts = _binseg(X, pen=1.5, min_len=3) + [n]
        segs = []
        for a, b in zip(cuts, cuts[1:]):
            lab = ("WORKING" if tool and mv[a:b].mean() > 0.4 and dp[a:b].mean() < 0.3 else
                   "ON" if on[a:b].mean() > 0.6 else
                   "SOURCE" if away[a:b].mean() < 0.5 / 3 and mv[a:b].mean() < 0.4 else "MOVED")
            if segs and segs[-1]["label"] == lab:
                segs[-1]["b"] = b
            else:
                segs.append({"label": lab, "a": a, "b": b})
        ev = []
        base = {"family": tr["family"], "track": tr}
        first = segs[0]
        src_s = S[first["a"]:first["b"]]
        if first["label"] == "SOURCE":
            src = {"ws": np.median([s["ws"] for s in src_s], 0).tolist(),
                   "box": np.median([s["box"] for s in src_s], 0).tolist(),
                   "size": float(np.median([s["size"] for s in src_s])),
                   "aspect": float(np.median([s["aspect"] for s in src_s])), "k": src_s[-1]["k"]}
            if len(segs) > 1:
                ev.append(dict(base, kind="PICKUP", t=S[first["b"] - 1]["t"], src=src, conf=0.8))
            elif S[-1]["t"] < t[-1] - JOIN_GAP_S:                # never moved here, then gone for good: carried off
                ev.append(dict(base, kind="PICKUP", t=S[-1]["t"], src=src, conf=0.6))
        for i, g in enumerate(segs):
            dur = S[g["b"] - 1]["t"] - S[g["a"]]["t"]
            if g["label"] == "ON" and dur >= ON_HOLD_S and (i > 0 or S[g["a"]]["t"] > t[0] + 3.0):
                rel = [s["rel"] for s in S[g["a"]:g["b"]] if s["rel"] is not None]
                ev.append(dict(base, kind="INSTALL", t=S[g["a"]]["t"], conf=0.8 if i > 0 else 0.6,
                               dst=np.median(rel, 0).tolist() if rel else None,
                               size=float(np.median([s["size"] for s in S[g["a"]:g["b"]]])),
                               aspect=float(np.median([s["aspect"] for s in S[g["a"]:g["b"]]]))))
                if i + 1 < len(segs) and segs[i + 1]["label"] == "MOVED" and \
                        S[segs[i + 1]["b"] - 1]["t"] - S[segs[i + 1]["a"]]["t"] >= 3.0:
                    ev.append(dict(base, kind="REMOVE", t=S[g["b"] - 1]["t"], conf=0.6))
            if g["label"] == "WORKING" and dur >= 2.0:
                rel = [s["rel"] for s in S[g["a"]:g["b"]] if s["rel"] is not None]
                ev.append(dict(base, kind="WORK", t=S[g["b"] - 1]["t"], t0=S[g["a"]]["t"], conf=0.7,
                               dst=np.median(rel, 0).tolist() if rel else None,
                               ws=np.median([s["ws"] for s in S[g["a"]:g["b"]]], 0).tolist()))
        return ev


    def release(self):
        self._dino = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


class EvidenceMixin:
    """Pipeline methods: learn_events (with 'Learn from good video') and evidence_shadow ([5x] in every run)"""




# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.places_spots   (from sopmon\places_spots.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""places_spots.py - the two learners that remember a NORMAL video's layout, taken out of pipeline.py
(Pipeline inherits them, so every call stays self.<method>):

  PLACE LEARNER   where each part's copies lie at the start and in which order their places empty
                  (learn_positions -> places/<prompt key>.json; apply_positions splits look-alike copies into a
                  step each; position_events / _pickups follow each copy from its place; place_lifecycle makes a
                  step done when its copy is on the product)
  WORK SPOTS      where the hand / tool works (workspots.py); [5w] work_spots adds them to the verdict, and
                  _spots_decide lets them decide the tool-work steps when they are reliable
"""
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np


places_spots_ROOT = home()
LAYOUT_DIFF = 30.0                  # a learned place this far (mean grey, 16x16) from its picture = laid out differently
places_spots_TIE_S = 1.0                         # copies of one part leaving their places closer than this: their order is unknown
ABSENT_S = 10.0                     # a followed copy not seen near where it was for this long = picked up
# a step's words that free the order of its look-alike copies ('mount the 4 wheels in any order')
ANY_ORDER = re.compile(r"\b(in )?(any|either|no particular|whatever) order\b|\border (of the \w+ )?(does not|doesn't) matter\b",
                       re.I)

def step_label(s):
    pass        # pipeline imports this module: looked up when called
    return pipeline_step_label(s)


class PlaceSpotMixin:
    """the place learner + work spots; needs Pipeline's frames, detect(), bank, log() ..."""

    @staticmethod
    def places_path(prompt, write=False):
        """learnt places are kept PER SOP PROMPT (places/<key>.json, the key as for the work spots): learning on one
        kind of video no longer replaces another's. A prompt without its own file still reads the old shared
        positions.json, so what was learnt before keeps working; writing never touches positions.json."""
        raise ImportError("sopmon.workspots is not part of this inference-only file (a model trained with it needs the full engine)")
        own = places_spots_ROOT / "places" / f"{app_name(prompt)}.json" if prompt else None
        if write:
            return own
        return own if own is not None and own.is_file() else places_spots_ROOT / "positions.json"

    def apply_positions(self, spec, prompt=None):
        """positions.json {part: [{"name", "box"}]} (or {part: {"ordered": false, "places": [...]}}): where the copies
        of a part lie on the table (box as fractions of the frame). A step on that part then expects one event per
        place, in the order they are listed (the square panel before the long one) - in any order if "ordered": false.
        A step on ONE copy ('mount the square grey side panel') takes the place whose name shares its extra word
        ('square panel'): the spec's own step order then orders the copies, and the place tells them apart.
        One prompt line on ALL the copies ('mount the grey side panels') is split into a step per copy, in the
        listed order ('mount the square grey side panel', 'mount the long grey side panel'), each judged on its own.
        The file: this prompt's own places (places_path), else the shared positions.json."""
        # a TRAINED MODEL's places (places_file) when the run was given one, else this prompt's own
        path = Path(self.places_file) if getattr(self, "places_file", None) else self.places_path(prompt)
        if not path.is_file():
            return
        words = lambda x: {stem(w) for w in keywords(str(x or ""))}
        known = []
        for k, v in json.loads(path.read_text(encoding="utf-8")).items():
            v = v if isinstance(v, dict) else {"places": v}
            pol = v.get("copy_order") or ({True: "reference", False: "any"}[bool(v["ordered"])] if "ordered" in v else None)
            known.append((words(k), v["places"], pol))
        out, renum = [], {}
        # steps on the SAME part, one per copy ('mount the square ... side panel', 'mount the long ... side panel' both
        # compile to part 'side panel'): the k-th of them is the k-th copy
        sib = {}
        for s in spec.get("steps") or []:
            if not (s.get("precondition") or s.get("groups")):
                sib.setdefault(" ".join(sorted(words(s.get("part")))), []).append(s["id"])
        for s in spec.get("steps") or []:
            out.append(s)
            renum.setdefault(s["id"], len(out))
            if s.get("precondition") or s.get("groups"):
                continue
            w = words(s.get("part"))
            mates = sib.get(" ".join(sorted(w)), [])
            for kw, pos, pol in known:
                if not kw or not w or not (kw <= w or w <= kw):     # 'grey side panels' ~ 'side panel'
                    continue
                # the COPY ORDER - which copy first, apart from the order of the steps: the prompt's word wins
                # ('mount the 4 wheels in any order'), else positions.json's, else the normal video's (reference)
                said = "any" if ANY_ORDER.search(str(s.get("text") or "")) else None
                s["copy_order"] = said or pol or "reference"
                ordered = s["copy_order"] != "any"
                one = [p for p in pos if words(p["name"]) & (w - kw)]
                if not one and len(pos) > 1:                 # the copy named in the step's text ('the square ...')
                    one = [p for p in pos if (words(p["name"]) - kw) & (words(s.get("text")) - kw)]
                if not one and len(mates) == len(pos) > 1:   # no name to go by: the k-th such step, the k-th place
                    one = [pos[mates.index(s["id"])]]
                if len(one) == 1 and s.get("count", 1) == 1:
                    # a step of ONE copy: the prompt names the copies in its own order
                    s["positions"], s["positions_ordered"], s["copy_order"] = one, True, "explicit"
                elif not one and s.get("count", 1) in (1, len(pos)) and ordered and len(pos) > 1:
                    out.pop()
                    whole = " ".join(stem(x) for x in str(s["part"]).split())
                    # the action without the part ('mount side panels' -> 'mount'): each copy's step names its copy
                    act = " ".join(x for x in str(s["action_query"]).split() if stem(x.lower()) not in w)
                    for p in pos:
                        extra = " ".join(x for x in keywords(p["name"]) if stem(x) not in w)
                        part = f"{extra} {whole}".strip()
                        text = re.sub(re.escape(str(s["part"])), part, s.get("text") or "", count=1, flags=re.I)
                        out.append(dict(s, part=part, action_query=act or s["action_query"], count=1, copy_of=s["part"],
                                        group_part=p["name"], slots_per_group=1,
                                        visible_per_group=1, text=re.sub(r"\bthem\b", "it", text),
                                        positions=[p], positions_ordered=True))
                    self.log(f"  {s['part']}: one step per remembered place - "
                             + ", ".join(step_label(x) for x in out[-len(pos):])
                             + f" (copy order: {s['copy_order']}"
                             + (" - as in the normal video)" if s["copy_order"] == "reference" else ")"))
                    break
                elif not one and s.get("count", 1) in (1, len(pos)):
                    s["positions"], s["positions_ordered"] = pos, ordered
                else:
                    continue
                self.log(f"  remembered places for {s['part']}: " + ", ".join(p["name"] for p in s["positions"])
                         + ("" if len(s["positions"]) < 2 else " (in this order)" if ordered else
                            " (copy order: any - the copies in any order, the step still in its place)"))
                break
        if len(out) != len(spec.get("steps") or []):          # split: number the steps again, checks follow them
            for i, s in enumerate(out, 1):
                s["id"] = i
            for q in spec.get("qa_checks") or []:
                if q.get("step_id") in renum:
                    q["step_id"] = renum[q["step_id"]]
            spec["steps"] = out




# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.pipeline   (from sopmon\pipeline.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""pipeline.py — the sequence architecture (sbs_arch), for any prompt, any video, any detection model.

  [SOP prompt]                               [Video]
       |                                        |
       v                                        v
  1. Spec-Compiler (VLM)                   2. Frame sampling @ 2-4 fps, exact timestamps
     steps / parts / counts / flags /
     QA checks / anomaly objects
       |________________________________________|
                          |
                          v
  2b. Detector bank: each object -> your own model whose class matches it, every other object ->
      the open-vocabulary detectors (fed the prompt's own visual phrases for it)
                          |
                          v
  3. Motion backbone + TAD: every part is detected in every frame; leaving the bench, moving,
     or being worked in place (optical flow / hands) IS the action -> {action, t_start, t_end}
                          |
                          v
  4a. VLM caption timeline -> an independent observed sequence
  4b. Targeted spatial verification at each t_end (VLM: seated / aligned / QA checks), rescue of
      motion misses, close-out judgement of late steps
                          |
                          v
  5. Deterministic sequence & rule aggregator -> SOP JSON (order judged by the longest in-order
     subsequence, so ONE late step is one anomaly, not a cascade)
  6. Annotated video + report

"monitor" prompts (a stream of items to inspect: conveyor QC, counting ...) take the monitor path:
detector + tracker (SAM2 if available) per frame, VLM defect inspection, off-path trajectories.

    python -m sopmon.pipeline --video V.mp4 --prompt "..." [--fps 3] [--out DIR]
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import threading
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


pipeline_ROOT = home()

MIN_HITS = 4                        # a part needs this many detections to be judged
GRACE_S = 2.0                       # sustained absence before a part counts as "consumed"
MOVE_FACTOR = 0.9                   # a part "moved" if its centroid shifts > this * its own size
FLOW_THRESH = 1.3                   # mean optical-flow magnitude (px/frame, 80x80 crop resample) that
                                    # counts as "being manipulated in place"
ORDER_TOL_S = 1.0                   # steps finishing within this of each other are not "out of order"
FRAME_CACHE = 160                   # decoded frames kept in memory (LRU)
MAX_FRAMES = 2400                   # long videos: sample fewer frames per second beyond this (13 min at 3 fps)
SPEC_VERSION = "17"             # bump when postprocess_spec changes, so cached specs are rebuilt
USE_RULE_SPEC = True            # numbered prompts: rule_spec (no VLM); False = always the VLM
_ORDER_RULE = re.compile(r"\b(?:before|after|until|unless|prior to|first|earlier than|later than)\b", re.I)
_NUM_WORDS = {w: i for i, w in enumerate("two three four five six seven eight nine ten eleven twelve".split(), 2)}
_NUMBERED = re.compile(r"(?:^|\s)(\d{1,2})[.)]\s+(?=[A-Za-z])")


_DEST = {"into", "onto", "inside", "in", "on", "at", "to"}


# a step the prompt FORBIDS - any wording: 'Don't close the lid', 'avoid touching the screen', 'the lid must not be
# closed', 'closing the lid is prohibited', 'no tape on the box', 'place the box and don't close the lid' (half a step)
pipeline__NEG = (r"(?:don'?t|don’t|do\s+not|never|must\s+not|mustn'?t|should\s+not|shouldn'?t|may\s+not|shall\s+not|"
        r"(?:is|are)?\s*not\s+(?:allowed|permitted)\s+to|avoid|refrain\s+from)")
_FORBID_PASSIVE = re.compile(r"^(.+?)\s+(?:must|should|may|can|shall|is|are|will)\s*(?:not|n'?t|’t)\s+(?:to\s+)?be\s+(.+)$", re.I)
_FORBID_AFTER = re.compile(r"^(.+?)\s+(?:is|are)\s+(?:not\s+allowed|not\s+permitted|prohibited|forbidden|banned)\b", re.I)
_FORBID_PRE = re.compile(r"(?:^|(?<=\s))(?:(?:you|they|the\s+operator|the\s+worker|please)\s+)?(?:" + pipeline__NEG +
                         r"|^(?:you\s+)?(?:cannot|can'?t|can\s+not))\s+", re.I)
_FORBID_NO = re.compile(r"^no\s+(.+)$", re.I)
_FORBID_WITHOUT = re.compile(r"^(.+?)\s+without\s+(.+)$", re.I)
# the prompt forbids something at all - without one of these a forbidden[] entry was invented
_FORBID_CUE = re.compile(r"\b(?:do\s+not|don'?t|never|must\s*n[o']t|mustn'?t|should\s*n[o']t|shouldn'?t|can'?t|cannot|"
                         r"not\s+(?:be|allowed|permitted)|prohibited|forbidden|banned|avoid|refrain|without|no\s+\w+)\b",
                         re.I)
# a sentence that sets a rule (what a QA check may come from), not one that describes the task
_RULE_CUE = re.compile(r"\b(?:flag\w*|check\w*|must|mustn'?t|should\w*|ensure\w*|verify|make\s+sure|need\w*|required?|"
                       r"always|never|don'?t|do\s+not|not\s+allowed|forbidden|prohibited|anomal\w*|wear\w*|inspect\w*)\b",
                       re.I)
# words any QA question has - they do not tie a check to the prompt ('is the operator wearing gloves?' -> 'glove')
_QA_GENERIC = {"is", "are", "doe", "do", "ha", "have", "operator", "worker", "person", "wear", "use", "there", "any",
               "visible", "correct", "correctly", "properly", "check", "it", "they", "their", "on", "in", "ok", "okay",
               "good", "fine", "yes", "look", "any", "seem", "appear"}
_JOIN_TAIL = re.compile(r"[\s,;:-]*(?:\b(?:and|but|then|and then|also|so)\b[\s,]*)*$", re.I)


class Cancelled(Exception):
    pass


# ======================================================================
# STAGE 1 - Spec-Compiler: prompt -> structured SOP spec
# ======================================================================
META = """You are a spec compiler for a video-inspection pipeline.

First, read the description and work out for yourself:
  - what kind of procedure it is (assembly, cooking, packing, lab work, maintenance...),
  - which physical objects the operator handles,
  - the discrete actions performed on those objects and whether their order is fixed,
  - WHICH problems the user wants flagged as an anomaly (they may say things like
    "flag when a part is not installed", "flag if the order is wrong", "flag any
    damaged item", "flag if no gloves"),
  - any stated rule about HOW an action must be done (PPE, grip, orientation, colour...).

Then output ONLY this JSON object (no prose). Write it COMPACT - one step per line, no
indentation, no // comments, no ```json fence - every extra space is a token to generate:

{
  "task": "<one line describing the procedure>",
  "mode": "sequence" | "monitor",  // "sequence": an operator performs discrete
                                  //   steps on parts, one after another (assembly,
                                  //   cooking, packing a kit...).
                                  // "monitor": a stream of items passes by and each
                                  //   must be checked for defects / going off-path
                                  //   (conveyor QC, inspection line, counting...).
  "ordered": true|false,          // true only if the description fixes the order
  "steps": [                      // for "sequence" mode.  For "monitor" mode this
                                  //   may be [] or a single {part: "<the item>"}.
    {"id": <int, from 1>,
     "action_query": "<short verb phrase for the action>",
     "part": "<the material/component being ADDED, APPLIED, or ATTACHED in
              THIS step - never the object it's being added to. 1-3 plain
              words a detector could label, no brand names, no adjectives.
              Disambiguating examples for 'verb X in/with/into Y' phrasing,
              where X and Y are different nouns:
                'install the camera module' -> part='camera module'
                  (the thing installed, NOT the phone it goes into)
                'wrap the product in bubble wrap' -> part='bubble wrap'
                  (the material being applied, NOT 'product')
                'seal the box with tape' -> part='tape'
                  (the material being applied, NOT 'box')
                'place the product in the box' -> part='box'
                  (here the box IS the thing being interacted with, since no
                   other new material is added in this step)
              Every distinct object named in the description must end up as
              the part of AT LEAST ONE step - never drop one silently.>",
     "count": <int, default 1 - how many identical items of "part" this step
               places.  If the description says a fixed number are loaded
               ("6 plates per spline", "30 plates in total", "fill all 20 slots")
               put that TOTAL here and make it ONE step, not many.  Each unit is
               then tracked as its own install in sequence.>,
     "groups": <int or 0 - if the items are loaded onto SEVERAL identical
                fixtures ("5 splines", "4 racks", "3 shelves"), the number of
                those fixtures.  0 if there is just one place.>,
     "group_part": "<the fixture noun a detector could label ("spline", "rack"),
                     only when groups > 0>",
     "slots_per_group": <int - count / groups when groups > 0>,
     "visible_per_group": <int - of slots_per_group, how many are visible to the
                camera at once.  If the text says "3 on the front and 3 on the
                back" and the view is from one side, only the front 3 are
                visible: put 3.  If all are visible put slots_per_group.>,
     "gate": <null, or - when the description says something is done to EACH
              fixture BEFORE its items go on it (scanned, labelled, opened,
              cleaned, inspected...) - {"action": "<that action as a short verb
              phrase, e.g. 'scan the spline'>", "tool": "<the object used to do
              it, a detector label, or null>"}>,
     "text": "<the step in full words>"}
  ],
  "forbidden": [                   // actions the description says must NOT be done ("don't close
                                  //   the lid", "do not touch the screen", "never stack them").
                                  //   They are NOT steps: leave them out of steps[] and list them
                                  //   here, written as the POSITIVE action that must not happen.
    {"action_query": "<the forbidden action, e.g. 'close the lid of the box'>",
     "part": "<the object it is done to / with, a detector label>",
     "text": "<the rule in full words>"}
  ],
  "monitor": {                     // ONLY for "monitor" mode, else null
    "item": "<the thing that passes by, 1-3 words>",
    "off_path": "<what 'leaving the expected path' looks like, or null>"
  },
  "flags": [ ... ],               // which conditions to treat as an anomaly.
                                  //   "missing_step" - a step / part was not performed
                                  //   "wrong_order"  - steps out of the stated order
                                  //   plus any qa_check id below.
                                  //   Include "missing_step"/"wrong_order" only if the
                                  //   user asks for it or it is the obvious intent.
                                  //   No flag instruction at all -> ["missing_step","wrong_order"].
  "qa_checks": [                   // ONLY visual yes/no rules (PPE, damage, orientation,
                                  //   colour...).  NEVER "missing_step" or "wrong_order"
                                  //   here - those are computed, not asked.  add each
                                  //   qa_check's id to "flags" too.
    {"id": "<slug>", "q": "<yes/no question about one action>",
     "fail_if": "<the answer that means non-compliant>"}
  ],
  "anomaly_objects": [             // Concrete physical objects a detector must put
                                  //   a bounding box around in order to judge the
                                  //   anomaly, that are a DIFFERENT kind of thing
                                  //   from the expected parts / items above - an
                                  //   intruder or wrong item named in the anomaly
                                  //   wording, or a negated catch-all for "not the
                                  //   expected item".  Each entry is a plain object
                                  //   noun ("crewmate figure", "glass bottle").
                                  //   NOT an action or event ("falling off the
                                  //   belt"), NOT the expected item with an
                                  //   adjective ("yellow milk", "other packet"),
                                  //   NOT a vague word ("label", "item").  Use []
                                  //   when the anomaly is only about the expected
                                  //   item's own colour / damage / size / position.
    "<object noun>"
  ]
}

How to choose "mode":
- "sequence" if the description is about a person carrying out steps in turn on a
  set of parts (assemble / install / attach / place / cook / pack a kit).
  steps[] = one entry per action; monitor = null.
- "monitor" if the description is about items that are produced or pass through
  continuously and each one must be inspected - for a defect (wrong colour,
  damage, contamination, wrong size...) or for leaving its expected path (falling
  off a belt, going the wrong way).  Set steps: [] and fill monitor{}.

Rules:
- sequence mode: 2-15 steps, one distinct "part" per step.
- Every stated concern the user wants flagged becomes a qa_check (visual yes/no
  question with fail_if) AND its id is added to "flags".  "leaving the path" gets
  the flag id "off_path".
- Do NOT invent qa_checks or flags the user did not ask for.
- An action the description forbids, in any wording ("don't ...", "do not ...", "never ...",
  "avoid ...", "X must not be ...", "... is not allowed / prohibited", "no ...", "without ...")
  goes in forbidden[] (written positively), never in steps[]. Doing it is an anomaly.
  A sentence with an allowed half and a forbidden half ("place the box and don't close the
  lid") gives a step for the allowed half and a forbidden[] entry for the other half.
  Name the object in the forbidden action instead of "it" ("stick the tape on the box").
  NEVER put an action that is one of the steps in forbidden[]. "Don't close the lid before the
  phone is in" / "closing the lid before the phone is in" is an ORDER rule (flag wrong_order),
  not a forbidden action.

FLAG-EXTRACTION EXAMPLES (mechanics only, any domain):
  "...flag when a part is not installed"   -> flags:["missing_step"]
  "...flag if the order is wrong"          -> flags:["wrong_order"]
  "...flag if the operator has no gloves"  -> flags:[...,"gloves"];
       qa_checks += {"id":"gloves","q":"is the operator wearing gloves?","fail_if":"no"}
  "...flag any <X> that looks <bad>"       -> flags:[...,"<x>_ok"];
       qa_checks += {"id":"<x>_ok","q":"does any <X> look <bad>?","fail_if":"yes"}
{known}
DESCRIPTION:
"""

_KNOWN = """
OBJECT NAMES YOUR DETECTION MODELS ALREADY KNOW: {labels}
When a step's part (or the monitored item) is one of these objects, write it EXACTLY as listed
here (spaces instead of underscores). Any other object is found by an open-vocabulary detector.
"""

_ORDER_WORDS = ("order", "sequence", "out of order", "wrong order", "out-of-order")
_MISS_WORDS = ("missing", "not installed", "not fitted", "skipped", "left out",
               "not performed", "not done", "not fixed", "omitted")
_MONITOR_HINTS = ("conveyor", "belt", "passes by", "passing by", "moving continuously",
                  "inspection line", "each item", "every item", "stream of", "one by one on",
                  "falls off", "fall off", "reject any", "discard any", "count how many",
                  "coming down the line", "on the line")
_BG = {"table", "desk", "workbench", "bench", "wall", "floor", "ceiling", "light", "lamp",
       "window", "door", "person", "hand", "arm", "man", "woman", "worker", "operator",
       "chair", "stool", "keyboard", "mouse", "monitor", "screen", "laptop", "computer",
       "wire", "cable", "cord", "drawer", "cabinet", "shelf", "watch", "glove", "room",
       "background", "surface", "mat", "paper", "sticker", "label", "box", "container",
       "tray", "bin", "machine", "device", "equipment", "tool", "camera"}
pipeline__NEG = ("non", "not", "no", "other", "wrong", "lacking", "without", "missing",
        "unlike", "different", "foreign")


_CHECK_VERBS = ("check", "confirm", "verify", "count", "ensure", "make sure")
_PRE_SENT = re.compile(r"precondition|must already|already (?:be )?(?:loaded|present|in place)|"
                       r"loaded with|check and confirm|must be present")


def _step_tools(text, objs):
    """the detected objects a step works WITH ('... using the red screwdriver' -> {'screwdriver red'})"""
    return {match(re.split(r"[,.;]", ph)[0], objs) for ph in re.findall(
        r"\b(?:with|using|by)\s+(?:the\s+)?([a-z][a-z \-]{2,40})", str(text).lower())} - {None}


def spec_objects(spec):
    """every object the run detects for this spec: step parts (or the monitored item), the fixtures and gate
    tools of grouped steps, and the anomaly objects. The window and the run use this same list."""
    if spec.get("mode") == "monitor":
        objs = [(spec.get("monitor") or {}).get("item") or "item"]
    else:
        objs = [s["part"] for s in spec.get("steps") or []]
        for s in spec.get("steps") or []:
            if s.get("groups"):
                objs.append(s["group_part"])
                if (s.get("gate") or {}).get("tool"):
                    objs.append(s["gate"]["tool"])
    objs += [f["part"] for f in spec.get("forbidden") or []]
    objs += list(spec.get("context_objects") or [])          # destinations named by the steps (models.specific_parts)
    out = []
    for o in objs + list(spec.get("anomaly_objects") or []):
        if o and not any(same_name(o, x) for x in out):          # 'yellow pad' == 'yellow pads'
            out.append(o)
    return out


def pipeline_step_label(s, art=""):
    """the name of a step: its action, plus its part only when the action does not name it already
    ('install the motherboard', not 'install the motherboard motherboard'). art: 'the ' -> 'install the battery'"""
    act = s["action_query"]
    return act if set(keywords(s["part"])) & set(keywords(act)) else f"{act} {art}{s['part']}"


_NEG_PREFIX = re.compile(r"^(?:no|not|non|without|bare|missing|un)[ _-]+")


def pipeline_stem(w):
    """plural + -ing / -ed: gloves -> glove, filled -> fill, loading -> load"""
    w = stem(w)
    for suf in ("ing", "ed"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    return w


def evidence_classes(spec, models, objects):
    """class names of your OWN models that the prompt mentions (also through a negation: 'no_glove' counts
    for a glove rule) but that are not already a spec object -> detected as evidence for the QA checks."""
    text = " ".join([str(spec.get("_raw_prompt") or ""), str(spec.get("task") or "")]
                    + [f"{c.get('id', '')} {c.get('q', '')}" for c in spec.get("qa_checks") or []]
                    + [str(s.get("text") or "") for s in spec.get("steps") or []])
    words = {pipeline_stem(w) for w in keywords(text)}
    out = []
    for m in models:
        if not m.get("enabled") or m.get("type") in OPEN_VOCAB or m.get("type") in (None, "", "auto"):
            continue
        free = [norm(c) for c in m.get("classes", [])
                if not (any(norm(o) == norm(c) for o in objects) or any(match(o, [c]) for o in objects))]
        # ALL of a class's words (not one: 'back' in "3 on the back" is not 'back cover')
        named = [n for n in free if {pipeline_stem(w) for w in keywords(_NEG_PREFIX.sub("", n))} <= words
                 and keywords(_NEG_PREFIX.sub("", n))]
        if not named and m.get("regions") and m["type"] in ("yolo_cls", "feat_cls"):
            # a slot classifier whose region groups ARE a grouped step's fixtures (30 slots in 5 groups = 5 splines):
            # it shows that step's items, whatever its class names ('empty' / 'filled') are
            try:
                ng = len({r["group"] for r in load_regions(m["regions"])})
            except Exception:
                ng = 0
            if ng and any(s.get("groups") == ng for s in spec.get("steps") or []):
                named = free
        if m.get("always"):
            named = free                    # 'Always use': every class, whatever the prompt says
        if not named:
            continue
        if m["type"] in ("yolo_cls", "feat_cls") or m.get("always"):
            named = free                    # a classifier's classes are alternatives: 'filled' brings 'empty'
        else:                               # a class's negated / plain counterpart: 'glove' <-> 'no_glove'
            cores = {_NEG_PREFIX.sub("", n) for n in named}
            named = [n for n in free if _NEG_PREFIX.sub("", n) in cores]
        # skip a class that is just another name for one already listed ('blue gloves' when 'glove' is there)
        out += [n for n in named if n not in out and match(n, out + list(objects)) is None]
    return out


# ======================================================================
# the pipeline
# ======================================================================
class Pipeline(PlaceSpotMixin, EvidenceMixin):
    def __init__(self, vlm, models, out_dir, fps=3.0, max_dim=1280, caption_points=40,
                 sam2_weights=None, spec_dir=None, keep_frames=False,
                 use_detections=True, audit=True, log=print, progress=None, stop=None, preview=None,
                 use_step_model=False, algorithm_only=False, vlm_sm=False, draw_boxes=True, use_action_model=True,
                 report_checks=True, write_video=True, use_workspots=False, use_evidence=True, evidence_tracker="lk",
                 places_file=None):
        self.vlm, self.models = vlm, models
        # [5x]: the evidence engine in shadow mode (evidence_engine.py) - shown next to the verdict, never deciding
        self.use_evidence, self.evidence_tracker = use_evidence, evidence_tracker
        self.places_file = places_file           # a trained model's places.json (train -> inference), else per prompt
        self.use_workspots = use_workspots       # [5w]: where the hand / tool works, learnt by 'Learn work spots...'
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.fps, self.max_dim, self.caption_points = float(fps), int(max_dim), int(caption_points)
        self.sam2_weights = sam2_weights
        self.spec_dir = Path(spec_dir) if spec_dir else pipeline_ROOT / "specs"
        self.keep_frames = keep_frames
        self.log = log
        self._progress = progress or (lambda stage, frac=None: None)
        self.stop = stop or threading.Event()
        self.use_detections = use_detections
        self.do_audit = audit
        self.use_step_model = use_step_model     # experimental: step times from the trained step model (stepmodel.py)
        self.algorithm_only = algorithm_only     # no VLM checks: detectors + the state machine decide alone
        # VLM + state machine: the VLM scores the single steps (one multiple-choice look per few seconds), the
        # detectors the counted / grouped ones, the state machine decides; no other VLM checks
        self.vlm_sm = bool(vlm_sm) and not algorithm_only and vlm is not None
        self.sm_only = self.algorithm_only or self.vlm_sm      # the verdict is the state machine's alone
        self.draw_boxes = draw_boxes             # False: the annotated video shows only the SOP panel, no detections
        # False: no VLM looks that only fill the report ([4a] confirm, [4b] boundaries, [5b] audit) when the state
        # machine gives the verdict - its own VLM looks (direction) stay
        self.report_checks = report_checks
        self.write_video = write_video           # False: no annotated video ([6])
        self.use_action_model = use_action_model  # the zero-shot video-text action model (videotext.py) in the decoder
        self._model_events = None
        self.evidence = []                 # prompt-named classes that are not step parts (glove, no_glove ...)
        self._events = None
        self._preview = preview            # preview(frame_rgb, dets or None, text) - the GUI canvas
        self._cache = OrderedDict()
        self.dets = {}                 # frame idx -> detections (each frame detected ONCE)
        self.scenery = []              # [(name, box)]: never-moving look-alikes (a printed picture of a part)
        self.frames = []
        self.bank = None
        self.n_vlm = 0

    # ── plumbing ──
    def _check(self):
        if self.stop.is_set():
            raise Cancelled()

    def progress(self, stage, frac=None):
        self._progress(stage, frac)


    def ask(self, prompt, images=None, max_new=400, tag="", max_side=896):
        self._check()
        self.n_vlm += 1
        return self.vlm.ask_text(prompt, images, max_new=max_new, tag=tag, max_side=max_side)

    def frgb(self, i):
        p = self.frames[i]["path"]
        if p in self._cache:
            self._cache.move_to_end(p)
            return self._cache[p]
        im = cv2.imread(p)
        if im is None:                                      # nearest surviving frame
            for j in list(range(i - 1, -1, -1)) + list(range(i + 1, len(self.frames))):
                im = cv2.imread(self.frames[j]["path"])
                if im is not None:
                    break
            else:
                raise FileNotFoundError(p)
        im = np.ascontiguousarray(im[:, :, ::-1])
        self._cache[p] = im
        if len(self._cache) > FRAME_CACHE:
            self._cache.popitem(last=False)
        return im

    def fpath(self, i):
        return self.frames[i]["path"]


    def spread(self, n):
        return [self.frames[int(k)]["idx"] for k in np.linspace(0, len(self.frames) - 1, min(n, len(self.frames)))]

    def detect(self, i):
        if i not in self.dets:
            self.dets[i] = [d for d in self.bank.detect_rgb(self.frgb(i)) if not self._is_scenery(d)]
        return self.dets[i]

    def _is_scenery(self, d, same=0.6):
        return any(d["name"] == nm and iou(d["box"], b) > same for nm, b in self.scenery)


    # ── STAGE 2b ──
    def prompt_vocab(self, spec, objects):
        """visual phrases for the open-vocabulary detectors, taken from the prompt's own words, plus
        intruder / anomaly classes named in the flag wording or seen on screen.
        -> (phrases {object: [phrases]}, extra anomaly objects)"""
        expected = list(objects)

        def _clean(x):
            x = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", str(x).lower()).strip())
            return x if (2 <= len(x) <= 40 and x not in _BG) else None

        def _dedupe_expected(x):
            core = re.sub(r"^(?:%s)[- ]" % "|".join(pipeline__NEG), "", x).strip()
            return bool(core) and any(sim(core, e) > 0.75 or set(keywords(core)) == set(keywords(e))
                                      for e in expected)

        def _novel(x):
            if not x:
                return False
            if any(n in x.split() or x.startswith(n + " ") or n + "-" in x for n in pipeline__NEG):
                return not _dedupe_expected(x)
            return not any(sim(x, e) > 0.6 or set(keywords(x)) & set(keywords(e)) for e in expected)

        raw = str(spec.get("_raw_prompt") or "")
        flag_text = " ".join([raw] + [c.get("q", "") for c in (spec.get("qa_checks") or [])]
                             + [str((spec.get("monitor") or {}).get("off_path") or "")])
        anom = [c for c in map(_clean, spec.get("anomaly_objects") or []) if _novel(c)]
        if "intruder" in (spec.get("flags") or []) or spec["mode"] == "monitor":
            fq = ("Expected objects for a video inspection: " f"{expected}\n"
                  "Read the anomaly instruction below. List every OTHER physical object / figure / kind of "
                  "thing it names as a reason to flag - the intruders or wrong items a detector must find - "
                  "including any negated catch-all (\"not one of the expected items\", \"anything lacking "
                  "<feature>\", \"other <type>\").\n"
                  f'INSTRUCTION: "{flag_text}"\n'
                  'Respond ONLY JSON: {"objects": ["...", "..."]}  (use [] if there are none; '
                  'name real objects only, never the words label/tag/item/object).')
            for c in map(_clean, (jparse(self.ask(fq, None, 220, "vocab-anom")) or {}).get("objects") or []):
                if _novel(c) and not any(sim(c, a) > 0.75 for a in anom):
                    anom.append(c)
            ci = self.spread(2)
            cq = (f"Expected object(s) for this inspection: {expected}.\n"
                  f"The user wants to flag: \"{flag_text}\"\n"
                  "Look at these frames. List every DISTINCT kind of figure / object actually visible in the "
                  "inspection area. Tag each as \"expected\" (one of the expected objects) or \"anomaly\" "
                  "(anything else the wording targets). Give a short plain detector label for each (no colours).\n"
                  'Respond ONLY JSON: {"expected": ["..."], "anomaly": ["..."]}  (use [] when a list is empty).')
            cen = jparse(self.ask(cq, [self.fpath(i) for i in ci], 320, "vocab-census")) or {}
            for c in map(_clean, cen.get("anomaly") or []):
                if _novel(c) and not any(sim(c, a) > 0.7 for a in anom):
                    anom.append(c)
            anom = list(dict.fromkeys(anom))[:8]
            if anom:
                gq = (f"Expected object(s): {expected}.\nCandidate anomaly classes: {anom}\n"
                      "For an object detector, KEEP a candidate only if it names a concrete physical object you "
                      "could draw a bounding box around AND it is a different kind of thing from the expected "
                      "object(s). DROP anything that is an action/event, a condition or colour of the expected "
                      "object, or a vague word.\n"
                      'Respond ONLY JSON: {"keep": ["...", ...]}')
                keep = {c for c in map(_clean, (jparse(self.ask(gq, None, 160, "vocab-gate")) or {}).get("keep") or [])}
                if keep:
                    anom = [a for a in anom if a in keep or any(sim(a, k) > 0.8 for k in keep)]
        ents = list(dict.fromkeys(expected + anom))

        dq = (f'Objects to detect: {ents}\nDescription text: "{raw}"\n'
              "For EACH object, quote the short visual phrase the DESCRIPTION uses for it - copy the user's own "
              "words for its colour, shape and material (e.g. \"grey rectangular plate\"). If the text gives no "
              "visual detail, just repeat the object name. Invent nothing.\n"
              'Respond ONLY JSON: {"<object>": "<phrase from the text>", ...}')
        desc = jparse(self.ask(dq, None, 240, "vocab-desc")) or {}
        sq = ("Objects an inspection must find in a video: " f"{ents}\nFor EACH, give up to 2 alternative noun "
              "phrases a detector could use for the SAME physical thing. At least two words each - no single "
              "generic word like 'panel' or 'frame'.\n"
              'Respond ONLY JSON: {"<object>": ["alt", "alt"], ...}')
        syn = jparse(self.ask(sq, [self.fpath(len(self.frames) // 2)], 260, "vocab-syn")) or {}

        colours = {"white", "black", "red", "green", "blue", "yellow", "orange", "purple", "pink", "brown",
                   "grey", "gray", "silver", "gold", "beige", "tan"}
        prompt_colours = colours & set(re.findall(r"[a-z]+", raw.lower()))
        generic = {"panel", "frame", "structure", "object", "item", "thing", "part", "piece", "unit", "board",
                   "shape", "block"}

        def _ok_phrase(s):
            s = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip())
            if not (2 <= len(s) <= 40) or s in _BG:
                return None
            toks = set(s.split())
            if toks & generic and len(toks) == 1:
                return None
            if (toks & colours) - prompt_colours:              # a colour the user didn't use
                return None
            return s

        phrases = {}
        for e in ents:
            got = []
            for cand in [desc.get(e), desc.get(e.rstrip("s"))] + list(syn.get(e) or syn.get(e.rstrip("s")) or [])[:2]:
                if isinstance(cand, str):
                    s = _ok_phrase(cand)
                    if s and s != e and s not in got:
                        got.append(s)
            phrases[e] = got
        return phrases, anom

    def build_bank(self, spec):
        objects = spec_objects(spec)
        self.anomaly_ents = list(spec.get("anomaly_objects") or [])
        if not self.use_detections:
            self.bank = NullBank(objects, log=self.log)
            return self.bank
        labels = own_labels(self.models)
        need_ov = [o for o in objects if not any(match(o, [c]) for c in labels)]
        has_ov = any(m.get("enabled") and m["type"] in OPEN_VOCAB for m in self.models)
        phrases, extra = {}, []
        if need_ov and has_ov and (self.vlm is None or self.algorithm_only):   # no VLM: plain object names
            self.log(f"[vocab] open-vocabulary objects {need_ov}: no VLM here - using their plain names")
        elif need_ov and has_ov:
            self.log(f"[vocab] open-vocabulary objects: {need_ov} - reading their visual phrases from the prompt")
            phrases, extra = self.prompt_vocab(spec, need_ov)
            extra = [a for a in extra if a not in objects]
            if extra:
                self.log(f"[vocab] anomaly / intruder classes: {extra}")
                spec["anomaly_objects"] = list(dict.fromkeys(list(spec.get("anomaly_objects") or []) + extra))
                if "intruder" not in spec["flags"]:
                    spec["flags"] = sorted(set(spec["flags"]) | {"intruder"})
            self.log("[vocab] phrases: " + json.dumps({k: v for k, v in phrases.items() if v}))
        elif need_ov:
            self.log(f"[models] WARNING: no enabled model knows {need_ov} and no open-vocabulary detector is "
                     "enabled - those objects can only be judged by the VLM")
        multi = set(spec.get("anomaly_objects") or [])
        if spec["mode"] == "monitor":
            multi |= set(objects)
        else:
            # (a step split into one per copy - 'first wheel' ... - still has its look-alikes all in view at once;
            # so has a part named by SEVERAL steps, one copy each: 'the square ... side panel', 'the long ... side
            # panel' - one box per class would keep only one of them)
            parts = [s["part"] for s in spec["steps"]]
            multi |= {s["part"] for s in spec["steps"] if s.get("count", 1) > 1 or s.get("copy_of")
                      or parts.count(s["part"]) > 1}
        # classes the prompt TALKS about that are not a step's part (glove / no_glove for a glove rule,
        # empty / filled for a fill check): detected as evidence for the QA checks
        self.evidence = evidence_classes(spec, self.models, objects + extra)
        if self.evidence:
            self.log(f"[models] evidence classes named by the prompt: {self.evidence}")
        multi |= set(self.evidence)
        self.bank = DetectorBank(self.models, objects + extra + self.evidence, phrases=phrases, multi=multi,
                                 log=self.log)
        self.anomaly_ents = list(spec.get("anomaly_objects") or [])
        return self.bank


    SIGNAL_KEYS = ("present", "count", "conf", "speed", "appear", "vanish")


    VSM_WIN_S, VSM_MAX = 3.0, 120          # VLM + state machine: seconds per look, most looks per video
    COND_KINDS = ("appears", "disappears", "inside", "off", "count_up", "count_down")
    REL_KINDS = ("inside", "off")                  # relations: [object, container or [containers]]


    def _product(self, spec, objs):
        """the detector objects the task line names as the product itself ('builds a red toy fire truck': the truck
        class that boxes the whole build once its first parts are joined) - never a step's part, never a tool"""
        task = str(spec.get("task") or "").lower()
        parts = {match(s["part"], objs) for s in spec.get("steps") or []}
        tools = _step_tools(task, objs)
        return {o for o in objs if o not in parts and o not in tools
                and re.search(r"\b" + re.escape(o.lower()) + r"s?\b", task)}

    def _same_class(self, names):
        """the names plus every object found on the same detector class ('gear head' = 'assembled gear head')"""
        al = getattr(self.bank, "alias", {}) or {}
        out = set(names)
        for o, others in al.items():
            if o in out or out & set(others):
                out |= {o, *others}
        return out




# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.trained   (from sopmon\trained.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""trained.py - TRAIN -> INFERENCE. A trained model is one SOP prompt's spec (the step sequence) filled in by one
NORMAL video: for each step, where its part lies at the start and when the good video takes it.

  trained/<name>/model.json   name, prompt, spec, trained-on video, date, detectors, the trained sequence, self-check
  trained/<name>/places.json  the learnt places (the place learner's file; inference reads only this one)

Inference loads a model: its spec is the sequence every test video is checked against, its places tell the steps'
parts apart. The SELF-CHECK: the good video's own times must follow the spec's order, else the model is marked
unreliable (a wrong 'good' video, or a part the detector misses).
"""
import json
import re
import time
import types
from pathlib import Path

trained_ROOT = home()
DIR = trained_ROOT / "trained"
trained_TIE_S = 1.0                     # steps of the good video closer than this are in order either way


def model_dir(name):
    return DIR / name


def trained_load(name):
    d = model_dir(name)
    m = json.loads((d / "model.json").read_text(encoding="utf-8"))
    m["places_file"] = str(d / "places.json")
    return m


def models_for(m, registry):
    """the detectors the model was trained with, switched on (whatever is enabled in the app now) - a model brings
    its own detector; one no longer in the registry is reported"""
    want = set(m.get("detectors") or [])
    out = [dict(x, enabled=x.get("name") in want) for x in registry or []]
    missing = want - {x.get("name") for x in out}
    if missing:
        raise RuntimeError(f"the model's detector(s) {sorted(missing)} are no longer in the model list")
    return out




# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.object_actions   (from sopmon\object_actions.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""object_actions.py - OBJECT-CONDITIONED ACTION MATCHER (a test: pack_test.py). A step is marked done only when ITS
objects show ITS interaction and ITS completed state - never because the video reached the time it happened in the
correct video.

  ROLES      from the SOP text, per step: the TARGET object (the step's part) and the other objects it names
             (destination, lid / sticker ...) -> detector classes by their words; generic, no rule per SOP
  FEATURES   per step and frame, only about ITS objects: each role class seen / count / movement / size change /
             appearing-disappearing; target inside / near / overlapping each other role; the work on the area.
             No clock, no position in the video, no step order
  TRAIN      one small TCN shared by all steps (receptive field ~10 s: a local action pattern, not the video's
             timeline), from the user's saved start / end labels: per step, POSITIVE inside its own interval,
             NEGATIVE everywhere else in the correct video (the other actions included - lid closing is not phone
             insertion) -> 'action happening' + 'action ends here'
  STATE      per step, the learnt completed state: the role signals that change from just before its start to just
             after its end (e.g. sticker_place seen, outer_box not seen)
  INFERENCE  per step, independently: candidate ends of its action -> each VERIFIED by its completed state
             appearing (and holding) at that end -> COMPLETED with its actual start / end; none verified ->
             MISSING_OR_UNCONFIRMED (no time, the reason); the completed state there from the start and never
             performed -> INITIAL_STATE. Steps whose evidence is identical take their verified occurrences in time
             order (1st -> the earlier step); nothing else uses the SOP order. Then the SOP graph on the end times.
"""
import json
import re
from pathlib import Path

import numpy as np


SLOTS = 3                     # role slots: the target + up to 2 other named objects
PER_ROLE = 6                  # seen, count, movement, size change, appearing (+), disappearing (-)
N_REL = 3                     # target inside / near / overlapping another role
D = SLOTS + SLOTS * PER_ROLE + (SLOTS - 1) * N_REL + 2      # slot flags + roles + relations + work, change
DIL = (1, 2, 4)               # ~10 s receptive field at 3 fps
END_PEAK = 0.35
REACH_S = 6.0                 # the completed state must be reached within this long after the candidate end
VERIFY = 0.5                  # at least this share of the learnt state change must happen at a candidate end


def _track(L, c):
    """per frame the largest box of class c (or None) and its count"""
    out, cnt = [], []
    for ds in L["dets"]:
        bs = [b for n, b in ds if n == c]
        cnt.append(len(bs))
        out.append(max(bs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if bs else None)
    return out, np.array(cnt, float)


def step_features(L, rs, work, change):
    """(T, D) the features of one step's roles"""
    t = np.array(L["t"], float)
    T = len(t)
    dt = float(np.median(np.diff(t))) if T > 1 else 1 / 3
    # 1 s in frames, but never more than the frames seen so far (live, early on, at a high fps: else the shifted
    # arrays below come out longer than T)
    w1 = max(1, min(int(round(1.0 / dt)), T - 1))
    diag = float(np.hypot(L["W0"], L["H0"]))
    X = np.zeros((T, D), np.float32)
    tracks = [_track(L, c) for c in rs]
    ctr = lambda b: np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
    area = lambda b: max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    for k, (bx, cnt) in enumerate(tracks):
        X[:, k] = 1.0                                            # slot used
        o = SLOTS + k * PER_ROLE
        seen = (cnt > 0).astype(float)
        sm = np.convolve(seen, np.ones(3) / 3, mode="same")
        X[:, o] = sm
        X[:, o + 1] = np.minimum(cnt, 3) / 3
        areas = [area(b) for b in bx if b is not None]
        ma = float(np.median(areas)) if areas else 1.0
        for i in range(T):
            b = bx[i]
            if b is None:
                continue
            j = i - 1
            if j >= 0 and bx[j] is not None:
                X[i, o + 2] = min(np.linalg.norm(ctr(b) - ctr(bx[j])) / np.sqrt(ma), 2.0)
            X[i, o + 3] = float(np.clip(np.log(area(b) / ma), -2, 2))
        fut = np.r_[sm[w1:], np.repeat(sm[-1:], w1)]
        past = np.r_[np.repeat(sm[:1], w1), sm[:-w1]]
        X[:, o + 4] = np.clip(fut - past, 0, 1)
        X[:, o + 5] = np.clip(past - fut, 0, 1)
    if tracks:
        bt = tracks[0][0]
        for k in range(1, len(tracks)):
            o = SLOTS + SLOTS * PER_ROLE + (k - 1) * N_REL
            br = tracks[k][0]
            for i in range(T):
                a, b = bt[i], br[i]
                if a is None or b is None:
                    continue
                c = ctr(a)
                X[i, o] = float(b[0] <= c[0] <= b[2] and b[1] <= c[1] <= b[3])
                X[i, o + 1] = float(np.clip(1.0 - np.linalg.norm(c - ctr(b)) / (0.25 * diag), 0, 1))
                ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
                X[i, o + 2] = float(ix / (area(a) + area(b) - ix))
    X[:, -2], X[:, -1] = work[:T], change[:T]
    return X


def _signals(video, L, log=print):
    """the work on the area (motion) and its change in the last second, from sopmon/boundaries.py"""
    S = signals(video, L, log)
    m = np.convolve(S["motion"], np.ones(3) / 3, mode="same")
    work = np.clip(m / max(float(np.percentile(m, 95)), 1e-6), 0, 2)
    cr = S["crops"]
    dt = float(np.median(np.diff(S["t"]))) if len(S["t"]) > 1 else 1 / 3
    w1 = max(1, int(round(1.0 / dt)))
    change = np.clip(np.array([np.abs(cr[k] - cr[max(0, k - w1)]).mean() for k in range(len(cr))]) / 20.0, 0, 2)
    T = len(L["t"])
    pad = lambda a: np.r_[a, np.repeat(a[-1:], max(0, T - len(a)))][:T] if len(a) else np.zeros(T)
    return pad(work), pad(change)


def state_vec(X):
    """the role STATE signals of a step's features: seen per role + the relations (no motion)"""
    idx = [SLOTS + k * PER_ROLE for k in range(SLOTS)] + list(range(SLOTS + SLOTS * PER_ROLE,
                                                                   SLOTS + SLOTS * PER_ROLE + (SLOTS - 1) * N_REL))
    return X[:, idx]


def _net(d_in, n_steps, h=48, kind="tcn"):
    """the matcher's temporal model: 'tcn' (3 dilated blocks, ~10 s) or 'lstr' (causal transformer, sopmon/lstr.py)
    - the same input (the step's object features + which step) and output (action happening, action ends here)"""
    if kind == "lstr":
        raise ImportError("sopmon.lstr is not part of this inference-only file (a model trained with it needs the full engine)")
        return LSTR(d_in + n_steps, 2, h=h)
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, c, d):
            super().__init__()
            self.c1 = nn.Conv1d(c, c, 3, padding=d, dilation=d)
            self.c2 = nn.Conv1d(c, c, 3, padding=d, dilation=d)
            self.act, self.drop = nn.ReLU(), nn.Dropout(0.2)

        def forward(self, x):
            return x + self.c2(self.drop(self.act(self.c1(self.act(x)))))

    class Matcher(nn.Module):
        """the original block, 3 dilations (a local pattern); input = the step's object features + which step it is"""
        def __init__(self):
            super().__init__()
            self.inp = nn.Conv1d(d_in + n_steps, h, 1)
            self.blocks = nn.Sequential(*[Block(h, d) for d in DIL])
            self.out = nn.Conv1d(h, 2, 1)

        def forward(self, x):
            return self.out(self.blocks(self.inp(x)))
    return Matcher()


object_actions_HOLD_S = 5.0                  # a completed state must hold this long (a moving average): a 2-s hand over it is not it
SETTLE = 0.5                  # the step's time: its main change this far reached (0.8 tried: later, not better)
MIN_HOLD_S = 1.5             # ... or as long as it held in the good video before the next step hid it, at least this
LIVE_HOLD_S = 2.0            # LIVE: a state step is shown once its result held this long (the full hold decides later)


def _live_hold(inf):
    """LIVE: how long a state step's result must hold before it is shown done. The step's OWN object changing (the box
    opens / closes): LIVE_HOLD_S. Judged from ANOTHER object (the tape step: the sticker place disappears - a hand over
    it for 2 s looks the same): its full learnt hold"""
    hold = float(inf.get("hold_s", REACH_S))
    key, roles = inf.get("key") or {}, inf.get("roles") or []
    own = bool(key.get("cls")) and bool(roles) and key["cls"] == roles[0] and not key.get("other")
    return min(hold, LIVE_HOLD_S) if own else hold


def _held(sv, t, hold=object_actions_HOLD_S):
    """each state signal averaged over the next `hold` seconds (what holds from a frame on)"""
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 3
    k = max(1, int(round(hold / dt)))
    cs = np.vstack([np.zeros((1, sv.shape[1])), np.cumsum(sv, 0)])
    idx = np.arange(len(sv))
    hi = np.minimum(idx + k, len(sv))
    return (cs[hi] - cs[idx]) / (hi - idx)[:, None]


def _settle(sv, t, te, delta, hold=object_actions_HOLD_S):
    """the frame (at or after te, within REACH_S) where the step's MAIN object reaches half of its learnt change
    and holds it - the step's time, not the candidate's"""
    dl = np.array(delta)
    k = int(np.argmax(np.abs(dl)))
    if abs(dl[k]) < 0.3:
        return te
    R = _held(sv, t, hold)
    # the signal ITSELF (3 frames) at the frame: the moment the change happens; the forward average R only confirms
    # it holds from there (R alone crosses half-way BEFORE the change - a step timed ~1 s early)
    sm = np.convolve(sv[:, k], np.ones(3) / 3, mode="same")
    b = (t >= te - 2.5) & (t <= te - 0.1)
    base = float(sv[b, k].mean()) if b.any() else float(sv[0, k])
    for j in np.flatnonzero((t >= te - 0.5) & (t <= te + REACH_S)):
        if (sm[j] - base) * np.sign(dl[k]) >= SETTLE * abs(dl[k]) and \
                (R[j, k] - base) * np.sign(dl[k]) >= SETTLE * abs(dl[k]):
            return float(t[j])
    return te


def _verify(sv, t, ts, te, delta, hold=object_actions_HOLD_S):
    """the learnt completed state appearing across a candidate action [ts, te]: its objects' state just BEFORE the
    action's start vs just AFTER its end (as the state was learnt) -> (share of the learnt change that happened, text)"""
    ev = np.abs(np.array(delta)) >= 0.3
    if not ev.any():
        return None, "its objects show no visible completed state - cannot be verified"
    b = (t >= ts - 2.5) & (t <= ts - 0.1)
    a = (t >= te + 0.1) & (t <= te + REACH_S)
    if not b.any() or not a.any():
        return 0.0, "too close to the video's edge to see before / after"
    # the completed state REACHED within REACH_S (one action can be two changes a few seconds apart: the box
    # disappears, then the sticker place shows) and HELD (a 3-s average: a hand over the sticker place for 2 s is
    # not the tape): per signal the furthest held value in the learnt direction
    dl = np.array(delta)
    R = _held(sv, t, hold)
    post = np.where(dl > 0, R[a].max(0), R[a].min(0))
    got = (post - sv[b].mean(0))[ev] * np.sign(dl[ev])
    share = float(got.sum() / np.abs(np.array(delta)[ev]).sum())
    return share, f"{share:.0%} of its learnt state change seen at {te:.1f}s"


_CK = {}


def _load(path):
    """the trained matcher, loaded once (live inference asks every second)"""
    import torch
    key = (str(path), path.stat().st_mtime)
    if key not in _CK:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = _net(D, len(ck["ids"]), kind=ck.get("kind", "tcn"))
        net.load_state_dict(ck["state"])
        net.eval()
        _CK.clear()
        _CK[key] = (ck, net)
    return _CK[key]


def infer(model, video, L, log=print, model_file="action_matcher.pt", signals=None, live=False):
    """-> {step: {"status", "start_sec", "end_sec", "score", "evidence"}}, the SOP verdict. signals: (work, change)
    per frame from the live frames (live inference / a camera); else read from the video file. live: the video so
    far - a candidate counts only once its whole 'after' window (REACH_S) has been seen (the newest seconds wait)"""
    import torch
    d = Path(model["places_file"]).parent
    ck, net = _load(d / model_file)
    ids, info = ck["ids"], ck["info"]
    work, change = signals if signals is not None else _signals(video, L, log)
    t = np.array(L["t"], float)
    T = len(t)
    pad = lambda a: np.r_[np.asarray(a, float), np.zeros(max(0, T - len(a)))][:T]
    work, change = pad(work), pad(change)
    steps = model["spec"]["steps"]
    raw = {}
    rel_log, L_rel = {}, None
    for k, s in enumerate(steps):
        inf = info.get(s["id"]) or info.get(str(s["id"]))
        if not inf or "end" not in inf:
            continue
        X = step_features(L, inf["roles"], work, change)
        oh = np.zeros((len(ids), len(t)), np.float32)
        oh[ids.index(s["id"])] = 1.0
        with torch.no_grad():
            o = torch.sigmoid(net(torch.tensor(np.concatenate([X.T, oh], 0)[None]))).numpy()[0]
        act = np.convolve(o[0], np.ones(3) / 3, mode="same")
        end = np.convolve(o[1], np.ones(3) / 3, mode="same")
        sv = state_vec(X)
        cands = []
        span = float(np.clip(inf["duration"], 2.0, 12.0))           # the action's own length, not its position
        # candidate ends: (a) the network's 'action ends here' peaks, (b) every moment one of THIS step's objects
        # changes (the detector) - the object states drive it, the network helps
        js = {j for j in range(len(t)) if end[j] >= END_PEAK and end[j] >= end[max(0, j - 3):j + 4].max() - 1e-6}
        key = inf.get("key")
        ev_ts = []                                           # this step's own changes (live: the newest is provisional)
        for e in events(L):
            if key:
                ok_e = e["cls"] == key["cls"] and e.get("other") == key["other"] and e["kind"] == key["kind"]
            else:
                ok_e = e["cls"] in inf["roles"] or (e.get("other") in inf["roles"] and e["cls"] in inf["roles"])
            if ok_e:
                js.add(int(np.argmin(np.abs(t - e["t"]))))
                ev_ts.append(float(e["t"]))
        if live and len(t):
            # only moments whose result has been seen to HOLD - live for at most LIVE_HOLD_S (the full learnt hold,
            # up to 5 s, is the whole video's check; a live step it later fails is taken back in the window)
            hold0 = _live_hold(inf)
            js = {j for j in js if t[j] <= t[-1] - hold0}
        # a PLACEMENT step (its object put INTO / ONTO another - learnt from 'X enters Y'): its candidates are the
        # object's stays in the destination, each decided by sopmon/relations.py (came to rest, stayed in - not
        # carried over it - + released / not held / the learnt completed state); the object's permanence while hidden
        place = USE_RELATIONS and bool(key and key.get("kind") == "enters" and key.get("other"))
        if place:
            if L_rel is None:
                L_rel = _with_hands(L, video, signals, log)
            F_rel, pls = placements(L_rel, key["cls"], key["other"], live=live, hand=hand_class(L_rel),
                                       zone=inf.get("zone") if USE_ZONE else None)
            js = set()
            rel_log[s["id"]] = {"target": key["cls"], "destination": key["other"], "F": F_rel, "placements": []}
        for j in (sorted(js) if not place else []):
            i0 = j
            while i0 > 0 and act[i0 - 1] >= 0.5:
                i0 -= 1
            st = float(t[i0]) if i0 < j else max(float(t[0]), float(t[j]) - span)
            # the completed state checked where it was learnt: around the end (automatic labels) or across the action
            hold = float(inf.get("hold_s", object_actions_HOLD_S))         # learnt per step (older models: HOLD_S)
            share, why = _verify(sv, t, float(t[j]) if inf.get("local_state") else st, float(t[j]), inf["delta"],
                                 hold)
            ok = share is not None and share >= VERIFY
            sc = float(end[max(0, j - 2):j + 3].max())
            te = _settle(sv, t, float(t[j]), inf["delta"], hold) if ok else float(t[j])
            if ok and live and len(t) and te > t[-1] - _live_hold(inf):
                ok = False                                   # its settled moment is too new to have held yet
            cands.append({"start": min(st, te), "end": te, "score": round(sc, 2), "verified": ok,
                          "unverifiable": share is None, "share": share, "why": why,
                          "rank": (share if share is not None else 0.0) + 0.5 * sc})
        for p in (pls if place else []):
            tc = p["completion"]
            j = int(np.argmin(np.abs(t - tc)))
            hold = float(inf.get("hold_s", object_actions_HOLD_S))
            # the learnt completed state across it (from the object's entry): the third kind of placement evidence
            t_v = p.get("entered_rect", p["entered"])        # before/after from when it first came over the box
            share, why_s = _verify(sv, t, t_v, t_v, inf["delta"], hold)
            # live: the learnt state counts only once its whole 'after' window has been seen (a few frames are not it)
            post_ok = share is not None and share >= VERIFY and (not live or float(t[-1]) >= t_v + REACH_S)
            verdict, why = decide(p, post_ok)
            i0 = int(np.argmin(np.abs(t - p["entered"])))
            while i0 > 0 and act[i0 - 1] >= 0.5 and t[i0 - 1] > p["entered"] - span:
                i0 -= 1
            st = min(float(t[i0]), p["entered"])
            sc = float(end[max(0, j - 2):j + 3].max())
            rel_log[s["id"]]["placements"].append(dict(p, verdict=verdict, why=why,
                                                       post_state_share=None if share is None else round(share, 2)))
            # live = the whole video's rule: a placement that STAYED IN (persists) but has no release / state evidence
            # (hidden by the hands, the next part over it) counts as done-unverified at once, as the final check does;
            # one not yet seen long enough waits
            if verdict == "NOT_PLACED" or (live and verdict == "UNCERTAIN" and not p.get("persists")):
                continue                     # live: only CONFIRMED placements are results; the rest is a CANDIDATE
                                             # (live_state) until the whole video decides it
            if p.get("removed_at") is not None:
                # TAKEN OUT AGAIN later (a card held still over the box, then put back on the table): it does not
                # count - the step is done when the object goes in and STAYS (kept only to report REMOVED)
                cands.append({"start": min(st, tc), "end": tc, "score": round(sc, 2), "verified": False,
                              "unverifiable": False, "removed": True, "removed_at": p["removed_at"], "share": share,
                              "why": why, "rank": -1.0})
                continue
            cands.append({"start": min(st, tc), "end": tc, "score": round(sc, 2), "verified": verdict == "CONFIRMED",
                          "unverifiable": False, "placement_uncertain": verdict == "UNCERTAIN",
                          "share": share, "why": f"{why} ({why_s})", "confirm": p["confirmation"],
                          "removed_at": p.get("removed_at"),
                          "rank": (share or 0.0) + 0.5 * sc + (1.0 if verdict == "CONFIRMED" else 0.0)})
        # an unverifiable step (no visible state of its own) needs the network to be sure
        cands = [c for c in cands if not c["unverifiable"] or c["score"] >= END_PEAK]
        init = bool(np.abs(sv[t <= t[0] + 2.0].mean(0) - np.array(inf["post"])).max() < 0.3) if len(t) else False
        if inf.get("start_state"):
            # a step that is a STATE at the start in the good video (the box already open), not an action: done if
            # its state is there when the recording starts; it takes no action moment of another step
            cands = []
            init = bool(np.abs(sv[t <= t[0] + 2.0].mean(0) - np.array(inf["post"])).max() < 0.35) if len(t) else False
        raw[s["id"]] = {"cands": cands, "init": init, "roles": inf["roles"], "start_state": inf.get("start_state"),
                        "ev_ts": ev_ts, "hold0": float(inf.get("hold_s", REACH_S)),
                        "sig": tuple(inf["roles"]) + tuple(np.round(np.sign(np.array(inf["delta"])) *
                                                                     (np.abs(inf["delta"]) >= 0.3), 0))}
    # steps with IDENTICAL evidence (same objects, same state change): their verified occurrences in time order
    groups = {}
    for sid, r in raw.items():
        if not r.get("start_state"):
            groups.setdefault(r["sig"], []).append(sid)
    out = {}
    for sig, sids in groups.items():
        sids = sorted(sids)
        if len(sids) > 1:
            occ = sorted({round(c["end"], 1): c for sid in sids for c in raw[sid]["cands"] if c["verified"]}.values(),
                         key=lambda c: c["end"])
            occ = [c for i, c in enumerate(occ) if i == 0 or c["end"] - occ[i - 1]["end"] > 3.0]
            for sid, c in zip(sids, occ):
                out[sid] = c
        else:
            v = [c for c in raw[sids[0]]["cands"] if c["verified"]]
            if v:
                out[sids[0]] = max(v, key=lambda c: c["rank"])
    # a step whose objects show no visible completed state (the tape strips stay on the table): its recognised action
    # is kept, with its time, but flagged - never counted as verified
    for sid, r in raw.items():
        if sid not in out:
            # + a placement that entered and came to rest but could not be confirmed (hidden too soon): UNCERTAIN
            u = [c for c in r["cands"] if c["unverifiable"] or c.get("placement_uncertain")]
            if u:
                out[sid] = dict(max(u, key=lambda c: c["score"]), unverified=True)
    # INITIAL STATE: only the SOP's first steps, whose completed state is already there when the recording starts
    # (e.g. the box is already open) - a later step cannot be 'done before the video' while earlier ones are not
    prefix = set()
    for s in steps:
        r = raw.get(s["id"])
        if r is not None and r["init"] and s["id"] not in out:
            prefix.add(s["id"])
        else:
            break
    res = {}
    for s in steps:
        sid = s["id"]
        r = raw.get(sid)
        if r is None:
            res[sid] = {"status": "NOT_TRAINED", "start_sec": None, "end_sec": None, "score": None,
                        "evidence": "no saved label for this step"}
            continue
        c = out.get(sid)
        rm = [cc for cc in r["cands"] if cc.get("removed")]
        if not c and rm:
            # its object went in, but was taken out again and never put back to stay
            x_ = max(rm, key=lambda cc: cc["end"])
            res[sid] = {"status": "REMOVED", "start_sec": round(x_["start"], 2), "end_sec": round(x_["end"], 2),
                        "score": x_["score"], "evidence": f"put in at {x_['end']:.1f}s, but taken out again at "
                                                          f"{x_['removed_at']:.1f}s - not in its destination at the end"}
            continue
        if c and not c.get("unverified"):
            # TAKEN OUT AGAIN: the object's LAST confirmed placement was undone (seen outside its destination later)
            last_c = max([c] + [cc for cc in r["cands"] if cc["verified"] and cc["end"] > c["end"]],
                         key=lambda cc: cc["end"])
            if last_c.get("removed_at") is not None:
                res[sid] = {"status": "REMOVED", "start_sec": round(last_c["start"], 2),
                            "end_sec": round(last_c["end"], 2), "score": last_c["score"],
                            "evidence": f"put in at {last_c['end']:.1f}s, but taken out again at "
                                        f"{last_c['removed_at']:.1f}s - not in its destination at the end"}
                continue
        if c and c.get("unverified"):
            res[sid] = {"status": "RECOGNISED_UNVERIFIED", "start_sec": round(c["start"], 2),
                        "end_sec": round(c["end"], 2), "score": c["score"],
                        "evidence": f"action of {r['roles']} recognised (score {c['score']}), but {c['why']}"}
        elif c:
            res[sid] = {"status": "COMPLETED", "start_sec": round(c["start"], 2), "end_sec": round(c["end"], 2),
                        "score": c["score"], "evidence": f"action of {r['roles']} recognised (score {c['score']}); "
                                                         + c["why"]}
        elif sid in prefix:
            res[sid] = {"status": "INITIAL_STATE", "start_sec": None, "end_sec": round(float(t[0]), 2),
                        "score": None, "evidence": "its completed state is there from the start of the recording - "
                                                   "satisfied before the video, not performed in it"}
        else:
            best = max(r["cands"], key=lambda c: c["score"]) if r["cands"] else None
            ev = ("no action of " + str(r["roles"]) + " recognised") if not best else \
                f"best candidate at {best['end']:.1f}s not confirmed: {best['why']}"
            pls_ = (rel_log.get(sid) or {}).get("placements") or []
            if not best and pls_:
                # a placement step: why its object's last time in the destination was not a placement
                p_ = pls_[-1]
                ev = (f"no confirmed placement - last in its destination {p_['entered']:.1f}-{p_['left']:.1f}s: "
                      f"{p_['why']}")
            res[sid] = {"status": "MISSING_OR_UNCONFIRMED", "start_sec": None, "end_sec": None,
                        "score": best["score"] if best else None, "evidence": ev}
    # SKIPPED = proven missing: the SOP's LAST step was completed (the box closed, the work finished) while this step
    # never was - the process went past it (live: only once the last step is seen, so never too early)
    last = steps[-1]["id"] if steps else None
    finished = last is not None and res.get(last, {}).get("status") == "COMPLETED"
    skipped = {s["id"] for s in steps[:-1] if finished and res[s["id"]]["status"] == "MISSING_OR_UNCONFIRMED"}
    skipped |= {sid for sid, x in res.items() if x["status"] == "REMOVED"}     # proven not done (undone)
    for sid in skipped:
        if res[sid]["status"] == "REMOVED":
            continue
        res[sid]["status"] = "MISSING"
        res[sid]["evidence"] = (f"never done before the last step (done at {res[last]['end_sec']:.1f}s) - "
                                + res[sid]["evidence"])
    # completion vs confirmation time, and the LIVE state (what is happening now: a placement under way / waiting
    # for its evidence) - a display only, it never confirms anything
    def _detected(c_):
        """how strongly a candidate was seen, 0-100: its network score or the share of its learnt state change seen"""
        if not c_:
            return None
        sh = c_.get("share")
        return int(round(100 * max(float(c_.get("score") or 0.0), min(1.0, float(sh)) if sh is not None else 0.0)))

    for s in steps:
        sid, x, c = s["id"], res[s["id"]], out.get(s["id"])
        r_ = raw.get(sid) or {"cands": []}
        if x["status"] == "INITIAL_STATE":
            x["detected"] = 100
        elif c:
            x["detected"] = _detected(c)
        else:                                # missing / removed: the best it got
            x["detected"] = max([_detected(cc) for cc in r_["cands"]] or [0])
        x["completion_time"] = x["end_sec"] if x["status"] in ("COMPLETED", "RECOGNISED_UNVERIFIED") else None
        x["confirmation_time"] = (c.get("confirm") if c and c.get("confirm") is not None else
                                  (round(min(float(t[-1]), c["end"] + float((info.get(sid) or {}).get("hold_s", object_actions_HOLD_S))), 2)
                                   if c and x["status"] == "COMPLETED" else None))
        x["live_state"], x["provisional_t"] = None, None
        rl = rel_log.get(sid)
        if rl is not None and x["status"] != "COMPLETED" and len(t):
            stt = timeline(rl["F"], [(p, p["verdict"]) for p in rl["placements"]])
            now = stt[-1] if stt else None
            if any(p["verdict"] == "UNCERTAIN" for p in rl["placements"]):
                x["live_state"] = "CANDIDATE"
            elif now == "ENTERING" or (now == "NEAR" and np.nan_to_num(rl["F"]["mot"][-1]) >= 0.6):
                x["live_state"] = "IN_PROGRESS"
            # PROVISIONAL (live): the object is in its destination now, or its stay is still being decided - shown at
            # once at its entry time; a stay seen OUTSIDE again (passed over) is not
            gone = float(t[-1]) - 2 * GAP_S
            F_ = rl["F"]
            out_after = lambda p: bool((F_["vis"] & ~F_["ins"] & F_["dseen"] & (F_["t"] > p["left"])).any())
            # ... also while it is HIDDEN after its stay (a hand over it) and not seen outside since: still going on
            open_ = [p for p in rl["placements"] if p["persists"] is not False and p["verdict"] != "CONFIRMED"
                     and (p["left"] >= gone or p["verdict"] == "UNCERTAIN"
                          or (float(t[-1]) - p["left"] <= 5.0 and not out_after(p)))]
            if open_ and live:
                x["provisional_t"] = max(p["entered"] for p in open_)
        elif live and len(t) and x["status"] not in ("COMPLETED", "INITIAL_STATE") and raw.get(sid):
            # a STATE step (box opened / closed): its own change just seen, not yet held long enough to confirm
            r = raw[sid]
            recent = [e for e in r["ev_ts"] if e > float(t[-1]) - r["hold0"] - 1.0]
            if recent:
                x["provisional_t"] = round(max(recent), 2)
                x["live_state"] = x["live_state"] or "IN_PROGRESS"
    # the RELATION LOG: per placement step its object-destination states over time + every stay, decided, and why
    vd_rel = {}
    for sid, rl in rel_log.items():
        stt = timeline(rl["F"], [(p, p["verdict"]) for p in rl["placements"]])
        vd_rel[str(sid)] = {"step": sid, "target": rl["target"], "destination": rl["destination"],
                            "timeline": segments(rl["F"]["t"], stt),
                            "placements": [{k: v for k, v in p.items()} for p in rl["placements"]]}
    evs = []
    for s in steps:
        x = res[s["id"]]
        te = x["end_sec"] if x["status"] in ("COMPLETED", "INITIAL_STATE", "RECOGNISED_UNVERIFIED") else None
        evs.append({"unit": f"o{s['id']}", "step": s["id"], "family": s.get("part"), "pickup": te,
                    "p_pickup": 0.8 if te is not None else 0.0, "place": None, "p_place": 0.0,
                    "pickups": [te] if te is not None else [], "at_place_end": s["id"] in skipped,
                    "not_placed": False, "absent": False})
    vd = decode_events(json.loads(json.dumps(model["spec"])), evs)
    lab = {s["id"]: s.get("action_query") or s.get("text") for s in steps}
    for a in vd["anomalies"]:
        for sid in skipped:
            if a["text"].startswith(f"MISSING: step {sid} "):
                a["text"] = f"{res[sid]['status']}: step {sid} {lab[sid]} - {res[sid]['evidence']}"
    for s in vd["steps"]:
        if s["step"] in skipped:
            s["why"] = res[s["step"]]["evidence"]
    for a in vd["anomalies"]:
        for sid, x in res.items():
            if x["status"] in ("MISSING_OR_UNCONFIRMED", "NOT_TRAINED") and a["text"].startswith(f"UNCERTAIN: step {sid} "):
                a["text"] = f"{x['status']}: step {sid} {lab[sid]} - {x['evidence']}"
    for s in vd["steps"]:
        x = res.get(s["step"])
        if x and x["status"] != "COMPLETED" and s["status"] == "UNCERTAIN":
            s["status"], s["why"] = x["status"].replace("_", " "), x["evidence"]
        if x:
            s["detected"] = x.get("detected")
    vd["relations"] = vd_rel
    return res, vd


def _with_hands(L, video, signals, log=print):
    """the record + MediaPipe hands as 'hand' detections (for the placement check's release evidence): the hands
    found live (L['hands']), else from the video file (once, cached); the record unchanged without MediaPipe or when
    the detector has its own hand class"""
    if not USE_HANDS or hand_class(L):
        return L
    if L.get("hands") is not None:
        return add_to(L, L["hands"])
    if signals is None and video and Path(str(video)).is_file() and available():
        try:
            return add_to(L, for_video(video, L, log))
        except Exception as e:
            log(f"  (hands not used: {type(e).__name__}: {e})")
    return L


USE_RELATIONS = True          # placement steps decided by sopmon/relations.py (False: the older state check only)
USE_HANDS = True              # + MediaPipe hands for release evidence (False: without hands)
USE_ZONE = True               # 'inside' = in the destination's LEARNT inside, not its whole rectangle (False: rectangle)


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.temporal   (from sopmon\temporal.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""temporal.py - the TEMPORAL MODEL of a trained model: per-frame event predictions from learnt features.

  training videos + SOP spec
        |  Qwen parser (spec), YOLO detections, SIFT alignment (evidence_engine)
        v
  FEATURES per UNIT and frame      a unit = one part copy at its learnt place (place learner), or one tool
     place picture change, the part seen at / near its place, point motion there, the part on the product (count and
     its change), the part seen at all, the scene's motion, the product seen, a tool working at the product
        v
  TCN (dilated 1-D convolutions)   -> per frame: NONE / PICKUP / PLACE / TOOL
        v
  events (peaks)  ->  sequence decoder (each step = its unit's pickup; look-alikes taken together -> their placing;
                      the longest in-order run of steps) -> NORMAL / ANOMALY / UNCERTAIN

LABELS: nobody labels frames. A training video's labels come from the place learner (when each copy leaves its
place) and the evidence engine (when it goes on the product, when a tool works) - the TCN learns to see those events
from the features, and gets better with more training videos. With one training video it mostly reproduces them.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["NONE", "PICKUP", "PLACE", "TOOL"]
temporal_D = 13                       # features per unit and frame (11, 12: the detector's appearance vectors)
LABEL_S = 0.7                # an event's label covers +- this around its time
PEAK = 0.5                   # an event needs this probability
WEAK = 0.3                   # below: no event; between: an uncertain one
temporal_TIE_S = 1.0


# ── features ──
def _units(spec, fam, W0, H0):
    """one unit per learnt place (a copy of a step's part), one per tool class"""
    units = []
    for s in spec.get("steps") or []:
        for p in s.get("positions") or []:
            b = np.array(p["box"]) * [W0, H0, W0, H0]
            units.append({"kind": "place", "step": s["id"], "name": p["name"], "box": b.tolist(),
                          "family": fam["part_of"].get(str(s["id"])), "t_ref": p.get("t_ref")})
    for t in fam["tools"]:
        units.append({"kind": "tool", "step": None, "name": t, "box": None, "family": t})
    return units


REF_S = 3.0                  # the first seconds of a video: where each part lies there (relocate)


def product_box(named, product):
    """the PRODUCT's box among one frame's detections [(class, box)] - the largest box of a product class that is
    not another part seen as the product: a product box lying ON a part's own box (IoU > 0.5 - the battery pack also
    called 'phone') is that part. Falls back to the largest product box when every one is such a look-alike."""
    pb = [b for n, b in named if n in product]
    if not pb:
        return None
    parts = [b for n, b in named if n not in product]
    real = [b for b in pb if not any(iou(b, q) > 0.5 for q in parts)]
    return max(real or pb, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


def relocate(units, start, fam, canon, W0, H0, log=print, min_seen=0.3, max_cost=2.5):
    """THIS video's layout: the parts may lie elsewhere than in the training video (another table, another order
    of the kit). start: [(t, dets)] of the first REF_S seconds. Each part's copies there (the detector's boxes of its
    class, seen in >= min_seen of those frames, off the product) are matched to the trained copies by position, size
    and shape; a copy found elsewhere MOVES its place there. A trained copy with nothing left for it is NOT ON THE
    TABLE at the start (already fitted, or missing from the kit) - nothing can be taken from its place.
    Changes units in place; -> a line per moved / absent copy"""
    from scipy.optimize import linear_sum_assignment
    n = max(1, len(start))
    diag = float(np.hypot(W0, H0))
    notes = []
    prod = []
    for _, ds in start:
        pb = product_box([(canon(d["name"]), d["box"]) for d in ds], fam["product"])
        if pb is not None:
            prod.append(pb)
    P = np.median(prod, 0) if prod else None
    # fitted already = WELL inside the product's box (its centre inside the box shrunk by 15% on each side); a part
    # lying against the product's edge (the camera module beside the phone) is still on the table
    def on_p(b):
        if P is None:
            return False
        mx, my = 0.15 * (P[2] - P[0]), 0.15 * (P[3] - P[1])
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        return P[0] + mx <= cx <= P[2] - mx and P[1] + my <= cy <= P[3] - my
    for f in sorted({u["family"] for u in units if u["kind"] == "place" and u["family"]}):
        us = [u for u in units if u["kind"] == "place" and u["family"] == f]
        groups = []
        for _, ds in start:
            for b in {tuple(np.round(d["box"], 1)) for d in ds if canon(d["name"]) == f}:
                g = max(groups, key=lambda g: iou(np.median(g, 0), b), default=None)
                if g is not None and iou(np.median(g, 0), b) > 0.3:
                    g.append(b)
                else:
                    groups.append([b])
        boxes = [np.median(g, 0) for g in groups if len(g) >= min_seen * n]
        sz = lambda b: max(b[2] - b[0], b[3] - b[1], 1.0)
        ar = lambda b: np.log((b[2] - b[0] + 1.0) / (b[3] - b[1] + 1.0))
        ctr = lambda b: np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
        # a copy ON the product at the start is fitted already - unless it lies right at a trained place (a place
        # that touches the product, as in the training video)
        at_trained = lambda b: any(np.linalg.norm(ctr(b) - ctr(u["box"])) < 1.0 * sz(u["box"]) for u in us)
        boxes = [b for b in boxes if not on_p(b) or at_trained(b)]
        # seen at all (even in a frame or two - a part that comes into view late in the first seconds)
        seen_any = [g for g in groups if not on_p(np.median(g, 0)) or at_trained(np.median(g, 0))]
        got = {}
        if boxes:
            C = np.array([[np.linalg.norm(ctr(u["box"]) - ctr(b)) / diag * 4 + abs(np.log(sz(b) / sz(u["box"])))
                           + abs(ar(b) - ar(u["box"])) for b in boxes] for u in us])
            for r, c in zip(*linear_sum_assignment(C)):
                if C[r, c] <= max_cost:
                    got[r] = boxes[c]
        for r, u in enumerate(us):
            if r not in got:
                if not seen_any:
                    # its part not seen anywhere off the product in the first seconds: nothing to take
                    u["absent"] = True
                    notes.append(f"  layout: {u['name']} (step {u['step']}) not on the table at the start")
                else:                                       # seen only now and then: its trained place is kept
                    notes.append(f"  layout: {u['name']} (step {u['step']}) seen only briefly at the start - its "
                                 "trained place is kept")
                continue
            b = [float(v) for v in got[r]]
            if np.linalg.norm(ctr(b) - ctr(u["box"])) > 0.5 * sz(u["box"]):
                notes.append(f"  layout: {u['name']} (step {u['step']}) lies elsewhere in this video - its place "
                             f"moved {[int(v) for v in u['box']]} -> {[int(v) for v in b]}")
                u["moved_from"] = u["box"]
            u["box"] = b
    for ln in notes:
        log(ln)
    return notes


class FeatureStream:
    """CAUSAL features: one frame in -> every unit's features for it (U, D), from this frame and the ones before only.
    Training (features) and live inference (LiveInference) both use it, so the TCN sees the same thing in both.
    Per frame: SIFT alignment to the first frame (detected boxes masked; a new camera position once two frames agree),
    residual point motion (LK grid minus the camera's motion), the detections, the product's box."""

    def __init__(self, units, fam, W0, H0, canon, fps, ref_s=3.0, align_every=3):
        self.units, self.fam, self.W0, self.H0, self.canon = units, fam, W0, H0, canon
        self.dt = 1.0 / float(fps)
        self.ref_s, self.s = ref_s, ALIGN_W / W0
        self.sift, self.bf = cv2.SIFT_create(800), cv2.BFMatcher()
        hs = int(round(H0 * self.s))
        ys, xs = np.mgrid[GRID // 2:hs:GRID, GRID // 2:ALIGN_W:GRID]
        self.pts, self.gshape = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32), ys.shape
        self.k, self.t0 = 0, None
        self.f0, self.prev, self.A, self.pend = None, None, np.array([[1, 0, 0], [0, 1, 0]], np.float64), None
        self.P, self.P_t = None, -1e9
        fams = sorted({u["family"] for u in units if u["family"] and u["kind"] == "place"})
        self.ncopy = {f: max(1, sum(1 for u in units if u["family"] == f and u["kind"] == "place")) for f in fams}
        self.on = {f: [] for f in fams}
        self.seen = {f: [] for f in fams}
        self.onf = {f: [] for f in fams}
        self.ref = [[] for _ in units]                      # each place's picture over the first ref_s
        self.refm = [None for _ in units]
        self.aref = [[] for _ in units]                     # each place's detector appearance at the start
        self.tool = {u["name"]: {"ws": None, "t": None} for u in units if u["kind"] == "tool"}
        self.w = max(1, int(round(1.5 / self.dt)))
        self.align_every = max(1, int(align_every))       # SIFT (CPU) every n-th frame: the camera rarely moves

    def _align(self, small, dets):
        mask = np.full(small.shape, 255, np.uint8)
        for d in dets:
            x0, y0, x1, y1 = (int(v * self.s) for v in d["box"])
            p = int(0.1 * max(x1 - x0, y1 - y0))
            mask[max(0, y0 - p):y1 + p, max(0, x0 - p):x1 + p] = 0
        kp, des = self.sift.detectAndCompute(small, mask)
        if self.f0 is None:
            self.f0 = (kp, des)
            return self.A
        k0, d0 = self.f0
        if des is None or d0 is None or len(kp) < 8 or len(k0) < 8:
            return self.A
        ms = [m for m, n in (x for x in self.bf.knnMatch(des, d0, k=2) if len(x) == 2) if m.distance < 0.75 * n.distance]
        if len(ms) < 8:
            return self.A
        T, inl = cv2.estimateAffinePartial2D(np.float32([kp[m.queryIdx].pt for m in ms]),
                                             np.float32([k0[m.trainIdx].pt for m in ms]),
                                             method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if T is None or inl is None or int(inl.sum()) < 40:
            return self.A
        near = lambda a, b: np.abs(a[:, 2] - b[:, 2]).max() < 3.0 and np.abs(a[:, :2] - b[:, :2]).max() < 0.01
        if near(T, self.A):
            self.A, self.pend = T, None
        elif self.pend is not None and near(T, self.pend):
            self.A, self.pend = T, None                     # two frames agree: the camera did move
        else:
            self.pend = T
        return self.A

    def push(self, rgb, dets, ts, app=None):
        """app: the detector's appearance vector of each detection (featstore), or None"""
        U = len(self.units)
        x = np.zeros((U, temporal_D), np.float32)
        if self.t0 is None:
            self.t0 = ts
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(g, (int(round(self.W0 * self.s)), int(round(self.H0 * self.s))), interpolation=cv2.INTER_AREA)
        A_prev = self.A.copy()
        A = self._align(small, dets) if self.k % self.align_every == 0 or self.f0 is None else self.A
        M = np.zeros(self.gshape, np.float32)
        if self.prev is not None:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, small, self.pts.reshape(-1, 1, 2), None,
                                                 winSize=(15, 15), maxLevel=3)
            R = np.linalg.inv(np.vstack([A, [0, 0, 1]])) @ np.vstack([A_prev, [0, 0, 1]])
            pred = self.pts @ R[:2, :2].T + R[:2, 2] - self.pts
            r = np.linalg.norm(p1.reshape(-1, 2) - self.pts - pred, axis=1)
            r[st.ravel() != 1] = 0
            M = r.reshape(self.gshape)
        self.prev = small
        gh, gw = self.gshape
        W0, H0 = self.W0, self.H0

        def mot(b):
            a, bb = int(b[0] / W0 * gw), max(int(b[0] / W0 * gw) + 1, int(np.ceil(b[2] / W0 * gw)))
            c, d = int(b[1] / H0 * gh), max(int(b[1] / H0 * gh) + 1, int(np.ceil(b[3] / H0 * gh)))
            v = M[c:d, a:bb]
            return float(v.mean()) if v.size else 0.0

        ds = list({(self.canon(d["name"]), tuple(np.round(d["box"], 1))) for d in dets})
        prod_b = product_box(ds, self.fam["product"])
        if prod_b is not None:
            self.P, self.P_t = prod_b, ts
        P = self.P if ts - self.P_t <= 1.0 else None
        inside = lambda b, box: box[0] <= (b[0] + b[2]) / 2 <= box[2] and box[1] <= (b[1] + b[3]) / 2 <= box[3]
        for f in self.on:
            self.on[f].append(sum(1 for n, b in ds if n == f and P is not None and inside(b, P)))
            self.seen[f].append(sum(1 for n, _ in ds if n == f))
            self.onf[f].append(float(np.median(self.on[f][-(2 * self.w + 1):])) / self.ncopy[f])
        # tools: followed from where they were last seen
        tool_act, tft = 0.0, {}
        for name, stt in self.tool.items():
            cs = [b for n, b in ds if n == name]
            ws = lambda b: (A @ np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 1.0])) / W0
            if cs:
                b = min(cs, key=lambda b: np.linalg.norm(ws(b) - stt["ws"])) if stt["ws"] is not None else cs[0]
                p = ws(b)
                spd = (np.linalg.norm(p - stt["ws"]) / max(ts - stt["t"], 1e-3)) if stt["ws"] is not None else 0.0
                stt["ws"], stt["t"] = p, ts
                if P is None:
                    dp = 1.0
                else:
                    pw, ph = max(P[2] - P[0], 1.0), max(P[3] - P[1], 1.0)
                    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    dp = 0.0 if inside(b, P) else min(np.hypot(max(P[0] - cx, 0, cx - P[2]),
                                                               max(P[1] - cy, 0, cy - P[3])) / np.hypot(pw, ph), 1.0)
                near = 1.0 if dp < 0.3 else 0.0
                v = (1.0, near, min(spd / 0.1, 2.0), min(mot(b) / 3.0, 2.0), dp)
            else:
                v = (0.0, 0.0, 0.0, 0.0, 1.0)
            tft[name] = v
            tool_act = max(tool_act, v[1] * min(v[2] + v[3], 1.0))
        g_mot = min(float(M.mean()) / 2.0, 2.0)
        prod = 1.0 if P is not None else 0.0
        early = ts - self.t0 < self.ref_s
        gb = None
        for i, u in enumerate(self.units):
            if u["kind"] == "place":
                if gb is None:
                    gb = cv2.GaussianBlur(g, (5, 5), 0)
                b = np.array(u["box"])
                x0, y0, x1, y1 = (int(round(v)) for v in b)
                crop = gb[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)].astype(np.float32)
                if early or self.refm[i] is None:
                    self.ref[i].append(crop)
                    self.refm[i] = np.median(self.ref[i], 0)
                diff = float(np.abs(crop - self.refm[i]).mean()) if crop.shape == self.refm[i].shape else 0.0
                ctr, size = (b[:2] + b[2:]) / 2, max(b[2] - b[0], b[3] - b[1], 1.0)
                grown = [b[0] - 0.3 * size, b[1] - 0.3 * size, b[2] + 0.3 * size, b[3] + 0.3 * size]
                f = u["family"]
                at = 1.0 if any(n == f and inside(bb, grown) for n, bb in ds) else 0.0
                dn = min([np.linalg.norm((np.array(bb[:2]) + bb[2:]) / 2 - ctr) / size for n, bb in ds if n == f] or [5.0])
                onf = self.onf[f] if f in self.onf else None
                x[i, 0] = min(diff / 50.0, 2.0)
                x[i, 1] = at
                x[i, 2] = min(dn, 5.0) / 5.0
                x[i, 3] = min(mot(b) / 3.0, 2.0)
                if onf:
                    x[i, 4] = onf[-1]
                    back = onf[-1 - self.w] if len(onf) > self.w else onf[0]
                    x[i, 5] = float(np.clip((onf[-1] - back) / (self.w * self.dt) * 5.0, -2, 2))
                    x[i, 6] = float(np.median(self.seen[f][-(2 * self.w + 1):])) / self.ncopy[f]
                x[i, 9] = tool_act
                # the DETECTOR'S APPEARANCE: the copy as it lay at its place at the start; is the thing at the place
                # still it (11), is it now on the product (12)
                if app is not None:
                    va = [(d["box"], v) for d, v in zip(dets, app) if v is not None and self.canon(d["name"]) == f]
                    here = [v for bb, v in va if inside(bb, grown)]
                    if (early or not self.aref[i]) and here:
                        self.aref[i].append(np.mean(np.array(here, np.float32), 0))
                    if self.aref[i]:
                        ref = np.mean(self.aref[i], 0)
                        ref = ref / (np.linalg.norm(ref) + 1e-6)
                        cos = lambda v: float(np.dot(ref, np.asarray(v, np.float32)))
                        x[i, 11] = max([cos(v) for v in here] or [0.0])
                        x[i, 12] = max([cos(v) for bb, v in va if P is not None and inside(bb, P)] or [0.0])
            else:
                vis, near, spd, tm, dp = tft[u["name"]]
                x[i, 0], x[i, 1], x[i, 2], x[i, 3], x[i, 6] = vis, near, spd, tm, dp
                x[i, 9], x[i, 10] = tool_act, 1.0
            x[i, 7], x[i, 8] = g_mot, prod
        self.k += 1
        return x


# ── the TCN ──
def temporal__net():
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, c, d):
            super().__init__()
            self.c1 = nn.Conv1d(c, c, 3, padding=d, dilation=d)
            self.c2 = nn.Conv1d(c, c, 3, padding=d, dilation=d)
            self.act, self.drop = nn.ReLU(), nn.Dropout(0.2)

        def forward(self, x):
            return x + self.c2(self.drop(self.act(self.c1(self.act(x)))))

    class TCN(nn.Module):
        """dilated 1-D convolutions over time; receptive field ~63 frames (21 s at 3 fps), both directions"""

        def __init__(self, d_in=temporal_D, h=64, n_cls=len(CLASSES)):
            super().__init__()
            self.inp = nn.Conv1d(d_in, h, 1)
            self.blocks = nn.Sequential(*[Block(h, d) for d in (1, 2, 4, 8, 16)])
            self.out = nn.Conv1d(h, n_cls, 1)

        def forward(self, x):                       # (B, D, T) -> (B, C, T)
            return self.out(self.blocks(self.inp(x)))

    return TCN()


KINDS = {"tcn": "TCN (dilated convolutions)", "lstr": "LSTR (long short-term transformer, causal)"}
KIND_KEY = "_kind_lstr"       # a marker in an LSTR model's state dict (a TCN's has none)


def make_net(kind="tcn", d_in=temporal_D, n_out=None):
    """the temporal model of a kind: 'tcn' (the original) or 'lstr' (sopmon/lstr.py)"""
    n_out = n_out or len(CLASSES)
    if kind == "lstr":
        raise ImportError("sopmon.lstr is not part of this inference-only file (a model trained with it needs the full engine)")
        return LSTR(d_in, n_out)
    return temporal__net() if (d_in, n_out) == (temporal_D, len(CLASSES)) else None


def kind_of(state):
    return "lstr" if KIND_KEY in state else "tcn"


class Predictor:
    """the trained temporal model (TCN or LSTR - the state says which) loaded once (live inference calls it every
    frame)"""

    def __init__(self, state):
        import torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.kind = kind_of(state)
        self.net = make_net(self.kind)
        self.net.load_state_dict({k: v for k, v in state.items() if k != KIND_KEY})
        self.net.eval().to(self.dev)

    def __call__(self, X):
        """X (U, T, D) -> (U, T, C) probabilities"""
        import torch
        with torch.no_grad():
            x = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32, device=self.dev)
            p = torch.softmax(self.net(x), 1)
        return p.cpu().numpy().transpose(0, 2, 1)


def decode_events(spec, events, labels=None):
    """the SOP VERIFIER over the detected events (the network only detects; this decides):
      1. ASSIGNMENT  each step <- the event(s) of its copy (its learnt place: the spatial evidence), reported
      2. EXISTENCE   OK; MISSING when absence is proven (never taken and still lying at its place at the end, or
                     taken but left lying - never placed); UNCERTAIN when not seen taken but not at its place either
                     (hidden by a hand, taken unseen); REPEATED when a copy was taken more than once
      3. ORDER       every broken precedence 'step j happened before step i' (i before j in the SOP); the steps to
                     blame = the fewest that explain them (not in the longest in-order run)
    events: per unit {"step", "family", "pickup", "p_pickup", "place", "p_place", "not_placed", "at_place_end",
    "pickups" (all its pickup times)} -> {"status", "steps", "anomalies", "precedence", "events"}"""
    steps = []
    for s in spec.get("steps") or []:
        evs = [e for e in events if e.get("step") == s["id"]]
        lab = (labels or {}).get(s["id"]) or pipeline_step_label(s)
        if not evs:
            steps.append({"step": s["id"], "label": lab, "status": "NOT JUDGED", "t": None, "p": 0.0,
                          "why": "no learnt place for its part", "assigned": []})
            continue
        miss = [e for e in evs if e["pickup"] is None]
        loose = [e for e in evs if e["pickup"] is not None and e.get("not_placed")]
        ts = [e["pickup"] for e in evs if e["pickup"] is not None]
        p = min(e["p_pickup"] for e in evs)
        rep = [e for e in evs if len(e.get("pickups") or []) > 1]
        if miss and all(e.get("absent") for e in miss):
            # not on the table when the video starts: nothing could be taken - fitted already, or not in the kit
            st, why = "UNCERTAIN", "not on the table at the start (fitted already, or not in the kit)"
        elif miss:
            # absence PROVEN only if the copy is still lying at its place at the end; else it may have been taken
            # unseen (a hand over it, the camera turned) - not proven missing
            proven = all(e.get("at_place_end") for e in miss) if any("at_place_end" in e for e in miss) else True
            st = "MISSING" if proven else "UNCERTAIN"
            why = ("never taken - still lying at its place at the end" if proven and any("at_place_end" in e for e in miss)
                   else "never seen taken" if proven else
                   "not seen taken, but not at its place at the end either - hidden or taken unseen")
        elif loose:
            st, why = "MISSING", "taken but never placed / assembled - left lying at the end"
        else:
            st, why = "OK", ""
        steps.append({"step": s["id"], "label": lab, "status": st, "why": why, "p": round(p, 2),
                      "t": max(ts) if ts and st == "OK" else None, "family": evs[0]["family"],
                      "t_taken": max(ts) if loose and ts else None,
                      "t_place": max((e["place"] for e in evs if e["place"] is not None), default=None),
                      "repeated": [round(x, 1) for e in rep for x in e["pickups"]] if rep else None,
                      # the hands' install evidence (shown, not deciding): when the last copy was fitted
                      "hands": any("installed" in e for e in evs),
                      "t_installed": max((e["installed"] for e in evs if e.get("installed") is not None), default=None)
                      if all(e.get("installed") is not None for e in evs if e["pickup"] is not None) else None,
                      "assigned": [{"copy": e.get("unit"), "pickup": e["pickup"], "p": round(e["p_pickup"], 2)}
                                   for e in evs]})
    by = {}
    for st in steps:
        if st["status"] == "OK" and st.get("family"):
            by.setdefault(st["family"], []).append(st)
    for grp in by.values():                       # look-alikes taken together: their placing tells the order
        grp.sort(key=lambda s: s["t"])
        k = 0
        while k < len(grp):
            j = k
            while j + 1 < len(grp) and grp[j + 1]["t"] - grp[j]["t"] < temporal_TIE_S:
                j += 1
            tie = grp[k:j + 1]
            if len(tie) > 1 and all(s["t_place"] is not None for s in tie):
                for s, tq in zip(sorted(tie, key=lambda s: s["step"]), sorted(s["t_place"] for s in tie)):
                    s["t"], s["tie"] = tq, True
            k = j + 1
    # ORDER - the SOP's PRECEDENCE GRAPH decides: step a must come before step b when the SOP lists a first, except
    # look-alike copies of one part whose order is free ('mount the 4 wheels in any order': copy_order any). Every
    # broken edge is reported ('step 3 happened before step 2'); the steps to blame are the fewest whose removal
    # leaves no broken edge. LIS (the longest in-order run) is kept only as a second opinion in the result.
    by_id = {s["id"]: s for s in spec.get("steps") or []}

    def required(a, b):                               # must a (listed first) come before b?
        sa, sb = by_id.get(a, {}), by_id.get(b, {})
        fam = sa.get("copy_of")
        return not (fam and fam == sb.get("copy_of") and "any" in (sa.get("copy_order"), sb.get("copy_order")))

    done = [k for k, s in enumerate(steps) if s["status"] == "OK"]
    edges = [(steps[a]["step"], steps[b]["step"]) for ia, a in enumerate(done) for b in done[ia + 1:]
             if required(steps[a]["step"], steps[b]["step"])]
    t_of = {steps[k]["step"]: steps[k]["t"] for k in done}
    broken_e = [(a, b) for a, b in edges if t_of[b] < t_of[a] - temporal_TIE_S]           # b happened before a
    prec = [(b, a, t_of[b], t_of[a]) for a, b in broken_e]
    blame, left = [], list(broken_e)
    p_of = {steps[k]["step"]: steps[k]["p"] for k in done}
    while left:                                        # the step in the most broken edges goes first
        cnt = {}
        for a, b in left:
            cnt[a] = cnt.get(a, 0) + 1
            cnt[b] = cnt.get(b, 0) + 1
        x = max(cnt, key=lambda v: (cnt[v], -p_of.get(v, 1.0), v))
        blame.append(x)
        left = [(a, b) for a, b in left if x not in (a, b)]
    lab_of = {s["step"]: s["label"] for s in steps}
    for k in done:
        s = steps[k]
        if s["step"] in blame:
            s["status"] = "OUT OF ORDER"
            s["broken"] = [(a, t_of[a]) for a, b in broken_e if b == s["step"]] + \
                          [(b, t_of[b]) for a, b in broken_e if a == s["step"]]
    # second opinion: the old longest-in-order-run blame (shown, not deciding)
    ts = [steps[k]["t"] for k in done]
    best, prev = [1] * len(done), [-1] * len(done)
    for a in range(len(done)):
        for b in range(a):
            if ts[b] <= ts[a] + temporal_TIE_S and best[b] + 1 > best[a]:
                best[a], prev[a] = best[b] + 1, b
    keep = set()
    if done:
        a = int(np.argmax(best))
        while a >= 0:
            keep.add(done[a])
            a = prev[a]
    lis_blame = [steps[k]["step"] for k in done if k not in keep]
    anomalies = []
    for s in steps:
        if s["status"] in ("MISSING", "OUT OF ORDER", "UNCERTAIN"):
            not_placed = s["status"] == "MISSING" and s.get("t_taken") is not None
            if s["status"] == "UNCERTAIN":
                weak = True
            elif s["status"] == "OUT OF ORDER":
                weak = s["p"] < PEAK
            else:
                weak = False
            if s["status"] == "OUT OF ORDER":
                rel = "; ".join((f"happened before step {i} ({lab_of[i]}, {ti:.1f}s)" if s["t"] < ti
                                 else f"happened after step {i} ({lab_of[i]}, {ti:.1f}s)") for i, ti in s["broken"])
                text = f"OUT OF ORDER: step {s['step']} {s['label']} at {s['t']:.1f}s " + (rel or "")
            elif not_placed:
                text = f"MISSING: step {s['step']} {s['label']} - taken at {s['t_taken']:.1f}s but never placed / assembled"
            else:
                text = f"{s['status']}: step {s['step']} {s['label']} - {s['why']}"
            anomalies.append({"at": s["t"] if s["t"] is not None else s.get("t_taken"), "step": "temporal model",
                              "uncertain": bool(weak), "text": text + (" (weak)" if weak and s["status"] != "UNCERTAIN"
                                                                       else "")})
        if s.get("repeated"):
            anomalies.append({"at": s["repeated"][-1], "step": "temporal model", "uncertain": False,
                              "text": f"REPEATED: step {s['step']} {s['label']} - its part was taken "
                                      f"{len(s['repeated'])} times ({', '.join(f'{x:.1f}s' for x in s['repeated'])})"})
    status = "ANOMALY" if any(not a["uncertain"] for a in anomalies) else "UNCERTAIN" if anomalies else "NORMAL"
    return {"status": status, "steps": steps, "anomalies": anomalies, "events": events,
            "precedence": [{"step": j, "before": i, "t": tj, "t_before": ti} for j, i, tj, ti in prec],
            "lis_out_of_order": lis_blame}


def _check_features(state):
    """a model trained on another feature set (an older version) cannot read these features"""
    w = state.get("inp.weight")
    if w is not None and int(w.shape[1]) != temporal_D:
        raise RuntimeError(f"this model's temporal model was trained on {int(w.shape[1])} features per frame, this "
                           f"version makes {temporal_D} - train the model again (Train model)")


# ── LIVE inference: the test video frame by frame, the sequence built as it plays ──
LAG_S = 3.0                  # an event is read this long after it (the TCN looks a little ahead)
CONFIRM_S = 3.0              # a pickup counts once its copy is NOT at its place for this long after it
ALL_FRAMES = True            # live: EVERY frame read + detected + shown; the step model gets its trained rate


def _candidates(pr, sep=6):
    """the TCN's pickup peaks (local maxima >= PEAK, `sep` frames apart), in time order"""
    out = []
    for j in range(len(pr)):
        if pr[j] >= PEAK and pr[j] >= pr[max(0, j - sep // 2):j + sep // 2 + 1].max() - 1e-6:
            if not out or j - out[-1] >= sep:
                out.append(j)
    return out


def confirm(t_c, t_all, at, now):
    """is the candidate pickup at t_c real? `at` (per frame: the detector sees the part at the copy's place).
    -> ("yes" | "no" | "wait", share of the frames after it with the place empty)"""
    a, b = t_c + 1.0, t_c + 1.0 + CONFIRM_S
    if now < b:
        return "wait", None
    w = (t_all >= a) & (t_all <= b)
    if not w.any():
        return "yes", 1.0
    m = float(np.mean(at[w]))
    return ("yes" if m <= 0.3 else "no"), 1.0 - m
LOOSE_S = 8.0                # a taken copy still lying off the product this long at the end = never placed
WIN = 200                    # frames the TCN sees each step (its receptive field is ~63)


def _ensure_bank(pl, model, sp):
    """the model's detectors in pl.bank (+ their appearance hooks pl._roi) - built only when pl has none for this
    model and these detectors yet, so a kept Pipeline loads its weights once"""
    key = (model.get("name"), model.get("trained_at"),
           tuple(sorted(m_.get("name", "") for m_ in pl.models if m_.get("enabled"))))
    if getattr(pl, "_bank_key", None) == key and pl.bank is not None and getattr(pl.bank, "backends", None):
        return False
    if getattr(pl, "_roi", None) is not None:
        pl._roi.close()
    pl.build_bank(sp)
    pl._roi = RoiFeatures(pl.bank)                         # hooks on the detectors: once per bank
    pl._bank_key = key
    return True


class FrameFeed:
    """a frame source filled by the HOST APPLICATION (SOPMonitor.process_frame): LiveInference reads it like a
    camera - put(frame, t) then one LiveInference.step() per frame. t = seconds since the first frame."""

    def __init__(self, fps: float = 30.0):
        self.fps, self.t, self._frame, self.closed = float(fps), None, None, False

    def put(self, frame_bgr, t: float) -> None:
        self._frame, self.t = frame_bgr, float(t)

    # the parts of cv2.VideoCapture that LiveInference uses
    def isOpened(self) -> bool:
        return not self.closed

    def read(self):
        f, self._frame = self._frame, None
        return (f is not None), f

    def grab(self) -> bool:
        return self._frame is not None

    def get(self, prop) -> float:
        return self.fps if prop == cv2.CAP_PROP_FPS else 0.0

    def release(self) -> None:
        self.closed = True


class LiveInference:
    """one test video against a trained model, frame by frame, as it plays: detect -> FeatureStream -> TCN on the
    frames so far -> each unit's PICKUP / PLACE read LAG_S behind -> the steps done so far, compared with the trained
    sequence. step() returns the state after each frame; finish() the final decode over the whole video (the same
    decoder as the report's [5t])."""

    def __init__(self, pl, model, video, fps=3.0, log=print, all_frames=None, step_tcn=None):
        """step_tcn: the step TCN already loaded (load_step_tcn) - else read from temporal.pt here"""
        all_frames = ALL_FRAMES if all_frames is None else all_frames
        self.pl, self.model, self.log = pl, model, log
        self.rec = []                                       # per frame (t, product box, [(class, box)]): record()
        # per frame the work on the product (motion) and a small picture of it - the action model's live signals
        self.sig_m, self._crops, self._sg_prev, self._lastP = [], [], None, None
        # MediaPipe hands per live frame (sopmon/mp_hands.py) - the placement check's release evidence; [] if absent
        self.hand_boxes = [] if available() else None
        sp = json.loads(json.dumps(model["spec"]))
        pl.places_file = model["places_file"]
        pl.apply_positions(sp, model["prompt"])
        self.spec = sp
        # the DETECTORS load once per model (kept loaded on the GPU across runs): a Pipeline whose bank was built for
        # this model (and these detectors) is used as it is
        _ensure_bank(pl, model, sp)
        self.roi = pl._roi
        # a FILE is read frame by frame (every step_n-th frame); a CAMERA / stream gives its newest frame every 1/fps
        # seconds and the rest are dropped - as a live check must
        if isinstance(video, FrameFeed):                   # frames handed in by the host application, one at a time
            self.camera, self.cap = True, video
        else:
            v = str(video).strip()
            self.camera = v.isdigit() or "://" in v
            self.cap = cv2.VideoCapture(int(v) if v.isdigit() else v)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open the {'camera / stream' if self.camera else 'video'}: {video}")
        self.fps = float(fps)
        src = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.src = src if 1.0 <= src <= 240.0 else 30.0
        self.step_n = 1 if self.camera else max(1, round(self.src / self.fps))
        # EVERY frame is read, detected and shown; every step_n-th (a camera: the newest every 1/fps s) also goes to
        # the step model, at the rate it was trained at
        self.all_frames = bool(all_frames)
        self.n_model = 0                                    # frames given to the step model so far
        self._model, self._state_cache = True, None
        self.read_fps = self.src if self.all_frames else (self.fps if self.camera else self.src / self.step_n)
        self.n_total = 0 if self.camera else int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration = self.n_total / self.src if self.n_total else None
        self.model_fps = float(model.get("fps") or 3.0)
        self.proc = []                                      # seconds per processed frame (the last 30)
        self.t_start = time.time()
        ok, fr = self.cap.read()
        if not ok:
            raise RuntimeError("no frames from the source")
        self.first = self._resize(fr)
        self.H0, self.W0 = self.first.shape[:2]
        pl.frames = [{"idx": 0, "ts": 0.0, "path": ""}]
        pl.frgb = lambda i: self.first                      # families() / _units() read the frame size only
        eng = SpatiotemporalEvidenceEngine(log=lambda s: None, appearance=False)
        self.fam = eng.families(pl, sp)
        self.units = _units(sp, self.fam, self.W0, self.H0)
        run_fps = self.fps if self.camera else self.src / self.step_n
        self.canon = lambda n: sorted(pl._same_class({n}))[0]
        self.stream, self.buf, self.layout = None, [], []     # started after REF_S seconds (_start)
        self.pred = step_tcn if step_tcn is not None else load_step_tcn(model)
        self.lag = max(1, int(round(LAG_S * run_fps)))
        self.run_fps = run_fps
        if abs(run_fps - self.model_fps) > 0.25 * self.model_fps:
            log(f"  WARNING: inference at {run_fps:.1f} fps, the model was trained at {self.model_fps:g} fps - the TCN "
                "counts frames, so its events are judged on another time scale. Train the model at this fps.")
        self.X, self.t, self.i = [], [], 0
        self.pending = fr                                   # the first frame is processed by the first step()
        self.picked = {}                                    # unit index -> (t, p)
        self.placed = {}
        self.pmax = {}                                      # the best pickup probability of a unit not (yet) seen
        self.hist = []                                      # (t, [(family, box)] off the product) - the last seconds
        self.repicks = {}                                   # unit -> every time it was taken (when more than once)
        self.rejected = {}                                  # unit -> its last candidate pickup that was not one
        self.found = {}                                     # unit -> (event t, video t when found, wall s)
        self.tim = {"read": [], "detect": [], "features": [], "tcn": []}   # ms per frame, the last 30
        self.last_ms = {}                                   # the newest frame's ms per stage
        self.walls = []                                     # wall times of the last frames (the real fps)
        self.q = None                                       # the reader thread's frames (files)
        self.labels = {s["id"]: pipeline_step_label(s) for s in sp["steps"]}
        same = [v for v in self.labels.values() if list(self.labels.values()).count(v) > 1]
        for s in sp["steps"]:                              # look-alike steps named by their copy ('square panel')
            ps = [p["name"] for p in s.get("positions") or []]
            if self.labels[s["id"]] in same and ps:
                self.labels[s["id"]] += f" ({', '.join(ps)})"
        self.trained = {r["step"]: r for r in model.get("sequence") or []}

    def _start(self):
        """the first seconds seen: this video's layout (relocate), then the feature stream over them"""
        self.layout = relocate(self.units, [(b[3], b[1]) for b in self.buf], self.fam, self.canon, self.W0, self.H0,
                               self.log)
        self.stream = FeatureStream(self.units, self.fam, self.W0, self.H0, self.canon, self.run_fps)
        for rgb, dets, app, ts in self.buf:
            self.X.append(self.stream.push(rgb, dets, ts, app))
            self.t.append(ts)
        self.buf = []

    def _now(self):
        """a live frame's time: the host's timestamp (FrameFeed), else the wall clock since the start"""
        if isinstance(self.cap, FrameFeed) and self.cap.t is not None:
            return float(self.cap.t)
        return time.time() - self.t_start

    def _resize(self, fr):
        h, w = fr.shape[:2]
        s = self.pl.max_dim / max(h, w)
        if s < 1:
            fr = cv2.resize(fr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(fr[:, :, ::-1])

    def _reader(self):
        """FILES: a second thread decodes, resizes AND DETECTS the coming frames (the GPU stage: detector + its
        appearance vectors) while the main thread builds features and runs the TCN on the current ones (the CPU
        stage) - the two overlap, so the GPU is not idle while the CPU works. A queue of 16 frames between them."""
        import queue
        import threading
        self.q = queue.Queue(maxsize=16)

        def run():
            i = 1                                          # the video's frame number
            while not self._stop_read:
                if not self.all_frames:                    # (old way) only the model's frames are read at all
                    for _ in range(self.step_n - 1):
                        if not self.cap.grab():
                            self.q.put(None)
                            return
                ok, fr = self.cap.read()
                if not ok:
                    self.q.put(None)
                    return
                t0 = time.time()
                rgb = self._resize(fr)
                if self.all_frames:
                    ts, model = i / self.src, i % self.step_n == 0
                else:
                    ts, model = i * self.step_n / self.src, True
                dets = self.pl.bank.detect_rgb(rgb)        # EVERY frame is detected
                app = None
                if model:                                  # the step model's frames: + appearance
                    app = self.roi.vectors(dets, rgb.shape) if self.roi is not None and self.roi.ok else None
                self.q.put((rgb, ts, dets, app, 1000.0 * (time.time() - t0), model))
                i += 1
        self._stop_read = False
        threading.Thread(target=run, daemon=True).start()

    def _read(self):
        """-> (frame, its time) or (None, None); a file's frames come from the reader thread already detected
        (self._pre = their detections, appearance vectors and detection ms)"""
        self._pre = None
        self._model = True
        if self.pending is not None:
            fr, self.pending = self.pending, None
            return fr, 0.0
        if not self.camera:
            if self.q is None:
                self._reader()
            item = self.q.get()
            if item is None:
                return None, None
            rgb, ts, dets, app, ms, self._model = item
            self._pre = (dets, app, ms)
            return rgb, ts
        if self.camera:
            due = self.n_model / self.fps
            if self.all_frames:                            # EVERY camera frame (detected); a model frame when due
                ok, fr = self.cap.read()
                if not ok:
                    return None, None
                now = self._now()
                self._model = now >= due
                return fr, now
            while True:                                    # (old way) only the newest frame once one is due
                ok, fr = self.cap.read()
                if not ok:
                    return None, None
                now = self._now()
                if now >= due:
                    return fr, now
        for _ in range(self.step_n - 1):
            if not self.cap.grab():
                return None, None
        ok, fr = self.cap.read()
        return (fr, self.i * self.step_n / self.src) if ok else (None, None)

    def speed(self):
        """how fast it runs: ms per frame (and per stage), the frames per second actually processed, how many times
        faster than the video plays, and whether that keeps up with the inference fps (a camera needs it to)"""
        if not self.proc:
            return {"ms": None, "max_fps": None, "keeps_up": None, "fps": self.run_fps}
        ms = 1000.0 * float(np.mean(self.proc))
        if self.q is not None and self.tim["detect"]:
            # a FILE: detection runs on the reader thread, overlapped with the rest - a frame costs the slower stage
            ms = max(ms, float(np.mean(self.tim["detect"])))
        mx = 1000.0 / ms
        w = self.walls
        real = (len(w) - 1) / (w[-1] - w[0]) if len(w) > 1 and w[-1] > w[0] else None
        fd = [f[1] - f[0] for f in self.found.values()]
        return {"ms": ms, "max_fps": mx, "keeps_up": mx >= self.read_fps - 0.01, "fps": self.read_fps,
                "all_frames": self.all_frames, "model_rate": self.run_fps,
                "model_fps": self.model_fps, "camera": self.camera, "real_fps": real,
                "x_real": real / self.read_fps if real else None,
                "stages": {k: float(np.mean(v)) for k, v in self.tim.items() if v},
                "found_after": float(np.mean(fd)) if fd else None,
                "late_s": (time.time() - self.t_start) - self.t[-1] if self.camera and self.t else None}

    def _tick(self, k, t0):
        ms = 1000.0 * (time.time() - t0)
        self.tim[k] = (self.tim[k] + [ms])[-30:]
        self.last_ms[k] = ms                               # THIS frame's time per stage (infer.py prints it)
        return time.time()

    def step(self):
        """one sampled frame -> the live state, or None at the end of the video / when the camera stops"""
        fr, ts = self._read()
        if fr is None:
            self.cap.release()
            return None
        self.i += 1
        tp = time.time()
        self.walls = (self.walls + [tp])[-30:]
        self.last_ms = {}
        # a file's frames come resized (RGB) from the reader thread; the camera's and the very first one do not
        rgb = self._resize(fr) if self.camera or self.i == 1 else fr
        t1 = self._tick("read", tp)
        if self._pre is not None:                          # detected ahead on the reader thread (overlapped)
            dets, app, ms = self._pre
            self.tim["detect"] = (self.tim["detect"] + [ms])[-30:]
            self.last_ms["detect"] = ms
        else:
            dets = self.pl.bank.detect_rgb(rgb)
            app = None
            if self._model:
                app = self.roi.vectors(dets, rgb.shape) if self.roi is not None and self.roi.ok else None
            t1 = self._tick("detect", t1)
        if not self._model:
            # a frame BETWEEN the step model's frames: detected and shown, not fed to the model (it learnt its timing
            # at its own fps - every frame would make each step look several times longer to it)
            self.proc = (self.proc + [time.time() - tp])[-30:]
            return {"rgb": rgb, "dets": dets, "t": ts, "probs": None, "model_frame": False,
                    "state": self._state_cache if self._state_cache is not None else self.state(),
                    "speed": self.speed()}
        self.n_model += 1
        st_ = self.stream
        cn = st_.canon if st_ is not None else self.canon
        Pnow = None if st_ is None or st_.P is None or ts - st_.P_t > 1.0 else [float(x) for x in st_.P]
        self.rec.append((float(ts), Pnow, [(cn(d["name"]), [round(float(x), 1) for x in d["box"]]) for d in dets]))
        self._live_sig(rgb, Pnow)
        if self.stream is None:
            # the first REF_S seconds: only watched - where does each part lie in THIS video (relocate)
            self.buf.append((rgb, dets, app, ts))
            if ts - self.buf[0][3] < REF_S:
                self.proc = (self.proc + [time.time() - tp])[-30:]
                self._state_cache = self.state()
                return {"rgb": rgb, "dets": dets, "t": ts, "probs": None, "state": self._state_cache,
                        "speed": self.speed(), "model_frame": True}
            self._start()
        else:
            self.X.append(self.stream.push(rgb, dets, ts, app))
            self.t.append(ts)
        t1 = self._tick("features", t1)
        # the parts' copies lying OFF the product (the last known product box), kept for the 'never placed' check
        P = self.stream.P
        fams = {u["family"] for u in self.units if u["kind"] == "place"}
        off = []
        for d in dets:
            f = self.stream.canon(d["name"])
            b = d["box"]
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if f in fams and not (P is not None and P[0] <= cx <= P[2] and P[1] <= cy <= P[3]):
                off.append((f, tuple(np.round(b, 1))))
        self.hist = [h for h in self.hist if ts - h[0] <= LOOSE_S + 1.0] + [(ts, list(set(off)))]
        X = np.stack(self.X[-WIN:], 1)
        P = self.pred(X)
        self._tick("tcn", t1)
        # an event's score grows as the frames after it come in (the TCN looks ahead): every frame, each unit not
        # yet seen is checked over ALL frames up to LAG_S ago, and takes the time of its best one once it is sure
        k = P.shape[1] - 1 - self.lag
        if k >= 1:
            off = len(self.t) - P.shape[1]                 # P covers the last WIN frames
            for u, unit in enumerate(self.units):
                if unit["kind"] != "place" or unit.get("absent"):
                    continue
                if u in self.picked:
                    self._repick(u, P, k, off)
                if u not in self.picked:
                    # the TCN's peaks are CANDIDATES; one is the pickup only once its copy is gone from its place for
                    # CONFIRM_S after it - a hand passing over it (reaching for a neighbour) leaves it there
                    pr = P[u, :k + 1, 1]
                    self.pmax[u] = max(self.pmax.get(u, 0.0), float(pr.max()) if len(pr) else 0.0)
                    at_all, t_all = np.stack(self.X, 1)[u][:, 1], np.array(self.t)
                    for j in _candidates(pr):
                        tj = self.t[off + j]
                        if tj <= self.rejected.get(u, -1e9):
                            continue
                        st, empty = confirm(tj, t_all, at_all, ts)
                        if st == "wait":
                            break
                        if st == "no":
                            self.rejected[u] = tj
                            self.log(f"  [{tj:6.1f}s] not a pickup: {unit['name']} is still at its place after it "
                                     "(a hand over it)")
                            continue
                        self.picked[u] = (tj, round(float(pr[j]) * empty, 2))
                        self.found[u] = (tj, ts, time.time() - self.t_start)   # seen this long after it happened
                        self.log(f"  [{tj:6.1f}s] PICKUP  {unit['name']}  (step {unit['step']}, TCN {pr[j]:.2f}, "
                                 f"place empty after it {empty:.0%})")
                        break
                if u not in self.placed:
                    q = np.convolve(P[u, :, 2], np.ones(3) / 3, mode="same")[:k + 1]
                    j = int(np.argmax(q))
                    if q[j] >= PEAK:
                        self.placed[u] = (self.t[off + j], float(q[j]))
        # PROVISIONAL (shown at once, 'doing…'): a copy whose pickup the TCN already sees in the newest frames (not
        # yet read LAG_S behind, not yet confirmed) AND that is gone from its place right now - confirmed later as a
        # pickup, or dropped again when the part is back at its place (a hand passing over it)
        self.prov = {}
        if P.shape[1] >= 2:
            off = len(self.t) - P.shape[1]
            tt = np.array(self.t[off:])
            recent = tt >= ts - (LAG_S + CONFIRM_S + 1.0)
            for u, unit in enumerate(self.units):
                if unit["kind"] != "place" or unit.get("absent") or u in self.picked or not recent.any():
                    continue
                pr = np.where(recent & (tt > self.rejected.get(u, -1e9)), P[u, :, 1], 0.0)
                j = int(np.argmax(pr))
                at_now = float(np.mean([x[u, 1] for x in self.X[-2:]]))
                if pr[j] >= PEAK and at_now < 0.5:
                    self.prov[u] = float(tt[j])
        self.proc = (self.proc + [time.time() - tp])[-30:]
        self._state_cache = self.state()
        return {"rgb": rgb, "dets": dets, "t": ts, "probs": P[:, -1], "state": self._state_cache,
                "speed": self.speed(), "model_frame": True}

    def _repick(self, u, P, k, off, gap_s=5.0, back=0.5):
        """REPEATED: a copy taken, back at its place for a while (put back), then taken again"""
        last = (self.repicks.get(u) or [self.picked[u][0]])[-1]
        q = np.convolve(P[u, :, 1], np.ones(3) / 3, mode="same")[:k + 1]
        tt = np.array(self.t[off:off + k + 1])
        cand = np.flatnonzero((tt > last + gap_s) & (q >= PEAK))
        if not len(cand):
            return
        j = int(cand[np.argmax(q[cand])])
        Xh = np.stack(self.X, 1)[u]                         # (T, D) this unit
        tall = np.array(self.t)
        between = Xh[(tall > last + 2.0) & (tall < tt[j] - 1.0), 1]   # seen back at its place in between?
        st, _ = confirm(float(tt[j]), tall, Xh[:, 1], self.t[-1])    # and gone again after (not a hand over it)
        if st != "yes":
            return
        if len(between) and between.mean() >= back:
            self.repicks.setdefault(u, [self.picked[u][0]]).append(float(tt[j]))
            self.log(f"  [{tt[j]:6.1f}s] TAKEN AGAIN  {self.units[u]['name']}  (step {self.units[u]['step']})")

    def release(self):
        self._stop_read = True                             # the reader thread ends at its next frame
        if self.q is not None:
            try:
                while True:
                    self.q.get_nowait()
            except Exception:
                pass
        try:
            self.cap.release()
        except Exception:
            pass

    def state(self):
        """the steps done so far vs the trained sequence: done (time), waiting, or AHEAD of a step still waiting"""
        rows, done_t = [], {}
        for s in self.spec["steps"]:
            us = [u for u, x in enumerate(self.units) if x.get("step") == s["id"]]
            got = [self.picked[u] for u in us if u in self.picked]
            if not us:
                st, tt = "no place", None
            elif all(self.units[u].get("absent") for u in us):
                st, tt = "not on table", None              # not there at the start: cannot be waited for
            elif len(got) == len(us):
                tt = max(g[0] for g in got)
                st, done_t[s["id"]] = "done", tt
            else:
                st, tt = ("partly" if got else "waiting"), None
            fa = [self.found[u][1] - self.found[u][0] for u in us if u in self.found]
            pp = [self.picked[u][1] if u in self.picked else self.pmax.get(u, 0.0) for u in us]
            pv = [getattr(self, "prov", {}).get(u) for u in us if u not in self.picked]
            pv = [x for x in pv if x is not None]
            rows.append({"step": s["id"], "label": self.labels[s["id"]], "status": st, "t": tt,
                         "provisional_t": min(pv) if pv and st != "done" else None,
                         "detected": int(round(100 * min(pp))) if pp else None,
                         "t_ref": (self.trained.get(s["id"]) or {}).get("t"), "units": us,
                         "found_after": max(fa) if fa and tt is not None else None})
        ids = [r["step"] for r in rows]
        bad = []
        for r in rows:                                     # done while an EARLIER step of the sequence is not yet
            if r["status"] == "done":
                # (an earlier step done out of order is still done: only a step not yet seen, or seen later, counts)
                before = [q for q in rows[:ids.index(r["step"])] if q["status"] not in ("no place", "not on table") and
                          (q["t"] is None or q["t"] > r["t"] + temporal_TIE_S)]
                if before:
                    r["status"] = "AHEAD"
                    r["before"] = [q["step"] for q in before]
                    bad.append(r)
        return {"rows": rows, "bad": bad, "order": [r["step"] for r in sorted(
            (r for r in rows if r["t"] is not None), key=lambda r: r["t"])]}

    def not_placed(self):
        """copies TAKEN from their place but lying LOOSE at the end: off the product, still for the last LOOSE_S, and
        not at a place of their part still untaken (that is a copy not yet taken). Each loose copy is the taken copy it
        looks most like (size and shape of its learnt place). -> {unit index: (box)}"""
        if not self.hist or self.hist[-1][0] - self.hist[0][0] < LOOSE_S * 0.8:
            return {}
        win = [h for h in self.hist if self.hist[-1][0] - h[0] <= LOOSE_S]
        out = {}
        for f in {u["family"] for u in self.units if u["kind"] == "place"}:
            taken = [i for i, u in enumerate(self.units) if u["kind"] == "place" and u["family"] == f and i in self.picked]
            if not taken:
                continue
            untaken = [self.units[i]["box"] for i, u in enumerate(self.units)
                       if u["kind"] == "place" and u["family"] == f and i not in self.picked]
            ctr = lambda b: np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            size = lambda b: max(b[2] - b[0], b[3] - b[1], 1.0)
            loose = []
            for b in [b for g, b in win[-1][1] if g == f]:
                if any(np.linalg.norm(ctr(b) - ctr(q)) < 0.7 * size(q) for q in untaken):
                    continue                               # a copy still at its own (untaken) place
                stay = np.mean([any(g == f and np.linalg.norm(ctr(b) - ctr(c)) < 0.5 * size(b) for g, c in h[1])
                                for h in win])
                if stay >= 0.8:
                    loose.append(b)
            # each loose copy -> the taken copy whose learnt place looks most like it (size, shape)
            free = list(taken)
            for b in loose:
                if not free:
                    break
                ar = lambda q: np.log((q[2] - q[0] + 1.0) / (q[3] - q[1] + 1.0))
                i = min(free, key=lambda i: abs(np.log(size(b) / size(self.units[i]["box"])))
                        + abs(ar(b) - ar(self.units[i]["box"])))
                free.remove(i)
                out[i] = b
                self.log(f"  [end] {self.units[i]['name']} (step {self.units[i]['step']}) taken but left lying at "
                         f"{[int(v) for v in b]} - never placed")
        return out

    def _live_sig(self, rgb, P, side=320):
        """this frame's work on the product (share of moving pixels in its box, grown 30 %) and its small picture"""
        s = side / max(rgb.shape[:2])
        g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
                                          cv2.COLOR_RGB2GRAY).astype(np.float32), (5, 5), 0)
        if P is not None:
            self._lastP = P
        b = self._lastP if self._lastP is not None else [0, 0, rgb.shape[1], rgb.shape[0]]
        w, h = b[2] - b[0], b[3] - b[1]
        x1, y1 = int(max(0, (b[0] - 0.3 * w) * s)), int(max(0, (b[1] - 0.3 * h) * s))
        x2, y2 = int(min(g.shape[1], (b[2] + 0.3 * w) * s)), int(min(g.shape[0], (b[3] + 0.3 * h) * s))
        prev = self._sg_prev
        mot = float((np.abs(g - prev)[y1:y2, x1:x2] > 12).mean()) if prev is not None and prev.shape == g.shape \
            and x2 > x1 and y2 > y1 else 0.0
        self.sig_m.append(mot)
        cx1, cy1 = int(max(0, b[0] * s)), int(max(0, b[1] * s))
        cx2, cy2 = int(min(g.shape[1], b[2] * s)), int(min(g.shape[0], b[3] * s))
        crop = g[cy1:max(cy2, cy1 + 2), cx1:max(cx2, cx1 + 2)]
        self._crops.append(cv2.resize(crop, (64, 32)))
        self._sg_prev = g
        if self.hand_boxes is not None:
            try:
                self.hand_boxes.append(hands(rgb, out_w=float(self.W0)))
            except Exception as e:
                self.log(f"  (hands off: {type(e).__name__}: {e})")
                self.hand_boxes = None

    def live_signals(self):
        """(work, change) per frame so far - the same signals the action model reads from a file, from live frames"""
        m = np.convolve(np.array(self.sig_m, float), np.ones(3) / 3, mode="same") if self.sig_m else np.zeros(0)
        work = np.clip(m / max(float(np.percentile(m, 95)) if len(m) else 1.0, 1e-6), 0, 2)
        t = [r[0] for r in self.rec]
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 3
        w1 = max(1, int(round(1.0 / dt)))
        cr = self._crops
        change = np.clip(np.array([np.abs(cr[k] - cr[max(0, k - w1)]).mean() for k in range(len(cr))]) / 20.0, 0, 2) \
            if cr else np.zeros(0)
        return work, change

    def record_so_far(self):
        """the run up to now, for the action model while the video plays"""
        return {"t": [r[0] for r in self.rec], "P": [r[1] for r in self.rec], "dets": [r[2] for r in self.rec],
                "W0": self.W0, "H0": self.H0, "units": [], "events": [], "status": "", "anomalies": [],
                **({"hands": list(self.hand_boxes)} if self.hand_boxes is not None else {})}

    def record(self, final):
        """this run as one record (per frame: time, product box, detections; the places in this video's layout; the
        final events) - what sopmon/completion.py reads to time each step's completion"""
        return {"t": [r[0] for r in self.rec], "P": [r[1] for r in self.rec], "dets": [r[2] for r in self.rec],
                "W0": self.W0, "H0": self.H0,
                "units": [dict(name=u["name"], step=u["step"], family=u["family"],
                               box=[float(x) for x in u["box"]], absent=bool(u.get("absent")))
                          for u in self.units if u["kind"] == "place"],
                "events": final["events"], "status": final["status"],
                "anomalies": [a["text"] for a in final["anomalies"]], "labels": self.labels,
                **({"hands": list(self.hand_boxes)} if self.hand_boxes is not None else {})}

    def finish(self):
        """the final verdict: the sequence decoder over the events SEEN LIVE (what the window showed - a second pass
        over the whole video could disagree with it), with copies taken but never placed as MISSING"""
        if self.stream is None and self.buf:              # a video shorter than the first seconds
            self._start()
        loose = self.not_placed()
        events = []
        # still lying at its place over the last LOOSE_S (the detector sees its part there): absence is PROVEN
        tall = np.array(self.t)
        endw = tall >= (tall[-1] - LOOSE_S if len(tall) else 0)
        Xa = np.stack(self.X, 1) if self.X else None
        for u, unit in enumerate(self.units):
            if unit["kind"] != "place":
                continue
            pk, pc = self.picked.get(u), self.placed.get(u)
            at_end = bool(Xa is not None and endw.any() and Xa[u, endw, 1].mean() >= 0.5)
            events.append({"unit": unit["name"], "step": unit["step"], "family": unit["family"],
                           "pickup": pk[0] if pk else None, "p_pickup": pk[1] if pk else self.pmax.get(u, 0.0),
                           "place": pc[0] if pc else None, "p_place": pc[1] if pc else 0.0,
                           "not_placed": u in loose, "at_place_end": at_end, "absent": bool(unit.get("absent")),
                           "pickups": self.repicks.get(u) or ([pk[0]] if pk else [])})
        return decode_events(self.spec, events, labels=self.labels)


def speed_line(sp):
    """the time per frame (each stage) and whether it keeps up with real time"""
    if not sp or sp.get("ms") is None:
        return "  per frame : -"
    st = sp.get("stages") or {}
    parts = " · ".join(f"{k} {st[k]:.0f}" for k in ("detect", "features", "tcn") if k in st)
    rate = (f"every frame read ({sp['fps']:.1f} fps), step model at {sp.get('model_rate', 0):.1f} fps"
            if sp.get("all_frames") else f"video sampled at {sp['fps']:.1f} fps")
    s = (f"  per frame : {sp['ms']:.0f} ms" + (f"  ({parts} ms)" if parts else "")
         + (f"  ·  {sp['real_fps']:.1f} fps processed" if sp.get("real_fps") else "")
         + f"  ·  {rate}  ->  " + ("real time OK" if sp["keeps_up"] else "slower than the video"))
    if sp.get("late_s") is not None:
        s += f"  ·  {sp['late_s']:.1f} s behind the camera"
    return s


def _pct(v):
    return f"{v:d}%" if isinstance(v, int) else (f"{100 * v:.0f}%" if isinstance(v, float) else "-")


def live_lines(model, video, st, t, duration=None, final=None, speed=None):
    """the Result panel: per step its status, the time it was done and how strongly it was detected; the verdict"""
    name = f"camera {video}" if str(video).strip().isdigit() or "://" in str(video) else Path(video).name
    L = [f"LIVE INFERENCE  {name}   model {model['name']}",
         f"  t = {t:6.1f} s" + (f" / {duration:.0f} s" if duration else ""), speed_line(speed), ""]
    pf = model.get("places_file")
    action = bool(pf and (Path(pf).parent / "action_matcher.pt").exists())
    if final is None and action:
        L.append("  the steps appear in the live window as they are confirmed")
        return L
    if final is None:
        L.append(f"  {'STEP':<44} {'STATUS':<14} {'TIME':>8}  {'DETECTED':>8}")
        for k, r in enumerate(st["rows"], 1):
            stt = {"done": "done", "AHEAD": "OUT OF ORDER", "partly": "partly", "no place": "-",
                   "not on table": "not on table"}.get(r["status"], "waiting")
            tt = f"{r['t']:.1f} s" if r.get("t") is not None else "-"
            L.append(f"  {k:2d}. {r['label']:<40} {stt:<14} {tt:>8}  {_pct(r.get('detected')):>8}")
        L += ["", "  SEQUENCE SO FAR : " + (" -> ".join(map(str, st["order"])) or "(nothing yet)")]
        return L
    L += [f"FINAL VERDICT: {final['status']}", f"  {'STEP':<44} {'STATUS':<14} {'TIME':>8}  {'DETECTED':>8}"]
    for s in final["steps"]:
        tt = f"{s['t']:.1f} s" if s.get("t") is not None else "-"
        det = s.get("detected", int(round(100 * s["p"])) if s.get("p") is not None else None)
        L.append(f"  {s['step']:2d}. {s['label'][:40]:<40} {s['status'][:14]:<14} {tt:>8}  {_pct(det):>8}")
    seq = [s["step"] for s in sorted((s for s in final["steps"] if s.get("t") is not None), key=lambda s: s["t"])]
    L += ["", "  SEQUENCE : " + (" -> ".join(map(str, seq)) or "(none)")]
    L += [f"  ! {a['text']}" for a in final["anomalies"]] or ["  no anomaly: the video follows the trained sequence"]
    return L


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sop_monitor.config   (from sop_monitor\config.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""
config.py - where the data lives, the inference settings, and the ACTIVE MODEL.

Everything is read from THIS folder (inference_code_only/):
    models.json          detector registry (paths relative to this folder)
    weights/             detector weights + the MediaPipe hand model
    trained/<name>/      trained sequence models;  trained/active_model.txt = the model used when none is named
    outputs/             run folders + the inspection log  (written)
    specs/               small caches                      (written)

Another data folder can be used by setting the environment variable SOP_MONITOR_HOME before starting Python.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ACTIVE_MODEL_FILE = "active_model.txt"

APP_ROOT = home()                                    # the data folder (see the first section)


@dataclass
class AppConfig:
    """Inference settings (defaults = the desktop app's). Paths follow APP_ROOT."""

    app_root: Path = field(default_factory=lambda: APP_ROOT)
    all_frames: bool = True            # read + detect EVERY frame; the step model gets its trained fps
    action_check_every_s: float = 0.5  # each step judged this often (seconds of video)
    output_dir: Optional[Path] = None  # default APP_ROOT/outputs

    def __post_init__(self):
        self.app_root = Path(self.app_root)
        self.output_dir = Path(self.output_dir) if self.output_dir else self.app_root / "outputs"

    @property
    def registry_file(self) -> Path:
        return self.app_root / "models.json"

    @property
    def weights_dir(self) -> Path:
        return self.app_root / "weights"

    @property
    def trained_dir(self) -> Path:
        return self.app_root / "trained"

    @property
    def spec_dir(self) -> Path:
        return self.app_root / "specs"

    @property
    def hand_model(self) -> Path:
        return self.weights_dir / "hand_landmarker.task"


def bootstrap(cfg: Optional[AppConfig] = None) -> AppConfig:
    """the data folder present + the engine importable -> the config. Every public function calls it."""
    cfg = cfg or AppConfig()
    if Path(cfg.app_root).resolve() != APP_ROOT:
        raise ValueError(f"AppConfig.app_root ({cfg.app_root}) differs from {APP_ROOT}; set SOP_MONITOR_HOME")
    if not cfg.registry_file.is_file():
        raise FileNotFoundError(f"no models.json in {APP_ROOT}")
    for d in ("specs", "outputs"):
        (APP_ROOT / d).mkdir(exist_ok=True)
    return cfg


def get_active_model(cfg: Optional[AppConfig] = None) -> str:
    """the active model's name (trained/active_model.txt), else the newest trained model"""
    cfg = cfg or AppConfig()
    d = cfg.trained_dir
    p = d / ACTIVE_MODEL_FILE
    if p.is_file():
        name = p.read_text(encoding="utf-8").strip()
        if name and (d / name / "model.json").is_file():
            return name
    models = sorted((x for x in d.iterdir() if (x / "model.json").is_file()), key=lambda x: x.stat().st_mtime,
                    reverse=True) if d.is_dir() else []
    if not models:
        raise FileNotFoundError(f"no trained model in {d}")
    return models[0].name


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sop_monitor.results   (from sop_monitor\results.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""
results.py - the typed objects training and inference return / pass to callbacks.
All are plain dataclasses: `dataclasses.asdict(x)` gives JSON-ready dicts.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── TRAINING ─────────────────────────────────────────────────────────────────


# ── INFERENCE ────────────────────────────────────────────────────────────────

@dataclass
class FrameInfo:
    """one frame, passed to `on_frame` while inference runs"""
    index: int                         # frames processed so far (1, 2, ...)
    video_frame: Optional[int]         # the video's own frame number (None for a camera)
    t: float                           # video time, seconds
    rgb: Any                           # the frame (H, W, 3) uint8 RGB, resized to <= 1280 px
    detections: List[Dict[str, Any]]   # [{"name", "conf", "box": [x1, y1, x2, y2], "model"}]
    model_frame: bool                  # this frame also went to the step model
    frame_ms: float                    # this frame's processing time
    stage_ms: Dict[str, float]         # detect / features / tcn / step_check ms of this frame
    fps: float                         # frames processed per second (the last 30)
    keeps_up: Optional[bool]           # fast enough for real time


@dataclass
class StepEvent:
    """a change in the step list, passed to `on_step` the moment it happens"""
    kind: str                          # "done" | "doing" | "taken_back"
    step: int
    label: str
    t: float                           # video time of the step (done) / its start (doing) / when undone
    in_order: Optional[bool] = None    # done: in the trained order so far
    detected: Optional[int] = None     # done: how strongly it was detected, 0-100
    note: str = ""                     # out-of-order / why taken back
    shown_at: Optional[float] = None   # video time it was shown


@dataclass
class StepResult:
    step: int
    label: str
    status: str                        # OK / MISSING / OUT OF ORDER / REMOVED / UNCERTAIN ...
    start_s: Optional[float]
    done_s: Optional[float]
    duration_s: Optional[float]        # how long the step took in the video
    shown_at_s: Optional[float]        # video time inference confirmed it live
    found_after_s: Optional[float]     # shown_at - done
    detected_pct: Optional[int]
    evidence: str = ""


@dataclass
class InferenceResult:
    video: str
    model: str
    verdict: str                       # NORMAL / ANOMALY / UNCERTAIN
    sequence: List[int]                # step ids in the order they were done
    steps: List[StepResult]
    anomalies: List[str]
    frames_read: int
    frames_to_model: int
    seconds: float                     # wall time of the run
    fps: float                         # frames processed per second
    frame_ms: Dict[str, float]         # whole frame: mean / median / p95 / max
    report: str                        # the same text the app shows
    out_dir: Optional[str] = None      # where results / CSVs / video were saved
    stopped: bool = False              # ended by stop() / the stop event


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sop_monitor.timing   (from sop_monitor\timing.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""
timing.py - per-frame and per-step timing: summaries, the per-step table, CSV files.
"""

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np



def frame_record(index: int, r: Dict, live, frame_ms: float, check_ms: Optional[float], camera: bool) -> Dict:
    """one frame's row of frame_times.csv: its numbers, whether it fed the step model, ms per stage"""
    lm = dict(getattr(live, "last_ms", {}) or {})
    return {"frame": index,
            "video_frame": None if camera else int(round(float(r["t"]) * live.src)),
            "model_frame": int(bool(r.get("model_frame", True))),
            "video_t": round(float(r["t"]), 2),
            "frame_ms": round(frame_ms, 1),
            "detect_ms": round(lm.get("detect", 0.0), 1),
            "features_ms": round(lm.get("features", 0.0), 1),
            "tcn_ms": round(lm.get("tcn", 0.0), 1),
            "step_check_ms": round(check_ms or 0.0, 1)}


def stage_ms(rec: Dict) -> Dict[str, float]:
    """a frame row -> {"detect", "features", "tcn", "step_check"} ms"""
    return {k: rec[f"{k}_ms"] for k in ("detect", "features", "tcn", "step_check")}


def stats(values) -> Dict[str, float]:
    v = np.asarray(list(values), float)
    if not len(v):
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {"mean": round(float(v.mean()), 1), "median": round(float(np.median(v)), 1),
            "p95": round(float(np.percentile(v, 95)), 1), "max": round(float(v.max()), 1)}


def frame_summary(frames: List[Dict], wall: float) -> List[str]:
    """mean / median / p95 / max of each stage over the run"""
    if not frames:
        return ["PER FRAME: no frames"]
    nm = sum(f.get("model_frame", 1) for f in frames)
    vfs = [f["video_frame"] for f in frames if f.get("video_frame") is not None]
    L = [f"PER FRAME  ({len(frames)} frames in {wall:.1f} s = {len(frames) / max(wall, 1e-6):.1f} fps processed"
         + (f";  {len(frames)} of the video's {vfs[-1] + 1} frames read + detected, {nm} given to the step model"
            if vfs else "") + ")",
         f"  {'stage':<18}{'mean':>8}{'median':>8}{'p95':>8}{'max':>8}   ms"]
    for k, lab in (("frame_ms", "whole frame"), ("detect_ms", "detect"), ("features_ms", "features"),
                   ("tcn_ms", "TCN"), ("step_check_ms", "step check")):
        v = [f[k] for f in frames if not (k in ("features_ms", "tcn_ms") and not f.get("model_frame", 1))]
        if k == "step_check_ms":
            v = [x for x in v if x > 0]
            lab += f" ({len(v)}x)"
        if not v:
            continue
        s = stats(v)
        L.append(f"  {lab:<18}{s['mean']:8.1f}{s['median']:8.1f}{s['p95']:8.1f}{s['max']:8.1f}")
    L.append("  (detect runs on its own thread while the rest works, so the whole frame is less than the sum)")
    return L


def step_results(model: Dict, final: Dict, acts: Optional[Dict], tracker) -> List[StepResult]:
    """per step: start / done / duration in the video, when inference showed it, detected %, evidence.
    The pickup model has no start per step: a step 'takes' the time since the step done before it in the video."""
    fin = {s["step"]: s for s in final.get("steps") or []}
    ends = sorted((f.get("t"), sid) for sid, f in fin.items() if f.get("t") is not None)
    before = {sid: (ends[k - 1][0] if k else 0.0) for k, (_, sid) in enumerate(ends)}
    out = []
    for s in model["spec"]["steps"]:
        sid = s["id"]
        a = (acts or {}).get(sid) or {}
        f = fin.get(sid) or {}
        end = a.get("end_sec") if acts is not None else f.get("t")
        start = a.get("start_sec") if acts is not None else before.get(sid)
        if acts is not None and a.get("status") == "INITIAL_STATE":
            start = end
        sh = tracker.shown_at.get(sid)
        det = f.get("detected", a.get("detected"))
        if det is None and f.get("p") is not None:
            det = int(round(100 * float(f["p"])))
        r2 = lambda v: None if v is None else round(float(v), 2)
        out.append(StepResult(step=sid, label=tracker.lab[sid], status=str(f.get("status", a.get("status", "-"))),
                              start_s=r2(start), done_s=r2(end),
                              duration_s=r2(end - start) if end is not None and start is not None else None,
                              shown_at_s=r2(sh[0]) if sh else None,
                              found_after_s=r2(sh[0] - end) if sh and end is not None else None,
                              detected_pct=det, evidence=str(a.get("evidence") or f.get("why") or "")))
    return out


def step_table(rows: List[StepResult], pickup: bool = False) -> List[str]:
    f = lambda v: "-" if v is None else f"{v:.1f}s"
    L = ["PER STEP  (video time: started -> done = how long the step took"
         + ("; this model has no start per step: 'took' = time since the step before it" if pickup else "") + ")",
         "           (shown = when inference confirmed it; after = shown - done)",
         f"  {'step':<34}{'status':<14}{'started':>9}{'done':>8}{'took':>8}{'shown':>8}{'after':>8}{'det':>6}"]
    for r in rows:
        lab = r.label if len(r.label) <= 29 else r.label[:28] + "."
        L.append(f"  {r.step:2d}. {lab:<29} {r.status[:13]:<14}{f(r.start_s):>9}{f(r.done_s):>8}{f(r.duration_s):>8}"
                 f"{f(r.shown_at_s):>8}{f(r.found_after_s):>8}"
                 f"{('-' if r.detected_pct is None else str(r.detected_pct) + '%'):>6}")
    done = [r for r in rows if r.duration_s is not None]
    if done:
        a, b = min(r.start_s for r in done), max(r.done_s for r in done)
        fa = [r.found_after_s for r in rows if r.found_after_s is not None]
        L.append(f"  whole sequence: {a:.1f} s -> {b:.1f} s = {b - a:.1f} s;  a step takes "
                 f"{np.mean([r.duration_s for r in done]):.1f} s on average"
                 + (f";  confirmed {np.mean(fa):.1f} s after it on average" if fa else ""))
    return L


def write_csv(path: Path, rows) -> None:
    rows = [asdict(r) if hasattr(r, "__dataclass_fields__") else r for r in rows]
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sop_monitor.detectors   (from sop_monitor\detectors.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""
detectors.py - the DETECTOR REGISTRY (models.json) and the check that every model file is present.

DETECTORS IN THIS FOLDER (weights/)
  detectors/box_assembly_yolo_detect.pt    "assembling_box_model (detect)"        packing the product box
  detectors/packing_yolo_segment.pt        "best (segment)"                        phone -> outer box packing
  detectors/phone_assembly_yolo_obb.pt     "assembling_moPdel_v2-2 (obb)"          phone assembly
  detectors/dataset3_truck_yolo_detect.pt  "Dataset3_assemble_original (detect)"   enabled in the phone model
  hand_landmarker.task                     MediaPipe hands (placement evidence: "the item was let go")

A trained model names the detectors it was trained with; inference loads exactly those.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional



def _resolve(entry: Dict, cfg: AppConfig) -> Dict:
    """an entry with its weights path made absolute (relative paths are relative to this folder)"""
    e = dict(entry)
    p = Path(str(e.get("path") or ""))
    if e.get("path") and not p.is_absolute():
        e["path"] = str((Path(cfg.app_root) / p).resolve())
    return e


def load_registry(cfg: Optional[AppConfig] = None) -> List[Dict]:
    """every detector, paths absolute"""
    cfg = bootstrap(cfg)
    entries = json.loads(Path(cfg.registry_file).read_text(encoding="utf-8"))
    return [_resolve(e, cfg) for e in entries]


def registry_for_model(model: Dict, cfg: Optional[AppConfig] = None) -> List[Dict]:
    """the registry with the detectors a trained model was trained with enabled"""
    bootstrap(cfg)
    return models_for(model, load_registry(cfg))


def check_models(model_name: Optional[str] = None, cfg: Optional[AppConfig] = None) -> Dict[str, object]:
    """-> {"ok", "problems", "found"}: the hand model, and for a trained model its files and its detectors' weights"""
    cfg = bootstrap(cfg)
    problems, found = [], {"hand_landmarker": cfg.hand_model.exists()}
    if not found["hand_landmarker"]:
        problems.append(f"MediaPipe hand model missing: {cfg.hand_model} (placement works with less evidence)")
    if model_name:
        d = Path(cfg.trained_dir) / model_name
        for f in ("model.json", "places.json", "temporal.pt", "action_matcher.pt"):
            found[f] = (d / f).exists()
            if not found[f]:
                problems.append(f"{model_name}: {f} missing"
                                + (" (the pickup model decides then)" if f == "action_matcher.pt" else ""))
        if found.get("model.json"):
            m = json.loads((d / "model.json").read_text(encoding="utf-8"))
            reg = {e.get("name"): e for e in load_registry(cfg)}
            for n in m.get("detectors") or []:
                p = (reg.get(n) or {}).get("path")
                found[f"detector: {n}"] = bool(p) and Path(p).exists()
                if not found[f"detector: {n}"]:
                    problems.append(f"detector '{n}' of {model_name}: " + ("not in models.json" if n not in reg
                                                                            else f"weights not found ({p})"))
    soft = ("less evidence", "pickup model decides")
    return {"ok": not [p for p in problems if not any(s in p for s in soft)], "problems": problems, "found": found}


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sop_monitor.step_tracker   (from sop_monitor\step_tracker.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""
step_tracker.py - the live step list: which steps are DONE (in order or not), which are being DONE NOW, which were
taken back. The same rules as the desktop app's live window:

  ACTION MODEL (box / packing models - action_matcher.pt):
    every `every_s` seconds of video each step is judged on the video so far (object_actions.infer, live):
    - done        its own objects' action + completed state seen, held (placements: came to rest / let go / vanished
                  inside and stayed)
    - doing       its object is in its destination / its change just seen, not yet confirmed
    - taken back  its object was taken out again, or its result did not hold in 2 checks in a row
  PICKUP MODEL (phone-type models without an action model):
    - done        its part left its place and stayed away (TCN + place check)
    - doing       the TCN sees the pickup now and the part is gone from its place
"""

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


DONE = ("COMPLETED", "INITIAL_STATE", "RECOGNISED_UNVERIFIED")


class StepTracker:
    def __init__(self, model: Dict, video: str, every_s: float = 0.5, on_step: Optional[Callable] = None,
                 log: Callable[[str], None] = print):
        self.model, self.video, self.log, self.on_step = model, video, log, on_step
        self.ids = [s["id"] for s in model["spec"]["steps"]]
        self.lab = {s["id"]: pipeline_step_label(s) for s in model["spec"]["steps"]}
        self.matcher = (Path(model["places_file"]).parent / "action_matcher.pt").exists()
        self.every = float(every_s)
        self.shown: List[Tuple[int, float, bool, Optional[int]]] = []   # (step, t, in_order, detected)
        self.doing: Dict[int, float] = {}
        self.retracted: List[Tuple[int, str]] = []
        self.shown_at: Dict[int, Tuple[float, float]] = {}             # step -> (video t shown, wall s)
        self.act_ms: List[float] = []
        self._next, self._miss, self._t0 = 0.0, {}, time.time()

    # ── events ──
    def _emit(self, ev: StepEvent):
        if self.on_step is not None:
            try:
                self.on_step(ev)
            except Exception as e:                         # a host callback must never stop the inference
                self.log(f"  (on_step callback failed: {type(e).__name__}: {e})")

    def _late(self, sid, t, shown_t):
        """the SOP's earlier steps that make step sid (done at t) out of order: done AFTER t, or not done and not
        being done either. An earlier step still 'doing' (its confirmation lags) is not held against sid yet - if it
        started before t (one that started after t is late whatever its confirmation says)."""
        return [e for e in self.ids[:self.ids.index(sid)]
                if (e in shown_t and shown_t[e] > t)
                or (e not in shown_t and (e not in self.doing or self.doing[e] > t))]

    def _recheck(self, now):
        """the order judged again from the step TIMES each time the list changes - a step confirmed later but done
        earlier (the card confirmed at 32 s, done at 18 s) clears the step after it"""
        shown_t = {s: t for s, t, *_ in self.shown}
        new = []
        for sid, t, ok, det in self.shown:
            late = self._late(sid, t, shown_t)
            if ok and late:                                # newly out of order: say it once
                note = f"step {', '.join(map(str, late))} " + ("done later" if all(e in shown_t for e in late)
                                                              else "not done")
                if self.on_step is None:
                    self.log(f"  [{t:6.1f}s] STEP {sid} {self.lab[sid]}: OUT OF ORDER ({note})   (at {now:.1f}s)")
                self._emit(StepEvent("done", sid, self.lab[sid], float(t), in_order=False, detected=det, note=note,
                                     shown_at=float(now)))
            new.append((sid, t, not late, det))
        self.shown = new

    def _done(self, sid, t, det, now):
        shown_t = {s: tt for s, tt, *_ in self.shown}
        before = self._late(sid, t, shown_t)
        self.shown.append((sid, t, not before, det))
        self.shown_at[sid] = (now, time.time() - self._t0)
        self.doing.pop(sid, None)
        self.retracted = [r for r in self.retracted if r[0] != sid]
        note = f"step {', '.join(map(str, before))} not done yet" if before else ""
        if self.on_step is None:                       # with a callback, the host reports the events itself
            self.log(f"  [{t:6.1f}s] STEP {sid} {self.lab[sid]}: " + ("done" if not before else f"OUT OF ORDER ({note})")
                     + (f"  detected {det}%" if det is not None else "") + f"   (shown at {now:.1f}s)")
        self._emit(StepEvent("done", sid, self.lab[sid], float(t), in_order=not before, detected=det, note=note,
                             shown_at=float(now)))

    def _set_doing(self, new: Dict[int, float], now: float):
        for sid in set(new) - set(self.doing):
            self._emit(StepEvent("doing", sid, self.lab[sid], float(new[sid]), shown_at=float(now)))
        self.doing = new

    # ── per frame ──
    def update(self, live, r) -> Optional[float]:
        """after each frame: the step list; -> ms of the step check run on this frame (None when none ran)"""
        shown = {s for s, *_ in self.shown}
        if not self.matcher:
            for x in sorted((x for x in r["state"]["rows"] if x["t"] is not None and x["step"] not in shown),
                            key=lambda x: x["t"]):
                self._done(x["step"], x["t"], x.get("detected"), r["t"])
            shown = {s for s, *_ in self.shown}
            self._set_doing({x["step"]: x["provisional_t"] for x in r["state"]["rows"]
                             if x.get("provisional_t") is not None and x["step"] not in shown}, r["t"])
            self._recheck(r["t"])
            return None
        if r["t"] < self._next or len(live.rec) < 10:
            return None
        self._next = r["t"] + self.every
        a = time.time()
        res, _ = infer(self.model, self.video, live.record_so_far(), log=lambda s: None,
                          signals=live.live_signals(), live=True)
        self.act_ms = (self.act_ms + [1000 * (time.time() - a)])[-20:]
        for sid in list(shown):                            # TAKE BACK
            st = res.get(sid, {}).get("status")
            self._miss[sid] = 0 if st in DONE else self._miss.get(sid, 0) + 1
            if st == "REMOVED" or self._miss[sid] >= 2:
                self.shown = [s for s in self.shown if s[0] != sid]
                self._miss.pop(sid, None)
                why = res[sid].get("evidence", "")[:100] if st == "REMOVED" else "its result did not hold"
                self.retracted.append((sid, why))
                if self.on_step is None:
                    self.log(f"  [{r['t']:6.1f}s] step {sid} {self.lab[sid]}: NOT done after all - {why}")
                self._emit(StepEvent("taken_back", sid, self.lab[sid], float(r["t"]), note=why,
                                     shown_at=float(r["t"])))
        shown = {s for s, *_ in self.shown}
        for te, sid in sorted((x["end_sec"], sid) for sid, x in res.items()
                              if x["status"] in DONE and x["end_sec"] is not None and sid not in shown):
            self._done(sid, te, res[sid].get("detected"), r["t"])
        shown = {s for s, *_ in self.shown}
        self._set_doing({sid: x["provisional_t"] for sid, x in res.items()
                         if x.get("provisional_t") is not None and sid not in shown}, r["t"])
        self._recheck(r["t"])
        return self.act_ms[-1]


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sop_monitor.inference   (from sop_monitor\inference.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""
inference.py - RUN A TRAINED MODEL on a video file or a live camera.

    from sop_monitor import InferenceEngine
    engine = InferenceEngine()              # no name = the ACTIVE model (the one trained last)
    engine.load()                           # ONCE: detectors on the GPU (half precision) + warm-up
    res = engine.run("test.mp4", on_frame=..., on_step=...)     # any number of videos / camera sessions
    res.verdict, res.sequence, res.steps, res.anomalies
    engine.close()

PER FRAME:  read -> DETECT (its own thread, overlapped) -> on every Nth frame (the model's trained fps): features ->
            step TCN -> twice a second of video: each step judged (StepTracker) -> on_frame / on_step callbacks
EVERY frame is read and detected; the step models get the rate they learnt at.
END:        the whole video's verdict (the action model on the full record), per-step timing, files.

THREADING: run() blocks; call it on a worker thread in a GUI. Callbacks run on that thread. One run at a time.
"""

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

Log = Callable[[str], None]


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING - one function per model; load_models() calls them in order, ONCE per trained model
#
#    trained/<name>/model.json + places.json   load_trained_model()   the steps, the detectors to use, the places
#    weights/detectors/*.pt  (models.json)     load_detectors()       the YOLO detector(s) on the GPU, half precision
#    trained/<name>/temporal.pt                load_step_tcn()        the step TCN (pickup / place events per frame)
#    trained/<name>/action_matcher.pt          load_action_model()    the action model (each step's own objects)
#    weights/hand_landmarker.task              load_hand_model()      MediaPipe hands (optional evidence)
#    (all of them)                             warm_up()              first CUDA calls made before the first frame
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class LoadedModels:
    """everything one trained model needs at inference time, loaded once and reused for every video"""
    name: str
    model: Dict                         # model.json (+ "places_file", "name")
    pipeline: Any                       # holds the detectors (pipeline.bank) and the learnt places
    step_tcn: Any                       # Predictor: the step TCN, on the GPU
    action_model: Optional[tuple]       # (checkpoint, network) or None: then the pickup model decides
    hand_model: Any                     # the MediaPipe HandLandmarker, or None
    seconds: Dict[str, float] = field(default_factory=dict)   # load time per model

    @property
    def detectors(self) -> List[str]:
        return list(self.model.get("detectors") or [])

    @property
    def steps(self) -> List[str]:
        return [s.get("action_query") for s in self.model["spec"]["steps"]]

    @property
    def fps(self) -> float:
        """the rate the step models were trained at (frames per second given to them)"""
        return float(self.model.get("fps") or 3.0)


def load_trained_model(name: str, cfg: Optional[AppConfig] = None) -> Dict:
    """trained/<name>/model.json -> the model dict (steps, detectors, fps, prompt, the places file)"""
    bootstrap(cfg)
    d = Path((cfg or AppConfig()).trained_dir) / name
    if not (d / "model.json").is_file():
        raise FileNotFoundError(f"no trained model '{name}' in {d.parent}")
    return dict(trained_load(name), name=name)


def load_detectors(model: Dict, cfg: Optional[AppConfig] = None) -> "Pipeline":
    """the model's YOLO detector(s) (the ones it was trained with, from models.json) on the GPU in half precision,
    with the learnt places applied -> the Pipeline that holds them (pipeline.bank = the detectors)"""
    cfg = bootstrap(cfg)
    registry = registry_for_model(model, cfg)                 # exactly the detectors this model was trained with
    pl = Pipeline(None, registry, cfg.output_dir / "_engine", fps=float(model.get("fps") or 3.0),
                  spec_dir=cfg.spec_dir, algorithm_only=True, log=lambda s: None)
    sp = json.loads(json.dumps(model["spec"]))
    pl.places_file = model["places_file"]
    pl.apply_positions(sp, model["prompt"])                   # where each part lies at the start (places.json)
    _ensure_bank(pl, model, sp)                               # loads the .pt files -> pl.bank, pl._roi
    return pl


def load_step_tcn(model: Dict) -> "Predictor":
    """trained/<name>/temporal.pt -> the step TCN on the GPU (per frame: pickup / place probabilities per part)"""
    import torch
    path = Path(model["places_file"]).parent / "temporal.pt"
    state = torch.load(path, map_location="cpu")
    _check_features(state)
    return Predictor(state)


def load_action_model(model: Dict) -> Optional[tuple]:
    """trained/<name>/action_matcher.pt -> (checkpoint, network), or None when the model has none (the pickup
    model decides then)"""
    path = Path(model["places_file"]).parent / "action_matcher.pt"
    return _load(path) if path.exists() else None


def load_hand_model() -> Any:
    """weights/hand_landmarker.task -> the MediaPipe HandLandmarker, or None (mediapipe not installed / no file:
    placement then works with less evidence)"""
    try:
        lm = _landmarker() if available() else None
    except Exception:
        return None
    if lm is not None:
        import atexit
        atexit.register(close_hand_model)          # closed before Python shuts MediaPipe down (no exit traceback)
    return lm


def close_hand_model() -> None:
    """close the MediaPipe hand model (unload_models and the interpreter exit call it)"""
    global _LM
    if _LM is not None:
        lm, _LM = _LM, None
        real_close = lm.close
        lm.close = lambda: None                    # MediaPipe's __del__ closes again at exit: a no-op from now on
        try:
            real_close()
        except Exception:                          # MediaPipe may already be shutting down
            pass


def warm_up(pl: "Pipeline", model: Dict, hand_model: Any = None, n: int = 3) -> float:
    """a few detections (+ hands) on a real frame so the first frame of a run is as fast as the rest -> seconds"""
    import torch
    # NOT cudnn.benchmark: the TCN's input grows by a frame every frame - benchmark would re-tune every frame
    torch.backends.cudnn.benchmark = False
    fr = None
    v = model.get("trained_on_path")
    if v and Path(v).is_file():
        cap = cv2.VideoCapture(str(v))
        ok, f = cap.read()
        cap.release()
        if ok:
            h, w = f.shape[:2]
            s = pl.max_dim / max(h, w)
            if s < 1:
                f = cv2.resize(f, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            fr = np.ascontiguousarray(f[:, :, ::-1])
    if fr is None:
        fr = np.zeros((720, 1280, 3), np.uint8)
    t0 = time.time()
    for _ in range(n):
        dets = pl.bank.detect_rgb(fr)
        if pl._roi is not None and pl._roi.ok:
            pl._roi.vectors(dets, fr.shape)
    if hand_model is not None:
        try:
            hands(fr)
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time() - t0


def load_models(name: Optional[str] = None, cfg: Optional[AppConfig] = None, log: Log = print) -> LoadedModels:
    """ALL the models of one trained model, each by its own function, then a warm-up. Call ONCE; reuse the result
    for every video / camera session (InferenceEngine does)."""
    cfg = bootstrap(cfg)
    name = name or get_active_model(cfg)
    secs: Dict[str, float] = {}

    def timed(key, fn, *a):
        t0 = time.time()
        out = fn(*a)
        secs[key] = round(time.time() - t0, 2)
        return out

    model = timed("trained_model", load_trained_model, name, cfg)
    pl = timed("detectors", load_detectors, model, cfg)
    tcn = timed("step_tcn", load_step_tcn, model)
    act = timed("action_model", load_action_model, model)
    hand = timed("hand_model", load_hand_model)
    secs["warm_up"] = round(warm_up(pl, model, hand), 2)
    lm = LoadedModels(name=name, model=model, pipeline=pl, step_tcn=tcn, action_model=act, hand_model=hand,
                      seconds=secs)
    log(f"[load] {name}: " + ", ".join(f"{k} {v:.2f}s" for k, v in secs.items()))
    return lm


def unload_models(lm: Optional[LoadedModels]) -> None:
    """free the detectors' GPU memory"""
    if lm is None:
        return
    pl = lm.pipeline
    if getattr(pl, "_roi", None) is not None:
        pl._roi.close()
    if getattr(pl, "bank", None) is not None and hasattr(pl.bank, "release"):
        pl.bank.release()
    if lm.hand_model is not None:
        close_hand_model()


class InferenceEngine:
    """one trained model, loaded once, run on as many videos / camera sessions as needed"""

    def __init__(self, model_name: Optional[str] = None, cfg: Optional[AppConfig] = None, log: Log = print):
        self.cfg = bootstrap(cfg)
        self.model_name = model_name or get_active_model(self.cfg)
        self.log = log
        self.models: Optional[LoadedModels] = None  # every model, loaded once by load()
        self.tracker: Optional[StepTracker] = None  # the current / last run's step list
        self._stop = threading.Event()
        self._busy = threading.Lock()

    # ── loading ──
    @property
    def loaded(self) -> bool:
        return self.models is not None

    @property
    def model(self) -> Optional[Dict]:
        return self.models.model if self.models else None

    @property
    def pl(self):
        return self.models.pipeline if self.models else None

    def load(self, models: Optional[LoadedModels] = None) -> Dict:
        """load every model ONCE (load_models), or use ones already loaded -> a summary"""
        t0 = time.time()
        self.models = models or load_models(self.model_name, self.cfg, log=lambda s: None)
        lm = self.models
        info = {"model": lm.name, "detectors": lm.detectors, "steps": lm.steps, "model_fps": lm.fps,
                "action_model": lm.action_model is not None, "hand_model": lm.hand_model is not None,
                "load_seconds": round(time.time() - t0, 1), "seconds_per_model": lm.seconds}
        self.log(f"[load] {lm.name}: " + ", ".join(f"{k} {v:.2f}s" for k, v in lm.seconds.items()))
        return info

    def close(self) -> None:
        """unload the detectors (frees the GPU)"""
        unload_models(self.models)
        self.models = None

    def stop(self) -> None:
        """end the current run (a camera session ends with its verdict)"""
        self._stop.set()

    # ── one run ──
    def run(self, video: str, on_frame: Optional[Callable[[FrameInfo], None]] = None,
            on_step: Optional[Callable[[StepEvent], None]] = None, stop_event: Optional[threading.Event] = None,
            real_time: bool = False, save: bool = True, out_dir: Optional[str] = None) -> InferenceResult:
        """
        video       a file path, a camera index ("0") or a stream URL
        on_frame    called for EVERY frame (FrameInfo: rgb, detections, timings, fps)
        on_step     called when a step is done / starts / is taken back (StepEvent)
        stop_event  set it, or call stop(), to end the run
        real_time   play a FILE at its own speed (a camera is real time anyway)
        save        write the report, result.json, frame_times.csv, step_times.csv, relations.json
        """
        if not self.loaded:
            self.load()
        if not self._busy.acquire(blocking=False):
            raise RuntimeError("this engine is already running - one run at a time")
        try:
            return self._run(str(video).strip(), on_frame, on_step, stop_event or self._stop, real_time, save,
                             out_dir)
        finally:
            self._busy.release()

    def _new_run_folder(self, name: str, out_dir: Optional[str]) -> Path:
        if out_dir:
            return Path(out_dir)
        now = datetime.now()
        return Path(self.cfg.output_dir) / now.strftime("%Y-%m-%d") / name / now.strftime("%H-%M-%S_infer")

    def _start_live(self, video: str):
        """the engine's per-frame inference on this video (the loaded detectors and step TCN are reused)"""
        pl = self.pl
        pl.dets, pl.frames, pl.scenery = {}, [], []
        pl._cache.clear()
        return LiveInference(pl, self.model, video, fps=self.models.fps, log=lambda s: None,
                             all_frames=self.cfg.all_frames, step_tcn=self.models.step_tcn)

    def _run(self, video, on_frame, on_step, stop_event, real_time, save, out_dir) -> InferenceResult:
        self._stop.clear()
        camera = video.isdigit() or "://" in video
        if not camera and not Path(video).is_file():
            raise FileNotFoundError(video)
        name = f"camera_{video}" if video.isdigit() else ("stream" if camera else Path(video).stem)
        out = self._new_run_folder(name, out_dir)
        if save:
            out.mkdir(parents=True, exist_ok=True)
        self.pl.out = out
        live = self._start_live(video)
        tracker = self.tracker = StepTracker(self.model, video, self.cfg.action_check_every_s, on_step=on_step,
                                             log=self.log)
        self.log(f"[run] {video}  (every frame read + detected; step model at {live.run_fps:.2f} fps)")
        frames: List[Dict] = []
        walls: List[float] = []
        stopped = False
        t_start, t_prev = time.time(), time.perf_counter()
        while True:
            if stop_event.is_set():
                stopped = True
                live.release()
                break
            r = live.step()                               # read + detect (+ features + TCN on model frames)
            if r is None:
                break
            check_ms = tracker.update(live, r)            # the steps, live
            slept = 0.0
            if real_time and not camera and r["t"] > time.time() - t_start:
                slept = r["t"] - (time.time() - t_start)
                time.sleep(slept)
            now = time.perf_counter()
            frame_ms = 1000 * (now - t_prev - slept)
            t_prev = now
            walls = (walls + [time.time()])[-30:]
            fps = (len(walls) - 1) / (walls[-1] - walls[0]) if len(walls) > 1 and walls[-1] > walls[0] else 0.0
            rec = frame_record(len(frames) + 1, r, live, frame_ms, check_ms, camera)
            frames.append(rec)
            if on_frame is not None:
                try:
                    on_frame(FrameInfo(index=rec["frame"], video_frame=rec["video_frame"], t=float(r["t"]),
                                       rgb=r["rgb"], detections=r["dets"], model_frame=bool(rec["model_frame"]),
                                       frame_ms=frame_ms, stage_ms=stage_ms(rec), fps=fps,
                                       keeps_up=(r.get("speed") or {}).get("keeps_up")))
                except Exception as e:
                    self.log(f"  (on_frame callback failed: {type(e).__name__}: {e})")
        return self._finish(video, live, tracker, frames, time.time() - t_start, camera, out, save, stopped)

    def _finish(self, video, live, tracker, frames, wall, camera, out, save, stopped) -> InferenceResult:
        """the whole video's verdict + the result object + the files"""
        final = live.finish()
        result, acts = final, None
        if tracker.matcher:                                # the action model decides (as the app)
            acts, result = infer(self.model, video, live.record(final), log=lambda s: None,
                                    signals=live.live_signals() if camera else None)
        steps = step_results(self.model, result, acts, tracker)
        report = "\n".join(live_lines(self.model, video, live.state(), live.t[-1] if live.t else 0.0,
                                               live.duration, final=result, speed=live.speed()))
        report += "\n\n" + "\n".join(frame_summary(frames, wall))
        report += "\n\n" + "\n".join(step_table(steps, pickup=acts is None))
        seq = [s["step"] for s in sorted((s for s in result.get("steps") or [] if s.get("t") is not None),
                                         key=lambda s: s["t"])]
        res = InferenceResult(video=video, model=self.model_name, verdict=str(result.get("status")), sequence=seq,
                              steps=steps, anomalies=[a["text"] for a in result.get("anomalies") or []],
                              frames_read=len(frames), frames_to_model=sum(f["model_frame"] for f in frames),
                              seconds=round(wall, 1), fps=round(len(frames) / max(wall, 1e-6), 1),
                              frame_ms=stats(f["frame_ms"] for f in frames), report=report,
                              out_dir=str(out) if save else None, stopped=stopped)
        if save:
            (out / "live_result.txt").write_text(report, encoding="utf-8")
            (out / "result.json").write_text(json.dumps(asdict(res), indent=1, default=str), encoding="utf-8")
            write_csv(out / "frame_times.csv", frames)
            write_csv(out / "step_times.csv", steps)
            if result.get("relations"):
                (out / "relations.json").write_text(json.dumps(result["relations"], indent=1, default=str),
                                                    encoding="utf-8")
        self.log(f"[run] {res.verdict}  ({res.frames_read} frames, {res.fps} fps)" + (f"  -> {out}" if save else ""))
        return res




# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sop_monitor.__init__   (from sop_monitor\__init__.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""
sop_monitor - SOP sequence INFERENCE with trained models (no training code in this folder).

    import sys; sys.path.insert(0, r"<path>\\inference_code_only")
    from sop_monitor import InferenceEngine, list_models

    engine = InferenceEngine()          # the ACTIVE model, or InferenceEngine("model name")
    engine.load()                       # ONCE
    res = engine.run("test.mp4", on_step=print)
    engine.run("next.mp4")              # the models stay loaded
    engine.close()

See ../README.md.
"""

import json
from typing import Dict, List, Optional



def list_models(cfg: Optional[AppConfig] = None) -> List[Dict]:
    """the trained models, newest first: name, trained_at, video, detectors, fps, self_check, active"""
    cfg = bootstrap(cfg)
    try:
        active = get_active_model(cfg)
    except FileNotFoundError:
        active = None
    out = []
    d = cfg.trained_dir
    for p in sorted(d.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True) if d.is_dir() else []:
        f = p / "model.json"
        if f.is_file():
            m = json.loads(f.read_text(encoding="utf-8"))
            out.append({"name": p.name, "trained_at": m.get("trained_at"), "video": m.get("trained_on"),
                        "detectors": m.get("detectors"), "fps": m.get("fps"),
                        "self_check": m.get("action_self_check", m.get("self_check")), "active": p.name == active})
    return out




# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: main script   (from infer.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""
SOP Sequence Inference
======================
Checks a video (or a live camera) against a TRAINED SOP model: every step done, in the trained order, none undone.
The trained model and its detectors are loaded ONCE; every video given is then run with them.

  python infer.py --list                                the trained models (* = active)
  python infer.py --video test.mp4                      the ACTIVE model on one video
  python infer.py --video a.mp4 b.mp4 --save_video      several videos, annotated videos saved
  python infer.py --video 0                             camera 0 (Q in the window ends the session)
  python infer.py --model Good_18Nov_1434 --video x.mp4 a specific trained model

Printed / written per run (outputs/<date>/<video>/<time>_infer/):
  per FRAME   ms per frame + per stage (detect | features | tcn | step check), FPS     -> frame_times.csv
  per STEP    started, done, took, confirmed at, detected %                            -> step_times.csv
  the report (live_result.txt), result.json, relations.json, the annotated video (--save_video),
  and one row per video in outputs/inspection_log.xlsx (.csv without openpyxl)

Architecture
  1. Configuration      PipelineConfig (paths / device / inspection / ui), YAML or CLI
  2. Constants          states, colours
  3. Model loading      GPU diagnostics; load_all_models(): each model by its own loader (see MODEL LOADING:
                        load_trained_model / load_detectors / load_step_tcn / load_action_model / load_hand_model /
                        warm_up), ONCE
  4. Model runners      per frame: detect -> step TCN (at the model's fps) -> step check  [InferenceEngine.run]
  5. HUD                production status bar, side panels (FPS, ms, stages, steps), final verdict card
  6. Logging            inspection_log.xlsx (or .csv) - one row per video
  7. Processing loop    process_video(): window, keys, annotated video
  8. Integration API    SOPMonitor: start() / process_frame() / stop(), run_video(), draw()
  9. Entry point        CLI

Keys in the preview window:  Q / Esc = stop   F = full screen   P = pause
"""

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np




# ══════════════════════════════════════════════════════════════════════════════
#  1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PathsConfig:
    """the model, the input(s), the output root"""
    model: str = ""                              # "" = the active model (trained last)
    videos: List[str] = field(default_factory=list)   # files, a camera index ("0") or stream URLs
    out_base: str = str(APP_ROOT / "outputs")


@dataclass
class DeviceConfig:
    require_gpu: bool = False                    # True: stop when CUDA is not available


@dataclass
class InspectionConfig:
    all_frames: bool = True                      # read + detect EVERY frame (the step model keeps its fps)
    check_every_s: float = 0.5                   # each step judged this often (seconds of video)
    real_time: bool = False                      # play files at their own speed


@dataclass
class UIConfig:
    show_preview: bool = True
    save_video: bool = False
    excel_log: bool = True                       # inspection_log.xlsx in the output root
    frame_ms_warn: float = 150.0                 # frames slower than this are printed
    print_frames: bool = False                   # one console line per frame


@dataclass
class PipelineConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    inspection: InspectionConfig = field(default_factory=InspectionConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """a YAML file with the sections above (paths / device / inspection / ui)"""
        import yaml
        cfg = cls()
        with open(path, "r", encoding="utf-8") as fh:
            cfg.update_from_dict(yaml.safe_load(fh) or {})
        return cfg

    def update_from_dict(self, d: Dict[str, Any]) -> None:
        for section, values in (d or {}).items():
            target = getattr(self, section, None)
            if target is not None and isinstance(values, dict):
                for k, v in values.items():
                    if hasattr(target, k):
                        setattr(target, k, v)

    def app_config(self) -> AppConfig:
        """the library's settings from this config"""
        return AppConfig(output_dir=Path(self.paths.out_base), all_frames=self.inspection.all_frames,
                         action_check_every_s=self.inspection.check_every_s)


# ══════════════════════════════════════════════════════════════════════════════
#  2. CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

STATE_INSPECTING = "INSPECTING"
STATE_STEP = "STEP IN PROGRESS"
STATE_OUT_OF_ORDER = "OUT OF ORDER"
STATE_NORMAL = "NORMAL"
STATE_ANOMALY = "ANOMALY"
STATE_UNCERTAIN = "UNCERTAIN"

# BGR
COL = {"ok": (90, 210, 90), "bad": (70, 70, 235), "warn": (60, 190, 240), "muted": (150, 150, 150),
       "fg": (235, 235, 235), "panel": (14, 14, 14), "border": (80, 80, 80)}
STATE_COL = {STATE_INSPECTING: COL["fg"], STATE_STEP: COL["warn"], STATE_OUT_OF_ORDER: COL["bad"],
             STATE_NORMAL: COL["ok"], STATE_ANOMALY: COL["bad"], STATE_UNCERTAIN: COL["warn"]}


# ══════════════════════════════════════════════════════════════════════════════
#  3. HARDWARE & MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def resolve_device(require_gpu: bool = False) -> str:
    """print the GPU / CUDA state -> "cuda:0" or "cpu" (the detectors run in half precision on CUDA)"""
    import torch
    print("=" * 78 + "\n  GPU / DEVICE\n" + "=" * 78)
    print(f"  torch {torch.__version__}   CUDA {torch.version.cuda}   available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        if require_gpu:
            raise RuntimeError("CUDA is not available (run without --require_gpu to allow the CPU)")
        print("  [WARN] running on the CPU - slow\n" + "=" * 78)
        return "cpu"
    free, total = torch.cuda.mem_get_info(0)
    print(f"  GPU 0: {torch.cuda.get_device_name(0)}   memory free {free / 2**30:.1f} / {total / 2**30:.1f} GB")
    print("=" * 78)
    return "cuda:0"


def check_model_files(name: str, app_cfg: AppConfig) -> None:
    """every file the model needs is there (else FileNotFoundError, listing what is missing)"""
    chk = check_models(name, app_cfg)
    for p in chk["problems"]:
        print(f"[CHECK] {p}")
    if not chk["ok"]:
        raise FileNotFoundError(f"model {name} is incomplete (see above)")


def load_all_models(name: str, app_cfg: AppConfig) -> LoadedModels:
    """each model by its own loader (see MODEL LOADING), printed as it loads -> LoadedModels (loaded ONCE)"""
    secs: Dict[str, float] = {}

    def step(label, key, fn, *a):
        t0 = time.time()
        out = fn(*a)
        secs[key] = round(time.time() - t0, 2)
        print(f"[LOAD] {label:<16} {secs[key]:6.2f} s")
        return out

    model = step("trained model", "trained_model", load_trained_model, name, app_cfg)
    print(f"[MODEL] {name}")
    print(f"[MODEL] steps: " + "  ->  ".join(f"{s['id']}. {s.get('action_query')}" for s in model["spec"]["steps"]))
    pl = step("detectors", "detectors", load_detectors, model, app_cfg)
    for m_, b in getattr(pl.bank, "backends", []) or []:
        print(f"         - {m_['name']:<38} {Path(str(m_.get('path'))).name}")
    tcn = step("step TCN", "step_tcn", load_step_tcn, model)
    act = step("action model", "action_model", load_action_model, model)
    if act is None:
        print("         - none: the pickup model decides")
    hand = step("hand model", "hand_model", load_hand_model)
    if hand is None:
        print("         - not available (mediapipe / weights/hand_landmarker.task): less placement evidence")
    secs["warm_up"] = round(warm_up(pl, model, hand), 2)
    print(f"[LOAD] {'warm-up':<16} {secs['warm_up']:6.2f} s   (step model at {float(model.get('fps') or 3):g} fps)")
    return LoadedModels(name=name, model=model, pipeline=pl, step_tcn=tcn, action_model=act, hand_model=hand,
                        seconds=secs)


def load_engine(cfg: PipelineConfig) -> InferenceEngine:
    """the trained model: files checked, every model loaded ONCE and warmed up -> the engine for all the videos"""
    app_cfg = cfg.app_config()
    name = cfg.paths.model or get_active_model(app_cfg)
    check_model_files(name, app_cfg)
    engine = InferenceEngine(name, app_cfg, log=lambda s: None)
    engine.load(load_all_models(name, app_cfg))
    return engine


# ══════════════════════════════════════════════════════════════════════════════
#  4. MODEL RUNNERS
#     The per-frame chain runs inside sop_monitor.InferenceEngine.run() (one call per video):
#       run_detectors   every frame: the YOLO detector(s), on their own thread        (sopmon.temporal.LiveInference)
#       run_step_model  every Nth frame: features -> step TCN (pickup / place)         (sopmon.temporal.FeatureStream)
#       run_step_check  every check_every_s of video: each step's action + state      (sop_monitor.step_tracker)
#     Their per-frame times arrive in FrameInfo.stage_ms (detect / features / tcn / step_check).
# ══════════════════════════════════════════════════════════════════════════════

def live_state(tracker) -> str:
    """the status bar's state from the live step list"""
    if tracker is None:
        return STATE_INSPECTING
    if any(not ok for _, _, ok, _ in tracker.shown):
        return STATE_OUT_OF_ORDER
    if tracker.doing:
        return STATE_STEP
    return STATE_INSPECTING


# ══════════════════════════════════════════════════════════════════════════════
#  5. HUD & OVERLAYS
# ══════════════════════════════════════════════════════════════════════════════

def _put(img, text, x, y, fs, col, th=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, col, th, cv2.LINE_AA)


def _panel(img, x1, y1, x2, y2, border):
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), COL["panel"], -1)
    cv2.addWeighted(ov, 0.75, img, 0.25, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), border, 1)


def draw_status_bar(img, model: str, state: str, done: int, total: int, verdict: Optional[str] = None):
    """the top bar: MODEL | STATUS | STEPS done/total"""
    H, W = img.shape[:2]
    S = W / 1280.0
    fs = max(0.55, 0.7 * S)
    status = verdict or state
    segs = [(f"MODEL: {model[:34]}", COL["fg"]), (f"STATUS: {status}", STATE_COL.get(status, COL["fg"])),
            (f"STEPS: {done}/{total}", COL["ok"] if done == total else COL["fg"])]
    widths = [cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0][0] for s, _ in segs]
    gap, pad, h = int(30 * S), int(16 * S), int(40 * S)
    tw = sum(widths) + gap * (len(segs) - 1) + 2 * pad
    x = max(8, (W - tw) // 2)
    _panel(img, x, 8, x + tw, 8 + h, STATE_COL.get(status, COL["border"]))
    cx = x + pad
    for (s, c), w in zip(segs, widths):
        _put(img, s, cx, 8 + h - int(13 * S), fs, c, 2)
        cx += w + gap


def draw_side_hud(img, fi: FrameInfo, tracker, ms_avg: float, warn_ms: float):
    """left: FPS, frame, ms (current / average), ms per stage;  right: the steps so far"""
    H, W = img.shape[:2]
    S = W / 1280.0
    fs, row, top = max(0.45, 0.52 * S), max(22, int(25 * S)), int(60 * S)
    st = fi.stage_ms
    left = [(f"FPS    {fi.fps:5.1f}", COL["ok"] if fi.keeps_up is not False else COL["bad"]),
            (f"FRAME  {fi.video_frame if fi.video_frame is not None else fi.index:06d}"
             + ("  [model]" if fi.model_frame else ""), COL["fg"]),
            (f"MS     {fi.frame_ms:5.0f}  (avg {ms_avg:5.0f})", COL["bad"] if ms_avg >= warn_ms else COL["fg"]),
            (f"       detect {st['detect']:.0f} | features {st['features']:.0f} | tcn {st['tcn']:.0f}", COL["muted"]),
            (f"       step check {np.mean(tracker.act_ms):.0f} ms" if tracker is not None and tracker.act_ms
             else "       step check -", COL["muted"]),
            (f"T      {fi.t:6.1f} s", COL["fg"])]
    _panel(img, 8, top, int(345 * S), top + 12 + row * len(left), COL["border"])
    for i, (s, c) in enumerate(left):
        _put(img, s, 18, top + row * (i + 1), fs, c, 2 if i == 0 else 1)
    if tracker is None:
        return
    short = lambda sid: tracker.lab[sid] if len(tracker.lab[sid]) <= 26 else tracker.lab[sid][:25] + "."
    lines = [("SEQUENCE", COL["muted"])]
    for sid, t, ok, det in tracker.shown:
        lines.append((f"{'OK ' if ok else 'BAD'} {sid}. {short(sid):<26} {t:5.1f}s"
                      + (f" {det:3d}%" if det is not None else ""), COL["ok"] if ok else COL["bad"]))
    for sid, t in sorted(tracker.doing.items(), key=lambda kv: kv[1]):
        lines.append((f"... {sid}. {short(sid):<26} doing", COL["warn"]))
    for sid, _ in tracker.retracted:
        lines.append((f"<-  {sid}. {short(sid):<26} undone", COL["warn"]))
    rw = min(W - int(370 * S), int(max(len(s) for s, _ in lines) * 11.0 * S) + 24)
    _panel(img, W - rw - 8, top, W - 8, top + 12 + row * len(lines), COL["border"])
    for i, (s, c) in enumerate(lines):
        _put(img, s, W - rw, top + row * (i + 1), fs, c)


def draw_final_verdict_overlay(img, res: InferenceResult):
    """the end-of-video card: the verdict, the sequence, the findings, the run's numbers"""
    H, W = img.shape[:2]
    bad = res.verdict == "ANOMALY"
    tint = np.zeros_like(img)
    tint[:] = (0, 0, 100) if bad else (0, 70, 10)
    out = cv2.addWeighted(img, 0.40, tint, 0.60, 0)
    col = COL["bad"] if bad else COL["ok"] if res.verdict == "NORMAL" else COL["warn"]
    fs = max(1.4, min(W, H) / 260.0)
    label = f"FINAL: {res.verdict}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs, 3)
    cv2.putText(out, label, ((W - tw) // 2, H // 4), cv2.FONT_HERSHEY_DUPLEX, fs, col, 3, cv2.LINE_AA)
    lines = [f"Sequence : {' -> '.join(map(str, res.sequence)) or '-'}"]
    lines += [f"  {s.step}. {s.label[:40]:<40} {s.status:<13} "
              f"{'-' if s.done_s is None else f'{s.done_s:.1f}s'}" for s in res.steps]
    lines += [f"! {a[:90]}" for a in res.anomalies[:3]]
    lines += [f"Frames {res.frames_read} ({res.frames_to_model} to the step model)   {res.fps} fps   "
              f"{res.frame_ms['mean']:.0f} ms/frame"]
    fs2, rh = max(0.5, fs * 0.33), max(22, int(H * 0.036))
    y = H // 4 + th + 20
    bw = int(W * 0.8)
    _panel(out, (W - bw) // 2 - 10, y - rh, (W + bw) // 2 + 10, y + rh * len(lines), col)
    for i, ln in enumerate(lines):
        _put(out, ln, (W - bw) // 2, y + rh * i, fs2, COL["bad"] if ln.startswith("!") else COL["fg"])
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  6. LOGGING (one row per video)
# ══════════════════════════════════════════════════════════════════════════════

LOG_COLUMNS = ["Sr No.", "Timestamp", "Video", "Model", "Verdict", "Sequence", "Findings", "Frames", "FPS",
               "Output folder"]


def log_row(res: InferenceResult) -> Dict[str, Any]:
    return {"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Video": Path(res.video).name,
            "Model": res.model, "Verdict": res.verdict, "Sequence": " > ".join(map(str, res.sequence)),
            "Findings": " | ".join(res.anomalies) or "-", "Frames": res.frames_read, "FPS": res.fps,
            "Output folder": res.out_dir or "-"}


def append_to_log(res: InferenceResult, out_base: str) -> Path:
    """inspection_log.xlsx (openpyxl) or inspection_log.csv -> the file written"""
    Path(out_base).mkdir(parents=True, exist_ok=True)
    row = log_row(res)
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill
        path = Path(out_base) / "inspection_log.xlsx"
        if path.exists():
            wb = openpyxl.load_workbook(path)
            ws = wb.active
        else:
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Inspection Log"
            ws.append(LOG_COLUMNS)
            for c in ws[1]:
                c.font = Font(bold=True, color="FFFFFF")
                c.fill = PatternFill("solid", start_color="1F4E78")
        ws.append([ws.max_row] + [row[k] for k in LOG_COLUMNS[1:]])
        fill = {"NORMAL": "C6EFCE", "ANOMALY": "FFC7CE"}.get(res.verdict, "FFEB9C")
        ws.cell(row=ws.max_row, column=LOG_COLUMNS.index("Verdict") + 1).fill = PatternFill("solid", start_color=fill)
        for _ in range(5):
            try:
                wb.save(path)
                return path
            except PermissionError:                      # open in Excel: retry
                time.sleep(2)
        raise PermissionError(f"{path} is locked")
    except ImportError:
        path = Path(out_base) / "inspection_log.csv"
        new = not path.exists()
        n = 1 if new else sum(1 for _ in open(path, encoding="utf-8"))
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(LOG_COLUMNS)
            w.writerow([n] + [row[k] for k in LOG_COLUMNS[1:]])
        return path


# ══════════════════════════════════════════════════════════════════════════════
#  7. PROCESSING LOOP
# ══════════════════════════════════════════════════════════════════════════════

class Display:
    """the preview window + the annotated video, fed from the engine's on_frame callback"""

    def __init__(self, engine: InferenceEngine, cfg: PipelineConfig, title: str, video_path: Optional[Path],
                 fps: float = 25.0):
        self.engine, self.cfg, self.title = engine, cfg, title
        self.video_path, self.writer, self.fps = video_path, None, fps
        self.ms: List[float] = []
        self.last = None
        self.paused = self.fullscreen = False
        if cfg.ui.show_preview:
            try:
                cv2.namedWindow(title, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(title, 1280, 720)
            except cv2.error:
                cfg.ui.show_preview = False

    def on_frame(self, fi: FrameInfo) -> None:
        self.ms = (self.ms + [fi.frame_ms])[-30:]
        if self.cfg.ui.print_frames or fi.frame_ms > self.cfg.ui.frame_ms_warn:
            print(f"  frame {fi.index:5d}  t {fi.t:6.2f}s  {fi.frame_ms:6.1f} ms  "
                  + " ".join(f"{k} {v:.0f}" for k, v in fi.stage_ms.items())
                  + ("  [model]" if fi.model_frame else "") + ("  SLOW" if fi.frame_ms > self.cfg.ui.frame_ms_warn
                                                              else ""))
        if not (self.cfg.ui.show_preview or self.cfg.ui.save_video):
            return
        tr = self.engine.tracker
        img = cv2.cvtColor(fi.rgb, cv2.COLOR_RGB2BGR)
        draw_status_bar(img, self.engine.model_name, live_state(tr), len(tr.shown) if tr else 0,
                        len(tr.ids) if tr else 0)
        draw_side_hud(img, fi, tr, float(np.mean(self.ms)), self.cfg.ui.frame_ms_warn)
        self.last = img
        if self.cfg.ui.save_video and self.video_path is not None:
            if self.writer is None:
                self.video_path.parent.mkdir(parents=True, exist_ok=True)
                self.writer = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                                              (img.shape[1], img.shape[0]))
            self.writer.write(img)
        if self.cfg.ui.show_preview:
            cv2.imshow(self.title, img)
            self._keys()

    def _keys(self) -> None:
        k = cv2.waitKey(1) & 0xFF
        while self.paused and k not in (ord("p"), ord("q"), 27):
            k = cv2.waitKey(50) & 0xFF
        if k == ord("p"):
            self.paused = not self.paused
        elif k == ord("f"):
            self.fullscreen = not self.fullscreen
            cv2.setWindowProperty(self.title, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
        elif k in (ord("q"), 27):
            print("[INFO] stopped from the window")
            self.engine.stop()

    def finish(self, res: InferenceResult) -> None:
        """the final verdict card: 2 s at the end of the saved video, shown until a key is pressed"""
        if self.last is None:
            return
        card = draw_final_verdict_overlay(self.last, res)
        if self.writer is not None:
            for _ in range(50):
                self.writer.write(card)
            self.writer.release()
            print(f"[SAVED] {self.video_path}")
        if self.cfg.ui.show_preview:
            cv2.imshow(self.title, card)
            print("[INFO] press any key in the window to continue")
            cv2.waitKey(0)


def on_step(ev: StepEvent) -> None:
    """step events -> the console"""
    if ev.kind == "done":
        print(f"  [{ev.t:6.1f}s] STEP {ev.step} {ev.label}: " + ("done" if ev.in_order else f"OUT OF ORDER ({ev.note})")
              + (f"  detected {ev.detected}%" if ev.detected is not None else ""))
    elif ev.kind == "taken_back":
        print(f"  [{ev.t:6.1f}s] STEP {ev.step} {ev.label}: NOT done after all - {ev.note}")


def video_fps(video: str) -> float:
    """the source's frames per second (for the annotated video)"""
    if video.isdigit() or "://" in video:
        return 25.0
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return float(fps) if 1.0 <= fps <= 240.0 else 25.0


def process_video(engine: InferenceEngine, video: str, cfg: PipelineConfig) -> InferenceResult:
    """one video / camera session: run, show, save, log -> the result"""
    name = f"camera_{video}" if video.isdigit() else Path(video).stem
    run_dir = Path(cfg.paths.out_base) / datetime.now().strftime("%Y-%m-%d") / name / \
        datetime.now().strftime("%H-%M-%S_infer")
    title = f"SOP inference | {name} | {engine.model_name}  (Q = stop, F = full screen, P = pause)"
    disp = Display(engine, cfg, title, run_dir / f"{name}_annotated.mp4" if cfg.ui.save_video else None,
                   fps=video_fps(video) if cfg.inspection.all_frames else float(engine.model.get("fps") or 3.0))
    print(f"\n[INFO] {video}")
    res = engine.run(video, on_frame=disp.on_frame, on_step=on_step, real_time=cfg.inspection.real_time,
                     out_dir=str(run_dir))
    print("\n" + res.report)
    disp.finish(res)
    if cfg.ui.excel_log:
        print(f"[LOG] {append_to_log(res, cfg.paths.out_base)}")
    return res


def run_inference(cfg: PipelineConfig) -> List[InferenceResult]:
    """load once, process every input -> the results"""
    resolve_device(cfg.device.require_gpu)
    engine = load_engine(cfg)
    results = []
    try:
        for v in cfg.paths.videos:
            results.append(process_video(engine, str(v), cfg))
    finally:
        engine.close()
        if cfg.ui.show_preview:
            cv2.destroyAllWindows()
    print("\n" + "=" * 62 + "\n  SUMMARY\n" + "=" * 62)
    for r in results:
        print(f"  {r.verdict:<10} {Path(r.video).name:<40} {' > '.join(map(str, r.sequence))}")
    print("=" * 62)
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  8. INTEGRATION API - SOPMonitor: THE ONE CLASS A HOST APPLICATION USES
#
#  A) Your application has the frames (its own camera / video loop):
#
#       import infer
#       mon = infer.SOPMonitor()                      # the active model (or SOPMonitor("Good_18Nov_1434")); loads ONCE
#       mon.start()                                   # a new inspection (one product / one cycle)
#       while running:
#           frame = grab_frame()                      # BGR numpy array (as OpenCV gives it)
#           st = mon.process_frame(frame, t)          # t = seconds since start (None = wall clock)
#           st.state            "INSPECTING" | "STEP IN PROGRESS" | "OUT OF ORDER"
#           st.steps_done       [{"step", "label", "t", "in_order", "detected"}]  in the order they were done
#           st.doing            steps happening right now;   st.events = what changed on THIS frame (StepEvent)
#           st.detections, st.frame_ms, st.stage_ms, st.fps
#           show(mon.draw(frame, st))                 # optional: the HUD drawn on the frame
#       res = mon.stop()                              # the verdict: res.verdict, res.sequence, res.steps, res.anomalies
#       mon.start() ...                               # the next product - same loaded models
#       mon.close()
#
#  B) A video file / camera index / stream URL that the monitor reads itself:
#       res = mon.run_video("test.mp4", on_step=..., on_frame=...)
#
#  Errors: FileNotFoundError (a model file / video missing), RuntimeError (no frames / used before start()).
#  Threads: call one monitor from one thread. Keep one SOPMonitor per model for the life of the application.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FrameStatus:
    """what SOPMonitor.process_frame() returns for every frame"""
    index: int                          # frames processed in this inspection (1, 2, ...)
    t: float                            # seconds since start()
    state: str                          # INSPECTING / STEP IN PROGRESS / OUT OF ORDER
    steps_done: List[Dict[str, Any]]    # [{"step", "label", "t", "in_order", "detected"}] in the order done
    doing: List[int]                    # steps happening now (not confirmed yet)
    taken_back: List[int]               # steps shown done earlier and then undone
    events: List[StepEvent]             # what changed on THIS frame
    detections: List[Dict[str, Any]]    # [{"name", "conf", "box": [x1, y1, x2, y2] in the ANALYSED frame, "model"}]
    frame_size: Tuple[int, int]         # (w, h) of the analysed frame (the boxes' coordinates; <= 1280 px)
    model_frame: bool                   # this frame also went to the step model
    frame_ms: float                     # processing time of this frame
    stage_ms: Dict[str, float]          # detect / features / tcn / step_check
    fps: float                          # frames processed per second (the last 30)

    @property
    def n_done(self) -> int:
        return len(self.steps_done)


class SOPMonitor:
    """Load a trained SOP model once; check inspections frame by frame (process_frame) or whole videos (run_video).
    See the comment block above for the usage."""

    def __init__(self, model: str = "", out_dir: Optional[str] = None, save_results: bool = True,
                 check_every_s: float = 0.5, log: Optional[Callable[[str], None]] = None):
        """model: a trained model's name ("" = the active one). out_dir: where results go (default outputs/).
        save_results: write the report / CSVs / result.json per inspection. log: a function for messages."""
        self.log = log or (lambda s: None)
        self.save_results = save_results
        self.engine = create_engine(model, out_dir=out_dir, check_every_s=check_every_s, log=self.log)
        self._reset()

    # ── about the model ──
    @property
    def model_name(self) -> str:
        return self.engine.model_name

    @property
    def steps(self) -> List[Dict[str, Any]]:
        """the trained sequence: [{"step", "label"}]"""
        return [{"step": s["id"], "label": pipeline_step_label(s)} for s in self.engine.model["spec"]["steps"]]

    @property
    def detectors(self) -> List[str]:
        return self.engine.models.detectors

    @property
    def load_seconds(self) -> Dict[str, float]:
        """how long each model took to load"""
        return dict(self.engine.models.seconds)

    # ── A) frames from the host application ──
    def start(self, name: str = "inspection") -> None:
        """begin a new inspection (the previous one, if not stopped, is dropped)"""
        if self._live is not None:
            self._live.release()
        self._reset()
        self._name = name
        self._started = True

    def process_frame(self, frame_bgr: np.ndarray, t: Optional[float] = None) -> FrameStatus:
        """one frame (BGR uint8, any size) -> its FrameStatus. t: seconds since start(); None = the wall clock.
        For a video FILE give t = frame_number / video_fps (frames may then come faster than real time)."""
        if not self._started:
            raise RuntimeError("call start() before process_frame()")
        if frame_bgr is None or getattr(frame_bgr, "ndim", 0) != 3:
            raise ValueError("process_frame needs a BGR image (H, W, 3)")
        tp = time.perf_counter()
        if self._wall0 is None:
            self._wall0 = time.time()
        ts = (time.time() - self._wall0) if t is None else float(t)
        eng = self.engine
        if self._live is None:                              # the first frame: the per-inspection state
            self._t0 = ts
            pl = eng.pl
            pl.dets, pl.frames, pl.scenery = {}, [], []
            pl._cache.clear()
            self._feed = FrameFeed()
            self._feed.put(frame_bgr, 0.0)
            self._live = LiveInference(pl, eng.model, self._feed, fps=eng.models.fps, log=self.log,
                                       all_frames=True, step_tcn=eng.models.step_tcn)
            self._tracker = eng.tracker = StepTracker(eng.model, self._name, eng.cfg.action_check_every_s,
                                                      on_step=self._events.append, log=self.log)
        else:
            self._feed.put(frame_bgr, max(0.0, ts - self._t0))
        r = self._live.step()
        if r is None:
            raise RuntimeError("the frame could not be processed")
        check_ms = self._tracker.update(self._live, r)
        frame_ms = 1000.0 * (time.perf_counter() - tp)
        self._walls = (self._walls + [time.time()])[-30:]
        w = self._walls
        fps = (len(w) - 1) / (w[-1] - w[0]) if len(w) > 1 and w[-1] > w[0] else 0.0
        rec = frame_record(len(self._frames) + 1, r, self._live, frame_ms, check_ms, True)
        self._frames.append(rec)
        tr = self._tracker
        events, self._events[:] = list(self._events), []
        self._last = FrameInfo(index=rec["frame"], video_frame=None, t=float(r["t"]), rgb=r["rgb"],
                               detections=r["dets"], model_frame=bool(rec["model_frame"]), frame_ms=frame_ms,
                               stage_ms=stage_ms(rec), fps=fps, keeps_up=(r.get("speed") or {}).get("keeps_up"))
        self._ms = (self._ms + [frame_ms])[-30:]
        return FrameStatus(index=rec["frame"], t=float(r["t"]), state=live_state(tr),
                           steps_done=[{"step": s, "label": tr.lab[s], "t": round(float(tt), 2), "in_order": ok,
                                        "detected": det} for s, tt, ok, det in tr.shown],
                           doing=sorted(tr.doing), taken_back=[s for s, _ in tr.retracted], events=events,
                           detections=r["dets"], frame_size=(r["rgb"].shape[1], r["rgb"].shape[0]),
                           model_frame=bool(rec["model_frame"]), frame_ms=frame_ms, stage_ms=stage_ms(rec), fps=fps)

    def stop(self) -> InferenceResult:
        """end the inspection -> its verdict (the action model over the whole inspection); files saved if asked"""
        if self._live is None:
            raise RuntimeError("no frames were processed since start()")
        eng = self.engine
        out = eng._new_run_folder(self._name, None)
        if self.save_results:
            out.mkdir(parents=True, exist_ok=True)
        eng.pl.out = out
        res = eng._finish(self._name, self._live, self._tracker, self._frames, time.time() - (self._wall0 or 0),
                          True, out, self.save_results, False)
        self._live.release()
        self._reset()
        return res

    def draw(self, frame_bgr: np.ndarray, status: Optional[FrameStatus] = None) -> np.ndarray:
        """a copy of the frame with the HUD (status bar, FPS / ms, the steps so far) - for the host's display"""
        img = frame_bgr.copy()
        tr = self._tracker
        draw_status_bar(img, self.model_name, status.state if status else live_state(tr),
                        len(tr.shown) if tr else 0, len(tr.ids) if tr else len(self.steps))
        if self._last is not None:
            draw_side_hud(img, self._last, tr, float(np.mean(self._ms)) if self._ms else 0.0, 150.0)
        return img

    # ── B) a file / camera / stream read by the monitor ──
    def run_video(self, source: str, on_step: Optional[Callable[[StepEvent], None]] = None,
                  on_frame: Optional[Callable[[FrameInfo], None]] = None, real_time: bool = False,
                  stop_event=None) -> InferenceResult:
        """a whole video file, a camera index ("0") or a stream URL -> its verdict (blocks until the end)"""
        return self.engine.run(str(source), on_frame=on_frame, on_step=on_step, real_time=real_time,
                               save=self.save_results, stop_event=stop_event)

    def cancel(self) -> None:
        """end a running run_video() early (from another thread)"""
        self.engine.stop()

    # ── cleanup ──
    def close(self) -> None:
        """free the GPU"""
        if self._live is not None:
            self._live.release()
        self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _reset(self) -> None:
        self._live = self._tracker = self._feed = self._last = None
        self._frames, self._events, self._walls, self._ms = [], [], [], []
        self._wall0, self._t0, self._started, self._name = None, 0.0, False, "inspection"

def create_engine(model: str = "", out_dir: Optional[str] = None, all_frames: bool = True,
                  check_every_s: float = 0.5, log=lambda s: None) -> InferenceEngine:
    """an engine with EVERY model of the trained model loaded once (model "" = the active model)"""
    app_cfg = AppConfig(output_dir=Path(out_dir) if out_dir else None, all_frames=all_frames,
                        action_check_every_s=check_every_s)
    name = model or get_active_model(app_cfg)
    check_model_files(name, app_cfg)
    engine = InferenceEngine(name, app_cfg, log=log)
    engine.load(load_models(name, app_cfg, log=log))
    return engine


# ══════════════════════════════════════════════════════════════════════════════
#  9. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_cli_args() -> Tuple[PipelineConfig, argparse.Namespace]:
    ap = argparse.ArgumentParser(description="SOP sequence inference with a trained model")
    ap.add_argument("--config", help="YAML config (sections paths / device / inspection / ui)")
    ap.add_argument("--model", help="trained model name (default: the active model)")
    ap.add_argument("--video", nargs="+", help="video file(s), a camera index (0) or a stream URL")
    ap.add_argument("--out", help="output root (default: outputs/ in this folder)")
    ap.add_argument("--real_time", action="store_true", help="play files at their own speed")
    ap.add_argument("--no_preview", action="store_true", help="no window")
    ap.add_argument("--save_video", action="store_true", help="save the annotated video")
    ap.add_argument("--no_excel", action="store_true", help="no inspection_log")
    ap.add_argument("--print_frames", action="store_true", help="a console line per frame")
    ap.add_argument("--model_frames_only", action="store_true", help="read only the step model's frames (faster)")
    ap.add_argument("--require_gpu", action="store_true")
    ap.add_argument("--list", action="store_true", help="list the trained models and exit")
    a = ap.parse_args()
    cfg = PipelineConfig.from_yaml(a.config) if a.config else PipelineConfig()
    if a.model:
        cfg.paths.model = a.model
    if a.video:
        cfg.paths.videos = list(a.video)
    if a.out:
        cfg.paths.out_base = a.out
    cfg.inspection.real_time |= a.real_time
    cfg.inspection.all_frames &= not a.model_frames_only
    cfg.ui.show_preview &= not a.no_preview
    cfg.ui.save_video |= a.save_video
    cfg.ui.excel_log &= not a.no_excel
    cfg.ui.print_frames |= a.print_frames
    cfg.device.require_gpu |= a.require_gpu
    return cfg, a


def main() -> None:
    cfg, a = parse_cli_args()
    if a.list or not cfg.paths.videos:
        print("Trained models (* = active, used when --model is not given):")
        for m in list_models():
            print(f"  {'*' if m['active'] else ' '} {m['name']:<55} {m['trained_at'] or '':<17} "
                  f"self-check {m['self_check']}")
        if not a.list:
            print("\nGive --video <file | camera index> to run inference.")
        return
    run_inference(cfg)


if __name__ == "__main__":
    main()
