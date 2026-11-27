"""
sop_engine.py - the SOP monitor's TRAINING engine in ONE file (generated - do not edit; rebuild with
tools/build_training_engine.py from the app's sopmon/).

Used by train_sequence.py:  import sop_engine as E;  E.trained.save(...), E.pipeline.Pipeline(...), E.label_gui.auto_train(...)
The data folder: the SOP_MONITOR_HOME environment variable (train_sequence.py sets it to the folder above).

SECTIONS (search for "MODULE:")
  _home
  textmatch
  jsonparse
  models
  featstore
  mp_hands
  boundaries
  relations
  state_events
  evidence_engine
  places_spots
  pipeline
  trained
  object_actions
  temporal
  interaction_features
  state_prototypes
  step_lifecycle
  lstr
  workspots
  sop_events
  objsteps
  hands
  direction
  stepmodel
  seqdecode
  videotext
  vlm_client
  action_model
  pack_actions
  semantic_events
  boundary_learning
  auto_labels
  ref_match
  completion
  label_gui
"""
from __future__ import annotations


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon._home   (from sopmon\_home.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""the DATA folder (models.json, weights/, trained/ read; specs/, outputs/ written), decided once:
  1. the SOP_MONITOR_HOME environment variable, when set
  2. FOLDER mode: this file's own folder when models.json is next to it (the inference_code_only folder), else
     its PARENT folder when models.json is there (sequence_training_inference/inference/infer.py: data one level up)
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
    if (here.parent / "models.json").is_file() and (here.parent / "trained").is_dir():
        return here.parent                                     # inference/ next to the shared data folder
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
pose / classify, RT-DETR) and reads its class names. (The open-vocabulary detectors - YOLO-World, YOLOE,
Grounding DINO, OWLv2 - make detection datasets in the full application; this runs trained detectors.)

Each object the spec names goes to an OWN model whose class name matches it; every other object
goes to the enabled open-vocabulary detectors (they find any text prompt, and are given the visual
phrases from the prompt for it). All boxes come back as one list, de-duplicated across models.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np

pass

os.environ.setdefault("YOLO_AUTOINSTALL", "false")          # never pip-install mid-run
os.environ.setdefault("ULTRALYTICS_AUTOINSTALL", "false")

TYPES = {
    "auto": "Auto-detect from the weights / id",
    "yolo": "Own YOLO / RT-DETR detector, segmenter, OBB, pose (.pt)",
    "yolo_cls": "Own YOLO classifier (.pt)",
    "feat_cls": "Own crop classifier (backbone features + trained head, .pt checkpoint)",
}
OPEN_VOCAB = ()                              # none here (trained detectors only)
DEFAULT_PATH = {}
DEFAULT_CONF = {"auto": 0.25, "yolo": 0.25, "yolo_cls": 0.50, "feat_cls": 0.50}


def _device():
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ── what is this model? ──
def _ultra(path):
    """Load ultralytics weights with the right class (YOLO / RTDETR)."""
    import ultralytics as U
    low = os.path.basename(str(path)).lower()
    if "rtdetr" in low or "rt-detr" in low:
        return U.RTDETR(path)
    m = U.YOLO(path)
    return m


def inspect_model(path):
    """-> {"type", "task", "classes", "imgsz"} for any weights file, HF id or HF folder."""
    p = str(path).strip().strip('"')
    low = p.lower()
    is_file = os.path.isfile(p)
    if os.path.isdir(p):
        cfg = Path(p) / "config.json"
        mt = json.loads(cfg.read_text(encoding="utf-8")).get("model_type", "") if cfg.is_file() else ""
        low = low + " " + mt.lower()
    if not is_file:
        if not low.endswith((".pt", ".pth")):      # (an official name like yolo11n.pt downloads itself)
            raise ValueError(f"not a weights file and not a known HuggingFace detector: {p}")
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


# (26 Nov, user: "only the necessary messages") a newer ultralytics warns on EVERY predict call ("'half' is
# deprecated ... Use 'quantize'"): its DEPRECATION notices are not shown at all; other warnings and errors pass
class _OnceDeprecation:
    def filter(self, record):
        return "deprecated" not in record.getMessage().lower()


def _quiet_deprecations():
    import logging
    for name in ("ultralytics", "ultralytics.utils"):
        lg = logging.getLogger(name)
        if not any(isinstance(f, _OnceDeprecation) for f in lg.filters):
            lg.addFilter(_OnceDeprecation())
        for h in lg.handlers:                       # ultralytics logs through its own handler
            if not any(isinstance(f, _OnceDeprecation) for f in h.filters):
                h.addFilter(_OnceDeprecation())


# ── backends: detect(frame_bgr) -> [(object, conf, (x1, y1, x2, y2))] ──
class _Ultra:
    def _predict(self, frame, **kw):
        if not getattr(_Ultra, "_quiet", False):
            _quiet_deprecations()
            _Ultra._quiet = True
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


BACKEND = {"yolo": _OwnYolo, "yolo_cls": _OwnYoloCls, "feat_cls": _FeatCls}


def resolve(m):
    """an 'auto' entry -> its concrete type, classes and task (done once, when the model is added)."""
    if m.get("type") in (None, "", "auto"):
        info = inspect_model(m["path"])
        m.update(type=info["type"], task=info["task"])
        if info["classes"]:
            m["classes"] = info["classes"]
    return m


def specific_parts(spec, classes, log=print):
    """each step's PART -> the most specific detector class its own text names: the parser may shorten 'the empty
    outer box' to 'box', and a detector that has both 'box' and 'outer_box' would then follow the wrong object.
    A class counts when all its words are in the step's text and it contains all the part's words; the one with the
    most words wins. -> a copy of spec"""
    import json as _j
    import re as _re
    w = lambda s: [x for x in _re.findall(r"[a-z]+", str(s or "").lower().replace("_", " "))]
    sp = _j.loads(_j.dumps(spec))
    cls = {" ".join(w(c)): c for c in classes or [] if w(c)}
    for s in sp.get("steps") or []:
        pw = set(w(s.get("part")))
        if not pw:
            continue
        tw = set(w(s.get("text"))) | set(w(s.get("action_query"))) | pw
        best = max((c for c in cls if pw < set(c.split()) and set(c.split()) <= tw),
                   key=lambda c: len(c.split()), default=None)
        if best is None:
            # (25 Nov) the step names the class WITHOUT its adjective: 'fit the battery cover' -> 'black battery
            # cover' (its last two words, together, in the text) - not 'battery' (packing: step 2 then took step 1's
            # 'battery disappears'). 'the box' never picks 'open box' / 'closed box' (two words needed)
            seq = " " + " ".join(w(s.get("text"))) + " "
            tail = [c for c in cls if len(c.split()) >= 3 and pw < set(c.split())
                    and f" {' '.join(c.split()[-2:])} " in seq]
            best = max(tail, key=lambda c: len(c.split()), default=None)
        if best and best != " ".join(w(s.get("part"))):
            log(f"  step {s['id']}: part '{s.get('part')}' -> '{best}' (its text names the detector class "
                f"'{cls[best]}')")
            s["part"] = best
    # DESTINATIONS: 'put the mouse INTO THE OPEN BOX' - the place an item goes must be detected too, or it can never
    # be seen going in (only the steps' parts are detected). A detector class named right after into / onto /
    # inside / in / on (within 4 words) is added as a context object
    parts = {" ".join(w(s.get("part"))) for s in sp.get("steps") or []}
    ctx = list(sp.get("context_objects") or [])
    for s in sp.get("steps") or []:
        tw = w(s.get("text")) + ["."] + w(s.get("action_query"))
        for i, x in enumerate(tw):
            if x not in ("into", "onto", "inside", "in", "on"):
                continue
            nxt = set(tw[i + 1:i + 5])
            best = max((c for c in cls if set(c.split()) <= nxt), key=lambda c: len(c.split()), default=None)
            if best and best not in parts and best not in ctx:
                ctx.append(best)
                log(f"  step {s['id']}: destination '{best}' detected too (detector class '{cls[best]}')")
    if ctx:
        sp["context_objects"] = ctx
    return sp


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

pass
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


def store_path(pl):
    """the store of this run's video, keyed like its detections (the same frames, detector, objects)"""
    c = pl._det_cache_path() if hasattr(pl, "_det_cache_path") else None
    return STORE / f"{c.stem}.npz" if c is not None else None


def load(pl):
    """-> {frame idx: [vector or None per detection, in pl.detect order]} or None"""
    p = store_path(pl)
    if p is None or not p.is_file():
        return None
    z = np.load(p, allow_pickle=False)
    V, off, has = z["V"], z["off"], z["has"]
    out, k = {}, 0
    for i, f in enumerate(pl.frames):
        n = int(off[i + 1] - off[i])
        out[f["idx"]] = [V[k + j] if has[k + j] else None for j in range(n)]
        k += n
    return out


def build(pl, log=print):
    """every frame of the run once more through the detector (its maps), the vectors of the CACHED detections cut
    out and saved -> {frame idx: [vector or None]}. Reads the store back when it exists."""
    got = load(pl)
    if got is not None:
        log("  feature store read back (this video's detector features were stored before)")
        return got
    roi = RoiFeatures(pl.bank)
    if not roi.ok:
        log("  feature store: no ultralytics detector to read features from - appearance left out")
        return None
    out, V, off, has = {}, [], [0], []
    dim = None
    try:
        for k, f in enumerate(pl.frames):
            if k % 50 == 0:
                pl._check()
                pl.progress(f"feature store {k}/{len(pl.frames)}", k / len(pl.frames))
            rgb = pl.frgb(k)
            pl.bank.detect_rgb(rgb)                         # the forward pass: its maps
            vs = roi.vectors(pl.detect(f["idx"]), rgb.shape)
            out[f["idx"]] = vs
            for v in vs:
                if v is not None:
                    dim = len(v)
            V.append(vs)
            off.append(off[-1] + len(vs))
    finally:
        roi.close()
    if dim is None:
        return out
    flat = [v if v is not None else np.zeros(dim, np.float16) for vs in V for v in vs]
    has = [v is not None for vs in V for v in vs]
    p = store_path(pl)
    if p is not None:
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, V=np.array(flat, np.float16).reshape(-1, dim), off=np.array(off),
                            has=np.array(has, bool))
        log(f"  feature store saved: {sum(has)} box vectors of {dim} values -> {p.name}")
    return out


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

pass
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
        close_at_exit(_LM)
    return _LM


def close_at_exit(lm):
    """close a MediaPipe task at exit, while Python's thread pools still run: its close() goes through one, and
    a close after their shutdown fails ("cannot schedule new futures"), then its __del__ prints "'NoneType' object
    is not callable". threading's exit hooks run before the pools stop (atexit's run after)."""
    import threading
    reg = getattr(threading, "_register_atexit", None)
    try:
        if reg:
            reg(_close_quiet, lm)
            return
    except RuntimeError:                                   # already shutting down
        pass
    import atexit
    atexit.register(_close_quiet, lm)


def _close_quiet(lm):
    """close a MediaPipe task once, ignoring a second close"""
    try:
        lm.close()
    except Exception:
        pass


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

pass
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


def change_point(x):
    """the index splitting x into a HIGH then a LOW segment with the least squared error (-> k: x[:k] high)"""
    n = len(x)
    if n < 3:
        return n
    cs, cs2 = np.cumsum(x), np.cumsum(x * x)
    best, bk = None, n
    for k in range(1, n):
        a, b = x[:k], x[k:]
        if a.mean() <= b.mean():
            continue
        cost = (cs2[k - 1] - cs[k - 1] ** 2 / k) + ((cs2[-1] - cs2[k - 1]) - (cs[-1] - cs[k - 1]) ** 2 / (n - k))
        if best is None or cost < best:
            best, bk = cost, k
    return bk


def boundaries(video, L, log=print):
    """-> {step id: {"start", "end", "bound", "work", "ok", "why"}} for every step with a confirmed pickup"""
    S = signals(video, L, log)
    t, m = S["t"], np.convolve(S["motion"], np.ones(3) / 3, mode="same")
    ref = float(np.percentile(m, 95)) if len(m) else 1.0
    picks = sorted((e["pickup"], e["step"]) for e in L["events"] if e.get("pickup") is not None)
    out = {}
    for i, (tp, sid) in enumerate(picks):
        later = [x for x, s in picks[i + 1:] if s != sid and x > tp + 0.5]
        bound = min(later) if later else min(float(t[-1]), tp + LAST_S)
        sel = np.flatnonzero((t >= tp) & (t <= bound))
        if len(sel) < 3:
            out[sid] = {"start": tp, "end": None, "bound": bound, "work": 0.0, "ok": False,
                        "why": "window too short"}
            continue
        x = m[sel]
        work = float(x.max()) / max(ref, 1e-6)
        if work < MIN_WORK:
            out[sid] = {"start": tp, "end": None, "bound": bound, "work": round(work, 2), "ok": False,
                        "why": "no work on the product seen after the pickup"}
            continue
        k = change_point(x)
        te = float(t[sel[min(k, len(sel) - 1)]])
        # the product changed for good across the window? (its picture at the end vs just before the pickup)
        before = np.flatnonzero((t >= tp - 1.5) & (t < tp))
        after = np.flatnonzero((t > te) & (t <= te + 1.5))
        chg = float(np.abs(S["crops"][after].mean(0) - S["crops"][before].mean(0)).mean()) \
            if len(before) and len(after) else None
        out[sid] = {"start": tp, "end": te, "bound": bound, "work": round(work, 2), "ok": True,
                    "change": None if chg is None else round(chg, 1),
                    "why": f"work high -> low at {te:.1f}s (window {tp:.1f}-{bound:.1f}s)"}
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
# ADAPTIVE CONFIRMATION (24 Nov test - ADAPTIVE = False goes back): the waits above are the AMBIGUOUS case; the more of
# the placement is SEEN, the shorter the window that must pass without contradiction (seen outside again):
ADAPTIVE = True
WIN_VISIBLE_S = 0.4    # visibly installed: came to rest + same size (stable) and released / not held when last seen
WIN_STRONG_S = 0.6     # strong: dropped into it (motion fell inside), destination consistent, not held when last seen
WIN_VANISH_S = 1.0     # vanished inside WITH support: came from outside, inside its LEARNT interior (zone), the
                       # destination in view - else (out of view in the hand, no learnt zone) VANISH_S as before


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


def learn_zone(L, target, dest, stays):
    """where the target came to REST inside the destination in the good video (its confirmed stays) -> the zone
    {"u": [lo, hi], "d": [lo, hi]} - the box's INSIDE, not the lid / the table behind it its rectangle also covers"""
    pts = []
    for p in stays:
        for i, t in enumerate(L["t"]):
            if p["completion"] <= t <= p["left"]:
                a, b = _largest(L["dets"][i], target), _largest(L["dets"][i], dest)
                if a and b:
                    pts.append(zone_xy(a, b))
    if len(pts) < 3:
        return None
    P = np.array(pts)
    return {"u": [float(P[:, 0].min()), float(P[:, 0].max())], "d": [float(P[:, 1].min()), float(P[:, 1].max())],
            "n": len(pts)}


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
        pre = mot[(t >= t[i0] - 0.4) & (t <= t[i0])]            # the approach: the last 0.4 s (was 4 frames)
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
            pre_on = F["hand_on"][(t >= t[i0] - 0.3) & (t <= t[i1])].any()   # in the stay or 0.3 s before
            rw = seg[(t[seg] >= t[rest]) & (t[seg] <= t[rest] + STABLE_S)] if rest is not None else seg[:0]
            release = bool(pre_on and len(rw) and not F["hand_on"][rw].any())
        # the WINDOW that must pass without contradiction - adaptive: shorter the more of the placement was seen
        post_w, how = POST_S, "ambiguous"
        if ADAPTIVE:
            # at rest + the same size + in view for STABLE_S IS 'visibly installed': a part really held moves or is
            # covered (the detector loses it) - fingertips working NEXT to it (reaching for the next part) do not undo
            # that (the card at rest in the box from 4.9 s waited the 2 s ambiguous window because a hand was near)
            if stable:
                post_w, how = WIN_VISIBLE_S, "visibly installed"
            elif candidate and dropped and dest_ok and not held_end:
                post_w, how = WIN_STRONG_S, "strong placement evidence"
        # PERSISTS: not seen OUTSIDE the destination for that window after it came to rest (hidden is fine)
        after = (t > tc) & (t <= tc + post_w)
        out_seen = after & clear_out
        n_out = int(out_seen.sum())
        seen_after = int((after & vis).sum())
        enough = t_end >= tc + post_w
        # STILL BEING HANDLED (live): the stay goes on at the newest frames - the target seen there but not yet at rest,
        # or just hidden (a hand over it, not yet long enough to call it pushed in): its time is not known yet
        handling = live and not (rest is not None and stable) and t_end - tb <= 2 * GAP_S \
            and (held_end or not ADAPTIVE)               # (24 Nov) 'still being handled' only with a hand ON it
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
            conf_t = round(min(t_end, max(tc + post_w, tc + (STABLE_S if stable else 0.0))), 2)
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
            # adaptive: vanished inside its LEARNT interior (zone), having come from outside and stayed in a while =
            # supported -> WIN_VANISH_S; else (no zone learnt, a brief touch) the full VANISH_S
            supported = ADAPTIVE and zone is not None and from_outside and tb - ta >= ENTER_S
            vw = WIN_VANISH_S if supported else VANISH_S
            w = (t > tl) & (t <= tl + vw)
            # its next sighting WITH the destination in view (one on top of the closed box, the destination no longer
            # detected, says nothing about in / out) must be inside again
            nxt_vis = np.flatnonzero(vis & F["dseen"] & (t > tl))
            if t_end >= tl + vw and w.any() and not (vis & w).any() and float(F["dseen"][w].mean()) >= 0.7 \
                    and not (clear_out & (t > tl)).any() and (not len(nxt_vis) or ins[nxt_vis[0]]) \
                    and from_outside and i1 > i0:
                vanished = round(tl, 2)
                candidate = True                             # went in and stayed in: its own proof of entering
                if persists is None and supported and not n_out:
                    persists = True                          # its supported vanish IS the 'stayed in' evidence
                if conf_t is None and persists:
                    conf_t = round(min(t_end, tl + vw), 2)
                if supported:
                    how = "vanished inside its learnt interior"
        kind = ("already_inside" if not from_outside else "passed_over" if persists is False else
                "candidate" if candidate else "inside_briefly")
        out.append({"target": target, "dest": dest, "entered": round(ta, 2), "left": round(tb, 2),
                    "completion": round(tc, 2), "confirmation": conf_t,
                    "candidate": candidate, "persists": persists, "stable_unheld": stable, "release": release,
                    "hidden_after": hidden_after, "kind": kind, "held_end": held_end,
                    "from_outside": from_outside, "removed_at": removed_at, "entered_rect": round(entered_rect, 2),
                    "vanished_inside": vanished, "confirm_window": (how, post_w),
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


_LAST = {}                   # (25 Nov) the last record's events: one action-model check asks once per step


def events(L):
    """-> [{"t", "cls", "kind", "other", "held"}] every sustained change of the detector's classes, by time
    (the same record asked again -> the same list, copied; ~0.1 s saved per call)"""
    d = L.get("dets")
    if _LAST.get("dets") is d and _LAST.get("n") == len(L["t"]):
        return [dict(e) for e in _LAST["evs"]]
    out = _events(L)
    _LAST.update(dets=d, n=len(L["t"]), evs=[dict(e) for e in out])
    return out


def _events(L):
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
    _mark_state_flips(L, out, t)
    return sorted(out, key=lambda e: e["t"])


# (25 Nov) a STATE CHANGE is not a disappearance: 'closed box' disappearing while 'open box' appears in the same
# place is the box OPENED, not the box gone. The two classes are one object's states when their names differ only
# by state words (open / closed / empty / full ...) and share the head noun. Such a disappearance gets "into": the
# class it became; a step learnt from a REAL disappearance (the box slid into its sleeve) no longer matches it -
# box top wrong order: step 5 ('closed box disappears', learnt at 23.2 s) was matched to the box opened at 5.6 s.
STATE_WORDS = {"open", "opened", "closed", "close", "empty", "full", "filled", "sealed", "unsealed", "folded",
               "unfolded", "packed", "unpacked", "assembled", "finished", "on", "off"}
FLIP_S = 1.0                 # the other state appears within this long of the disappearance ...
FLIP_NEAR = 0.5              # ... where the object was (centre distance <= this x its size)


def same_object(a, b):
    """two class names that are one object's states ('closed box' / 'open box')"""
    wa, wb = (a or "").lower().replace("_", " ").split(), (b or "").lower().replace("_", " ").split()
    if not wa or not wb or wa == wb or wa[-1] != wb[-1]:
        return False
    ra, rb = [w for w in wa[:-1] if w not in STATE_WORDS], [w for w in wb[:-1] if w not in STATE_WORDS]
    return ra == rb and (set(wa[:-1]) | set(wb[:-1])) & STATE_WORDS != set()


def _same_place(a, b):
    """two boxes of one object in its two states: centres within FLIP_NEAR x size, or mostly overlapping (an open
    box's lid makes its box much taller than the closed box's)"""
    size = max(1.0, float(np.hypot(a[2] - a[0], a[3] - a[1])))
    if np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2) <= FLIP_NEAR * size:
        return True
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return small > 0 and ix / small >= 0.5


def _mark_state_flips(L, out, t):
    def box_near(c, t0, before):
        ks = [k for k in range(len(t)) if (t[k] <= t0 if before else t[k] >= t0)]
        for k in (reversed(ks) if before else ks):
            bs = [b for n, b in L["dets"][k] if n == c]
            if bs:
                return bs[0]
        return None
    for e in out:
        if e["kind"] != "disappears" or e.get("other"):
            continue
        b0 = box_near(e["cls"], e["t"], True)
        for f in out:
            if f["kind"] != "appears" or f.get("other") or abs(f["t"] - e["t"]) > FLIP_S \
                    or not same_object(e["cls"], f["cls"]):
                continue
            b1 = box_near(f["cls"], f["t"], False)
            if b0 is None or b1 is None:
                continue
            size = max(1.0, float(np.hypot(b0[2] - b0[0], b0[3] - b0[1])))
            d = float(np.hypot((b0[0] + b0[2] - b1[0] - b1[2]) / 2, (b0[1] + b0[3] - b1[1] - b1[3]) / 2))
            if d <= FLIP_NEAR * size:
                e["into"] = f["cls"]
                f["from"] = e["cls"]                     # ... and the other state appeared FROM it
                break
    # (26 Nov) the other state seen AT ITS PLACE right after, with no 'appears' change of its class (a false 'box'
    # elsewhere kept that class's count at 1: packing missing battery, 'open box' -> 'box' at 9.5 s was never an
    # event) - found by place: a STATE CHANGE event is added
    classes = sorted({n for ds in L["dets"] for n, _ in ds})
    for e in list(out):
        if e["kind"] != "disappears" or e.get("other") or e.get("into"):
            continue
        b0 = box_near(e["cls"], e["t"], True)
        if b0 is None:
            continue
        size = max(1.0, float(np.hypot(b0[2] - b0[0], b0[3] - b0[1])))
        near = lambda b: float(np.hypot((b0[0] + b0[2] - b[0] - b[2]) / 2,
                                        (b0[1] + b0[3] - b[1] - b[3]) / 2)) <= FLIP_NEAR * size
        for c in classes:
            if not same_object(e["cls"], c):
                continue
            after = [k for k in range(len(t)) if e["t"] - 0.05 <= t[k] <= e["t"] + FLIP_S
                     and any(n == c and near(b) for n, b in L["dets"][k])]
            before = [k for k in range(len(t)) if e["t"] - FLIP_S <= t[k] < e["t"] - 0.2
                      and any(n == c and near(b) for n, b in L["dets"][k])]
            if after and not before:
                e["into"] = c
                out.append({"t": float(t[after[0]]), "cls": c, "kind": "appears", "other": None,
                            "held": e["held"], "from": e["cls"], "by_place": True})
                break
    # (26 Nov) an APPEARANCE is a state change when the partner state was seen AT THAT PLACE just before - even if
    # the partner's detections flicker on afterwards (box top wrong order: 'close box' appears at 17.8 s while an
    # 'open box' is still reported on and off until 23.2 s - the close was never a state change)
    for f in out:
        if f["kind"] != "appears" or f.get("other") or f.get("from"):
            continue
        b1 = box_near(f["cls"], f["t"], False)
        if b1 is None:
            continue
        for c in classes:
            if not same_object(f["cls"], c):
                continue
            seen = [b for k in range(len(t)) if f["t"] - FLIP_S <= t[k] < f["t"] - 0.02
                    for n, b in L["dets"][k] if n == c and _same_place(b, b1)]
            if seen:
                f["from"], f["by_place"] = c, True
                break


def describe(e):
    return f"{e['cls']} {e['kind']}" + (f" {e['other']}" if e.get("other") else "")


def ambiguous_steps(spec, L):
    """the steps whose pickup time cannot be trusted -> {step id: reason}"""
    by_step, out = {}, {}
    for u in L["units"]:
        by_step.setdefault(u["step"], []).append(tuple(round(x, 1) for x in u["box"]))
    shared = {}
    for sid, bs in by_step.items():
        for b in bs:
            shared.setdefault(b, set()).add(sid)
    tp = {e["step"]: e["pickup"] for e in L["events"] if e.get("pickup") is not None}
    for s in spec["steps"]:
        sid = s["id"]
        if sid not in by_step:
            out[sid] = "no learnt place"
        elif any(len(shared[b]) > 1 for b in by_step[sid]):
            out[sid] = "its place is shared with step(s) " + ", ".join(
                str(x) for b in by_step[sid] for x in sorted(shared[b] - {sid}))
        elif sid in tp and any(o != sid and abs(tp[o] - tp[sid]) < TIE_S for o in tp):
            out[sid] = "its pickup is at the same time as another step's"
    # a step tied with an ambiguous one is not trustworthy either (one hidden-at-once moment)
    for sid in list(tp):
        if sid not in out and any(o in out and abs(tp[o] - tp[sid]) < TIE_S for o in tp if o != sid):
            out[sid] = "its pickup is at the same time as another step's"
    return out


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

pass
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


class PointMotion:
    """[E2] class-agnostic point motion between consecutive frames. backend 'lk' (OpenCV pyramidal Lucas-Kanade on a
    point grid, no download) or 'cotracker3' (torch.hub facebookresearch/co-tracker, used if it loads)"""

    def __init__(self, backend="lk", log=print):
        self.backend, self.log = backend, log
        self.model = None
        if backend == "cotracker3":
            try:
                import torch
                self.model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").eval()
                self.dev = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = self.model.to(self.dev)
            except Exception as e:
                self.log(f"  CoTracker3 not available ({type(e).__name__}: {e}) - OpenCV point tracking instead")
                self.backend = "lk"

    def grid(self, h, w):
        ys, xs = np.mgrid[GRID // 2:h:GRID, GRID // 2:w:GRID]
        return np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32), ys.shape

    def flows(self, greys):
        """greys: list of (h, w) uint8 -> (T, n, 2) displacement of each grid point from the frame before (0 first),
        (T, n) valid"""
        h, w = greys[0].shape
        pts, _ = self.grid(h, w)
        T = len(greys)
        D, V = np.zeros((T, len(pts), 2), np.float32), np.zeros((T, len(pts)), bool)
        if self.backend == "cotracker3" and self.model is not None:
            return self._cotracker(greys, pts)
        for k in range(1, T):
            p1, st, _ = cv2.calcOpticalFlowPyrLK(greys[k - 1], greys[k], pts.reshape(-1, 1, 2), None,
                                                 winSize=(15, 15), maxLevel=3)
            D[k] = (p1.reshape(-1, 2) - pts)
            V[k] = st.ravel() == 1
        return D, V

    def _cotracker(self, greys, pts, win=16):
        import torch
        T = len(greys)
        D, V = np.zeros((T, len(pts), 2), np.float32), np.zeros((T, len(pts)), bool)
        q = torch.from_numpy(np.concatenate([np.zeros((len(pts), 1), np.float32), pts], 1))[None].to(self.dev)
        for a in range(0, T - 1, win - 1):
            b = min(T, a + win)
            clip = np.stack([np.repeat(g[..., None], 3, 2) for g in greys[a:b]])
            vid = torch.from_numpy(clip).permute(0, 3, 1, 2)[None].float().to(self.dev)
            with torch.no_grad():
                self.model(video_chunk=vid, is_first_step=True, queries=q)
                tr, vis = None, None
                for i in range(0, vid.shape[1] - self.model.step, self.model.step):
                    tr, vis = self.model(video_chunk=vid[:, i:i + self.model.step * 2])
            if tr is None:
                continue
            tr, vis = tr[0].cpu().numpy(), vis[0].cpu().numpy() > 0.5
            for j in range(1, tr.shape[0]):
                D[a + j], V[a + j] = tr[j] - tr[j - 1], vis[j] & vis[j - 1]
        return D, V


class SpatiotemporalEvidenceEngine:
    """src: the Pipeline (frames, detect(), frgb(), bank, _same_class(), _product(), video_path, log ...).
    extract(src, spec) -> the evidence: tracks, events, operations (see the module text)"""

    def __init__(self, tracker="lk", log=print, progress=None, check=None, appearance=True, refine=True):
        self.tracker, self.log = tracker, log
        self.progress = progress or (lambda m, f=None: None)
        self.check = check or (lambda: None)
        self.use_app, self.use_refine = appearance, refine
        self._dino = None

    # ── public ──
    def extract(self, src, spec):
        t0 = time.time()
        frames = src.frames
        t = np.array([f["ts"] for f in frames], float)
        H0, W0 = src.frgb(0).shape[:2]
        fam = self.families(src, spec)
        A, M = self._register(src)                               # [E1] + [E2]
        prod = fam["product"]
        P = self._product_boxes(src, prod, t)
        tracks, events = [], []
        for name in sorted(set(fam["part_of"].values()) | set(fam["tools"])):
            trs = self._tracks(src, name, A, W0)                 # [E3]
            for tr in trs:
                self._features(tr, P, M, W0, H0, t)              # [E4]
            tracks += trs
            if name in fam["tools"]:
                for tr in trs:
                    events += [e for e in self._events(tr, t, tool=True) if e["kind"] == "WORK"]   # [E5] work
            else:
                events += [e for tr in trs for e in [self._pickup(tr, t)] if e]     # [E5] source -> taken
                events += self._installs(src, name, P, t)        # [E5] -> on the product
        events += self._place_pickups(src, spec, fam, events, t, W0)   # [E5] scene change at a learnt place
        ops = self._operations(events)
        if self.use_refine:
            self._refine(src, ops, A, W0)                       # [E6]
        if self.use_app:
            self._appearance(src, ops)                          # [E7]
        self.log(f"  evidence: {len(tracks)} tracks, {len(events)} events, {len(ops)} operations "
                 f"({self.tracker} points, {time.time() - t0:.0f}s)")
        return {"t": t.tolist(), "W": W0, "H": H0, "families": fam, "ops": ops,
                "events": [{k: v for k, v in e.items() if k != "track"} for e in events],
                "backend": self.tracker}

    def families(self, src, spec):
        """which detector names are one class (look-alike copies), which are steps' parts, the product, the tools"""
        pass
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

    # ── [E1] + [E2] ──
    def _register(self, src):
        """-> A (T, 2, 3): frame k pixel -> first-frame pixel (similarity), M (T, gh, gw): residual point motion"""
        frames = src.frames
        key = hashlib.sha1(json.dumps([getattr(src, "video_path", ""), len(frames), frames[-1]["ts"],
                                       self.tracker, ALIGN_W, GRID, "v2"]).encode()).hexdigest()[:16]
        cp = evidence_engine_CACHE / f"{key}.npz"
        if cp.is_file():
            z = np.load(cp)
            return z["A"], z["M"]
        H0, W0 = src.frgb(0).shape[:2]
        s = ALIGN_W / W0
        greys = []
        for k in range(len(frames)):
            g = cv2.cvtColor(src.frgb(k), cv2.COLOR_RGB2GRAY)
            greys.append(cv2.resize(g, (ALIGN_W, int(round(H0 * s))), interpolation=cv2.INTER_AREA))
        sift = cv2.SIFT_create(800)
        bf = cv2.BFMatcher()
        feats = []
        for k, g in enumerate(greys):
            if k % 25 == 0:
                self.check()
                self.progress(f"evidence: aligning {k}/{len(greys)}", k / len(greys))
            # only the bench: every detected box (the product held up close, hands, parts) is masked out, so the
            # alignment follows the camera, not what moves in front of it
            mask = np.full(g.shape, 255, np.uint8)
            for d in src.detect(frames[k]["idx"]):
                x0, y0, x1, y1 = (int(v * s) for v in d["box"])
                p = int(0.1 * max(x1 - x0, y1 - y0))
                mask[max(0, y0 - p):y1 + p, max(0, x0 - p):x1 + p] = 0
            feats.append(sift.detectAndCompute(g, mask))

        def sim(a, b):                                    # b -> a, (2x3 at ALIGN_W, inliers)
            (ka, da), (kb, db) = feats[a], feats[b]
            if da is None or db is None or len(ka) < 8 or len(kb) < 8:
                return None, 0
            ms = [m for m, n in (x for x in bf.knnMatch(db, da, k=2) if len(x) == 2) if m.distance < 0.75 * n.distance]
            if len(ms) < 8:
                return None, 0
            pb = np.float32([kb[m.queryIdx].pt for m in ms])
            pa = np.float32([ka[m.trainIdx].pt for m in ms])
            T, inl = cv2.estimateAffinePartial2D(pb, pa, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            return T, int(inl.sum()) if inl is not None else 0

        I = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
        A, pend = [I.copy()], None
        near = lambda a, b: np.abs(a[:, 2] - b[:, 2]).max() < 3.0 and np.abs(a[:, :2] - b[:, :2]).max() < 0.01
        for k in range(1, len(greys)):
            # straight to the first frame (no drift); a NEW camera position only once two frames in a row agree
            d, n_d = sim(0, k)
            if d is None or n_d < 40:
                A.append(A[-1].copy())
            elif near(d, A[-1]):
                A.append(d)
                pend = None
            elif pend is not None and near(d, pend):
                A[-1] = pend
                A.append(d)
                pend = None
            else:
                A.append(A[-1].copy())
                pend = d
        A = np.array(A)
        self.progress("evidence: point motion", None)
        pm = PointMotion(self.tracker, self.log)
        self.tracker = pm.backend
        D, V = pm.flows(greys)
        pts, gshape = pm.grid(*greys[0].shape)
        M = np.zeros((len(greys),) + gshape, np.float32)
        for k in range(1, len(greys)):
            # the camera's own motion between k-1 and k, predicted at each point, taken off
            Ak = np.vstack([A[k], [0, 0, 1]])
            Ak1 = np.vstack([A[k - 1], [0, 0, 1]])
            R = np.linalg.inv(Ak) @ Ak1                         # frame k-1 -> frame k
            pred = pts @ R[:2, :2].T + R[:2, 2] - pts
            r = np.linalg.norm(D[k] - pred, axis=1)
            r[~V[k]] = 0
            M[k] = r.reshape(gshape)
        A_full = A.copy()                                      # to full-size pixels: scale the translation
        A_full[:, :, 2] /= s
        evidence_engine_CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cp, A=A_full, M=M)
        return A_full, M

    # ── [E3] ──
    def _tracks(self, src, family, A, W0):
        from scipy.optimize import linear_sum_assignment
        canon = lambda n: sorted(src._same_class({n}))[0]
        live, done = [], []
        for k, f in enumerate(src.frames):
            # look-alike copies ('first wheel' ... 'fourth wheel') carry the same box once per name: one box once
            ds = list({tuple(np.round(d["box"], 1)): d for d in src.detect(f["idx"])
                       if canon(d["name"]) == family}.values())
            cs = []
            for d in ds:
                x0, y0, x1, y1 = d["box"]
                c = A[k] @ np.array([(x0 + x1) / 2, (y0 + y1) / 2, 1.0])
                cs.append((c / W0, d))
            for tr in list(live):
                if f["ts"] - tr["s"][-1]["t"] > TRACK_GAP_S:
                    live.remove(tr)
                    done.append(tr)
            if live and cs:
                C = np.array([[np.linalg.norm(c - tr["s"][-1]["ws"]) / max(tr["s"][-1]["size"], 1e-3)
                               for c, _ in cs] for tr in live])
                ri, ci = linear_sum_assignment(C)
                took = set()
                for r, c in zip(ri, ci):
                    if C[r, c] <= GATE:
                        live[r]["s"].append(self._sample(k, f["ts"], cs[c], W0))
                        took.add(c)
                cs = [x for j, x in enumerate(cs) if j not in took]
            for c in cs:
                live.append({"family": family, "s": [self._sample(k, f["ts"], c, W0)]})
        tracks = done + live
        # a hand over a copy ends its track; a later track starting where it ended is the same copy
        tracks.sort(key=lambda tr: tr["s"][0]["t"])
        out = []
        for tr in tracks:
            a = tr["s"][0]
            prev = [p for p in out if 0 < a["t"] - p["s"][-1]["t"] <= JOIN_GAP_S
                    and np.linalg.norm(a["ws"] - p["s"][-1]["ws"]) < 0.7 * max(p["s"][-1]["size"], 1e-3)]
            if prev:
                p = min(prev, key=lambda p: a["t"] - p["s"][-1]["t"])
                p["s"] += tr["s"]
            else:
                out.append(tr)
        return [tr for tr in out if len(tr["s"]) >= 3]

    @staticmethod
    def _sample(k, ts, cd, W0):
        c, d = cd
        x0, y0, x1, y1 = d["box"]
        return {"k": k, "t": float(ts), "ws": np.asarray(c, float), "box": [float(v) for v in d["box"]],
                "size": max(x1 - x0, y1 - y0) / W0, "aspect": (x1 - x0) / max(y1 - y0, 1.0), "conf": float(d["conf"])}

    def _product_boxes(self, src, prod, t, hold_s=1.0):
        """per frame the product's box (the largest product detection; the last one for `hold_s` when unseen)"""
        canon = lambda n: sorted(src._same_class({n}))[0]
        out, last = [], (None, -1e9)
        for k, f in enumerate(src.frames):
            bs = [d["box"] for d in src.detect(f["idx"]) if canon(d["name"]) in prod]
            if bs:
                b = max(bs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                last = (b, f["ts"])
                out.append(b)
            else:
                out.append(last[0] if f["ts"] - last[1] <= hold_s else None)
        return out

    # ── [E4] ──
    def _features(self, tr, P, M, W0, H0, t):
        gh, gw = M.shape[1:]
        for j, s in enumerate(tr["s"]):
            x0, y0, x1, y1 = s["box"]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            gx0, gx1 = int(x0 / W0 * gw), max(int(x0 / W0 * gw) + 1, int(np.ceil(x1 / W0 * gw)))
            gy0, gy1 = int(y0 / H0 * gh), max(int(y0 / H0 * gh) + 1, int(np.ceil(y1 / H0 * gh)))
            s["motion"] = float(M[s["k"], gy0:gy1, gx0:gx1].mean()) if M[s["k"], gy0:gy1, gx0:gx1].size else 0.0
            b = P[s["k"]]
            if b is None:
                s["dprod"], s["rel"] = None, None
            else:
                pw, ph = max(b[2] - b[0], 1.0), max(b[3] - b[1], 1.0)
                inside = b[0] <= cx <= b[2] and b[1] <= cy <= b[3]
                dx = max(b[0] - cx, 0, cx - b[2])
                dy = max(b[1] - cy, 0, cy - b[3])
                s["dprod"] = 0.0 if inside else float(np.hypot(dx, dy) / np.hypot(pw, ph))
                s["rel"] = [float((cx - b[0]) / pw), float((cy - b[1]) / ph)]
            if j:
                p = tr["s"][j - 1]
                s["speed"] = float(np.linalg.norm(s["ws"] - p["ws"]) / max(s["t"] - p["t"], 1e-3))
            else:
                s["speed"] = 0.0

    # ── [E5] ──
    def _pickup(self, tr, t, start_s=5.0, leave=2.0, hold_s=1.0):
        """a copy that lies at its SOURCE (seen from the first `start_s`, off the product) is taken when it is
        carried off (> `leave` x its size away for `hold_s` - a copy nudged by the hand taking its neighbour stays
        nearer) or is gone from it for good. Hands passing over it are no pickup: only its position counts.
        -> a PICKUP event or None"""
        S = tr["s"]
        if S[0]["t"] > t[0] + start_s or S[0]["dprod"] == 0.0:
            return None
        n0 = max(3, min(len(S) // 5, 9))
        p0 = np.median([s["ws"] for s in S[:n0]], 0)
        size = max(float(np.median([s["size"] for s in S[:n0]])), 1e-3)
        away = np.array([np.linalg.norm(s["ws"] - p0) / size > leave for s in S])
        ts = np.array([s["t"] for s in S])
        k = None
        for i in range(len(S)):
            if away[i] and away[i:][ts[i:] <= ts[i] + hold_s].all():
                k = i
                break
        if k is None and ts[-1] < t[-1] - JOIN_GAP_S:
            k = len(S)                                       # never moved off it here, then gone for good
        if k is None or k == 0:
            return None
        at = S[:k]
        # its look is taken at the first clean view (just before the pickup the hand is already on it)
        src = {"ws": p0.tolist(), "box": np.median([s["box"] for s in at[:n0]], 0).tolist(), "size": size,
               "aspect": float(np.median([s["aspect"] for s in at])), "k": at[min(len(at), n0) // 2]["k"]}
        return {"family": tr["family"], "kind": "PICKUP", "t": at[-1]["t"], "src": src,
                "conf": 0.8 if k < len(S) else 0.65}

    def _place_pickups(self, src, spec, fam, events, t, W0, ref_s=3.0, thr=12.0, hold_s=10.0, share=0.8):
        """SCENE CHANGE: a learnt source place (the place learner's spatial anchors) whose copy the detector does not
        see there - its picture differs from the first `ref_s` for `share` of the next `hold_s` = taken (a hand
        passing over it changes it for less)"""
        H0 = src.frgb(0).shape[0]
        out = []
        have = [e["src"]["ws"] for e in events if e["kind"] == "PICKUP"]
        grey = None
        for s in spec.get("steps") or []:
            fam_s = fam["part_of"].get(str(s["id"]))
            for p in s.get("positions") or []:
                b = np.array(p["box"]) * [W0, H0, W0, H0]
                c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]) / W0
                size = max(b[2] - b[0], b[3] - b[1]) / W0
                if fam_s is None or any(np.linalg.norm(np.array(h) - c) < 0.7 * size for h in have):
                    continue                                 # the detector followed this copy from its place
                if grey is None:
                    grey = [cv2.GaussianBlur(cv2.cvtColor(src.frgb(k), cv2.COLOR_RGB2GRAY), (5, 5), 0)
                            for k in range(len(src.frames))]
                x0, y0, x1, y1 = (int(round(v)) for v in b)
                crop = np.array([g[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)] for g in grey], np.float32)
                n_ref = max(1, int(np.searchsorted(t, t[0] + ref_s)))
                d = np.abs(crop - np.median(crop[:n_ref], 0)).mean((1, 2)) > thr
                k = next((i for i in range(n_ref, len(t)) if d[i] and d[i:][t[i:] <= t[i] + hold_s].mean() >= share
                          and t[i] + hold_s <= t[-1] + 1e-6), None)
                if k is None:
                    continue
                out.append({"family": fam_s, "kind": "PICKUP", "t": float(t[k - 1]), "conf": 0.7, "how": "scene change",
                            "src": {"ws": c.tolist(), "box": b.tolist(), "size": size,
                                    "aspect": float((b[2] - b[0]) / max(b[3] - b[1], 1.0)), "k": k - 1}})
                have.append(c.tolist())
        return out

    def _installs(self, src, family, P, t, win_s=3.0, hold_s=ON_HOLD_S, drop_s=10.0):
        """the STATE of the product: how many copies of the part are on it (boxes inside its box), median over
        `win_s`. Each new level held for `hold_s` = one INSTALL (the copy that is new there gives where on the
        product); a level lost for `drop_s` = a REMOVE. Copies there from the start are no install."""
        canon = lambda n: sorted(src._same_class({n}))[0]
        T = len(t)
        boxes = []
        for k, f in enumerate(src.frames):
            b = P[k]
            ds = {tuple(np.round(d["box"], 1)) for d in src.detect(f["idx"]) if canon(d["name"]) == family}
            inside = [] if b is None else [d for d in ds if b[0] <= (d[0] + d[2]) / 2 <= b[2]
                                           and b[1] <= (d[1] + d[3]) / 2 <= b[3]]
            rel = [((d[0] + d[2]) / 2 - b[0]) / max(b[2] - b[0], 1.0) for d in inside], \
                  [((d[1] + d[3]) / 2 - b[1]) / max(b[3] - b[1], 1.0) for d in inside]
            boxes.append((inside, list(zip(*rel))))
        n = np.array([len(x[0]) for x in boxes], float)
        dt = float(np.median(np.diff(t))) if T > 1 else 1.0
        w = max(1, int(round(win_s / dt)))
        lev = np.array([np.median(n[max(0, i - w // 2):i + w // 2 + 1]) for i in range(T)])
        h = max(1, int(round(hold_s / dt)))
        held = np.array([lev[i:i + h].min() if i + h <= T else lev[i:].min() for i in range(T)])
        base = int(held[t <= t[0] + 3.0].max()) if (t <= t[0] + 3.0).any() else 0
        out, top = [], base
        for i in range(T):
            while held[i] >= top + 1:
                top += 1
                # where the new copy sits: the box farthest from every box seen there before
                before = [r for j in range(max(0, i - 3 * w), max(0, i - w)) for r in boxes[j][1]]
                cand = boxes[i][1] or [r for j in range(i, min(T, i + h)) for r in boxes[j][1]]
                dst = max(cand, key=lambda r: min((np.hypot(r[0] - q[0], r[1] - q[1]) for q in before), default=1.0)) \
                    if cand else None
                k = next((j for j in range(i, min(T, i + h)) if boxes[j][0]), i)
                bx = max(boxes[k][0], key=lambda d: d[2] - d[0]) if boxes[k][0] else None
                out.append({"family": family, "kind": "INSTALL", "t": float(t[i]), "conf": 0.8,
                            "dst": list(dst) if dst else None,
                            "size": (max(bx[2] - bx[0], bx[3] - bx[1]) / src.frgb(0).shape[1]) if bx else None,
                            "aspect": ((bx[2] - bx[0]) / max(bx[3] - bx[1], 1.0)) if bx else None})
        d = max(1, int(round(drop_s / dt)))
        for i in range(T - d):
            if top > base and lev[i] >= 1 and lev[i + 1:i + 1 + d].max() < lev[i] and lev[-1] < lev[i]:
                out.append({"family": family, "kind": "REMOVE", "t": float(t[i]), "conf": 0.6})
                break
        return out

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

    def _operations(self, events):
        """a pickup and the install of the same class after it = one operation (source -> destination)"""
        ops = []
        by = {}
        for e in sorted(events, key=lambda e: e["t"]):
            by.setdefault(e["family"], []).append(e)
        for fam, evs in by.items():
            open_p = []
            for e in evs:
                if e["kind"] == "PICKUP":
                    open_p.append(e)
                elif e["kind"] == "INSTALL":
                    # the k-th copy taken is the k-th on the product (first taken, first fitted)
                    cand = [p for p in open_p if 0 <= e["t"] - p["t"] <= LINK_S]
                    p = cand[0] if cand else None
                    if p is not None:
                        open_p.remove(p)
                    ops.append(self._op(fam, "INSTALL", p, e))
                elif e["kind"] in ("REMOVE", "WORK"):
                    ops.append({"family": fam, "kind": e["kind"], "t": e["t"], "t0": e.get("t0"),
                                "dst": e.get("dst"), "ws": e.get("ws"), "conf": e["conf"]})
            ops += [self._op(fam, "INSTALL", p, None) for p in open_p]
        return sorted(ops, key=lambda o: o["t"])

    @staticmethod
    def _op(fam, kind, p, e):
        # its time is the PICKUP's where there is one: the source tells the copies apart, 'on the product' is one
        # count for the whole class (which copy went on is not seen) - the install only makes it surer
        src = p["src"] if p else None
        return {"family": fam, "kind": kind, "t": p["t"] if p else e["t"],
                "t_pick": p["t"] if p else None, "t_install": e["t"] if e else None,
                "src": src["ws"] if src else None, "src_box": src["box"] if src else None,
                "src_k": src["k"] if src else None,
                "dst": e.get("dst") if e else None,
                "size": src["size"] if src else e.get("size"), "aspect": src["aspect"] if src else e.get("aspect"),
                "conf": 0.9 if p and e else 0.6 if e else 0.55}

    # ── [E6] ──
    def _refine(self, src, ops, A, W0, rate=10.0, back_s=0.8, thr=12.0):
        """a pickup at the 3 fps sampling is +-0.3 s: the source video read at `rate` around it, the moment its
        place's picture changes"""
        video = getattr(src, "video_path", None)
        if not video or not os.path.isfile(video):
            return
        cap = cv2.VideoCapture(video)
        try:
            vw = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or W0
            sc = vw / W0
            off = (getattr(src, "seg", None) or (0.0, None))[0] or 0.0
            for o in ops:
                if o.get("t_pick") is None or o.get("src_box") is None:
                    continue
                self.check()
                x0, y0, x1, y1 = (int(round(v * sc)) for v in o["src_box"])
                ref, best = None, None
                # one seek, then read on (a seek per frame decodes from the last keyframe each time)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                every = max(1, int(round(fps / rate)))
                t_a = max(o["t_pick"] - back_s, 0.0)
                cap.set(cv2.CAP_PROP_POS_MSEC, (off + t_a) * 1000.0)
                for j in range(int((back_s + 1.0) * fps) + 1):
                    if j % every:
                        if not cap.grab():
                            break
                        continue
                    tt = t_a + j / fps
                    ok, fr = cap.read()
                    if not ok:
                        break
                    g = cv2.cvtColor(fr[max(y0, 0):y1, max(x0, 0):x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
                    if g.size == 0:
                        break
                    if ref is None:
                        ref = g
                        continue
                    if np.abs(g - ref).mean() > thr:
                        best = float(tt)
                        break
                if best is not None:
                    o["t_pick_refined"] = o["t"] = round(best, 2)
        finally:
            cap.release()

    # ── [E7] ──
    def _appearance(self, src, ops):
        need = [o for o in ops if o.get("src_box") is not None]
        if not need:
            return
        try:
            import timm
            import torch
            if self._dino is None:
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0,
                                      img_size=224).eval().to(dev)
                cfg = timm.data.resolve_data_config({}, model=m)
                self._dino = (m, dev, np.array(cfg["mean"], np.float32), np.array(cfg["std"], np.float32))
            m, dev, mean, std = self._dino
            ims = []
            for o in need:
                rgb = src.frgb(o["src_k"])
                x0, y0, x1, y1 = o["src_box"]
                p = 0.15 * max(x1 - x0, y1 - y0)
                c = rgb[int(max(0, y0 - p)):int(y1 + p), int(max(0, x0 - p)):int(x1 + p)]
                c = cv2.resize(c, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                ims.append(((c - mean) / std).transpose(2, 0, 1))
            with torch.no_grad():
                f = m(torch.from_numpy(np.stack(ims)).to(dev)).float()
                f = torch.nn.functional.normalize(f, dim=1).cpu().numpy()
            for o, v in zip(need, f):
                o["app"] = np.round(v, 4).tolist()
        except Exception as e:
            self.log(f"  (appearance vectors skipped: {type(e).__name__}: {e})")

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

pass
pass
pass

pass
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
        pass
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


    def _pickups(self, copies, grey, thr=12.0, ref_s=3.0, reach=1.5, away=3.0):
        """copies [(key, part, box)] - box as frame fractions: the copy's source place. Each copy is FOLLOWED from its
        place: frame by frame it takes the nearest detection of its part within `reach` x its size of where it was
        last seen (copies of one part never take the same detection - nearest pairs first), so a wheel that tips over
        and rolls aside stays that wheel. It is PICKED UP at the start of its first stretch of >= ABSENT_S not seen
        near where it was last, or when it is seen more than `away` x its size from its place (carried off) - the
        finished truck later set down on its empty spot cannot bring it back. Without detections: its place's picture
        differs from the start for good. -> {key: (frame index or None, why)}"""
        t = np.array([f["ts"] for f in self.frames])
        objs = list(getattr(self.bank, "objects", None) or [])
        if not objs or isinstance(self.bank, NullBank):
            out = {}
            for key, _, box in copies:
                d, _ = self._place_diff(box, grey, ref_s)
                last = int(np.flatnonzero(d < thr).max()) if (d < thr).any() else -1
                out[key] = ((None, "not on its place at the start" if last < 0 else "never left its place")
                            if last < 0 or last >= len(t) - 1 else (last + 1, "its place differs from then on"))
            return out
        H, W = grey[0].shape[:2]
        n_ref = max(1, int(np.searchsorted(t, t[0] + ref_s)))
        cen = lambda b: np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
        names, src, pos, size = {}, {}, {}, {}
        for key, part, box in copies:
            o = match(part, objs)
            names[key] = self._same_class({o}) if o else set()
            b = np.array(box) * [W, H, W, H]
            src[key] = pos[key] = cen(b)
            size[key] = max(b[2] - b[0], b[3] - b[1], 1.0)
        seen = {k: False for k in names}
        gone, done = {k: None for k in names}, {}
        self._followed = {}                                 # frame idx -> {box key: copy key}
        for i, f in enumerate(self.frames):
            live = [k for k in names if k not in done]
            if not live:
                break
            dets = self.detect(f["idx"])
            # (one box is one detection: look-alike copies' names each carry the same boxes)
            pairs = sorted((float(np.linalg.norm(cen(d["box"]) - pos[k])), k, tuple(np.round(d["box"], 1)))
                           for k in live for d in dets if d["name"] in names[k])
            boxes = {tuple(np.round(d["box"], 1)): d["box"] for d in dets}
            took_k, took_j = set(), set()
            for dist, k, j in pairs:
                if k in took_k or j in took_j or dist > reach * size[k]:
                    continue
                took_k.add(k)
                took_j.add(j)
                pos[k], seen[k], gone[k] = cen(boxes[j]), True, None
                self._followed.setdefault(f["idx"], {})[j] = k     # which copy this box is (the video's label)
                if np.linalg.norm(pos[k] - src[k]) > away * size[k]:
                    done[k] = (i, "carried away from its place")
            for k in live:
                if k in took_k or k in done:
                    continue
                if not seen[k] and i >= n_ref:
                    done[k] = (None, "not on its place at the start")
                elif seen[k] and gone[k] is None:
                    gone[k] = i
                elif seen[k] and t[i] - t[gone[k]] >= ABSENT_S:
                    done[k] = (gone[k], f"not seen near it for {ABSENT_S:g} s from then")
        return {k: done.get(k, (None, "never left its place")) for k in names}

    def _greys(self):
        return [cv2.GaussianBlur(cv2.cvtColor(self.frgb(f["idx"]), cv2.COLOR_RGB2GRAY), (5, 5), 0) for f in self.frames]

    def _place_diff(self, box, grey, ref_s=3.0):
        """a place (box as frame fractions) over the video: how far its picture is from the first `ref_s` in each
        frame, and that first picture -> ((T,) mean grey difference, (h, w) float32 reference patch)"""
        H, W = grey[0].shape[:2]
        t = np.array([f["ts"] for f in self.frames])
        n_ref = max(1, int(np.searchsorted(t, t[0] + ref_s)))
        x0, y0, x1, y1 = (int(round(v * d)) for v, d in zip(box, (W, H, W, H)))
        x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
        crop = np.array([im[y0:y1, x0:x1] for im in grey], np.float32)
        ref = np.median(crop[:n_ref], 0)
        return np.abs(crop - ref).mean((1, 2)), ref

    def learn_positions(self, video, prompt, spec=None, ref_s=4.0, min_seen=0.3, thr=12.0):
        """a NORMAL video -> this prompt's places file (places_path; positions.json is no longer written): where
        every step's part lies at the start (the detector's boxes of that
        part over the first `ref_s`, one place per copy), the copies listed in the order their places empty in this
        video (the right order). Places that never empty here are left out. Each place keeps a small picture of how
        it looks at the start, so a test video laid out differently is not judged by it. Names of an earlier file's
        places are kept (nearest box); new copies are named first, second, ... The old file -> positions.json.bak.
        -> {part: {"ordered", "places", "learned_from"}}"""
        if spec is None:
            spec = self.compile_spec(prompt, own_labels(self.models), False)
        spec = json.loads(json.dumps(spec))
        self.log(f"\n[learn places] {Path(video).name}")
        self.progress("sampling the frames")
        self.sample_frames(str(video))
        self.progress("loading the detectors")
        self.build_bank(spec)
        t = np.array([f["ts"] for f in self.frames])
        first = [f for f in self.frames if f["ts"] < t[0] + ref_s]
        grey = self._greys()
        H, W = grey[0].shape[:2]
        # training a MODEL writes its own places file (places_file); else this prompt's
        own = getattr(self, "places_file", None)
        path = Path(own) if own else self.places_path(prompt, write=True)
        src = path if own and path.is_file() else self.places_path(prompt)   # earlier names are kept
        old = json.loads(src.read_text(encoding="utf-8")) if src.is_file() else {}
        words = lambda x: {stem(w) for w in keywords(str(x or ""))}
        ords = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
        out = {}
        pass
        # steps on the SAME part, one per copy ('the square ... side panel', 'the long ... side panel'): learned
        # together, all their copies at once - the normal video takes them in the steps' order
        sib = {}
        for s in spec.get("steps") or []:
            if not (s.get("precondition") or s.get("groups")):
                sib.setdefault(s["part"], []).append(s)
        for s in spec.get("steps") or []:
            self._check()
            if s.get("precondition") or s.get("groups") or s["part"] in out:
                continue
            # places are for parts that LEAVE a start spot (pick / place / install); a work step on a part that stays
            # put ('tighten the bolt') would 'leave' whenever the camera turns away from it - work spots learn those
            prim = primitive(s.get("text") or f"{s.get('action_query') or ''} {s['part']}")
            if prim in ("MANIPULATE", "SCAN", "INSPECT", "REMOVE"):
                self.log(f"  {s['part']}: a {prim.lower()} step - its place is where the work happens "
                         "('Learn work spots'), not where a part starts - not learned here")
                continue
            obj = match(s["part"], self.bank.objects)
            if not obj:
                self.log(f"  {s['part']}: no detector object - not learned")
                continue
            mates = sib.get(s["part"]) or [s]
            n = sum(max(int(m.get("count", 1) or 1), int(m.get("slots_per_group") or 1)) for m in mates)
            groups = []                                   # the boxes of each copy over the first seconds
            for f in first:
                for d in self.detect(f["idx"]):
                    if d["name"] != obj:
                        continue
                    g = max(groups, key=lambda g: iou(np.median(g, 0), d["box"]), default=None)
                    if g is not None and iou(np.median(g, 0), d["box"]) > 0.3:
                        g.append(d["box"])
                    else:
                        groups.append([d["box"]])
            groups = sorted([g for g in groups if len(g) >= min_seen * len(first)], key=len, reverse=True)[:n]
            if not groups:
                self.log(f"  {s['part']}: not seen in the first {ref_s:g} s - not learned")
                continue
            boxes = [[round(float(np.clip(v / dd, 0, 1)), 4) for v, dd in zip(np.median(g, 0), (W, H, W, H))]
                     for g in groups]
            pick = self._pickups([(i, s["part"], box) for i, box in enumerate(boxes)], grey, thr)
            places = []
            for i, box in enumerate(boxes):
                k, why = pick[i]
                if k is None:
                    self.log(f"  {s['part']}: a copy at {box} - {why} here - left out")
                    continue
                _, ref = self._place_diff(box, grey)
                places.append({"box": box, "t": float(t[k]),
                               "thumb": np.round(cv2.resize(ref, (16, 16))).astype(int).tolist()})
            if not places:
                continue
            places.sort(key=lambda p: p["t"])
            w = words(s["part"])
            prev = next((v if isinstance(v, list) else v.get("places") or [] for k, v in old.items()
                         if w and words(k) and (words(k) <= w or w <= words(k))), [])
            last_w = stem(keywords(s["part"])[-1])
            # one step per copy, each text with its own word ('square' / 'long'): the k-th copy taken in this
            # (normal) video is the k-th step's - named by that word, so the step finds it again
            own = [[x for x in keywords(m.get("text") or "") if stem(x) not in w and
                    not any(stem(x) in words(o.get("text")) for o in mates if o is not m)] for m in mates]
            if len(mates) == len(places) > 1 and all(own):
                names = [f"{o[0]} {last_w}" for o in own]
            elif len(places) == 1:
                names = [prev[0]["name"] if len(prev) == 1 else str(s["part"])]
            elif len(prev) == len(places) and not any(q["name"].split()[0] in ords for q in prev):
                # hand-given names ('square panel') follow their copy; 'first' ... 'fourth' always name the
                # pickup order of THIS video
                c = lambda bx: np.array([(bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2])
                left, names = list(prev), []
                for p in places:
                    q = min(left, key=lambda q: np.linalg.norm(c(q["box"]) - c(p["box"])))
                    left.remove(q)
                    names.append(q["name"])
            else:
                names = [f"{ords[i] if i < len(ords) else i + 1} {last_w}" for i in range(len(places))]
            for p, nm in zip(places, names):
                p["name"] = nm
                self.log(f"  {s['part']}: {nm} at {p['box']} - leaves its place at {p['t']:.1f}s")
            if len(places) < n:
                self.log(f"  {s['part']}: only {len(places)} of {n} copies found on the table")
            # t_ref: when the good video takes it - the trained sequence a model shows
            out[s["part"]] = {"ordered": True, "learned_from": Path(video).name,
                              "places": [dict({k: p[k] for k in ("name", "box", "thumb")}, t_ref=round(p["t"], 1))
                                         for p in places]}
        freed = self.bank.release() if hasattr(self.bank, "release") else 0   # the GPU back for the VLM
        if freed:
            self.log(f"  detectors unloaded - {freed:.0f} MB of GPU memory freed")
        if not out:
            self.log("  nothing learned - the places file left as it was")
            return out
        path.parent.mkdir(parents=True, exist_ok=True)
        had = path.is_file()
        if had:
            shutil.copyfile(path, path.with_suffix(".json.bak"))
        path.write_text(json.dumps(out, indent=1), encoding="utf-8")
        self.log(f"  saved {sum(len(v['places']) for v in out.values())} places of {len(out)} parts -> {path} "
                 "(this prompt only)" + (f"; the old one kept as {path.with_suffix('.json.bak').name}" if had else ""))
        return out


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

pass
pass
pass
pass
pass
pass

pass
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
# an un-numbered list of steps: words that only join steps, sentences that set rules (not steps), words a new action
# never starts with (after 'and' they mean the same action goes on: 'put the mouse and the card into the box')
_LIST_LEAD = re.compile(r"^(?:and\s+then|and|then|first(?:ly)?|next|finally|lastly|after\s+that)\b[\s,]*", re.I)
_LIST_RULE = re.compile(r"^(?:flag|an?\s+anomal|anomal|report|alert|raise|the\s+sop\s+is)\w*", re.I)
_LIST_THEN = re.compile(r",|\b(?:and\s+)?then\b", re.I)
_NOT_ACTION = {"the", "a", "an", "its", "it", "his", "her", "their", "this", "that", "these", "those", "all", "both",
               "each", "every", "one", "two", "three", "four", "five", "some", "any", "then"}
_SUBJECTS = {"operator", "worker", "user", "person", "technician", "he", "she", "they", "you", "we", "i", "there",
             "when", "if", "while", "after", "before", "once", "is", "are", "was", "were"}


def pipeline__words(s):
    return re.findall(r"[a-z]+", str(s).lower().replace("_", " "))


def _find_name(ws, lab):
    """where the name 'lab' (words) starts in the word list ws (plurals count), or None"""
    lw = lab.split()
    for i in range(len(ws) - len(lw) + 1):
        if all(ws[i + j] in (x, x + "s", x + "es") for j, x in enumerate(lw)):
            return i
    return None


def _label_names(labels):
    """the detector classes as plain words -> class; plus the HEAD noun several classes share ('open box' /
    'close box' -> 'box'), so a step may just say 'the box' (postprocess_spec's _state_labels then picks open or
    close box from the step's own words)"""
    labs = {" ".join(pipeline__words(l)): l for l in labels if pipeline__words(l)}
    heads = {}
    for n in labs:
        if len(n.split()) > 1:
            heads.setdefault(n.split()[-1], []).append(n)
    shared = {h: h for h, ns in heads.items() if len(ns) >= 2 and h not in labs}
    return labs, shared


def list_steps(prompt, labels=()):
    """an UN-numbered list of steps -> one text per step ([] when it is not one).
    'Do these steps: insert the battery into the mouse, fit the battery cover onto the mouse, then close the box.'
    -> ['insert the battery into the mouse', 'fit the battery cover onto the mouse', 'close the box'].
    Steps are cut at , ; . new lines and 'then'; at 'and' only when both halves name their own object and the second
    half starts a new action ('put the mouse into the box and put the card into the box' = 2 steps, 'slide the box into
    its sleeve and leave it closed' = 1, 'put the mouse and the card into the box' = 1). The intro before a ':' (or a
    title line followed by a blank line) and rule sentences ('Flag an anomaly if ...') are not steps. Every step must
    name a detector class (or a head noun classes share) - else [] and the VLM reads the prompt"""
    if not labels or _numbered_steps(prompt):
        return []
    labs, shared = _label_names(labels)
    names = list(labs) + list(shared)
    has_obj = lambda t: any(_find_name(pipeline__words(re.sub(r"\([^)]*\)", " ", t)), n) is not None for n in names) or \
        any(" ".join(pipeline__words(b)) in labs for b in re.findall(r"\(([^)]*)\)", t))
    text = str(prompt or "").strip()
    if ":" in text:
        head, rest = text.split(":", 1)
        if rest.strip() and "\n\n" not in head.strip():
            text = rest
    else:
        parts = re.split(r"\n\s*\n", text, maxsplit=1)
        if len(parts) == 2 and "\n" not in parts[0].strip() and not re.search(r"[.,;]\s*$", parts[0].strip()):
            text = parts[1]                               # a title line, then a blank line
    out = []
    for sent in re.split(r"(?<=[.!?;])\s+|[.!?;]\s*$|\n+", text):
        sent = (sent or "").strip(" .!?;\t")
        if not sent or _LIST_RULE.match(sent):
            continue
        for piece in _LIST_THEN.split(sent):
            piece = _LIST_LEAD.sub("", piece.strip(" .;\t")).strip(" .;\t")
            if not piece:
                continue
            halves = re.split(r"\s+and\s+", piece)
            cur = halves[0]
            for h in halves[1:]:
                fw = (pipeline__words(h) or [""])[0]
                if has_obj(cur) and has_obj(h) and fw not in _NOT_ACTION and not any(
                        _find_name(pipeline__words(h), n) == 0 for n in names):
                    out.append(cur)
                    cur = h
                else:
                    cur = f"{cur} and {h}"
            out.append(cur)
    out = [re.sub(r"\s+", " ", o).strip(" ,") for o in out if o.strip(" ,")]
    # every piece must read as an instruction ('insert the battery ...'): one that starts with 'the', 'an operator',
    # or an object ('the mouse and the card start on the table') is a description - the VLM reads the prompt then
    starts_ok = lambda o: (pipeline__words(o) or [""])[0] not in _NOT_ACTION | _SUBJECTS and not any(
        _find_name(pipeline__words(o), n) == 0 for n in names)
    if len(out) < 2 or not all(has_obj(o) and starts_ok(o) for o in out):
        return []
    return out


def as_numbered(prompt, labels=()):
    """the prompt with its steps numbered: a numbered prompt as it is, an un-numbered list of steps (list_steps) as
    '1. ...\\n2. ...'; anything else as it is (the VLM reads it)"""
    items = list_steps(prompt, labels)
    if not items:
        return prompt
    return "\n".join(f"{i}. {t[0].upper() + t[1:]}." for i, t in enumerate(items, 1))


def rule_spec(prompt, labels=()):
    """a NUMBERED prompt ('1. open box, 2. put the mouse into the open box, ...') -> the compiler's JSON without the
    VLM (milliseconds, the same answer every time). Per numbered line: its text, a short action, and its PART = a
    detector class named in it - one in brackets in the line ('(close_box)'), else the first class named before the
    place it goes to ('put THE MOUSE into the open box'), else anywhere in the line; else a head noun several
    classes share ('close the box' -> 'box', made open / close box by postprocess_spec). An un-numbered list of steps
    ('insert the battery into the mouse, fit the cover onto the mouse, then close the box') is numbered first
    (as_numbered).
    -> dict in the VLM's format (postprocess_spec does the rest), or None when the prompt is not a list of steps or a
    line names no detector class (the VLM then)"""
    prompt = as_numbered(prompt, labels)
    n = _numbered_steps(prompt)
    if n < 2:
        return None
    ms = [m for m in _NUMBERED.finditer(prompt) if int(m.group(1)) <= n]
    pos = {}
    for m in ms:                                          # the first '1.', then the first '2.' after it ...
        k = int(m.group(1))
        if k not in pos and (k == 1 or (k - 1 in pos and m.start() > pos[k - 1].start())):
            pos[k] = m
    if len(pos) != n:
        return None
    word = pipeline__words
    labs, shared = _label_names(labels)
    find = _find_name

    steps = []
    for k in range(1, n + 1):
        a = pos[k].end()
        b = pos[k + 1].start() if k < n else len(prompt)
        txt = prompt[a:b].strip()
        if k == n:                                        # the last line ends at its sentence (then 'Flag ...')
            mm = re.search(r"[.!?]\s+(?=[A-Z])", txt)
            txt = txt[:mm.start()] if mm else txt
        txt = txt.strip(" ,;.\n\t")
        brackets = re.findall(r"\(([^)]*)\)", txt)
        bare = re.sub(r"\([^)]*\)", " ", txt)
        ws = word(bare)
        part = None
        for br in brackets:                               # '(close_box)': the line names its class itself
            bl = " ".join(word(br))
            if bl in labs:
                part = bl
                break
        if part is None:
            cut = next((i for i, x in enumerate(ws) if x in _DEST and i > 0), len(ws))
            for region in (ws[:cut], ws):
                hits = [(find(region, l), -len(l.split()), l) for l in labs if find(region, l) is not None]
                if hits:
                    part = min(hits)[2]
                    break
        if part is None:
            # a head noun several classes share ('close the box' -> 'box'), only as the object acted on: before the
            # place it goes to ('put the charger into THE BOX' is not about the box) and not the first half of a
            # compound ('the cover plate', 'the box lid' are other objects)
            ends = _DEST | {"and", "with", "its", "his", "her", "their", "from", "out", "off", "back", "down", "up"}
            hits = []
            for h in shared:
                i = find(ws[:cut], h)
                if i is not None and (i + 1 >= cut or ws[i + 1] in ends):
                    hits.append((i, h))
            if hits:
                part = min(hits)[1]
        if part is None:
            # no enabled detector names this step's object: the VLM reads the prompt instead (it knows that
            # 'the lid of the outer box' is about the outer box; plain rules would guess)
            return None
        aq = re.split(r",|;|\bwith\b", bare)[0]
        aq = " ".join(aq.split()[:8]).strip().lower()
        steps.append({"id": k, "action_query": aq or f"handle the {part}", "part": part, "count": 1,
                      "visible_per_group": 1})
    first = re.split(r"(?<=[.!?])\s", prompt.strip(), maxsplit=1)[0]
    return {"task": " ".join(first.split()[:14]).rstrip("."), "mode": "sequence", "ordered": True, "steps": steps,
            "forbidden": [], "flags": ["missing_step", "wrong_order"], "qa_checks": [], "anomaly_objects": []}


def _numbered_steps(prompt):
    """how many steps the prompt numbers 1., 2., 3. ... in a row (0 when it does not number them)"""
    seen = {int(n) for n in _NUMBERED.findall(prompt or "")}
    n = 0
    while n + 1 in seen:
        n += 1
    return n if n >= 2 else 0
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


def split_forbidden(text):
    """one step's wording -> (the part that is still a step or '', the forbidden action written positively or '').
    'place the box and don't close the lid' -> ('place the box', 'close the lid'); 'the lid must not be closed' ->
    ('', 'the lid is closed'); a plain step -> (text, '')"""
    t = re.sub(r"\s+", " ", str(text or "")).strip(" .")
    if m := _FORBID_PASSIVE.match(t):
        return "", f"{m.group(1)} is {m.group(2)}".strip(" .")
    if m := _FORBID_AFTER.match(t):
        return "", m.group(1).strip(" .")
    if m := _FORBID_NO.match(t):
        return "", m.group(1).strip(" .")
    if m := _FORBID_PRE.search(t):
        keep = _JOIN_TAIL.sub("", t[:m.start()]).strip(" ,.")
        return (keep if len(keep.split()) >= 2 else ""), t[m.end():].strip(" ,.")
    if m := _FORBID_WITHOUT.match(t):
        return m.group(1).strip(" ,."), m.group(2).strip(" .")
    return t, ""


def _mp_close_quiet(lm):
    """close a MediaPipe task once at exit, ignoring a second close"""
    try:
        lm.close()
    except Exception:
        pass


def _mp_close_at_exit(lm):
    """close a MediaPipe task at exit while Python's thread pools still run (its close() needs one; after their
    shutdown it fails and its __del__ prints "'NoneType' object is not callable"). threading's exit hooks run
    before the pools stop; atexit's run after."""
    import threading
    reg = getattr(threading, "_register_atexit", None)
    try:
        if reg:
            reg(_mp_close_quiet, lm)
            return
    except RuntimeError:
        pass
    import atexit
    atexit.register(_mp_close_quiet, lm)


class Cancelled(Exception):
    pass


def _centroid(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


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


def resolve_flags(prompt, vlm_flags, check_ids, mode="sequence"):
    """Deterministic keyword pass over the RAW prompt to fix the flag set the VLM guessed -
    a small model is unreliable at 'only flag X / don't flag Y'."""
    p = prompt.lower()
    flags = {str(f) for f in vlm_flags} | set(check_ids)
    if any(w in p for w in ("fall off", "falls off", "off the belt", "off the line",
                            "leaves the", "goes the wrong way", "off-path", "off path")):
        flags.add("off_path")
    if mode == "monitor":
        flags.discard("missing_step")
        flags.discard("wrong_order")
        return sorted(flags or set(check_ids) or {"defect"})
    mentions_order = any(w in p for w in _ORDER_WORDS)
    mentions_miss = any(w in p for w in _MISS_WORDS)
    exclude = any(w in p for w in ("do not flag", "don't flag", "dont flag",
                                   "not flag", "ignore", "no need to flag", "without flagging"))
    restrict = any(w in p for w in ("only flag", "just flag", "flag only"))
    if restrict:                                    # keep only what's explicitly named
        keep = set()
        if mentions_order:
            keep.add("wrong_order")
        if mentions_miss:
            keep.add("missing_step")
        for cid in check_ids:
            if cid in p or any(t in p for t in cid.split("_") if len(t) > 2):
                keep.add(cid)
        if keep:
            flags = keep
    if exclude:                                     # drop the excluded condition
        for w in ("do not flag", "don't flag", "dont flag", "not flag", "ignore"):
            i = p.find(w)
            if i < 0:
                continue
            tail = p[i:i + 80]
            if any(x in tail for x in _MISS_WORDS):
                flags.discard("missing_step")
            if any(x in tail for x in _ORDER_WORDS):
                flags.discard("wrong_order")
    flags.discard("")
    if not flags:
        flags = {"missing_step", "wrong_order"}
    return sorted(flags)


def _fold_gated(steps):
    """A, B, A, B, ... (A: the same one-off action each time, B: the same part each time) -> one grouped B step
    gated by A. Purely structural - works for scan/install, label/pack, open/fill, inspect/mount ..."""
    def verb(x):
        return (str(x.get("action_query") or "").split() or [""])[0].lower()

    out, i = [], 0
    while i < len(steps):
        a = steps[i]
        b = steps[i + 1] if i + 1 < len(steps) else None
        reps = 0
        if b is not None and a.get("count", 1) <= 1 and (b["part"] != a["part"] or verb(b) != verb(a)):
            k = i
            while k + 1 < len(steps) and verb(steps[k]) == verb(a) and steps[k].get("count", 1) <= 1 \
                    and steps[k + 1]["part"] == b["part"] and verb(steps[k + 1]) == verb(b):
                reps += 1
                k += 2
        if reps >= 2:
            item = dict(b)
            item["count"] = sum(steps[i + 2 * r + 1].get("count", 1) for r in range(reps))
            item["groups"], item["slots_per_group"] = reps, item["count"] // reps
            act = a["action_query"]
            item["gate"] = {"action": act if a["part"] in act else f"{act} the {a['part']}", "part": a["part"],
                            "tool": None}
            item["group_part"] = item.get("group_part") or (a["part"] if a["part"] != b["part"] else "")
            out.append(item)
            i += 2 * reps
        else:
            out.append(a)
            i += 1
    return out


_CHECK_VERBS = ("check", "confirm", "verify", "count", "ensure", "make sure")
_PRE_SENT = re.compile(r"precondition|must already|already (?:be )?(?:loaded|present|in place)|"
                       r"loaded with|check and confirm|must be present")


def precondition_counts(p):
    """[(n, noun)] from the prompt's precondition sentence(s): '... loaded with 5 splines and 20 yellow pads
    BEFORE ...' -> [(5, 'splines'), (20, 'yellow pads')]"""
    out = []
    for sent in re.split(r"(?<=[.!?])\s+|\n", p.lower()):
        if not _PRE_SENT.search(sent):
            continue
        for m in re.finditer(r"\b(\d{1,4})\s+([a-z][a-z ]{1,30}?)(?=\s+(?:and|before|on|in|are|is|must|-|have|has)\b|"
                             r"[,.;:)]|$)", sent):
            noun = m.group(2).strip()
            if noun.split()[0] in ("s", "sec", "secs", "seconds", "minutes", "times", "steps", "of", "per"):
                continue
            if not any(noun == x for _, x in out):
                out.append((int(m.group(1)), noun))
    return out


def _fold_gate_step(steps):
    """The per-fixture action written as ITS OWN step right before a grouped step ('scan the spline x5' before
    'place plate x30 = 5 splines x 6') is that group's gate: fold it in, so it is not expected a second time.
    A step counts as the gate when it is done once per fixture (count == groups, or it acts on the fixture)
    and is not a precondition check (check / confirm / verify / count)."""
    out = []
    for s in steps:
        prev = out[-1] if out else None
        if s.get("groups") and prev is not None and not prev.get("groups") and not prev.get("precondition"):
            act = str(prev.get("action_query") or "").lower()
            per_fixture = prev.get("count", 1) == s["groups"] or \
                stem(str(prev["part"]).lower()) == s["group_part"]
            if per_fixture and prev["part"] != s["part"] and not any(v in act for v in _CHECK_VERBS):
                out.pop()
                gate = s.get("gate") or prev.get("gate") or {}       # the compiler may put it on either step
                tool = gate.get("tool") or (prev["part"] if stem(str(prev["part"]).lower()) != s["group_part"]
                                            else None)
                tool = str(tool).strip().lower() if tool else None
                action = gate.get("action") or (prev["action_query"] if s["group_part"] in act
                                                else f"{prev['action_query']} the {s['group_part']}")
                if action.split()[-1].lower() in ("with", "using", "by") and tool:     # 'scan spline with'
                    action = f"{action} the {tool}"
                s["gate"] = {"action": action, "part": prev["part"], "tool": tool}
        out.append(s)
    return out


def _numbered_texts(prompt):
    """the prompt's numbered steps, in order: '1. Do a. 2. Do b.' -> ['Do a.', 'Do b.'] ([] when not numbered 1..n)"""
    n = _numbered_steps(prompt)
    if not n:
        return []
    marks = {}
    for m in _NUMBERED.finditer(prompt):
        k = int(m.group(1))
        if 1 <= k <= n and k not in marks and (k == 1 or (k - 1) in marks):
            marks[k] = (m.start(1), m.end())
    if len(marks) != n:
        return []
    out = []
    for k in range(1, n + 1):
        end = marks[k + 1][0] if k < n else len(prompt)
        txt = prompt[marks[k][1]:end].strip()
        if k == n:                                         # the last one ends at its sentence (rules may follow)
            txt = re.split(r"(?<=[.!?])\s+(?=[A-Z])", txt)[0]
        out.append(txt)
    return out


def _first_named(low, names):
    """(position, name) of the name mentioned first in the text, or None. A name counts only when ALL its words are
    in the text ('screw the black side handle' names 'side handle', not 'black rubber ring'); only when no name is
    there in full does one word do"""
    for full in (True, False):
        best = None
        for p in names:
            hits = [[m.start() for m in re.finditer(r"\b%s\w*" % re.escape(pipeline_stem(w)), low)] for w in keywords(p)]
            if full and not all(hits):
                continue
            pos = [x for h in hits for x in h]
            if pos and (best is None or min(pos) < best[0]):
                best = (min(pos), p)
        if best:
            return best
    return None


def _step_tools(text, objs):
    """the detected objects a step works WITH ('... using the red screwdriver' -> {'screwdriver red'})"""
    return {match(re.split(r"[,.;]", ph)[0], objs) for ph in re.findall(
        r"\b(?:with|using|by)\s+(?:the\s+)?([a-z][a-z \-]{2,40})", str(text).lower())} - {None}


def _ground_numbered(steps, prompt, labels=()):
    """a prompt that numbers its steps IS the spec's text: step k gets the prompt's sentence k (a small VLM may copy
    another step's words - '7. Put the grey seat back on' written as 'attach the yellow cab frame' - or leave steps
    out: 6 numbered, 4 written). A numbered sentence no step covers becomes a step of its own, its part the known part
    (a step's, or a detector class) it names first. When the VLM's part is not in its sentence, the part (and the
    action words before it) come from the sentence. -> the steps, in the prompt's order"""
    texts = _numbered_texts(prompt)
    if not texts:
        return steps
    parts = list(dict.fromkeys([s["part"] for s in steps] +
                               [re.sub(r"\s+", " ", str(x).lower().replace("_", " ")).strip() for x in labels]))
    if len(texts) != len(steps):
        # a step goes to the sentence that names its whole part first ('assembled gear head' is not 'screw the side
        # handle into the gear head'), only then to one sharing a word with it
        sks = [{pipeline_stem(w) for w in keywords(t.lower())} for t in texts]
        pks = [{pipeline_stem(w) for w in keywords(s["part"])} for s in steps]
        hit, used = {}, set()
        for test in (lambda pk, sk: pk <= sk, lambda pk, sk: pk & sk):
            for i, sk in enumerate(sks):
                if i in hit:
                    continue
                j = next((j for j in range(len(steps)) if j not in used and pks[j] and test(pks[j], sk)), None)
                if j is not None:
                    hit[i] = j
                    used.add(j)
        out = []
        for i, txt in enumerate(texts):
            if i in hit:
                out.append(steps[hit[i]])
                continue
            best = _first_named(txt.lower(), parts)
            if best is None:
                return steps                                   # a sentence names no known part: keep the VLM's
            aq = re.sub(r"\s*\band$", "", re.sub(r"\b(?:the|a|an)\b.*$", "", txt.lower()[:best[0]]).strip(" ,")
                        ).strip(" ,") or "handle"
            out.append({"part": best[1], "action_query": aq, "text": txt, "count": 1})
        steps = out
    for s, txt in zip(steps, texts):
        s["text"] = txt
        low = txt.lower()
        sk = {pipeline_stem(w) for w in keywords(low)}
        # the WHOLE part must be in its sentence: 'black rubber ring' is not named by 'screw the black side handle'
        if {pipeline_stem(w) for w in keywords(s["part"])} <= sk:
            continue
        best = _first_named(low, parts)
        if best:
            s["action_query"] = re.sub(r"\s*\band$", "", re.sub(r"\b(?:the|a|an)\b.*$", "", low[:best[0]]).strip(" ,")
                                       ).strip(" ,") or s["action_query"]
            s["part"] = best[1]
    return steps


def _state_labels(steps, labels=()):
    """a generic part that several detector classes share ('box' -> 'open box' / 'close box') gets the class its own
    sentence names: '1. Open box' -> open box, '5. close the box' -> close box ('closed' counts for 'close'). Without
    this 'box' matches only one of them and the other class is never looked for. A tie keeps the part as it is."""
    names = [re.sub(r"\s+", " ", str(x).lower().replace("_", " ")).strip() for x in labels]
    for s in steps:
        if s["part"] in names:
            continue
        pk = {pipeline_stem(w) for w in keywords(s["part"])}
        cands = [n for n in names if pk and pk < {pipeline_stem(w) for w in keywords(n)}]
        if len(cands) < 2:
            continue
        low = f"{s.get('text') or ''} {s.get('action_query') or ''}".lower().replace("_", " ")
        score = {n: sum(len(re.findall(r"\b%s\w*" % re.escape(pipeline_stem(w)), low))
                        for w in keywords(n) if pipeline_stem(w) not in pk) for n in cands}
        best = sorted(cands, key=lambda n: -score[n])
        if score[best[0]] > score[best[1]]:
            s["part"] = best[0]
    return steps


def postprocess_spec(spec, prompt, labels=()):
    """Deterministic clean-up of the compiler's JSON (the VLM is small; code keeps it sane)."""
    spec.setdefault("ordered", True)
    spec["steps"] = [s for s in (spec.get("steps") or []) if isinstance(s, dict) and str(s.get("part") or "").strip()]
    spec["_raw_prompt"] = prompt
    for s in spec["steps"]:
        s["part"] = re.sub(r"\s+", " ", str(s["part"]).lower().replace("_", " ")).strip()
        s["action_query"] = str(s.get("action_query") or s.get("text") or f"handle the {s['part']}").strip()
        s["text"] = str(s.get("text") or s["action_query"]).strip()
    spec["steps"] = _state_labels(_ground_numbered(spec["steps"], prompt, labels), labels)

    # FORBIDDEN actions are not steps: the compiler's forbidden[] plus any step it still wrote with 'don't ...' in
    # any wording (kept as the positive action - 'close the lid of the outer box' - that must NOT be seen). Half a
    # step ('place the box and don't close the lid') stays a step: 'place the box'.
    forb, steps = [], []
    cand = [(x, False) for x in (spec.get("forbidden") or []) if isinstance(x, dict)]
    for s in spec["steps"]:
        ka, fa = split_forbidden(s["action_query"])
        kt, ft = split_forbidden(s["text"])
        if not (fa or ft):
            steps.append(s)
            continue
        cand.append(({"action_query": fa or ft, "part": s["part"], "text": s["text"]}, True))
        if ka or kt:                                  # the allowed half is still a step
            steps.append(dict(s, action_query=ka or kt, text=kt or ka))
    for f, _ in cand:
        raw = str(f.get("action_query") or f.get("text") or "")
        aq = split_forbidden(raw)[1] or raw.strip(" ,.")   # the compiler may have kept the 'don't' too
        part = re.sub(r"\s+", " ", str(f.get("part") or "").lower().replace("_", " ")).strip()
        if aq and part and not any(sim(aq, x["action_query"]) > 0.85 for x in forb):
            forb.append({"action_query": aq, "part": part,
                         "text": str(f.get("text") or aq).strip()})
    spec["steps"] = steps
    # what the SOP asks for can never be forbidden: 'don't close the lid BEFORE the phone is in' is an ORDER rule (the
    # state machine checks the order), not 'never close the lid'. Drop a forbidden action that is an expected step,
    # or that only holds before / after / until something.
    spec["forbidden_dropped"] = [f for f in forb if _ORDER_RULE.search(f["text"] + " " + f["action_query"])
                                 or any(sim(f["action_query"], s["action_query"]) > 0.7
                                        or sim(f["action_query"], s["text"]) > 0.7 for s in steps)]
    forb = [f for f in forb if f not in spec["forbidden_dropped"]]
    # a small VLM copies the compiler's own examples ('close the lid', gloves) into any task: a forbidden action is
    # kept only when the prompt forbids something AND names its action (a word beyond its part's)
    pk_s = {pipeline_stem(w) for w in keywords(prompt)}
    forbids = bool(_FORBID_CUE.search(prompt))
    invented = [f for f in forb if not forbids or not ({pipeline_stem(w) for w in keywords(f["action_query"])}
                                                        - {pipeline_stem(w) for w in keywords(f["part"])}) & pk_s]
    spec["forbidden_dropped"] += [dict(f, why="not in the prompt") for f in invented]
    forb = [f for f in forb if f not in invented]
    for k, f in enumerate(forb, 1):
        f["id"] = f"forbidden{k}"
    spec["forbidden"] = forb
    # a step about an object the prompt never names was invented (e.g. from a model's class list: 'empty'): drop it
    pk = set(keywords(prompt))
    named = [s for s in spec["steps"] if kw_overlap(set(keywords(s["part"])), pk)]
    if named:
        spec["steps"] = named

    # sequence vs monitor - trust the VLM, fall back to keywords
    mode = str(spec.get("mode", "")).lower()
    p = prompt.lower()
    if mode not in ("sequence", "monitor"):
        mode = "monitor" if any(h in p for h in _MONITOR_HINTS) else "sequence"
    if len(spec["steps"]) <= 1 and any(h in p for h in _MONITOR_HINTS):
        mode = "monitor"
    spec["mode"] = mode
    if mode == "monitor":
        mon = spec.get("monitor") if isinstance(spec.get("monitor"), dict) else {}
        if not mon.get("item") and spec["steps"]:
            mon["item"] = spec["steps"][-1]["part"]
        mon["item"] = str(mon.get("item") or "item").lower().strip()
        mon.setdefault("off_path", None)
        spec["monitor"] = mon
    else:
        spec["monitor"] = None

    # qa_checks are EXTRA visual rules only (PPE, damage, orientation, colour). Drop structural
    # slugs, per-part "is X installed/present" checks, and checks that just name a step's part.
    part_words = set()
    for s in spec["steps"]:
        part_words.update(re.findall(r"\w+", s["part"]))
    clean, dropped_qa = [], []
    steps_txt = set(_numbered_texts(prompt))
    rule_k = {pipeline_stem(w) for sent in re.split(r"(?<=[.!?;])\s+|\n+", prompt)
              if sent.strip() and not any(sent.strip() in s or s in sent for s in steps_txt)
              and _RULE_CUE.search(sent) for w in keywords(sent)}
    for c in (spec.get("qa_checks") or []):
        if not isinstance(c, dict):
            continue
        cid = re.sub(r"[^a-z0-9_]", "_", str(c.get("id", "")).lower()).strip("_")
        q = str(c.get("q", "")).lower()
        if not cid or cid in ("missing_step", "wrong_order") or not (c.get("q") and c.get("fail_if")):
            continue
        if set(re.findall(r"[a-z0-9]+", cid.replace("_", " "))) & part_words and \
                any(w in q for w in ("installed", "present", "missing", "done")):
            continue
        if any(w in q for w in ("installed", "present", "in place", "attached", "fitted",
                                "mounted", "seated")) and "flush" not in q:
            continue                                           # that's the structural check
        qk = {pipeline_stem(w) for w in keywords(cid) + keywords(q)} - _QA_GENERIC
        if not qk & pk_s:
            dropped_qa.append(dict(c, id=cid, why="not in the prompt"))
            continue                                           # invented (copied from the compiler's examples)
        root = lambda ws: {w.rstrip("e")[:6] for w in ws}
        # a check must come from a sentence that ASKS for something (flag / must / check / never ...), not from the
        # task's description ('all parts start loose on the table' is no 'does a wheel look loose?' rule)
        if not root(qk) <= root(rule_k):
            dropped_qa.append(dict(c, id=cid, why="no rule in the prompt asks for it"))
            continue
        # a check that only restates what a step does ('is the box closed?' - step 3 closes it; 'is the phone in the
        # box?' - step 2 puts it there) is the order / missing check's job; as a QA rule it is judged on the wrong
        # moment or with the wrong sign ('fail if closed' on a box that must end closed)
        if any(root(qk) <= root({pipeline_stem(w) for w in keywords(s["text"])}) for s in spec["steps"]):
            dropped_qa.append(dict(c, id=cid, why="restates a step - the sequence check judges it"))
            continue
        c["id"] = cid
        clean.append(c)
    spec["qa_checks"] = clean
    spec["qa_dropped"] = dropped_qa
    if dropped_qa:                                             # and the flag the VLM set for it
        gone = {c["id"] for c in dropped_qa}
        spec["flags"] = [f for f in (spec.get("flags") or []) if str(f) not in gone]

    # scope each check to the ONE step it is about (keyword overlap with the step's part) so
    # "is the back cover clipped on?" is not asked at - and failed by - every earlier step.
    for c in spec["qa_checks"]:
        ck = set(keywords(c["id"])) | set(keywords(c["q"]))
        best_id, best_score = None, 0
        for s in spec["steps"]:
            score = len(ck & set(keywords(s["part"])))
            if score > best_score:
                best_id, best_score = s.get("id"), score
        c["step_id"] = best_id

    exp_names = [s["part"] for s in spec["steps"]]
    it = str((spec.get("monitor") or {}).get("item") or "").lower()
    if it:
        exp_names.append(it)
    ao = []
    for x in (spec.get("anomaly_objects") or []):
        x = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", str(x).lower()).strip())
        if not (2 <= len(x) <= 40) or x in _BG:
            continue
        if any(sim(x, e) > 0.85 for e in exp_names):
            continue
        core = re.sub(r"^(?:%s)[- ]" % "|".join(pipeline__NEG), "", x).strip()
        if core != x and any(sim(core, e) > 0.8 for e in exp_names):
            continue
        ao.append(x)
    spec["anomaly_objects"] = list(dict.fromkeys(ao))

    spec["flags"] = resolve_flags(prompt, spec.get("flags") or [], [c["id"] for c in spec["qa_checks"]], mode)
    if spec["anomaly_objects"] and "intruder" not in spec["flags"]:
        spec["flags"] = sorted(set(spec["flags"]) | {"intruder"})
    if spec["forbidden"]:
        spec["flags"] = sorted(set(spec["flags"]) | {"forbidden_action"})
    else:
        spec["flags"] = [f for f in spec["flags"] if f != "forbidden_action"]
    for s in spec["steps"]:
        try:
            s["count"] = max(1, int(s.get("count", 1) or 1))
        except (TypeError, ValueError):
            s["count"] = 1
    # a numbered sentence that names ONE part ('put the black rubber ring onto ...': no number, no plural) installs
    # one - the VLM's 'count: 2' was a guess
    numbered = set(_numbered_texts(prompt))
    for s in spec["steps"]:
        low = s["text"].lower()
        noun = (keywords(s["part"]) or [""])[-1]
        if s["count"] > 1 and s["text"] in numbered and noun and re.search(r"\b%s\b" % re.escape(noun), low) \
                and not re.search(r"\b(?:\d+|%s|all|each|every|both|several)\b" % "|".join(_NUM_WORDS), low):
            s["count"] = 1
    # a compound part ('grey seat and grey cabin roof', 'loader bucket and its yellow arm') is tracked by its FIRST
    # part when the detectors know it - the whole phrase matches whichever class shares the most words (the roof)
    for s in spec["steps"]:
        first = re.split(r"\s+and\s+", str(s.get("part") or ""), 1)[0]
        first = re.sub(r"^(?:the|a|an)\s+", "", first.strip(), flags=re.I)
        if first != s["part"] and labels and match(first, list(labels)):
            s["part"] = first

    # GROUPED work, found by STRUCTURE (no task words): a repeating pattern A, B, A, B, ... where every A is the
    # same one-off action (the gate: done to a fixture first) and every B handles the same part, becomes ONE
    # grouped step - B's part x (repeats x B's count), gated by A. The run then reports '<A> (<fixture> M)' ->
    # '<B> <part> N of <fixture> M' itself.
    grp = re.search(r"(\d{1,3})\s+([a-z][a-z ]{1,28}?)\s+(?:per|on each|for each|on every|in each)\s+([a-z]{3,20})", p)

    # precondition counts stated in the prompt ("must be loaded with 5 splines and 20 yellow pads BEFORE ...")
    # are check steps at the start - added back if the compiler dropped them, marked so they are never folded
    pre = []
    for n, noun in precondition_counts(p):
        hit = next((s for s in spec["steps"] if kw_overlap(set(keywords(s["part"])), set(keywords(noun)))
                    and any(v in str(s.get("action_query") or "").lower() for v in _CHECK_VERBS)), None)
        if hit is not None:
            hit["precondition"], hit["count"] = True, max(hit.get("count", 1), n)
        else:
            part = " ".join(stem(w) for w in noun.split())
            pre.append({"id": f"pre{len(pre) + 1}", "action_query": "check", "part": part, "count": n,
                        "text": f"confirm {n} {noun} are present (precondition)", "precondition": True})
    spec["steps"] = pre + spec["steps"]
    spec["steps"] = _fold_gated(spec["steps"])

    # a run of consecutive steps on the SAME part -> one count step - never two steps the prompt numbers apart
    # ('5. put the rubber ring ...' and '6. screw the side handle ...' are two steps, whatever the VLM named them)
    merged = []
    for s in spec["steps"]:
        same_verb = (merged and (merged[-1]["action_query"].split() or [""])[0].lower()
                     == (s["action_query"].split() or [""])[0].lower())
        apart = merged and s["text"] in numbered and merged[-1]["text"] in numbered and s["text"] != merged[-1]["text"]
        if merged and same_verb and not apart and \
                kw_overlap(set(keywords(merged[-1]["part"])), set(keywords(s["part"]))):
            merged[-1]["count"] += s["count"]
        else:
            merged.append(dict(s))
    spec["steps"] = merged

    # an explicit TOTAL in the prompt ("30 plates in total", "load all 20") wins
    for s in spec["steps"]:
        pw = set(keywords(s["part"]))
        for m in re.finditer(r"(\d{1,4})\s+([a-z ]{2,30}?)\s*(?:in total|total|altogether|in all)", p):
            if kw_overlap(pw, set(keywords(m.group(2)))):
                s["count"] = max(s["count"], int(m.group(1)))
        for m in re.finditer(r"(?:all|total of)\s+(\d{1,4}|%s)\s+([a-z ]{2,30})" % "|".join(_NUM_WORDS), p):
            if kw_overlap(pw, set(keywords(m.group(2)))):
                s["count"] = max(s["count"], int(_NUM_WORDS.get(m.group(1), m.group(1))))

    for s in spec["steps"]:
        s["groups"] = 0
        if s["count"] <= 1 or s.get("precondition"):
            continue
        pw = set(keywords(s["part"]))
        try:
            groups, spg = int(s.get("groups") or 0), int(s.get("slots_per_group") or 0)
        except (TypeError, ValueError):
            groups, spg = 0, 0
        gpart = str(s.get("group_part") or "").strip().lower()
        if grp and kw_overlap(pw, set(keywords(grp.group(2)))):
            spg, gpart = int(grp.group(1)), grp.group(3)
        if spg and not groups:
            groups = s["count"] // spg
        if groups and not spg:
            spg = s["count"] // groups
        if groups >= 2 and spg >= 1 and gpart:
            s["count"] = max(s["count"], groups * spg)
            try:
                vis = int(s.get("visible_per_group") or spg)
            except (TypeError, ValueError):
                vis = spg
            gpart = stem(re.sub(r"[^a-z ]", "", gpart).strip())
            s.update(groups=groups, slots_per_group=spg, visible_per_group=max(1, min(spg, vis)), group_part=gpart)
            # the gate: an action done to each fixture before its items (from the compiler, or from the folded
            # A,B,A,B pattern). Its tool is whatever object it names that is not the fixture itself.
            gate = s.get("gate") if isinstance(s.get("gate"), dict) else None
            if gate is None and s.get("scan_gate"):                 # older compiler field (yes / no)
                gate = {"action": f"scan the {gpart}", "tool": None}
            if gate:
                gate["action"] = str(gate.get("action") or f"prepare the {gpart}").strip()
                tool = gate.get("tool")
                if not tool and gate.get("part") and stem(str(gate["part"]).lower()) != gpart:
                    tool = gate["part"]
                tool = str(tool or "").strip().lower()
                gate["tool"] = tool if tool and tool not in ("null", "none") else None
            s["gate"] = gate
        else:
            s["groups"] = 0                 # (keep any gate on it: _fold_gate_step hands it to the grouped step)
    spec["steps"] = _fold_gate_step(spec["steps"])

    old2new = {}
    for i, s in enumerate(spec["steps"], 1):
        old2new[s.get("id")] = i
        s["id"] = i
        s["timestamp"] = None          # filled with the real completion time after the run
    for c in spec["qa_checks"]:
        c["step_id"] = old2new.get(c["step_id"]) if c["step_id"] is not None else None
    return spec


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


# ======================================================================
# optional helpers: hands (MediaPipe) and SAM2 tracking - both degrade gracefully
# ======================================================================
class HandTracker:
    """MediaPipe Hands - 'is a hand actually on this part' signal."""
    _URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

    def __init__(self, model_path, max_hands=2, min_det_conf=0.5):
        import urllib.request
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self._mp = mp
        mp_ = Path(model_path)
        if not mp_.is_file():
            mp_.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(self._URL, str(mp_))
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(mp_)),
            num_hands=max_hands, min_hand_detection_confidence=min_det_conf,
            running_mode=vision.RunningMode.IMAGE)
        self._lm = vision.HandLandmarker.create_from_options(options)
        _mp_close_at_exit(self._lm)                        # (see _mp_close_at_exit: no shutdown TypeError)

    def detect(self, frame_rgb, pad_frac=0.15):
        h, w = frame_rgb.shape[:2]
        res = self._lm.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                             data=np.ascontiguousarray(frame_rgb)))
        boxes = []
        for hl in res.hand_landmarks:
            xs, ys = [p.x * w for p in hl], [p.y * h for p in hl]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            pad = pad_frac * max(x2 - x1, y2 - y1, 1)
            boxes.append([x1 - pad, y1 - pad, x2 + pad, y2 + pad])
        return boxes


_NEG_PREFIX = re.compile(r"^(?:no|not|non|without|bare|missing|un)[ _-]+")


def pipeline_stem(w):
    """plural + -ing / -ed: gloves -> glove, filled -> fill, loading -> load"""
    pass
    w = stem(w)
    for suf in ("ing", "ed"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    return w


def evidence_classes(spec, models, objects):
    """class names of your OWN models that the prompt mentions (also through a negation: 'no_glove' counts
    for a glove rule) but that are not already a spec object -> detected as evidence for the QA checks."""
    pass
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
                 sam2_weights=None, hand_model=None, spec_dir=None, keep_frames=False,
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
        self.hand_model = hand_model or str(pipeline_ROOT / "weights" / "hand_landmarker.task")
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
        self._hands = None
        self.n_vlm = 0

    # ── plumbing ──
    def _check(self):
        if self.stop.is_set():
            raise Cancelled()

    def progress(self, stage, frac=None):
        self._progress(stage, frac)

    def preview(self, i_or_rgb, dets=None, text=""):
        if self._preview is None:
            return
        try:
            rgb = self.frgb(i_or_rgb) if isinstance(i_or_rgb, (int, np.integer)) else i_or_rgb
            self._preview(rgb, dets, text)
        except Exception:
            pass

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

    def find_scenery(self, cover=0.6, edge=0.05, same=0.6, touched=0.02):
        """SCENERY - a detection that is no part: the same box, never moving, there from the first to the last
        seconds (`cover` of the frames), and no other object ever on it - a printed picture of the part on the
        instruction sheet or a poster. A part that never moves marks no step anyway; a container that is filled (a
        box, a rack) is kept, the parts going into it touch it. Its detections are dropped from every frame.
        -> [(name, median box)]"""
        idx = [f["idx"] for f in self.frames]
        T = len(idx)
        if T < 20:
            return []
        cands = []
        for nm in sorted({d["name"] for i in idx for d in self.dets.get(i, [])}):
            per = [[d["box"] for d in self.dets.get(i, []) if d["name"] == nm] for i in idx]
            seeds = [b for bs in per for b in bs]
            tried = []
            for b0 in seeds[::max(1, len(seeds) // 150)]:
                if any(iou(b0, b) > same for b in tried):
                    continue
                tried.append(b0)
                hit = np.array([any(iou(b0, b) > same for b in bs) for bs in per])
                k = np.flatnonzero(hit)
                if hit.mean() >= cover and k[0] <= edge * T and k[-1] >= (1 - edge) * T - 1:
                    box = np.median([b for bs in per for b in bs if iou(b0, b) > same], axis=0).tolist()
                    cands.append((nm, [round(v, 1) for v in box]))

        def on(b, c):
            cx, cy = _centroid(b)
            return c[0] <= cx <= c[2] and c[1] <= cy <= c[3]
        out = []
        for nm, c in cands:
            n = sum(any(on(d["box"], c) and not any(d["name"] == m and iou(d["box"], b) > same for m, b in cands)
                        for d in self.dets.get(i, []) if d["name"] != nm) for i in idx)
            if n <= touched * T:
                out.append((nm, c))
        if out:
            self.scenery = out
            for i in idx:
                if i in self.dets:
                    self.dets[i] = [d for d in self.dets[i] if not self._is_scenery(d)]
            self.log("  scenery (never moves, nothing ever on it - e.g. a printed picture; ignored): " +
                     ", ".join(f"{nm} at {[int(v) for v in b]}" for nm, b in out))
        return out

    def hands(self):
        if self._hands is None:
            try:
                self._hands = HandTracker(self.hand_model)
                self.log("[hands] MediaPipe hands ready")
            except Exception as e:
                self.log(f"[hands] MediaPipe unavailable ({type(e).__name__}) - hand-contact signal off")
                self._hands = False
        return self._hands or None

    # ── STAGE 1 ──
    def compile_spec(self, prompt, labels=(), regenerate=False):
        labels = sorted({str(x).replace("_", " ").strip() for x in labels if str(x).strip()})
        # SPEC_VERSION changes whenever the clean-up rules change: old cached specs are then rebuilt
        key = hashlib.sha1((prompt + "|" + ",".join(labels) + "|" + str(getattr(self.vlm, "model_name", ""))
                            + "|" + SPEC_VERSION)
                           .encode("utf-8")).hexdigest()[:12]
        cache = self.spec_dir / f"arch_{key}.json"
        n_num = _numbered_steps(prompt)
        if cache.is_file() and not regenerate:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if cached.get("mode") != "sequence" or len(cached.get("steps") or []) >= n_num:
                self.log(f"[spec] from the cache ({cache.name})")
                return cached
            self.log(f"[spec] the cached spec has {len(cached.get('steps') or [])} steps but the prompt numbers "
                     f"{n_num} - writing it again")
        # a NUMBERED prompt is written by rules - no VLM, milliseconds; the VLM only for free-form descriptions
        if USE_RULE_SPEC:
            rprompt = as_numbered(prompt, labels)         # an un-numbered list of steps: numbered first
            n_rule = _numbered_steps(rprompt)
            if rprompt is not prompt:
                self.log(f"[spec] the prompt lists {n_rule} steps without numbers - read as: "
                         + " | ".join(_numbered_texts(rprompt)))
            rs = rule_spec(rprompt, labels)
            if rs is not None:
                vlm_steps = json.loads(json.dumps(rs["steps"]))
                spec = postprocess_spec(rs, rprompt, labels)
                if spec["mode"] == "sequence" and len(spec["steps"]) == n_rule:
                    spec["_known_labels"] = labels
                    spec["_vlm_steps"] = vlm_steps
                    spec["_written_by"] = "rules"
                    self.log(f"[spec] {n_rule} numbered steps - written by rules (no VLM): "
                             + ", ".join(f"{s['id']} {s['part']}" for s in spec["steps"]))
                    self.spec_dir.mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps(spec, indent=2), encoding="utf-8")
                    return spec
                self.log("[spec] the rules could not read every numbered step - asking the VLM")
        q = META.replace("{known}", _KNOWN.format(labels=", ".join(labels)) if labels else "") + prompt.strip()
        spec, raw, short = None, "", None
        for attempt in range(3):
            if attempt == 0:
                ask = q
            elif spec is None:
                ask = q + "\n\nYour previous answer was not a valid JSON object with a 'steps' list. Return ONLY the JSON object."
            else:
                ask = (q + f"\n\nYour previous answer had only {len(spec['steps'])} steps, but the description numbers "
                       f"{n_num} steps (1 to {n_num}). Output exactly {n_num} steps - one per number, in the same order, "
                       "each with its own action_query and part. Return ONLY the JSON object.")
            raw = self.ask(ask, None, 1000, "spec")
            spec = jparse(raw)
            if not (isinstance(spec, dict) and ("steps" in spec or "monitor" in spec)):
                spec = None
                continue
            # a small VLM sometimes keeps one numbered step and drops the rest: ask again, keep the longest answer
            if spec.get("mode", "sequence") == "sequence" and len(spec.get("steps") or []) < n_num:
                self.log(f"[spec] the VLM wrote {len(spec.get('steps') or [])} steps, the prompt numbers {n_num} - asking again")
                if short is None or len(spec["steps"]) > len(short["steps"]):
                    short = spec
                continue
            break
        else:
            spec = short or spec
            if short is not None:
                self.log(f"[spec] WARNING: after 3 tries the VLM still wrote only {len(short['steps'])} of the "
                         f"{n_num} numbered steps - a larger VLM (7B) writes this prompt correctly")
        if spec is None:
            raise RuntimeError(f"spec compile failed - the VLM answer was not usable:\n{raw[:600]}")
        vlm_steps = json.loads(json.dumps(spec.get("steps") or []))
        spec = postprocess_spec(spec, prompt, labels)
        if spec["mode"] == "sequence" and not spec["steps"]:
            raise RuntimeError("spec compile failed - no steps with a part found in the prompt")
        spec["_known_labels"] = labels
        spec["_vlm_steps"] = vlm_steps                   # what the VLM wrote, before the clean-up (to trace a bad spec)
        self.spec_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        return spec

    # ── STAGE 2 ──
    def sample_frames(self, video, seg=None):
        """seg: (t0, t1) seconds - sample only that part of the video (one cycle); timestamps then start at 0
        at t0, so every later stage sees a short video of its own"""
        fdir = self.out / "_frames"
        if fdir.exists():
            shutil.rmtree(fdir, ignore_errors=True)
        fdir.mkdir(parents=True)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"could not open the video: {video}")
        src = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not 1.0 <= src <= 240.0:
            src = 30.0
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        f0 = f1 = None
        if seg:
            f0 = int(round(seg[0] * src))
            f1 = int(round(seg[1] * src)) if seg[1] is not None else None
            if f0 > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
            n_total = (min(f1, n_total) if f1 and n_total else f1 or n_total) - f0
        step = max(1, round(src / self.fps))
        # an event classifier (scan / slot) sees events of ~1 s: keep the full rate rather than miss them
        events_cls = any(m.get("enabled") and m.get("type") in ("yolo_cls", "feat_cls") for m in self.models)
        if n_total and n_total / step > MAX_FRAMES and events_cls:
            self.log(f"  long video ({n_total / src / 60:.1f} min) - kept at {src / step:.2f} fps because an event "
                     f"classifier is used (short scans would fall between frames); split it into cycles to go faster")
        elif n_total and n_total / step > MAX_FRAMES:     # a very long video: fewer frames per second, not hours
            step = int(np.ceil(n_total / MAX_FRAMES))
            self.log(f"  long video ({n_total / src / 60:.1f} min): sampling every {step} frames "
                     f"({src / step:.2f} fps) to stay under {MAX_FRAMES} frames")
            self.fps = src / step
        self.eff_fps = src / step
        out, i, si = [], 0, 0
        while True:
            if f1 is not None and f0 + si >= f1:
                break
            if si % step == 0:
                ok, fr = cap.read()
                if not ok:
                    break
                h, w = fr.shape[:2]
                s = self.max_dim / max(h, w)
                if s < 1:
                    fr = cv2.resize(fr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
                p = str(fdir / f"{i:06d}.jpg")
                if cv2.imwrite(p, fr, [cv2.IMWRITE_JPEG_QUALITY, 93]):
                    out.append({"idx": i, "ts": round(si / src, 3), "path": p})
                    i += 1
                if i % 50 == 0:
                    self._check()
                    self.progress("sampling frames", si / n_total if n_total else None)
            elif not cap.grab():
                break
            si += 1
        cap.release()
        if not out:
            raise RuntimeError("frame sampling produced nothing - is the video readable?")
        for k, f in enumerate(out):                       # idx == position, always
            f["idx"] = k
        self.frames = out
        self.src_fps = src
        self.video_path, self.seg = str(video), seg       # the zero-shot action model reads the video itself
        return out

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


    # ── STAGE 3 ──
    def _det_cache_path(self):
        """outputs/_detcache/<key>.json - the detections of this video as sampled now, with these detectors: the key
        changes with the video file, the sampled frames, the frame size, the objects and the models"""
        video = getattr(self, "video_path", None)
        if not video or not os.path.isfile(video) or not self.frames:
            return None
        st = os.stat(video)
        mods = [{k: m.get(k) for k in ("name", "type", "path", "conf", "classes", "enabled", "regions")}
                for m in self.models if m.get("enabled")]
        for m in mods:                                    # retrained weights at the same path = a new key
            p = m.get("path")
            m["mtime"] = os.path.getmtime(p) if p and os.path.isfile(p) else None
        key = json.dumps([os.path.abspath(video), st.st_size, st.st_mtime, self.max_dim, self.frgb(0).shape[:2],
                          [f["idx"] for f in self.frames], sorted(self.bank.objects),
                          getattr(self.bank, "mapping", {}), mods,
                          sorted(getattr(self.bank, "multi", ()) or ())], sort_keys=True, default=str)
        return pipeline_ROOT / "outputs" / "_detcache" / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"

    def detect_all(self):
        seen = {}
        n = len(self.frames)
        t0 = time.time()
        cache = self._det_cache_path() if self.use_detections else None
        if cache is not None and cache.is_file():
            try:
                self.dets = {int(k): v for k, v in json.loads(cache.read_text(encoding="utf-8")).items()}
                self.log(f"  detections read from the cache ({cache.name}) - this video was detected before")
            except Exception as e:
                self.log(f"  (detection cache not read: {type(e).__name__}: {e})")
                self.dets = {}
        for k, f in enumerate(self.frames):
            if f["idx"] in self.dets:
                continue
            if k % 10 == 0:
                self._check()
                el = time.time() - t0
                eta = el / k * (n - k) if k else 0
                self.progress(f"detecting {k}/{n} frames" + (f" (~{eta:.0f}s left)" if k else ""), k / n)
            dets = self.detect(f["idx"])
            self.preview(f["idx"], dets, f"detecting  t={f['ts']:.1f}s")
        if cache is not None and not cache.is_file():
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps({str(k): v for k, v in self.dets.items()}, default=float), encoding="utf-8")
            except Exception as e:
                self.log(f"  (detection cache not written: {type(e).__name__}: {e})")
        self.find_scenery()
        for f in self.frames:
            for d in self.detect(f["idx"]):
                seen.setdefault(d["name"], []).append((f["ts"], d["box"], d["conf"]))
        return seen


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

pass
trained_ROOT = home()
DIR = trained_ROOT / "trained"
trained_TIE_S = 1.0                     # steps of the good video closer than this are in order either way


def model_dir(name):
    return DIR / name


def default_name(video):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", Path(video).stem)[:40] + time.strftime("_%d%b_%H%M")


def trained_load(name):
    d = model_dir(name)
    m = json.loads((d / "model.json").read_text(encoding="utf-8"))
    m["places_file"] = str(d / "places.json")
    return m


def sequence(spec, places_file, prompt=""):
    """the spec's steps as inference will check them (look-alike copies split, each with its learnt place) and when
    the good video takes each one -> [{"step", "label", "places": [(name, t_ref)], "t"}]"""
    pass
    sp = json.loads(json.dumps(spec))
    me = types.SimpleNamespace(log=lambda s: None, places_file=places_file, places_path=Pipeline.places_path)
    Pipeline.apply_positions(me, sp, prompt)
    out = []
    for s in sp.get("steps") or []:
        ps = [(p["name"], p.get("t_ref")) for p in s.get("positions") or []]
        ts = [t for _, t in ps if t is not None]
        out.append({"step": s["id"], "label": pipeline_step_label(s), "part": s.get("part"), "places": ps,
                    "t": max(ts) if ts else None})
    return out


def self_check(seq):
    """the good video must follow the spec: each step's time not before an earlier step's (TIE_S) -> (status, why)"""
    if not any(r["places"] for r in seq):
        return "NO PLACES", "no step's part was found at a place on the table"
    last, why = None, []
    for r in seq:
        if r["t"] is None:
            continue
        if last is not None and r["t"] < last[1] - trained_TIE_S:
            why.append(f"{r['label']} at {r['t']:.1f}s comes before {last[0]} at {last[1]:.1f}s")
        if last is None or r["t"] >= last[1]:
            last = (r["label"], r["t"])
    return ("NORMAL", "the good video follows the spec's order") if not why else ("UNRELIABLE", "; ".join(why))


def save(name, prompt, spec, video, models, places_file):
    """write model.json next to the learnt places.json -> the model"""
    seq = sequence(spec, places_file, prompt)
    status, why = self_check(seq)
    m = {"name": name, "prompt": prompt, "spec": spec, "trained_on": Path(video).name, "trained_on_path": str(video),
         "trained_at": time.strftime("%Y-%m-%d %H:%M"),
         "detectors": [x.get("name") for x in models or [] if x.get("enabled")],
         "sequence": seq, "self_check": status, "self_check_why": why}
    d = model_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.json").write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
    m["places_file"] = places_file
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


def update(name, **kw):
    """add to a saved model (the temporal model's training info ...) -> the model"""
    p = model_dir(name) / "model.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    m.update(kw)
    p.write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
    m["places_file"] = str(model_dir(name) / "places.json")
    return m


def action_model(m):
    """the model's action model (steps timed by their own objects' changes) if it has one -> its labels, or None"""
    d = model_dir(m["name"])
    if not (d / "action_matcher.pt").exists():
        return None
    for f in ("marked_labels.json", "suggested_labels.json"):
        try:
            return {int(k): v for k, v in json.loads((d / f).read_text(encoding="utf-8")).items()}
        except (OSError, ValueError):
            pass
    return {}


def sequence_lines(m):
    """the trained sequence as text (the result panel)"""
    tm = m.get("temporal")
    am = action_model(m)
    L = [f"TRAINED MODEL  {m['name']}", f"  trained on   {', '.join(m.get('train_videos') or [m['trained_on']])}"
                                       f"  ({m['trained_at']})",
         f"  detectors    {', '.join(m.get('detectors') or []) or '-'}"]
    if am is not None:
        # the ACTION MODEL decides the verdict: its step times and its check first; the pickup method is extra
        L.append(f"  self-check   {m.get('action_self_check', '?')} (action model, on the training video)")
        L += ["", "STEP TIMES ON THE GOOD VIDEO (the action model: when each step's own objects change - this "
                  "decides the verdict)"]
        chk = m.get("action_self_check_steps") or {}
        for s in (m.get("spec") or {}).get("steps") or []:
            lb, c = am.get(s["id"]), chk.get(str(s["id"])) or {}
            tt = f"{lb['start']:.1f} - {lb['end']:.1f}s" if lb else "no label"
            L.append(f"  {s['id']:2d}. {(s.get('action_query') or s.get('text') or '')[:40]:<40} {tt:<14} "
                     f"{(lb or {}).get('change', ''):<32} check: {c.get('status', '-')}")
        # the pickup method's times (when a part left its place - for a part put into a box: when the lid hid it)
        # would only mislead next to these; they stay in model.json, not on screen
        L += ["", "Inference checks each video against these STEP TIMES: a step missing, or done out of this order, "
                  "is an ANOMALY."]
        return L
    else:
        L.append(f"  self-check   {m['self_check']} - {m.get('self_check_why', '')}")
    if tm:
        st = tm.get("stats") or {}
        L += [f"  temporal     TCN ({', '.join(tm.get('classes') or [])}) - {st.get('units')} units, "
              f"{st.get('iters')} iterations, {st.get('seconds')}s on {st.get('device')}; labels "
              + ", ".join(f"{k} {v}" for k, v in (st.get("label_counts") or {}).items()),
              "  TCN check    " + ", ".join(f"{c['video']} {c['status']}" for c in tm.get("self_checks") or [])]
    else:
        L.append("  temporal     none (the places only)")
    L += ["",
         "TRAINED SEQUENCE (the spec's steps; when the good video takes each part from its place)"]
    for k, r in enumerate(m["sequence"], 1):
        where = ", ".join(f"{n} {t:.1f}s" if t is not None else n for n, t in r["places"]) or \
            ("no place (not needed - the action model times it)" if am is not None else "no learnt place")
        L.append(f"  {k:2d}. {r['label']:<36} {where}")
    L += ["", "Inference checks each video against this sequence: a step missing, or done out of this order, is an "
              "ANOMALY."]
    return L


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

pass

SLOTS = 3                     # role slots: the target + up to 2 other named objects
PER_ROLE = 6                  # seen, count, movement, size change, appearing (+), disappearing (-)
N_REL = 3                     # target inside / near / overlapping another role
D = SLOTS + SLOTS * PER_ROLE + (SLOTS - 1) * N_REL + 2      # slot flags + roles + relations + work, change
DIL = (1, 2, 4)               # ~10 s receptive field at 3 fps
# FEATURE FUSION (24 Nov, training modules 1-3 - FUSION = False trains the 32 original features as before):
# + interaction_features.py (hand-object contact / grasp / release, relative motion, interaction memory)
# + state_prototypes.py (frozen DINOv2 on the part + its destination vs the step's learnt pre / during / post)
FUSION = False               # 24 Nov test: fusion changed NOTHING on 10 videos (+20-60 ms per check) - off; the visual
                             # prototypes are learnt anyway (VISUAL_PROTOS) for step_lifecycle.py's verification branch
VISUAL_PROTOS = True
N_INT_, N_VIS_ = 10, 4       # = interaction_features.N_INT, state_prototypes.N_VIS
D_FUSED = D + N_INT_ + N_VIS_


def _fuse(X, L, rs, video, proto, learn_span=None):
    """the original features + interaction + visual-state features -> (X fused, the step's prototypes). learn_span
    (start, end) on the good video: the prototypes are learnt from it; else the model's saved ones are used"""
    pass
    t = np.asarray(L["t"], float)
    try:
        prepare(video, L, [rs])
        E = embeddings(video, L, rs)
    except Exception:                                      # no video file to read (a camera): visual features 0
        E = np.zeros((len(t), 384), np.float32)
    if learn_span is not None:
        proto = state_prototypes_learn(E, t, *learn_span)
    V = state_prototypes_features(E, proto)
    if len(t) > 2:
        V = np.stack([_smooth_s(V[:, i], t) for i in range(V.shape[1])], 1)
    return np.concatenate([X, interaction_features_features(L, rs), V], 1).astype(np.float32), proto
END_PEAK = 0.35
REACH_S = 6.0                 # the completed state must be reached within this long after the candidate end
VERIFY = 0.5                  # at least this share of the learnt state change must happen at a candidate end
NET_ONLY_VERIFY = 0.8         # ... and this much at a candidate the network alone proposes (no own detector change near)
EVENT_NEAR_S = 1.0            # 'near': the step's own change within this of the candidate


def object_actions__words(s):
    return {w for w in re.findall(r"[a-z]+", (s or "").lower().replace("_", " ")) if len(w) > 2}


def object_actions_roles(step, classes):
    """-> [target class, other named classes ...] for one SOP step (by the words of its part and text)"""
    part_w, text_w = object_actions__words(step.get("part")), object_actions__words(step.get("text")) | object_actions__words(step.get("action_query"))
    cw = {c: object_actions__words(c) for c in classes}
    score = lambda c, ws: len(cw[c] & ws) / max(1, len(cw[c]))
    # (25 Nov) a tie on the part is broken by the MORE words shared with it ('black battery cover': 3, not 'battery':
    # 1) before the text - the text may name the class without its adjective ('fit the battery cover')
    tgt = max(classes, key=lambda c: (score(c, part_w), len(cw[c] & part_w), score(c, text_w), -len(cw[c]))) \
        if classes else None
    if tgt is not None and score(tgt, part_w) == 0 and score(tgt, text_w) == 0:
        tgt = None
    others = sorted((c for c in classes if c != tgt and score(c, text_w) >= 0.99),
                    key=lambda c: -score(c, text_w))[:SLOTS - 1]
    return [r for r in [tgt] + others if r]


def _track(L, c):
    """per frame the largest box of class c (or None) and its count"""
    out, cnt = [], []
    for ds in L["dets"]:
        bs = [b for n, b in ds if n == c]
        cnt.append(len(bs))
        out.append(max(bs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if bs else None)
    return out, np.array(cnt, float)


PREFIX_TRAIN = True   # (24 Nov, training P0) every 2nd pass: the video CUT at a random moment (the live view)
PREFIX_MIN = 0.2      # ... between this share of the video and its end
DROPOUT_P = 0.3       # ... and in this share of the passes a short detection dropout of the step's object (0.3-1 s)
SMOOTH_S = 0.3        # TIME-BASED windows (24 Nov; were 3 / +-3 / +-2 FRAMES = these at 10 fps): signal smoothing,
PEAK_S = 0.3          # the 'action ends here' peak neighbourhood,
SCORE_S = 0.2         # and the window a candidate's network score is read in
MOTION_UNIT_S = 0.1   # motion features: displacement per this much time (was per FRAME - depended on the fps)


def _nf(t, sec):
    """seconds -> frames at this record's frame rate (at least 1)"""
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 3
    return max(1, int(round(sec / max(dt, 1e-3))))


def _smooth_s(x, t, sec=SMOOTH_S):
    n = _nf(t, sec)
    return np.convolve(x, np.ones(n) / n, mode="same") if n > 1 else np.asarray(x, float)


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
        sm = _smooth_s(seen, t)
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
                # per MOTION_UNIT_S of elapsed time (not per frame): the same hand movement, the same value at any fps
                X[i, o + 2] = min(np.linalg.norm(ctr(b) - ctr(bx[j])) / np.sqrt(ma)
                                  * MOTION_UNIT_S / max(float(t[i] - t[j]), 1e-3), 2.0)
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
    pass
    S = signals(video, L, log)
    m = _smooth_s(np.asarray(S["motion"], float), np.asarray(S["t"], float))
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


def _net(d_in, n_steps, h=48, kind="tcn", fps=None):
    """the matcher's temporal model: 'tcn' (3 dilated blocks, ~10 s) or 'lstr' (causal transformer, sopmon/lstr.py)
    - the same input (the step's object features + which step) and output (action happening, action ends here).
    fps (LSTR): its windows in seconds at this frame rate (lstr.for_fps); None = the old frame counts (old models)"""
    if kind == "lstr":
        pass
        if fps:
            lg, sh = for_fps(fps)
            return LSTR(d_in + n_steps, 2, h=h, long=lg, short=sh)
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


def train(model, video, L, labels, log=print, iters=900, out_name="action_matcher.pt", local_state=False,
          kind="tcn"):
    """local_state: the labels' END is the state-change moment itself (automatic labels): the completed state is
    learnt - and checked - right around the end, not from before the start"""
    """the user's saved labels -> trained/<model>/action_matcher.pt (+ the learnt roles and completed states)"""
    import torch
    d = Path(model["places_file"]).parent
    classes = sorted({n for ds in L["dets"] for n, _ in ds})
    work, change = _signals(video, L, log)
    t = np.array(L["t"], float)
    steps = model["spec"]["steps"]
    ids = [s["id"] for s in steps]
    info, xs, ya, ye, ends = {}, [], [], [], []
    lines = []
    seen_of = {c: _smooth_s((_track(L, c)[1] > 0).astype(float), t) for c in classes}
    for k, s in enumerate(steps):
        rs = object_actions_roles(s, classes)
        lb = labels.get(s["id"])
        # a label carrying the detector change it was found from (automatic) is learnt around its END
        loc = bool(local_state or (lb and lb.get("kind") is not None))
        if lb and lb.get("kind") is not None:
            # an AUTOMATIC label: its evidence is the detector change it was found from (the objects its text names
            # + that change's objects) - not whatever else changed nearby (a flickering class)
            rs = rs + [c for c in (lb.get("cls"), lb.get("other")) if c and c not in rs]
            rs = rs[:SLOTS]
        elif lb and lb.get("end") is not None:
            # + the classes that visibly CHANGED across this step in the correct video (the tape covers the sticker
            #   place): its evidence, learnt - not only the objects its text names
            a0, e0 = float(lb["start"]), float(lb["end"])
            pre_m, post_m = (t >= a0 - 2.0) & (t < a0), (t > e0) & (t <= e0 + 2.0)
            if loc:
                pre_m, post_m = (t >= e0 - 2.5) & (t <= e0 - 0.2), (t >= e0 + 0.2) & (t <= e0 + 2.7)
            if pre_m.any() and post_m.any():
                ch = sorted(((abs(seen_of[c][post_m].mean() - seen_of[c][pre_m].mean()), c) for c in classes
                             if c not in rs), reverse=True)
                rs = rs + [c for d_, c in ch if d_ >= 0.4][:max(0, SLOTS - len(rs))]
        X = step_features(L, rs, work, change)
        info[s["id"]] = {"roles": rs}
        if lb and lb.get("kind") == "holds":
            # (25 Nov) a STATE-HOLDS step (no change of its own: 'leave it closed on the table' after 'close the box'):
            # decided by a rule at inference - the earlier step done + its object then held - not by the network
            info[s["id"]].update(holds={"cls": lb["cls"], "after": lb["after"], "hold_s": float(lb["hold_s"])},
                                 subactions=lb.get("subactions"))
            lines.append(f"  step {s['id']}: done when its {lb['cls']} HOLDS {lb['hold_s']:g} s after step "
                         f"{lb['after']} (no change of its own)")
            continue
        if not lb or lb.get("end") is None:
            lines.append(f"  step {s['id']}: roles {rs} - no label, not trained")
            continue
        a, e = float(lb["start"]), float(lb["end"])
        A = ((t >= a) & (t <= e)).astype(np.float32)
        iv = lb.get("interval")                              # a HIDDEN completion (boundary_learning): the target is
        if iv:                                               # spread over the interval, not one made-up moment
            dist = np.maximum(0.0, np.maximum(float(iv[0]) - t, t - float(iv[1])))
            E = np.exp(-0.5 * (dist / 0.7) ** 2).astype(np.float32)
        else:
            E = np.exp(-0.5 * ((t - e) / 0.7) ** 2).astype(np.float32)
        ends.append(e)
        if FUSION:
            # (24 Nov, modules 1-3) + object relationship / interaction history + visual state prototypes (learnt here
            # from this step's own before / during / after) as extra input channels of the action model
            X, info[s["id"]]["proto"] = _fuse(X, L, rs, video, None, (a, e))
        elif VISUAL_PROTOS:
            # the visual state prototypes only (not a model input): step_lifecycle.py verifies completions with them
            try:
                pass
                prepare(video, L, [rs])
                info[s["id"]]["proto"] = state_prototypes_learn(embeddings(video, L, rs), t, a, e)
            except Exception as ex:
                lines.append(f"  step {s['id']}: visual prototypes not learnt ({type(ex).__name__})")
        sv = state_vec(X)
        bm, am =((t >= e - 2.5) & (t <= e - 0.2), (t >= e + 0.2) & (t <= e + 2.7)) if loc else \
            ((t >= a - 2.0) & (t < a), (t > e) & (t <= e + 2.0))
        pre = sv[bm].mean(0) if bm.any() else sv[:max(1, int(len(t) * .02))].mean(0)
        post = sv[am].mean(0) if am.any() else sv[-3:].mean(0)
        delta = post - pre
        if lb.get("kind") in ("appears", "disappears") and lb.get("cls") in rs:
            # an automatic label's own change is a FULL appear / disappear of its object: expect all of it (the
            # smoothed windows of the good video under-measure it, and then a short dip would pass)
            k_ = rs.index(lb["cls"])
            delta[k_] = 1.0 if lb["kind"] == "appears" else -1.0
        info[s["id"]].update({"start": a, "end": e, "duration": e - a, "pre": pre.tolist(), "post": post.tolist(),
                              "local_state": loc, "start_state": bool(lb.get("kind") == "start"),
                              # the detector change an automatic label was found from: at inference only the SAME
                              # change is a candidate end (the card step: 'card enters box', not 'box appears')
                              "key": ({"cls": lb.get("cls"), "other": lb.get("other"), "kind": lb.get("kind"),
                                       # (25 Nov) learnt from a state change ('closed box' became 'open box')?
                                       "into": lb.get("into"), "from": lb.get("from")}
                                      if lb.get("kind") not in (None, "start") and lb.get("cls") else None),
                              "delta": delta.tolist(),
                              # (25 Nov) the step's sub-actions (semantic_events): the unobservable ones are reported
                              "subactions": lb.get("subactions"),
                              "initial": bool(np.abs(sv[:max(1, int(2.0 / max(1e-3, t[1] - t[0])))].mean(0) - post).max() < 0.3)})
        oh = np.zeros((len(ids), len(t)), np.float32)
        oh[k] = 1.0
        xs.append(np.concatenate([X.T, oh], 0))
        ya.append(A)
        ye.append(E)
        # how long the result STAYED visible in the good video before a later step changed it (the pen in the box:
        # ~2.5 s, then the lid hides it) - the check waits that long, not HOLD_S, for such a step
        ev_ = np.abs(delta) >= 0.3
        hold = object_actions_HOLD_S
        if ev_.any():
            sm = np.apply_along_axis(lambda c: _smooth_s(c, t), 0, sv)
            C = (t >= e) & (np.abs(sm[:, ev_] - post[ev_]).max(1) < 0.35)
            j1 = int(np.searchsorted(t, e))
            while j1 < len(t) and (C[j1] or t[j1] <= e + 0.7):
                j1 += 1
            hold = float(np.clip(t[min(j1, len(t) - 1)] - e, MIN_HOLD_S, object_actions_HOLD_S))
        info[s["id"]]["hold_s"] = hold
        ch = [f"{n} {'+' if v > 0 else '-'}{abs(v):.1f}" for n, v in zip(_state_names(rs), delta) if abs(v) >= 0.3]
        lines.append(f"  step {s['id']}: roles {rs}; labelled {a:.1f}-{e:.1f}s; completed state: "
                     + (", ".join(ch) if ch else "NO visible change of its objects (completion cannot be verified)"))
    if not xs:
        raise ValueError("no saved label - nothing to train")
    # the DESTINATION'S INSIDE, learnt: where the items came to rest in it in the good video (all steps putting
    # something into the same destination pooled) - its rectangle also covers a raised lid / the table behind it
    try:
        pass
        by_dest = {}
        for sid, inf_ in info.items():
            k_ = inf_.get("key")
            if k_ and k_.get("kind") == "enters" and k_.get("other"):
                _, ps_ = placements(L, k_["cls"], k_["other"])
                ok_ = [p for p in ps_ if decide(p, True)[0] == "CONFIRMED"]
                z = learn_zone(L, k_["cls"], k_["other"], ok_)
                if z:
                    by_dest.setdefault(k_["other"], []).append(z)
        zones = {d_: {"u": [min(z["u"][0] for z in zs), max(z["u"][1] for z in zs)],
                      "d": [min(z["d"][0] for z in zs), max(z["d"][1] for z in zs)],
                      "n": sum(z["n"] for z in zs)} for d_, zs in by_dest.items()}
        for sid, inf_ in info.items():
            k_ = inf_.get("key")
            if k_ and k_.get("other") in zones and k_.get("kind") == "enters":
                inf_["zone"] = zones[k_["other"]]
        for d_, z in zones.items():
            lines.append(f"  inside of '{d_}' learnt from {z['n']} resting frames: across {z['u'][0]:.2f}-"
                         f"{z['u'][1]:.2f}, up from its bottom {z['d'][0]:.2f}-{z['d'][1]:.2f} box widths "
                         f"(+{ZONE_MARGIN} margin)")
    except Exception as e:
        lines.append(f"  (destination inside not learnt: {type(e).__name__}: {e})")
    try:                                                # the lifecycle's normal overlaps, learnt from the good video
        pass
        learn_open_when_taken(model, _with_hands(L, video, None, log), info)
    except Exception as ex:
        lines.append(f"  (lifecycle overlaps not learnt: {type(ex).__name__}: {ex})")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    lstr_fps = (1.0 / float(np.median(np.diff(t)))) if kind == "lstr" and len(t) > 1 else None
    net = _net(D_FUSED if FUSION else D, len(ids), kind=kind, fps=lstr_fps).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3 if kind == "lstr" else 2e-3, weight_decay=1e-3)
    xb = torch.tensor(np.stack(xs), device=dev)
    yA, yE = torch.tensor(np.stack(ya), device=dev), torch.tensor(np.stack(ye), device=dev)
    pa = ((1 - yA.mean()) / yA.mean().clamp_min(1e-3)).clamp(1, 20)
    tt = torch.tensor(t, dtype=torch.float32, device=dev)
    te_ = torch.tensor(ends, dtype=torch.float32, device=dev)
    gen = torch.Generator(device="cpu").manual_seed(0)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    nfr = lambda s: max(1, int(round(s / max(float(t[1] - t[0]), 1e-3)))) if len(t) > 1 else 1
    for it in range(iters):
        x, yEe, w = xb, yE, None
        if PREFIX_TRAIN and it % 2 == 1:
            # PREFIX: the video cut at a random moment, nothing after it (what the live model sees) - a completion
            # after the cut is NOT seen yet: its 'ends here' target there is 0 (an incomplete step, a hard negative)
            u = torch.rand(len(ends), generator=gen).to(dev) * (1 - PREFIX_MIN) + PREFIX_MIN
            cut = tt[0] + u * (tt[-1] - tt[0])
            w = (tt[None, :] <= cut[:, None]).float()                     # (steps, T) frames seen
            x = xb * w[:, None, :]
            yEe = yE * (te_[:, None] <= cut[:, None]).float()
        if DROPOUT_P and torch.rand(1, generator=gen).item() < DROPOUT_P and len(t) > 4:
            # DETECTION DROPOUT: the step's object not detected for a short span - its labels unchanged (a lost box is
            # not proof the part was installed)
            ln = int(torch.randint(nfr(0.3), nfr(1.0) + 1, (1,), generator=gen))
            s0 = int(torch.randint(0, max(1, len(t) - ln), (1,), generator=gen))
            x = x.clone()
            x[:, SLOTS:SLOTS + 2, s0:s0 + ln] = 0.0                       # role 0: seen + count
        o = net(x)
        la = bce(o[:, 0], yA, pos_weight=pa, reduction="none")
        le = bce(o[:, 1], yEe, pos_weight=pa * 2, reduction="none")
        loss = ((la + le) * w).sum() / w.sum().clamp_min(1.0) if w is not None else (la + le).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 200 == 0 or it == iters - 1:                # (26 Nov) progress, like the step model's TCN iter lines
            log(f"    Action iter {it:4d}  loss {loss.item():.3f}")
    torch.save({"state": {k: v.cpu() for k, v in net.state_dict().items()}, "ids": ids, "classes": classes,
                "info": info, "kind": kind, "lstr_fps": lstr_fps, "feat_ver": 2 if FUSION else 1}, d / out_name)
    lines.append(f"  action matcher ({kind.upper()}) trained: {len(xs)} steps x {len(t)} frames (positives = each "
                 f"step's own interval, negatives = everything else), final loss {loss.item():.3f} -> {out_name}")
    return lines


def _state_names(rs):
    """the names of state_vec's columns: each role slot 'seen', then the target's relations to the other slots"""
    name = lambda k: rs[k] if k < len(rs) else f"(slot {k})"
    out = [f"{name(k)} seen" for k in range(SLOTS)]
    for k in range(1, SLOTS):
        out += [f"{name(0)} inside {name(k)}", f"{name(0)} near {name(k)}", f"{name(0)} overlaps {name(k)}"]
    return out


LEFT_TAIL_S = 2.0             # a step's object still seen this long at the very end of a video = left out (MISSING)
object_actions_HOLD_S = 5.0                 # a completed state must hold this long (a moving average): a 2-s hand over it is not it
SETTLE = 0.5                  # the step's time: its main change this far reached (0.8 tried: later, not better)
MIN_HOLD_S = 1.5             # ... or as long as it held in the good video before the next step hid it, at least this
LIVE_HOLD_S = 2.0            # LIVE: a state step is shown once its result held this long (the full hold decides later)
# ONLINE CONFIRMATION (24 Nov, test - switch off to go back): live, a state step is confirmed as soon as its completed
# state is seen and has held for a time that depends on how STRONG the evidence is, not a fixed LIVE_HOLD_S / learnt
# hold. Each step on its own evidence - never waiting for the next step. A result that does not hold is taken back by
# the app ('NOT done after all'). Placement steps keep relations.py's rest / vanish rules (hand occlusion != installed).
ONLINE_CONFIRM = True
FAST_HOLD_S = 0.6            # the whole learnt change seen (share >= STRONG): held this long
MED_HOLD_S = 1.0             # most of it (share >= MEDIUM): held this long; weaker evidence waits the full hold
STRONG, MEDIUM = 1.0, 0.8


def _online_hold(inf, share, L=None, t=None, te=None):
    """LIVE: how long this candidate's completed state must have held before it is shown done. OCCLUSION GUARD: a
    step whose evidence is its part DISAPPEARING is not confirmed fast while a hand is over the place the part was
    last seen (hidden by the hand != installed - the spring under the fingers at 5 s); then the full hold applies"""
    full = _live_hold(inf)
    if not ONLINE_CONFIRM or share is None:
        return full
    fast = min(full, FAST_HOLD_S if share >= STRONG else MED_HOLD_S if share >= MEDIUM else full)
    rs = inf.get("roles") or []
    if rs and L is not None and "delta" in inf and inf["delta"][_state_names(rs).index(f"{rs[0]} seen")] <= -0.7:
        # its result is 'the part is GONE': gone INTO / ONTO its destination with the whole lifecycle seen (attachment
        # evidence) - quick, even with a hand still on it and only part of the learnt change seen yet (the caller
        # already required VERIFY of it); only hidden by a hand somewhere (no such evidence) - the full hold
        if len(rs) > 1 and _attach_evidence(L, rs[0], rs[1], te, t)[0]:
            return min(full, ATTACH_HOLD_S)
        if fast < full and _hand_over_last_seen(L, rs[0], te, t):
            return full
    return fast


AT_DEST_GAP = 0.5            # 'at its destination': boxes overlapping or apart by less than this x the smaller's size
ATTACH_STRAY = 0.15         # (on: sightings AT the destination do not count against 'gone')
ATTACH_STRAY_MAX = 2         # ... and at most this many sightings AWAY from it (one-frame stray boxes)
STATE_FLIPS = True          # (25 Nov) a 'disappears' step never matched to its object turning into its other state
PLACE_NEEDS_MOVE = True     # (25 Nov) a placement needs its part to have LEFT where it lay first (>= MOVE_FRAC x size)
# a compound step's sub-action the detector cannot observe ('slide it into its sleeve': no class for the sleeve):
#   "ignore"    the step is decided by its observable sub-actions only (the report says what was not checked)
#   "uncertain" the step is not confirmed as a whole -> RECOGNISED_UNVERIFIED
UNOBSERVABLE = "ignore"
MOVE_FRAC = 0.5


HOLDS_END_S = 1.0           # (25 Nov) the final check: the held state must still be there in the video's last second


def _state_holds_result(h, res, L, t, live):
    """a STATE-HOLDS step (auto_labels._state_holds): done when step h['after'] is done and its object h['cls'] is
    then seen steadily h['hold_s'] - and (whole video) still there at the end (not reopened / taken away)"""
    c, after, hold = h["cls"], h["after"], float(h["hold_s"])
    xp = res.get(after) or res.get(str(after)) or {}
    base = {"start_sec": None, "end_sec": None, "score": None}
    if xp.get("status") not in ("COMPLETED", "INITIAL_STATE", "RECOGNISED_UNVERIFIED") or xp.get("end_sec") is None:
        return dict(base, status="MISSING_OR_UNCONFIRMED", evidence=f"step {after} is not done, so its {c} cannot "
                                                                     f"hold yet")
    te = float(xp["end_sec"])
    seen = np.array([any(n == c for n, _ in ds) for ds in L["dets"]], bool)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
    th, k0, gap = None, None, 0.0
    for j in range(len(t)):
        if t[j] < te:
            continue
        if seen[j]:
            k0, gap = (j if k0 is None else k0), 0.0
            if t[j] - t[k0] >= hold:
                th = float(t[j])
                break
        elif k0 is not None:
            gap += dt
            if gap > 0.3:
                k0, gap = None, 0.0
    if th is None:
        return dict(base, status="MISSING_OR_UNCONFIRMED",
                    evidence=f"its {c} was not seen held {hold:g} s after step {after} ({te:.1f}s)")
    if not live:
        tail = t >= t[-1] - HOLDS_END_S
        if tail.sum() >= 3 and seen[tail].mean() < 0.5:
            return dict(base, status="MISSING_OR_UNCONFIRMED",
                        evidence=f"its {c} held after step {after} ({th:.1f}s) but is not there at the end of the "
                                 f"video (reopened / taken away)")
    return dict(base, status="COMPLETED", start_sec=round(te, 2), end_sec=round(th, 2), holds_ok=True,
                evidence=f"its {c} held {hold:g} s after step {after} (done at {te:.1f}s)"
                         + ("" if live else " and is still there at the end"))


def _moved_before(L, c, t_end):
    """did object c leave the place where it first lay (its centre >= MOVE_FRAC x its size away) before t_end?
    Unknown (never seen / seen only once) counts as moved - only a part seen lying still is held against it"""
    t = np.asarray(L["t"], float)
    P = [(float(t[k]), [b for n, b in L["dets"][k] if n == c]) for k in range(len(t)) if t[k] <= t_end + 0.5]
    P = [(tk, bs[0]) for tk, bs in P if bs]
    if len(P) < 3:
        return True
    t0 = P[0][0]
    first = [b for tk, b in P if tk <= t0 + 1.0]
    ctr = np.median([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in first], 0)
    size = float(np.median([np.hypot(b[2] - b[0], b[3] - b[1]) for b in first]))
    return any(np.hypot((b[0] + b[2]) / 2 - ctr[0], (b[1] + b[3]) / 2 - ctr[1]) >= MOVE_FRAC * size
               for _, b in P)
ATTACH_PROOF = True         # (25 Nov) live + final: a step's part seen going INTO / ONTO another of its objects
END_STATE_PROOF = True      # (25 Nov) the final verdict: a step proven done by the END of the video (see infer)
END_GONE_S = 1.0             # ... its part gone into its destination and not seen for at least this long at the end
END_STATE_S = 1.0            # ... its completed state measured over the last this-many seconds
END_STATE_SHARE = 0.7        # ... the end matching the learnt completed state at least this much
END_STATE_GAIN = 0.3         # ... and this much closer to it than the start was (the work happened in the video)
POST_LIVE_S = 1.5           # LIVE: a placement's learnt completed state counts this long after its entry (was REACH_S)
ATTACH_HOLD_S = 0.5         # LIVE: a part gone with full ATTACHMENT evidence (_attach_evidence) is shown after this
ATTACH_BACK_S = 8.0          # how far back its handling is looked for


def _boxes_of(L, c, k):
    return [d[1] for d in L["dets"][k] if d[0] == c]


_ATT = {}


def _attach_evidence(L, obj, dest, te, t, stray=0.0):
    """cached per live check (the same record, length and question) - see _attach_evidence_"""
    key = (id(L["dets"]), len(t), obj, dest, None if te is None else round(float(te), 2), stray)
    if key not in _ATT:
        if len(_ATT) > 256:
            _ATT.clear()
        _ATT[key] = _attach_evidence_(L, obj, dest, te, t, stray)
    return _ATT[key]


def _attach_evidence_(L, obj, dest, te, t, stray=0.0):
    """ATTACHMENT, from the object's lifecycle - all of it must be seen (else the step waits for the full hold):
      1 handled      it MOVED from where it lay (picked up / manipulated) in the ATTACH_BACK_S before
      2 approached   it came CLOSER to its destination over that handling
      3 at the dest  its LAST sighting is on / at the destination's box (overlapping, or within half its size)
      4 gone there   it vanished there - not seen anywhere since (allowing one stray frame)
      5 no contra    not seen back at the place it lay before
    obj / dest: the step's own objects from the SOP ('put THE SPRING onto THE REFILL'). -> (ok, why)"""
    if te is None or not len(t):
        return False, "no time"
    k_te = int(np.searchsorted(t, te))
    ks = [k for k in range(min(k_te, len(t) - 1), -1, -1) if t[k] >= te - ATTACH_BACK_S]
    track = [(k, _boxes_of(L, obj, k)[0]) for k in sorted(ks) if _boxes_of(L, obj, k)]
    if len(track) < 3:
        return False, "too few sightings before it went"
    cen = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    diag = lambda b: max(1.0, ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5)
    (k0, b0), (kl, bl) = track[0], track[-1]
    moved = max(np.hypot(cen(b)[0] - cen(b0)[0], cen(b)[1] - cen(b0)[1]) for _, b in track)
    if moved < 0.5 * diag(b0):
        return False, "not handled (it did not move from where it lay)"

    def dest_near(k):                                    # the destination's box at frame k (or the nearest within 0.5 s)
        for dk in sorted(range(-15, 16), key=abs):
            j = k + dk
            if 0 <= j < len(t) and abs(t[j] - t[k]) <= 0.5 and _boxes_of(L, dest, j):
                return _boxes_of(L, dest, j)[0]
        return None

    def gap(a, b):                                       # 0 if the boxes overlap, else the gap between them
        return max(0.0, max(a[0], b[0]) - min(a[2], b[2]), max(a[1], b[1]) - min(a[3], b[3]))

    dl, d0 = dest_near(kl), dest_near(k0)
    if dl is None:
        return False, "destination not seen at its last sighting"
    if gap(bl, dl) > 0.5 * min(diag(bl), diag(dl)):
        return False, "last seen away from its destination"
    if d0 is not None and gap(b0, d0) > 0 and gap(bl, dl) >= gap(b0, d0):
        return False, "did not come closer to its destination"
    # seen again AWAY from its destination - a part still showing AT it (a refill end sticking out of the pen until the
    # cap closes it) does not undo going in; stray: a few one-frame re-detections allowed
    def away(k):
        dk = dest_near(k)
        return dk is None or all(gap(b, dk) > AT_DEST_GAP * min(diag(b), diag(dk)) for b in _boxes_of(L, obj, k))
    after = [k for k in range(kl + 1, len(t)) if _boxes_of(L, obj, k) and (not stray or away(k))]
    # away from its destination: at most ATTACH_STRAY_MAX one-frame strays (a cap carried in the hand is seen often)
    if len(after) > (ATTACH_STRAY_MAX if stray else 1):
        return False, "seen again after it went"
    if any(gap(b, b0) == 0 for k in after for b in _boxes_of(L, obj, k)):
        return False, "back at its old place"
    return True, f"handled, brought to the {dest}, gone there at {t[kl]:.1f}s"


def _hand_over_last_seen(L, obj, te, t, back_s=3.0):
    """a hand box over the object's last seen box (> 30 % of it) at any frame from te to now"""
    hands = L.get("hands")
    if not hands or te is None:
        return False
    k_te = int(np.searchsorted(t, te))
    box = None
    for k in range(min(k_te, len(t) - 1), -1, -1):
        if t[k] < te - back_s:
            break
        b = [d[1] for d in L["dets"][k] if d[0] == obj]
        if b:
            box = b[0]
            break
    if box is None:
        return False
    a = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    for k in range(k_te, len(t)):
        for h in hands[k] or []:
            hb = h["box"]
            iw = min(box[2], hb[2]) - max(box[0], hb[0])
            ih = min(box[3], hb[3]) - max(box[1], hb[1])
            if iw > 0 and ih > 0 and iw * ih / a > 0.3:
                return True
    return False


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
    sm = _smooth_s(sv[:, k], t)
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
        d_in = D_FUSED if ck.get("feat_ver", 1) >= 2 else D        # older models: the 32 original features
        net = _net(d_in, len(ck["ids"]), kind=ck.get("kind", "tcn"), fps=ck.get("lstr_fps"))
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
    pass
    d = Path(model["places_file"]).parent
    ck, net = _load(d / model_file)
    ids, info = ck["ids"], ck["info"]
    fused = ck.get("feat_ver", 1) >= 2
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
        if fused:
            X, _ = _fuse(X, L, inf["roles"], video, inf.get("proto"))
        oh = np.zeros((len(ids), len(t)), np.float32)
        oh[ids.index(s["id"])] = 1.0
        with torch.no_grad():
            o = torch.sigmoid(net(torch.tensor(np.concatenate([X.T, oh], 0)[None]))).numpy()[0]
        act = _smooth_s(o[0], t)
        end = _smooth_s(o[1], t)
        pk, sk = _nf(t, PEAK_S), _nf(t, SCORE_S)
        sv = state_vec(X)
        cands = []
        span = float(np.clip(inf["duration"], 2.0, 12.0))           # the action's own length, not its position
        # candidate ends: (a) the network's 'action ends here' peaks, (b) every moment one of THIS step's objects
        # changes (the detector) - the object states drive it, the network helps
        js = {j for j in range(len(t)) if end[j] >= END_PEAK and end[j] >= end[max(0, j - pk):j + pk + 1].max() - 1e-6}
        key = inf.get("key")
        ev_ts = []                                           # this step's own changes (live: the newest is provisional)
        for e in events(L):
            if key:
                ok_e = e["cls"] == key["cls"] and e.get("other") == key["other"] and e["kind"] == key["kind"]
                # (26 Nov) a STATE CHANGE is one event from either side: 'open box disappears into close box' =
                # 'close box appears from open box' (the open box's detections may flicker on after the close)
                if not ok_e and STATE_FLIPS and not e.get("other") and (
                        (key.get("into") and key["kind"] == "disappears" and e["kind"] == "appears"
                         and e["cls"] == key["into"] and e.get("from") == key["cls"])
                        or (key.get("from") and key["kind"] == "appears" and e["kind"] == "disappears"
                            and e["cls"] == key["from"] and e.get("into") == key["cls"])):
                    ok_e = True
                    ev_ts.append(float(e["t"]))
                    js.add(int(np.argmin(np.abs(t - e["t"]))))
                    continue
                # (25 Nov) a step learnt from a REAL disappearance does not match a STATE CHANGE of that object
                # (the box opened: 'closed box' became 'open box' in place) - and one learnt from a state change
                # matches only that change. Older models (no 'into' in the key) count as learnt from a real
                # disappearance
                if ok_e and STATE_FLIPS and key["kind"] == "disappears" and \
                        (e.get("into") or None) != (key.get("into") or None):
                    ok_e = False
                # (26 Nov) ... and one learnt from 'X appears FROM its other state' (the box closed: 'open box' ->
                # 'box') matches only that state change - not an X appearing somewhere else (a false 'box' at the
                # table's edge at 1.6 s). Older models (no 'from' in the key): any appearance, as before
                if ok_e and STATE_FLIPS and key["kind"] == "appears" and key.get("from") and \
                        e.get("from") != key.get("from"):
                    ok_e = False
            else:
                ok_e = e["cls"] in inf["roles"] or (e.get("other") in inf["roles"] and e["cls"] in inf["roles"])
            if ok_e:
                js.add(int(np.argmin(np.abs(t - e["t"]))))
                ev_ts.append(float(e["t"]))
        # a candidate the DETECTOR backs: this step's own change within EVENT_NEAR_S of it (else: the network alone)
        ev_arr = np.array(ev_ts, float)
        backed = lambda j: bool(len(ev_arr)) and float(np.min(np.abs(ev_arr - t[j]))) <= EVENT_NEAR_S
        if live and len(t):
            # only moments whose result has been seen to HOLD - live for at most LIVE_HOLD_S (the full learnt hold,
            # up to 5 s, is the whole video's check; a live step it later fails is taken back in the window)
            hold0 = FAST_HOLD_S if ONLINE_CONFIRM else _live_hold(inf)   # online: each candidate's own hold below
            js = {j for j in js if t[j] <= t[-1] - hold0}
        # a PLACEMENT step (its object put INTO / ONTO another - learnt from 'X enters Y'): its candidates are the
        # object's stays in the destination, each decided by sopmon/relations.py (came to rest, stayed in - not
        # carried over it - + released / not held / the learnt completed state); the object's permanence while hidden
        place = USE_RELATIONS and bool(key and key.get("kind") == "enters" and key.get("other"))
        if place:
            pass
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
            # a moment ONLY the network proposes (no change of the step's own objects there) must show most of the
            # learnt change (MEDIUM): a weak 'ends here' peak with 60 % of it was a premature confirmation (24 Nov)
            ok = share is not None and share >= (VERIFY if backed(j) else max(VERIFY, NET_ONLY_VERIFY))
            # (26 Nov) a step learnt from a STATE CHANGE ('box' appears FROM 'open box') is confirmed only where that
            # change happened (an event backs it) - not by the network + 'a box is visible' (a false 'box' at the
            # table's edge, packing missing battery: 'closed' at 1.6 s, before the mouse went in)
            if ok and STATE_FLIPS and key and (key.get("from") or key.get("into")) and not backed(j):
                ok = False
            sc = float(end[max(0, j - sk):j + sk + 1].max())
            te = _settle(sv, t, float(t[j]), inf["delta"], hold) if ok else float(t[j])
            pending = False
            if ok and live and len(t) and te > t[-1] - _online_hold(inf, share, L, t, te):
                ok = False                                   # its settled moment is too new to have held yet
                pending = True                               # (26 Nov) ... shown at once, provisional, at te
            cands.append({"start": min(st, te), "end": te, "score": round(sc, 2), "verified": ok,
                          "unverifiable": share is None, "share": share, "why": why, "pending": pending,
                          "rank": (share if share is not None else 0.0) + 0.5 * sc})
        for p in (pls if place else []):
            tc = p["completion"]
            j = int(np.argmin(np.abs(t - tc)))
            hold = float(inf.get("hold_s", object_actions_HOLD_S))
            # the learnt completed state across it (from the object's entry): the third kind of placement evidence
            t_v = p.get("entered_rect", p["entered"])        # before/after from when it first came over the box
            share, why_s = _verify(sv, t, t_v, t_v, inf["delta"], hold)
            # live: the learnt state counts only once its whole 'after' window has been seen (a few frames are not it)
            # live: the learnt state counts once it has been seen long enough - REACH_S (6 s) after the entry before;
            # with ONLINE_CONFIRM once it HELD for POST_LIVE_S (the marker in the box waited 6 s, until the lid closed)
            post_wait = POST_LIVE_S if ONLINE_CONFIRM else REACH_S
            post_ok = share is not None and share >= VERIFY and (not live or float(t[-1]) >= t_v + post_wait)
            verdict, why = decide(p, post_ok)
            # (25 Nov) a part that NEVER LEFT where it lay was not put in: the destination came to it (the pen marker
            # lying against the box's side while the lid swings shut - the open box's box grows over it). Box top
            # missing sequence: the marker never moved, 'placed' at 10.2 s
            if PLACE_NEEDS_MOVE and verdict != "NOT_PLACED" and not _moved_before(L_rel, key["cls"], tc):
                verdict, why = "NOT_PLACED", (f"its {key['cls']} never left where it lay - the {key['other']} came to "
                                              f"it (not put in)")
            i0 = int(np.argmin(np.abs(t - p["entered"])))
            while i0 > 0 and act[i0 - 1] >= 0.5 and t[i0 - 1] > p["entered"] - span:
                i0 -= 1
            st = min(float(t[i0]), p["entered"])
            sc = float(end[max(0, j - sk):j + sk + 1].max())
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
                          "pending": bool(live and verdict == "UNCERTAIN" and p.get("persists")),
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
        inf_h = (info.get(sid) or info.get(str(sid)) or {}).get("holds")
        if r is None and inf_h:
            res[sid] = _state_holds_result(inf_h, res, L, t, live)
            continue
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
    # ATTACHMENT PROOF (25 Nov, live AND final): a step not confirmed whose learnt result is one of ITS parts GONE (the
    # refill into the pen, the cap onto the pen) is confirmed the moment that part's whole lifecycle is seen - handled,
    # brought to another of the step's objects, last seen AT it, gone there for ATTACH_HOLD_S, not back
    # (_attach_evidence). The same proof the spring uses, now for EVERY part of the step that disappears and for
    # placement steps too (a refill pushed into a pen held in the hand never 'comes to rest' in it).
    if ATTACH_PROOF and len(t) > 3:
        for s in steps:
            sid = s["id"]
            x, inf = res.get(sid), (info.get(sid) or info.get(str(sid)) or {})
            if not x or x["status"] != "MISSING_OR_UNCONFIRMED" or not inf.get("roles") or "delta" not in inf:
                continue
            rs, dl = inf["roles"], inf["delta"]
            names = _state_names(rs)
            done = None
            # a part that is the MAIN part of an EARLIER step proves that step, not this one (the battery going into the
            # mouse is step 1's - step 2 'fit the cover' also names the battery)
            earlier_main = set()
            for s2 in steps[:steps.index(s)]:
                i2 = info.get(s2["id"]) or info.get(str(s2["id"])) or {}
                k2 = i2.get("key") or {}
                if k2.get("cls") or i2.get("roles"):
                    earlier_main.add(k2.get("cls") or i2["roles"][0])
            for r in rs:
                if f"{r} seen" not in names or dl[names.index(f"{r} seen")] > -0.7 or r in earlier_main:
                    continue                                 # does not disappear in the step / an earlier step's part
                seen_r = [k for k in range(len(t)) if any(d_[0] == r for d_ in L["dets"][k])]
                if not seen_r:
                    continue
                # the FIRST sighting after which the part is (nearly) gone - a few one-frame re-detections (a thin
                # refill end sticking out, a stray box) allowed - for at least ATTACH_HOLD_S
                n_after = np.searchsorted(seen_r, np.arange(len(t)), side="right")
                for k in seen_r[-200:]:
                    if t[-1] - t[k] < ATTACH_HOLD_S:
                        continue
                    for d in rs:
                        if d != r:
                            ok_, why_ = _attach_evidence(L, r, d, float(t[k]) + 1e-3, t, stray=ATTACH_STRAY)
                            if ok_:
                                done = (float(t[k]), r, d, why_)
                                break
                    if done:
                        break
                if done:
                    break
            if done:
                tl, r, d, why_ = done
                x.update(status="COMPLETED", start_sec=round(max(float(t[0]), tl - float(inf.get("duration", 2.0))), 2),
                         end_sec=round(tl, 2), evidence=f"its {r} {why_} - attached to the {d}", end_state_proof=True,
                         detected=100)
    # END-STATE PROOF (25 Nov, the final verdict only): a step not confirmed during the video is DONE when the end of the
    # video proves it - (a) its part provably went INTO its destination and never came back (handled, brought there,
    # last seen there, gone ever since: the refill pushed into the pen fast, never 'at rest'), or (b) its learnt
    # completed state IS there at the end AND its action was seen (a candidate with VERIFY of the change: the cap fitted,
    # 59 % at 6.6 s). A part still lying there is the LEFT-OUT rule's (MISSING), never this one's.
    if not live and len(t) > 3 and END_STATE_PROOF:
        for s in steps:
            sid = s["id"]
            x, inf = res.get(sid), (info.get(sid) or info.get(str(sid)) or {})
            if not x or x["status"] != "MISSING_OR_UNCONFIRMED" or not inf.get("roles") or "delta" not in inf:
                continue
            key = inf.get("key") or {}
            rs = inf["roles"]
            tgt = key.get("cls") or rs[0]
            dest = key.get("other") or next((r for r in rs if r != tgt), None)
            # (a) went into its destination for good
            seen_k = [k for k in range(len(t)) if any(d_[0] == tgt for d_ in L["dets"][k])]
            if dest and seen_k and t[-1] - t[seen_k[-1]] >= END_GONE_S:
                ok_a, why_a = _attach_evidence(L, tgt, dest, float(t[seen_k[-1]]) + 1e-3, t)
                if ok_a:
                    tl = float(t[seen_k[-1]])
                    x.update(status="COMPLETED", start_sec=round(max(float(t[0]), tl - float(inf.get("duration", 2.0))), 2),
                             end_sec=round(tl, 2), evidence=f"end of video: its {tgt} {why_a}, never seen again "
                                                            f"({t[-1] - tl:.1f} s) - it went into the {dest}",
                             end_state_proof=True)
                    continue
            # (b) the END of the video IS its learnt completed state (close to the learnt 'post' on every signal the step
            # changes), the START was not (the work happened in the video), and its action was seen - a candidate with
            # VERIFY of the change, or its part TAKEN during the video (step_lifecycle)
            X_ = step_features(L, rs, work, change)
            sv_ = state_vec(X_)
            dl, post = np.array(inf["delta"]), np.array(inf["post"])
            ev = np.abs(dl) >= 0.3
            if not ev.any():
                continue
            start_v = sv_[t <= t[0] + 2.0].mean(0)
            end_v = sv_[t >= t[-1] - END_STATE_S].mean(0)
            near = lambda v: float(1.0 - np.mean(np.clip(np.abs(v - post)[ev] / np.abs(dl[ev]), 0, 1)))
            end_match, start_match = near(end_v), near(start_v)
            cands = [c for c in (raw.get(sid) or {}).get("cands", []) if c.get("share") is not None
                     and c["share"] >= VERIFY and not c.get("removed")]
            taken = None
            if not cands:
                try:
                    pass
                    taken = interaction(L, tgt, dest)["taken"]
                except Exception:
                    taken = None
            # moved TOWARDS the completed state during the video (signals that are 0 before and after - inside /
            # overlap - make the start half-match already: the GAIN is what counts)
            if end_match >= END_STATE_SHARE and end_match - start_match >= END_STATE_GAIN and (cands or taken is not None):
                if cands:
                    c = max(cands, key=lambda c: c["share"])
                    ts_, te_, how = c["start"], c["end"], f"its action was seen at {c['end']:.1f}s ({c['share']:.0%})"
                else:
                    te_ = float(t[seen_k[-1]]) if seen_k else float(taken)
                    ts_, how = float(taken), f"its {tgt} was taken at {taken:.1f}s"
                x.update(status="COMPLETED", start_sec=round(ts_, 2), end_sec=round(te_, 2),
                         evidence=f"end of video: its learnt completed state is there ({end_match:.0%} match, "
                                  f"{start_match:.0%} at the start) and {how}", end_state_proof=True)
    # STEP LIFECYCLE (24 Nov): interaction (taken / put back / released at the destination) + the visual state (DINOv2
    # prototypes) verify the completions - a premature one vetoed, a hidden placement released + looking done confirmed
    try:
        pass
        lifecycles = step_lifecycle_apply(model, video, L, res, info, live=live, log=log)
    except Exception as ex:
        lifecycles = {}
        log(f"  (step lifecycle not applied: {type(ex).__name__}: {ex})")
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
    # LEFT OUT = proven missing at the END of the video: the step's completed state is its own object GONE (fitted /
    # put in: 'spring seen -1.0'), but that object is still there, steadily seen, in the last seconds (the spring left
    # on the paper). Not live: the work may still be going on.
    if not live and len(t) > 3:
        tail = t >= t[-1] - LEFT_TAIL_S
        for s in steps:
            sid = s["id"]
            x, inf = res[sid], (info.get(sid) or info.get(str(sid)) or {})
            rs = inf.get("roles") or []
            if x["status"] != "MISSING_OR_UNCONFIRMED" or not rs or "delta" not in inf:
                continue
            tgt = rs[0]
            if inf["delta"][_state_names(rs).index(f"{tgt} seen")] > -0.7:
                continue                                     # its result is not 'the object is gone'
            seen = np.array([any(d_[0] == tgt for d_ in L["dets"][k]) for k in range(len(t))], float)
            share = float(seen[tail].mean()) if tail.sum() >= 3 else 0.0
            if share >= 0.8:
                x["status"] = "MISSING"
                x["evidence"] = (f"its {tgt} is still there at the end of the video (seen in {share:.0%} of the last "
                                 f"{LEFT_TAIL_S:g} s) - once this step is done it is gone (good video) - "
                                 + x["evidence"])
                skipped.add(sid)
    order_viol = [] if live else _order_from_preconditions(steps, info, rel_log, res, L, t)
    for v in order_viol:
        j = v["step"]
        skipped.discard(j)
        res[j].update(status="OUT_OF_ORDER", start_sec=round(v["t"], 2), end_sec=None, evidence=v["why"])
    # (25 Nov, P0) COMPOUND steps: a step with a sub-action the detector cannot observe ('slide it into its sleeve' -
    # no class for the sleeve) is not confirmed as a whole: its observable sub-action verified -> RECOGNISED_UNVERIFIED,
    # with which part is uncertain and why (after the rules above, which use its completion as done)
    for s in steps:
        x, inf = res[s["id"]], (info.get(s["id"]) or info.get(str(s["id"])) or {})
        un = [u for u in (inf.get("subactions") or []) if not u.get("observable")]
        if un and x["status"] == "COMPLETED" and UNOBSERVABLE == "ignore":
            # (25 Nov, user) the step is decided by its OBSERVABLE sub-actions only (box step 5 = 5a 'Close the box');
            # what is not checked is still said
            ok_ = [u for u in inf["subactions"] if u.get("observable")]
            x["evidence"] = ("checked: " + "; ".join(f"'{u['text']}'" for u in ok_) + " - not checked (cannot be "
                             "observed): " + "; ".join(f"'{u['text']}'" for u in un) + " | " + x["evidence"])
            continue
        if un and x["status"] == "COMPLETED":
            ok_ = [u for u in inf["subactions"] if u.get("observable")]
            x["status"] = "RECOGNISED_UNVERIFIED"
            x["subactions_unverified"] = [u["text"] for u in un]
            x["evidence"] = ("; ".join(f"'{u['text']}' verified" for u in ok_) + " - " + "; ".join(
                f"'{u['text']}' UNCERTAIN: {u.get('why')}" for u in un) + " | " + x["evidence"])
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
        if x["status"] == "INITIAL_STATE" or x.get("holds_ok"):
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
        # (26 Nov) LIVE: its action has just ENDED (recognised + verified / in its destination and staying) but has
        # not held long enough yet - the moment it ended, so the step is shown at once (provisional) at its time
        x["pending_end"] = None
        if live and x["status"] not in ("COMPLETED", "INITIAL_STATE", "RECOGNISED_UNVERIFIED"):
            pend = [cc for cc in r_["cands"] if cc.get("pending")]
            if pend:
                x["pending_end"] = round(max(float(cc["end"]) for cc in pend), 2)
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
    # LIVE lifecycle: 'in progress' from the moment a step's part is taken; early order warnings (provisional)
    order_warnings = []
    if live and lifecycles:
        try:
            order_warnings = live_status(model, res, info, lifecycles)
        except Exception as ex:
            log(f"  (lifecycle live status not applied: {type(ex).__name__}: {ex})")
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
    vd["order_warnings"] = order_warnings                # live, provisional ('taken while an earlier step is not done')
    vd["lifecycle"] = {str(k): v for k, v in lifecycles.items()}
    if order_viol:                                       # the precondition proof replaces 'missing' for that step
        for v in order_viol:
            j, i = v["step"], v["before"]
            vd["anomalies"] = [a for a in vd["anomalies"] if not a["text"].startswith(
                (f"MISSING: step {j} ", f"MISSING_OR_UNCONFIRMED: step {j} ", f"UNCERTAIN: step {j} "))]
            vd["anomalies"].append({"at": v["t"], "step": "action model", "uncertain": False,
                                    "text": f"OUT OF ORDER: step {j} {lab[j]} at {v['t']:.1f}s happened before step "
                                            f"{i} ({lab[i]}, {v['t_before']:.1f}s) - {v['why']}"})
            for s in vd["steps"]:
                if s["step"] == j:
                    s.update(status="OUT OF ORDER", t=v["t"], why=v["why"], broken=[(i, v["t_before"])])
            vd.setdefault("precedence", []).append({"step": j, "before": i, "t": v["t"], "t_before": v["t_before"]})
        vd["status"] = "ANOMALY"
    return res, vd


ENGAGE_S = 1.5        # a placement step's object held IN / AT its destination this long = its work has begun
AT_PLACE_S = 1.0      # ... while an earlier step's object still lay at its place this long just before = out of order


def _order_from_preconditions(steps, info, rel_log, res, L, t):
    """OUT OF ORDER from a broken PRECONDITION. A placement step j (its object into / onto another) is under way -
    its object engaged in its destination (near / entering / inside) for >= ENGAGE_S - while the object of an
    EARLIER step i, whose completed state is 'that object gone' (fitted / put in), still lies at its place
    (steadily seen, not moving). In the good video step i's object is always gone by then. The pen, wrong order:
    the refill goes into the pen at 8.1 s while the spring still lies on the paper (until 8.6 s) - step 2 before
    step 1; the refill sticks out of the pen, so step 2 is never 'placed' and was reported missing.
    -> [{"step": j, "before": i, "t": engaged, "t_before": i's object left its place, "why"}]"""
    out = []
    if len(t) < 5:
        return out
    dt = float(np.median(np.diff(t)))
    pos = {}

    def track(c):                                        # per frame: the object's box centre or None
        if c not in pos:
            P = []
            for ds in L["dets"]:
                b = [d_[1] for d_ in ds if d_[0] == c]
                P.append(None if not b else ((b[0][0] + b[0][2]) / 2, (b[0][1] + b[0][3]) / 2))
            pos[c] = P
        return pos[c]

    gone_obj = {}
    for s in steps:
        inf = info.get(s["id"]) or info.get(str(s["id"])) or {}
        rs = inf.get("roles") or []
        if rs and "delta" in inf and inf["delta"][_state_names(rs).index(f"{rs[0]} seen")] <= -0.7:
            gone_obj[s["id"]] = rs[0]
    ids = [s["id"] for s in steps]
    for j, rl in rel_log.items():
        if j not in ids or (res.get(j, {}).get("status") == "COMPLETED" and not res[j].get("end_state_proof")):
            continue                                     # a placed step's order is judged by its time (a step proven
            #                                              only by the END of the video: done, but WHEN in the order?)
        stt = timeline(rl["F"], [(p, p["verdict"]) for p in rl["placements"]])
        Ft = np.asarray(rl["F"]["t"], float)
        eng = np.array([x in ("NEAR", "ENTERING", "INSIDE") for x in stt], bool)
        # the first ENGAGEMENT: engaged frames, gaps up to 0.35 s bridged, lasting >= ENGAGE_S
        t_eng, k0, gap = None, None, 0.0
        for k in range(len(Ft)):
            if eng[k]:
                k0 = k if k0 is None else k0
                gap = 0.0
                if Ft[k] - Ft[k0] >= ENGAGE_S:
                    t_eng = float(Ft[k0])
                    break
            elif k0 is not None:
                gap += dt
                if gap > 0.35:
                    k0, gap = None, 0.0
        if t_eng is None:
            continue
        # the step's object must have been HANDLED (moved from where it first lay) for its engagement to count: a part
        # lying still against its destination's edge (the pen marker beside the opened box) is 'near', not put in
        tgt = rl.get("target")
        Pt = track(tgt)
        first = next((p for p in Pt if p is not None), None)
        k_e = int(np.searchsorted(t, t_eng))
        k_e2 = int(np.searchsorted(t, t_eng + ENGAGE_S))
        bt = next((d_[1] for k in range(len(t)) for d_ in L["dets"][k] if d_[0] == tgt), None)
        size = max(1.0, ((bt[2] - bt[0]) ** 2 + (bt[3] - bt[1]) ** 2) ** 0.5) if bt else 1.0
        moved = first is not None and any(Pt[k] is not None and np.hypot(Pt[k][0] - first[0], Pt[k][1] - first[1])
                                          > 0.5 * size for k in range(k_e, min(len(t), k_e2 + 1)))
        if not moved:
            continue
        for i in ids[:ids.index(j)] if j in ids else []:
            c = gone_obj.get(i)
            if not c or c in (rl.get("target"), rl.get("destination")):
                continue
            if res.get(i, {}).get("status") == "MISSING":    # never done at all (left out): missing, not order
                continue
            P = track(c)
            win = [k for k in range(len(t)) if t_eng - AT_PLACE_S <= t[k] <= t_eng + 0.2]
            pts = [P[k] for k in win if P[k] is not None]
            if len(win) < 3 or len(pts) < 0.8 * len(win):
                continue
            xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
            if max(xs.std(), ys.std()) > 12.0:               # moving = already being handled, not lying there
                continue
            k_eng = int(np.searchsorted(t, t_eng))
            left = next((float(t[k]) for k in range(k_eng, len(t)) if P[k] is None
                         or np.hypot(P[k][0] - xs.mean(), P[k][1] - ys.mean()) > 40), float(t[-1]))
            out.append({"step": j, "before": i, "t": t_eng, "t_before": left,
                        "why": f"its {rl.get('target')} went into the {rl.get('destination')} at {t_eng:.1f}s while the "
                               f"{c} of step {i} still lay at its place (until {left:.1f}s) - in the good video the "
                               f"{c} is gone before this"})
            break
    return out


def _with_hands(L, video, signals, log=print):
    """the record + MediaPipe hands as 'hand' detections (for the placement check's release evidence): the hands
    found live (L['hands']), else from the video file (once, cached); the record unchanged without MediaPipe or when
    the detector has its own hand class"""
    pass
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
    pass
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
    pass
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
        pass
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


def features(pl, spec, eng=None, log=print):
    """the per-frame features of every unit over the run's frames (FeatureStream, causal)
    -> (X (U, T, D) float32, units, t, {"fam"})"""
    pass
    eng = eng or SpatiotemporalEvidenceEngine(log=log, appearance=False, refine=False)
    t = np.array([f["ts"] for f in pl.frames], float)
    H0, W0 = pl.frgb(0).shape[:2]
    fam = eng.families(pl, spec)
    units = _units(spec, fam, W0, H0)
    fps = 1.0 / float(np.median(np.diff(t))) if len(t) > 1 else 3.0
    canon = lambda n: sorted(pl._same_class({n}))[0]
    # this video's layout first (where each part lies in its first seconds) - as live inference does
    relocate(units, [(f["ts"], pl.detect(f["idx"])) for f in pl.frames if f["ts"] < t[0] + REF_S], fam, canon,
             W0, H0, log)
    st = FeatureStream(units, fam, W0, H0, canon, fps)
    pass
    vecs = build(pl, log) or {}                  # the detector's per-box appearance (stored per video)
    X = np.zeros((len(units), len(t), temporal_D), np.float32)
    for k, f in enumerate(pl.frames):
        if k % 50 == 0:
            pl._check()
            pl.progress(f"temporal features {k}/{len(t)}", k / len(t))
        X[:, k] = st.push(pl.frgb(k), pl.detect(f["idx"]), f["ts"], vecs.get(f["idx"]))
    return X, units, t, {"fam": fam}


def labels(units, t, ev):
    """the training labels: PICKUP when the place learner saw the copy taken (t_ref, else the engine's pickup from
    that place), PLACE when the engine saw it go on the product, TOOL while a tool worked -> (U, T) int"""
    Y = np.zeros((len(units), len(t)), np.int64)

    def mark(i, a, b, c):
        Y[i, (t >= a) & (t <= b)] = c

    ops = ev.get("ops") or []
    for i, u in enumerate(units):
        if u["kind"] == "place":
            b = np.array(u["box"]) / ev["W"]
            ctr, size = (b[:2] + b[2:]) / 2, max(b[2] - b[0], b[3] - b[1])
            near = [o for o in ops if o.get("src") is not None and o["family"] == u["family"]
                    and np.linalg.norm(np.array(o["src"]) - ctr) < max(0.7 * size, 0.03)]
            o = min(near, key=lambda o: np.linalg.norm(np.array(o["src"]) - ctr)) if near else None
            tp = u.get("t_ref") if u.get("t_ref") is not None else (o.get("t_pick") if o else None)
            if o is not None and o.get("t_install") is not None:
                mark(i, o["t_install"] - LABEL_S, o["t_install"] + LABEL_S, 2)
            if tp is not None:
                mark(i, tp - LABEL_S, tp + LABEL_S, 1)
        else:
            for o in ops:
                if o["kind"] == "WORK" and o["family"] == u["family"]:
                    mark(i, o.get("t0") or o["t"] - 2.0, o["t"], 3)
    return Y


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
        pass
        return LSTR(d_in, n_out)
    return temporal__net() if (d_in, n_out) == (temporal_D, len(CLASSES)) else None


def kind_of(state):
    return "lstr" if KIND_KEY in state else "tcn"


def temporal_train(data, iters=600, win=240, batch=16, log=print, seed=0, kind="tcn"):
    """data: [(X (U, T, D), Y (U, T))] over the training videos -> the trained net (state dict) + its stats.
    kind: 'tcn' or 'lstr'"""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    seqs = [(x, y) for X, Y in data for x, y in zip(X, Y)]
    if not seqs:
        raise ValueError("no units to train on (no learnt place, no tool)")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = make_net(kind).to(dev)
    # a transformer wants a gentler step than the convolutions
    opt = torch.optim.AdamW(net.parameters(), 1e-3 if kind == "lstr" else 2e-3, weight_decay=1e-3)
    cnt = np.bincount(np.concatenate([y for _, y in seqs]), minlength=len(CLASSES)).astype(float)
    wts = torch.tensor(np.where(cnt > 0, (cnt.sum() / np.maximum(cnt, 1)) ** 0.5, 0.0), dtype=torch.float32, device=dev)
    lossf = torch.nn.CrossEntropyLoss(weight=wts / wts[wts > 0].mean())
    t0 = time.time()
    net.train()
    for it in range(iters):
        xb, yb = [], []
        for _ in range(batch):
            # events are rare: half the windows are centred on one
            x, y = seqs[rng.integers(len(seqs))]
            T = len(y)
            sc = rng.uniform(0.8, 1.25)                     # speed: a slower / faster operator
            n = max(8, int(round(win * sc)))
            ev = np.flatnonzero(y > 0)
            if len(ev) and rng.random() < 0.5:
                a = int(np.clip(rng.choice(ev) - rng.integers(n), 0, max(0, T - n)))
            else:
                a = int(rng.integers(max(1, T - n + 1)))
            xs, ys = x[a:a + n], y[a:a + n]
            idx = np.clip(np.round(np.linspace(0, len(ys) - 1, win)).astype(int), 0, len(ys) - 1)
            xs, ys = xs[idx], ys[idx]
            xs = xs * rng.uniform(0.8, 1.2, (1, temporal_D)) + rng.normal(0, 0.03, xs.shape)     # gain + noise
            xs[:, rng.random(temporal_D) < 0.1] = 0.0                                              # a feature missing
            xb.append(xs.T)
            yb.append(ys)
        xb = torch.tensor(np.array(xb), dtype=torch.float32, device=dev)
        yb = torch.tensor(np.array(yb), dtype=torch.long, device=dev)
        loss = lossf(net(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 200 == 0 or it == iters - 1:
            log(f"    {kind.upper()} iter {it:4d}  loss {loss.item():.3f}")
    stats = {"iters": iters, "seconds": round(time.time() - t0, 1), "units": len(seqs), "kind": kind,
             "label_counts": {c: int(n) for c, n in zip(CLASSES, cnt)}, "device": dev}
    state = {k: v.cpu() for k, v in net.state_dict().items()}
    if kind == "lstr":
        state[KIND_KEY] = torch.tensor(1)
    return state, stats


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


def predict(state, X):
    """-> (U, T, C) probabilities"""
    return Predictor(state)(X)


# ── events + the sequence decoder ──
def _peak(p, t):
    """the strongest moment of one event class for one unit -> (t, prob) or (None, max prob)"""
    if len(p) == 0:
        return None, 0.0
    q = np.convolve(p, np.ones(3) / 3, mode="same")
    k = int(np.argmax(q))
    return (float(t[k]), float(q[k])) if q[k] >= WEAK else (None, float(q[k]))


def decode(spec, units, probs, t, labels=None, X=None):
    """the whole video's TCN output -> events -> the SOP verifier. With the features X, a pickup is the first TCN
    peak CONFIRMED by its copy gone from its place after it (as live inference does); without them, the best peak.
    -> {"status", "steps", "anomalies", "events"}"""
    events = []
    t = np.asarray(t, float)
    for i, u in enumerate(units):
        if u["kind"] == "place":
            if X is not None and len(t):
                tp, sp = None, float(probs[i, :, 1].max()) if probs.shape[1] else 0.0
                for j in _candidates(probs[i, :, 1]):
                    st, empty = confirm(float(t[j]), t, X[i, :, 1], float(t[-1]) + CONFIRM_S + 1.0)
                    if st == "yes":
                        tp, sp = float(t[j]), float(probs[i, j, 1]) * empty
                        break
            else:
                tp, sp = _peak(probs[i, :, 1], t)
            tq, sq = _peak(probs[i, :, 2], t)
            events.append({"unit": u["name"], "step": u["step"], "family": u["family"], "pickup": tp, "p_pickup": sp,
                           "place": tq, "p_place": sq})
        else:
            on = probs[i, :, 3] >= PEAK
            events.append({"unit": u["name"], "tool": True, "seconds": round(float(on.sum() * np.median(np.diff(t))), 1)
                           if len(t) > 1 else 0.0})
    return decode_events(spec, events, labels)


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
    pass
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


# ── train / run, on a Pipeline with the model's places ──
def temporal_build(pl, video, prompt, spec, log=print):
    """frames, detections (the split spec: one step per copy) and the features of `video` -> (X, units, t, ev, sp)"""
    pass
    sp = json.loads(json.dumps(spec))
    pl.apply_positions(sp, prompt)
    if not pl.frames or getattr(pl, "video_path", None) != str(video):
        pl.sample_frames(str(video))
    pl.dets = {}
    pl.build_bank(sp)
    pl.detect_all()
    eng = SpatiotemporalEvidenceEngine(log=log, progress=pl.progress, check=pl._check, appearance=False)
    ev = eng.extract(pl, sp)
    X, units, t, _ = features(pl, sp, eng, log)
    if hasattr(pl.bank, "release"):
        pl.bank.release()
    return X, units, t, ev, sp


def train_model(pl, videos, prompt, spec, model_dir, log=print, kind="tcn"):
    """the training videos -> model_dir/temporal.pt + train_data.npz; then the self-check on each training video.
    kind: the temporal model - 'tcn' or 'lstr'"""
    import torch
    model_dir = Path(model_dir)
    data, per = [], []
    for v in videos:
        log(f"\n[temporal] features + labels: {Path(v).name}")
        X, units, t, ev, sp = temporal_build(pl, v, prompt, spec, log)
        Y = labels(units, t, ev)
        n = {c: int((Y == k).sum()) for k, c in enumerate(CLASSES)}
        log(f"  {len(units)} units, {len(t)} frames, labels {n}")
        data.append((X, Y))
        per.append((v, X, units, t, sp))
    log(f"[temporal] training the {KINDS.get(kind, kind)} ...")
    state, stats = temporal_train(data, log=log, kind=kind)
    torch.save(state, model_dir / "temporal.pt")
    np.savez_compressed(model_dir / "train_data.npz", **{f"X{i}": X for i, (X, _) in enumerate(data)},
                        **{f"Y{i}": Y for i, (_, Y) in enumerate(data)})
    checks = []
    for v, X, units, t, sp in per:
        r = decode(sp, units, predict(state, X), t, X=X)
        checks.append({"video": Path(v).name, "status": r["status"]})
        log(f"  self-check on {Path(v).name}: {r['status']}"
            + ("" if r["status"] == "NORMAL" else "  (" + "; ".join(a["text"] for a in r["anomalies"]) + ")"))
    return {"stats": stats, "self_checks": checks, "classes": CLASSES, "features": temporal_D, "kind": kind}


def _check_features(state):
    """a model trained on another feature set (an older version) cannot read these features"""
    w = state.get("inp.weight")
    if w is not None and int(w.shape[1]) != temporal_D:
        raise RuntimeError(f"this model's temporal model was trained on {int(w.shape[1])} features per frame, this "
                           f"version makes {temporal_D} - train the model again (Train model)")


# ── LIVE inference: the test video frame by frame, the sequence built as it plays ──
LAG_S = 3.0                  # an event is read this long after it (the TCN looks a little ahead)
CONFIRM_S = 3.0              # a pickup counts once its copy is NOT at its place for this long after it
ALL_FRAMES = True            # live: EVERY frame read + shown; the step model gets its trained rate
CAM_KEEP_ALL = True          # (25 Nov) a CAMERA: every frame queued + processed (none skipped), results may lag
DETECT_ALL = False          # (24 Nov) ...and DETECTED only when it goes to the step model: the frames between are shown
                             # with the last boxes - the detector at the model's rate (10 fps), not the video's (30)


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
# INSTALL evidence from the hands (sopmon/hands.py) - shown, not deciding: on the phone videos the installs of two parts
# fitted in one go fall in one hand run, so ordering by them lost a swap the pickups catch (16 Nov, 5/6 vs 6/6)
HAND_WORK = 20               # hand cells (of the 96 x 54 grid) over the product: the hands work on it
HAND_GAP = 1.0               # gaps this short inside a run are bridged
HAND_RUN = 1.0               # a run shorter than this is a hand passing over
HAND_SEARCH = 12.0           # the install run starts within this long after its pickup


def hand_runs(t, cells, closed_before=None):
    """[(start, end)] the runs of hand work on the product; closed_before: only runs ended before this time"""
    t, on = np.asarray(t, float), np.asarray(cells) >= HAND_WORK
    segs, a = [], None
    for i, x in enumerate(on):
        if x and a is None:
            a = i
        if not x and a is not None:
            segs.append([a, i - 1])
            a = None
    if a is not None:
        segs.append([a, len(on) - 1])
    runs = []
    for s in segs:
        if runs and t[s[0]] - t[runs[-1][1]] <= HAND_GAP:
            runs[-1][1] = s[1]
        else:
            runs.append(s)
    out = [(float(t[a]), float(t[b])) for a, b in runs if t[b] - t[a] >= HAND_RUN - 1e-6]
    if closed_before is not None:
        out = [r for r in out if r[1] < closed_before - HAND_GAP]
    return out


def install_time(pickup, runs):
    """the end of the first hand run on the product after a pickup (the part fitted, the hands leave), or None"""
    if pickup is None:
        return None
    for a, b in runs:
        if b > pickup + 0.5 and a < pickup + HAND_SEARCH:
            return b
    return None


def _ensure_bank(pl, model, sp):
    """the model's detectors in pl.bank (+ their appearance hooks pl._roi) - built only when pl has none for this
    model and these detectors yet, so a kept Pipeline loads its weights once"""
    pass
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

    def set(self, prop, value) -> bool:
        return False

    def release(self) -> None:
        self.closed = True


class LiveInference:
    """one test video against a trained model, frame by frame, as it plays: detect -> FeatureStream -> TCN on the
    frames so far -> each unit's PICKUP / PLACE read LAG_S behind -> the steps done so far, compared with the trained
    sequence. step() returns the state after each frame; finish() the final decode over the whole video (the same
    decoder as the report's [5t])."""

    def __init__(self, pl, model, video, fps=3.0, log=print, hands="off", all_frames=None, step_tcn=None):
        """video: a file, a camera index, a stream URL or a FrameFeed (frames pushed by the host);
        step_tcn: the step TCN already loaded (load_step_tcn) - else read from temporal.pt here"""
        all_frames = ALL_FRAMES if all_frames is None else all_frames
        import torch
        pass
        self.pl, self.model, self.log = pl, model, log
        # the HANDS (install evidence, not deciding): YOLOE per frame, or EgoHOS run ahead on the file
        self.hands = None
        if hands and hands != "off":
            pass
            self.hands = Hands(hands, video, log, check=getattr(pl, "_check", None))
        self.hand_c, self.installed = [], {}               # hand cells on the product per frame; unit -> install t
        self.rec = []                                       # per frame (t, product box, [(class, box)]): record()
        # per frame the work on the product (motion) and a small picture of it - the action model's live signals
        self.sig_m, self._crops, self._sg_prev, self._lastP = [], [], None, None
        # MediaPipe hands per live frame (sopmon/mp_hands.py) - the placement check's release evidence; [] if absent
        pass
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
        if isinstance(video, FrameFeed):                   # frames handed in by the host application
            self.camera, self.cap = True, video
        else:
            v = str(video).strip()
            self.camera = v.isdigit() or "://" in v
            self.cap = cv2.VideoCapture(int(v) if v.isdigit() else v)
        if self.camera:
            # a BOUNDED buffer: the driver keeps 1 frame, so a slow step drops old frames instead of falling behind
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except cv2.error:
                pass
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
        if step_tcn is not None:                            # loaded once (load_step_tcn), reused per run
            self.pred = step_tcn
        else:
            state = torch.load(Path(model["places_file"]).parent / "temporal.pt", map_location="cpu")
            _check_features(state)
            self.pred = Predictor(state)
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
        pass
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
        for rgb, dets, app, ts, hm in self.buf:
            self.X.append(self.stream.push(rgb, dets, ts, app))
            self.t.append(ts)
            self._hand_cells(hm, ts)
        self.buf = []

    def _hand_masks(self, rgb, ts):
        return self.hands.masks(rgb, ts) if self.hands is not None else None

    def _hand_cells(self, hm, ts):
        """this frame's hands vs the product box -> self.hand_c; a pickup whose install run has ended -> installed"""
        if self.hands is None:
            return
        pass
        P = self.stream.P if ts - self.stream.P_t <= 1.0 else None
        self.hand_c.append(on_product(hm or [], P, self.W0, self.H0)[1])
        runs = None
        for u, (tp, _) in self.picked.items():
            if u in self.installed:
                continue
            runs = hand_runs(self.t[:len(self.hand_c)], self.hand_c, closed_before=ts) if runs is None else runs
            ti = install_time(tp, runs)
            if ti is not None:
                self.installed[u] = ti
                self.log(f"  [{ti:6.1f}s] INSTALLED  {self.units[u]['name']}  (step {self.units[u]['step']}, the hands "
                         f"left the product)")

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
            last_dets = None
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
                if model or DETECT_ALL or last_dets is None:
                    dets = last_dets = self.pl.bank.detect_rgb(rgb)
                else:
                    dets = last_dets                       # a display frame: shown with the last boxes, not detected
                app = hm = None
                if model:                                  # the step model's frames: + appearance + hands
                    app = self.roi.vectors(dets, rgb.shape) if self.roi is not None and self.roi.ok else None
                    # the hands from the FULL frame (far, small hands need it), on this thread too (GPU)
                    hm = self._hand_masks(np.ascontiguousarray(fr[:, :, ::-1]), ts)
                self.q.put((rgb, ts, dets, app, 1000.0 * (time.time() - t0), hm, model))
                i += 1
        self._stop_read = False
        threading.Thread(target=run, daemon=True).start()

    def _cam_start(self):
        """the camera's capture thread: reads as fast as the camera gives frames. CAM_KEEP_ALL (25 Nov): EVERY frame is
        queued and processed - none skipped, even when processing is slower than the camera (the results then lag
        behind the live picture); otherwise only the newest frame is kept"""
        import collections
        import threading
        self._cam_lock, self._cam, self._cam_id, self._cam_seen = threading.Lock(), None, 0, 0
        self._cam_q = collections.deque()

        def run():
            while True:
                ok, fr = self.cap.read()
                with self._cam_lock:
                    if not ok:
                        self._cam_id = -1
                        return
                    item = (fr, time.time() - self.t_start)
                    if CAM_KEEP_ALL:
                        self._cam_q.append(item)
                    self._cam = item
                    self._cam_id += 1
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
            rgb, ts, dets, app, ms, hm, self._model = item
            self._pre = (dets, app, ms, hm)
            return rgb, ts
        if self.camera:
            # CAPTURE on its own thread (24 Nov): it keeps only the NEWEST frame - inference takes that one when it is
            # free, so a slow step skips frames instead of building a backlog (capture / inference / display apart)
            if isinstance(self.cap, FrameFeed):
                # frames PUSHED by the host (SOPMonitor.process_frame): one per step(), at the host's time; at
                # 30 fps every one goes to the step model too (none skipped)
                ok, fr = self.cap.read()
                if not ok:
                    return None, None
                now = float(self.cap.t) if self.cap.t is not None else time.time() - self.t_start
                self._model = (not self.all_frames) or self.fps >= self.src - 0.5 \
                    or now >= self.n_model / self.fps
                return fr, now
            if getattr(self, "_cam_lock", None) is None:
                self._cam_start()
            due = self.n_model / self.fps
            if CAM_KEEP_ALL:
                # EVERY captured frame, in order (none skipped); each one goes to the step model when the model runs at
                # the camera's rate (30 fps), else one when due
                while True:
                    with self._cam_lock:
                        item = self._cam_q.popleft() if self._cam_q else None
                        ended = self._cam_id < 0
                    if item is not None:
                        fr, now = item
                        self._model = (not self.all_frames) or self.fps >= self.src - 0.5 or now >= due
                        return fr, now
                    if ended:
                        return None, None
                    time.sleep(0.002)
            while True:
                with self._cam_lock:
                    cid, item = self._cam_id, self._cam
                if cid < 0:
                    return None, None
                if cid > self._cam_seen and item is not None:
                    self._cam_seen = cid
                    fr, now = item
                    if self.all_frames:                    # every NEW frame shown; a model frame when one is due
                        self._model = now >= due
                        return fr, now
                    if now >= due:                         # (old way) only a frame once one is due
                        return fr, now
                time.sleep(0.002)
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
        full = fr
        rgb = self._resize(fr) if self.camera or self.i == 1 else fr
        t1 = self._tick("read", tp)
        if self._pre is not None:                          # detected ahead on the reader thread (overlapped)
            dets, app, ms, hm = self._pre
            self.tim["detect"] = (self.tim["detect"] + [ms])[-30:]
            self.last_ms["detect"] = ms
        else:
            if self._model or DETECT_ALL or getattr(self, "_last_dets", None) is None:
                dets = self._last_dets = self.pl.bank.detect_rgb(rgb)
            else:
                dets = self._last_dets                     # a camera display frame: the last boxes, not detected
            app = hm = None
            if self._model:
                app = self.roi.vectors(dets, rgb.shape) if self.roi is not None and self.roi.ok else None
                hm = self._hand_masks(np.ascontiguousarray(full[:, :, ::-1]) if full is not rgb else rgb, ts)
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
            self.buf.append((rgb, dets, app, ts, hm))
            if ts - self.buf[0][3] < REF_S:
                self.proc = (self.proc + [time.time() - tp])[-30:]
                self._state_cache = self.state()
                return {"rgb": rgb, "dets": dets, "t": ts, "probs": None, "state": self._state_cache,
                        "speed": self.speed(), "model_frame": True}
            self._start()
        else:
            self.X.append(self.stream.push(rgb, dets, ts, app))
            self.t.append(ts)
            self._hand_cells(hm, ts)
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
            pass
            try:
                self.hand_boxes.append(hands(rgb, out_w=float(self.W0)))
            except Exception as e:
                self.log(f"  (hands off: {type(e).__name__}: {e})")
                self.hand_boxes = None


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
        runs = hand_runs(self.t[:len(self.hand_c)], self.hand_c) if self.hands is not None else []
        for u, unit in enumerate(self.units):
            if unit["kind"] != "place":
                continue
            pk, pc = self.picked.get(u), self.placed.get(u)
            if self.hands is not None and pk and u not in self.installed:    # a run still open at the end
                ti = install_time(pk[0], runs)
                if ti is not None:
                    self.installed[u] = ti
            at_end = bool(Xa is not None and endw.any() and Xa[u, endw, 1].mean() >= 0.5)
            events.append({"unit": unit["name"], "step": unit["step"], "family": unit["family"],
                           "pickup": pk[0] if pk else None, "p_pickup": pk[1] if pk else self.pmax.get(u, 0.0),
                           "place": pc[0] if pc else None, "p_place": pc[1] if pc else 0.0,
                           "not_placed": u in loose, "at_place_end": at_end, "absent": bool(unit.get("absent")),
                           "pickups": self.repicks.get(u) or ([pk[0]] if pk else []),
                           **({"installed": self.installed.get(u)} if self.hands is not None else {})})
        return decode_events(self.spec, events, labels=self.labels)


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.interaction_features   (from sopmon\interaction_features.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""interaction_features.py - OBJECT RELATIONSHIP + INTERACTION HISTORY features per frame (24 Nov, training module 1).

For one SOP step's TARGET part and its DESTINATION (the step's first two roles), from the detector's boxes and the
MediaPipe hands already in the record (L["hands"]: per frame [{"box", "tips"}]). Deterministic - no trainable graph
(one good video would overfit one); a dynamic 'scene graph' reduced to the edges that matter for the step:

  hand -> target   0 touch       a finger tip on the target's box (grown 10 %)
                   1 hand over   a hand box overlapping it
                   2 grasp       touched AND the target moving
                   3 release     touched in the last 0.5 s, not now (decays over 1 s)
  target -> dest   4 together    both moving the same way (held together / fitted)
                   5 approach    the distance closing (+) or opening (-), per second, in target sizes, clipped
  memory           6 unseen      seconds since the target was last seen (/ MEM_S, clipped)
                   7 untouched   seconds since a hand last touched it (/ MEM_S, clipped)
                   8 moved       how far it is (or was last seen) from where it first lay, in its sizes (clipped)
                   9 at dest     last seen AT the destination (boxes overlapping / within half a size) - kept while
                                 it is unseen: 'went in there' vs 'hidden somewhere else'
All times in seconds (any fps). -> (T, N_INT) float32; zeros without hands / roles.
"""
import numpy as np

N_INT = 10
MEM_S = 5.0
MOVE_THR = 0.15        # centre shift per 0.1 s, in target sizes = moving


def interaction_features__track(L, c):
    out = []
    for ds in L["dets"]:
        bs = [d[1] for d in ds if d[0] == c]
        out.append(max(bs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if bs else None)
    return out


def _ctr(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], float)


def _size(b):
    return max(1.0, float(np.hypot(b[2] - b[0], b[3] - b[1])))


def _gap(a, b):
    return max(0.0, max(a[0], b[0]) - min(a[2], b[2]), max(a[1], b[1]) - min(a[3], b[3]))


def interaction_features_features(L, rs):
    """(T, N_INT) for the step's roles rs = [target, destination, ...]"""
    t = np.asarray(L["t"], float)
    T = len(t)
    X = np.zeros((T, N_INT), np.float32)
    if not rs or T == 0:
        return X
    tg = interaction_features__track(L, rs[0])
    ds = interaction_features__track(L, rs[1]) if len(rs) > 1 else [None] * T
    hands = L.get("hands")
    if hands is None and L.get("hand_tips") is not None:
        # hands added from the video file (mp_hands.add_to): 'hand' boxes in the detections + all finger tips per frame
        hands = [[{"box": d[1], "tips": L["hand_tips"][k] or []} for d in L["dets"][k] if d[0] == "hand"]
                 for k in range(T)]
    hands = hands or [None] * T
    first = next((b for b in tg if b is not None), None)
    last_seen_t, last_touch_t, last_tb, last_at = None, None, None, 0.0
    prev = None                                      # (k, target box) of the last sighting
    prev_d = None                                    # (k, distance) target-destination
    for k in range(T):
        b, d = tg[k], ds[k]
        hs = hands[k] or []
        if b is not None:
            gx, gy = 0.1 * (b[2] - b[0]), 0.1 * (b[3] - b[1])
            touch = any(b[0] - gx <= x <= b[2] + gx and b[1] - gy <= y <= b[3] + gy
                        for h in hs for x, y in (h.get("tips") or []))
            over = any(_gap(b, h["box"]) == 0 for h in hs if h.get("box"))
            X[k, 0], X[k, 1] = float(touch), float(over)
            moving, v = False, None
            if prev is not None and t[k] > t[prev[0]]:
                v = (_ctr(b) - _ctr(prev[1])) / max(t[k] - t[prev[0]], 1e-3)
                moving = float(np.linalg.norm(v)) * 0.1 / _size(b) > MOVE_THR
            X[k, 2] = float(touch and moving)
            if touch:
                last_touch_t = t[k]
            if d is not None and v is not None and prev_d is not None:
                vd = None
                kd = prev_d[0]
                if ds[kd] is not None and t[k] > t[kd]:
                    vd = (_ctr(d) - _ctr(ds[kd])) / max(t[k] - t[kd], 1e-3)
                if vd is not None and moving:
                    cos = float(v @ vd / max(np.linalg.norm(v) * np.linalg.norm(vd), 1e-6))
                    X[k, 4] = max(0.0, cos) * float(np.linalg.norm(vd) * 0.1 / _size(d) > MOVE_THR)
            if d is not None:
                dist = float(np.linalg.norm(_ctr(b) - _ctr(d))) / _size(b)
                if prev_d is not None and t[k] > t[prev_d[0]]:
                    X[k, 5] = float(np.clip((prev_d[1] - dist) / (t[k] - t[prev_d[0]]), -2, 2)) / 2
                prev_d = (k, dist)
                last_at = float(_gap(b, d) <= 0.5 * min(_size(b), _size(d)))
            else:
                last_at = 0.0
            last_seen_t, last_tb, prev = t[k], b, (k, b)
        # release: touched until recently, not now - 1 right after the last touch, decaying to 0 over 1 s
        if last_touch_t is not None and not X[k, 0]:
            X[k, 3] = float(max(0.0, 1.0 - (t[k] - last_touch_t) / 1.0))
        X[k, 6] = min(1.0, (t[k] - last_seen_t) / MEM_S) if last_seen_t is not None else 1.0
        X[k, 7] = min(1.0, (t[k] - last_touch_t) / MEM_S) if last_touch_t is not None else 1.0
        if first is not None and last_tb is not None:
            X[k, 8] = min(2.0, float(np.linalg.norm(_ctr(last_tb) - _ctr(first))) / _size(first)) / 2
        X[k, 9] = last_at
    return X


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.state_prototypes   (from sopmon\state_prototypes.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""state_prototypes.py - VISUAL STATE PROTOTYPES per SOP step (24 Nov, training module 2).

Frozen DINOv2 (ViT-S/14, Apache 2.0) on ONE region per step and frame: the step's TARGET part together with its
DESTINATION (their boxes' union, last known boxes carried up to CARRY_S, padded) - not the whole frame, so the background
does not count. From the good video each step learns three prototypes (mean embeddings, unit length):

  pre     the region just before the step's action starts       (start - 2 s .. start)
  during  while the action happens                               (start .. completion)
  post    just after its completion                              (completion .. + 2 s)

Every frame then gets its cosine similarity to the three + 'post minus the closer of pre / during'. These are INPUT
features of the action model (and evidence), never the only proof of installation: two physical states can look the
same. Embeddings are cached per video frame and region, so the live check (every 0.5 s, on the record so far) only
embeds the new frames. A camera (no file to read again) gets zeros.
"""
from pathlib import Path

pass

import cv2
import numpy as np

N_VIS = 4
CARRY_S = 2.0
PAD = 0.15
PRE_S, state_prototypes_POST_S = 2.0, 2.0
SEEK_GAP = 30                # wanted frames further apart than this: a new seek instead of decoding in between
_M, _C = {}, {}


def _dino():
    if "m" not in _M:
        import timm
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        # weights/dinov2_vits14.safetensors when shipped (no internet needed), else the timm hub download
        local = home() / "weights" / "dinov2_vits14.safetensors"
        kw = {"pretrained_cfg_overlay": {"file": str(local)}} if local.is_file() else {}
        m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0,
                              dynamic_img_size=True, **kw).to(dev).eval()
        if dev == "cuda":
            m = m.half()
        _M.update(m=m, torch=torch, dev=dev)
    return _M["m"], _M["torch"], _M["dev"]


def _embed(crops):
    """[RGB uint8 crops] -> (n, 384) unit vectors"""
    m, torch, dev = _dino()
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    out = []
    for i in range(0, len(crops), 64):
        x = np.stack([(cv2.resize(c, (224, 224), interpolation=cv2.INTER_AREA) / 255.0 - mean) / std
                      for c in crops[i:i + 64]]).transpose(0, 3, 1, 2)
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float16 if dev == "cuda" else torch.float32, device=dev)
            f = m(xt).float().cpu().numpy()
        out.append(f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-6))
    return np.concatenate(out, 0) if out else np.zeros((0, 384), np.float32)


def _regions(L, rs):
    """per frame the region box (L's coordinates) of the step's target + destination, or None"""
    t = np.asarray(L["t"], float)
    out, last = [], {}
    for k, ds in enumerate(L["dets"]):
        bs = []
        for c in rs[:2]:
            b = [d[1] for d in ds if d[0] == c]
            if b:
                last[c] = (t[k], max(b, key=lambda q: (q[2] - q[0]) * (q[3] - q[1])))
            if c in last and t[k] - last[c][0] <= CARRY_S:
                bs.append(last[c][1])
        if not bs:
            out.append(None)
            continue
        x0, y0 = min(b[0] for b in bs), min(b[1] for b in bs)
        x1, y1 = max(b[2] for b in bs), max(b[3] for b in bs)
        w, h = x1 - x0, y1 - y0
        out.append((x0 - PAD * w, y0 - PAD * h, x1 + PAD * w, y1 + PAD * h))
    return out


def prepare(video, L, rs_list, windows=None):
    """embed the frames' region for each step's roles (only what is not cached yet). windows: [(t0, t1)] - only the
    frames in them (the live check looks at a second after a completion, not at the whole video: 25 Nov, a live
    check embedding every new frame stalled the video 7 s)"""
    v = str(video)
    if not v or v.strip().isdigit() or "://" in v or not Path(v).is_file():
        return
    c = _C.setdefault(v, {"emb": {}})
    if "src" not in c:
        cap = cv2.VideoCapture(v)
        c["src"] = cap.get(cv2.CAP_PROP_FPS) or 30.0
        c["W"], c["H"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    sx, sy = c["W"] / max(1, L.get("W0") or c["W"]), c["H"] / max(1, L.get("H0") or c["H"])
    t = np.asarray(L["t"], float)
    todo = {}                                            # video frame index -> [(key, region in pixels)]
    for rs in rs_list:
        if not rs:
            continue
        key = tuple(rs[:2])
        for k, r in enumerate(_regions(L, rs)):
            if windows is not None and not any(a <= t[k] <= b for a, b in windows):
                continue
            fi = int(round(t[k] * c["src"]))
            if r is None or (fi, key) in c["emb"]:
                continue
            todo.setdefault(fi, []).append((key, (max(0, int(r[0] * sx)), max(0, int(r[1] * sy)),
                                                  min(c["W"], int(r[2] * sx)), min(c["H"], int(r[3] * sy)))))
    if not todo:
        return
    # read run by run (a seek per stretch of wanted frames, not every frame in between)
    fis = sorted(todo)
    runs, cur = [], [fis[0]]
    for f in fis[1:]:
        if f - cur[-1] > SEEK_GAP:
            runs.append(cur)
            cur = [f]
        else:
            cur.append(f)
    runs.append(cur)
    cap = cv2.VideoCapture(v)
    crops, keys = [], []
    for run in runs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, run[0])
        fi = run[0]
        while fi <= run[-1]:
            ok, fr = cap.read()
            if not ok:
                break
            for key, (x0, y0, x1, y1) in todo.get(fi, []):
                if x1 - x0 >= 8 and y1 - y0 >= 8:
                    crops.append(np.ascontiguousarray(fr[y0:y1, x0:x1, ::-1]))
                    keys.append((fi, key))
            fi += 1
    cap.release()
    for (fk, key), e in zip(keys, _embed(crops)):
        c["emb"][(fk, key)] = e.astype(np.float32)


def embeddings(video, L, rs):
    """(T, 384) the step's region embedding per frame (zeros where none) - call prepare() first"""
    t = np.asarray(L["t"], float)
    E = np.zeros((len(t), 384), np.float32)
    c = _C.get(str(video))
    if not c or not rs or "src" not in c:
        return E
    key = tuple(rs[:2])
    for k in range(len(t)):
        e = c["emb"].get((int(round(t[k] * c["src"])), key))
        if e is not None:
            E[k] = e
    return E


def state_prototypes_learn(E, t, start, end):
    """the step's pre / during / post prototypes from the good video -> {name: list} or None"""
    t = np.asarray(t, float)
    ok = np.linalg.norm(E, axis=1) > 0.5
    out = {}
    for name, m in (("pre", (t >= start - PRE_S) & (t < start)), ("during", (t >= start) & (t <= end)),
                    ("post", (t > end) & (t <= end + state_prototypes_POST_S))):
        m = m & ok
        if m.sum() < 2:
            return None
        p = E[m].mean(0)
        out[name] = (p / max(np.linalg.norm(p), 1e-6)).tolist()
    return out


def state_prototypes_features(E, proto):
    """(T, N_VIS): cosine to pre / during / post, and post minus the closer of pre / during (0 where no region)"""
    X = np.zeros((len(E), N_VIS), np.float32)
    if not proto:
        return X
    ok = np.linalg.norm(E, axis=1) > 0.5
    P = np.array([proto["pre"], proto["during"], proto["post"]], np.float32)
    S = E @ P.T
    X[:, :3] = np.where(ok[:, None], S, 0.0)
    X[:, 3] = np.where(ok, S[:, 2] - np.maximum(S[:, 0], S[:, 1]), 0.0)
    return X


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.step_lifecycle   (from sopmon\step_lifecycle.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""step_lifecycle.py - the STEP LIFECYCLE MODULE (24 Nov): three independent evidence branches per SOP step -> early
order detection + completion verification, for the live status and the final verdict.

  action evidence   the action model's result (object_actions.infer: candidates verified by the learnt state change,
                    placements, attachment) - what was decided so far
  interaction       interaction_features.py on the step's TARGET part and its DESTINATION:
                      TAKEN      it moved from where it first lay (> TAKE_MOVE x its size, for TAKE_HOLD_S) with a hand on
                                 / over it around then
                      PUT BACK   later back where it lay (within BACK_NEAR x its size) for BACK_HOLD_S - the taking undone
                      RELEASED   a hand let go of it while it was AT its destination
  visual state      state_prototypes.py: the part + destination region vs the step's learnt pre / during / post look
                    (frozen DINOv2) - 'looks before' / 'looks done'

  COMPLETION VERIFICATION
    veto     a step confirmed while its region still clearly LOOKS LIKE BEFORE it (seen in most of the next VETO_S) is
             not confirmed yet (premature) - only with the region in view; never vetoes a step whose look does not change
    confirm  a step not confirmed (a placement hidden by the hand / the next part) whose part was RELEASED AT ITS
             DESTINATION and whose region then LOOKS DONE for POST_HOLD_S -> completed at the release
  LIVE       a step whose part is TAKEN (not put back, not done) is shown 'in progress' from that moment
  EARLY ORDER (live, provisional) a step's part TAKEN while an earlier step is not done - unless that part is one of the
             earlier step's own objects (the refill the spring goes onto) -> a warning; put back = withdrawn. The final
             order verdict stays with the completion times and the precondition rule (object_actions).
"""
import numpy as np

ENABLED = True
TAKE_MOVE, TAKE_HOLD_S = 0.5, 0.3
BACK_NEAR, BACK_HOLD_S = 0.6, 1.0      # 'put back': within 0.6 x its size of where it lay (not the exact spot)
HAND_NEAR_S = 0.5
VETO_S, VETO_SEEN, VETO_PRE = 1.0, 0.6, 0.6
VETO_UNLESS_PLACED = True              # (25 Nov) never veto a part seen taken + let go at its destination (not put back)
AT_DEST_SHARE = 0.8                    # ... and at its destination in this share of the veto window
VISUAL_WINDOWS = True                  # (25 Nov) DINOv2 only on the frames the checks use (speed; same results)
PRE_MARGIN, POST_MARGIN = 0.05, 0.02
POST_HOLD_S, POST_SHARE = 0.6, 0.7
DONE = ("COMPLETED", "INITIAL_STATE", "RECOGNISED_UNVERIFIED")


def step_lifecycle__track(L, c):
    out = []
    for ds in L["dets"]:
        bs = [d[1] for d in ds if d[0] == c]
        out.append(max(bs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if bs else None)
    return out


def step_lifecycle__ctr(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], float)


def step_lifecycle__size(b):
    return max(1.0, float(np.hypot(b[2] - b[0], b[3] - b[1])))


def roles_of(inf):
    key = inf.get("key") or {}
    rs = list(inf.get("roles") or [])
    tgt = key.get("cls") or (rs[0] if rs else None)
    dest = key.get("other") or next((r for r in rs if r != tgt), None)
    return tgt, dest


def interaction(L, tgt, dest):
    """-> {"taken": t | None, "put_back": t | None, "released": t | None} for the step's part (the LAST taking that was
    not put back), from the boxes + hands"""
    pass
    t = np.asarray(L["t"], float)
    out = {"taken": None, "put_back": None, "released": None}
    if not tgt or len(t) < 3:
        return out
    tr = step_lifecycle__track(L, tgt)
    seen = [k for k, b in enumerate(tr) if b is not None]
    if not seen:
        return out
    first = [tr[k] for k in seen if t[k] <= t[seen[0]] + 1.0]
    rest = np.median([step_lifecycle__ctr(b) for b in first], 0)
    size = float(np.median([step_lifecycle__size(b) for b in first]))
    F = interaction_features_features(L, [tgt] + ([dest] if dest else []))
    hand = (F[:, 0] > 0) | (F[:, 1] > 0)
    away = np.array([b is not None and np.linalg.norm(step_lifecycle__ctr(b) - rest) > TAKE_MOVE * size for b in tr])
    back = np.array([b is not None and np.linalg.norm(step_lifecycle__ctr(b) - rest) <= BACK_NEAR * size for b in tr])
    k = 0
    while k < len(t):
        if away[k]:
            j = k
            while j + 1 < len(t) and (away[j + 1] or tr[j + 1] is None) and t[j + 1] - t[k] < TAKE_HOLD_S:
                j += 1
            near_hand = hand[(t >= t[k] - HAND_NEAR_S) & (t <= t[k] + HAND_NEAR_S)].any()
            if t[j] - t[k] >= TAKE_HOLD_S - 1e-6 or (j == len(t) - 1 and near_hand):
                if near_hand or not hand.any():          # a hand on it then (or no hands in this record at all)
                    out["taken"], out["put_back"] = float(t[k]), None
                    # put back: back at its place for BACK_HOLD_S later
                    m = k + 1
                    while m < len(t):
                        if back[m]:
                            n = m
                            while n + 1 < len(t) and (back[n + 1] or tr[n + 1] is None) and t[n + 1] - t[m] < BACK_HOLD_S:
                                n += 1
                            if t[n] - t[m] >= BACK_HOLD_S - 1e-6:
                                out["put_back"] = float(t[m])
                                break
                            m = n + 1
                        else:
                            m += 1
                    if out["put_back"] is None:
                        break
                    k = int(np.searchsorted(t, out["put_back"])) + 1
                    continue
            k = j + 1
        else:
            k += 1
    if out["taken"] is not None and dest:
        rel = np.flatnonzero((F[:, 3] >= 0.6) & (F[:, 9] > 0) & (t >= out["taken"]))
        if len(rel):
            out["released"] = float(t[rel[0]])
    return out


def visual(video, L, inf, windows=None):
    """-> (V (T, 4): cos pre / during / post / post-margin, region seen mask) or (None, None) without prototypes.
    windows [(t0, t1)]: only those frames are embedded (the rest: not seen) - the frames the checks below look at"""
    proto = inf.get("proto")
    if not proto:
        return None, None
    pass
    rs = list(inf.get("roles") or [])
    try:
        prepare(video, L, [rs], windows)
        E = embeddings(video, L, rs)
    except Exception:
        return None, None
    return state_prototypes_features(E, proto), np.linalg.norm(E, axis=1) > 0.5


def step_lifecycle_apply(model, video, L, res, info, live=False, log=print):
    """the lifecycle on the action model's results (in place): completion verification (veto / confirm) -> the
    statuses; and per step res[sid]["lifecycle"] (taken, put back, released, looks). -> {sid: lifecycle}"""
    if not ENABLED:
        return {}
    t = np.asarray(L["t"], float)
    if len(t) < 3:
        return {}
    t_end = float(t[-1])
    out = {}
    if VISUAL_WINDOWS:
        # (25 Nov) every step's frames read in ONE pass of the video (a pass per step opened + sought the file 4x)
        try:
            pass
            rs_all, wins_all = [], [(t_end - 0.2, t_end)]
            for s in model["spec"]["steps"]:
                inf_ = info.get(s["id"]) or info.get(str(s["id"])) or {}
                x_ = res.get(s["id"])
                if not inf_.get("proto") or x_ is None:
                    continue
                rs_all.append(list(inf_.get("roles") or []))
                if x_["status"] in ("COMPLETED", "RECOGNISED_UNVERIFIED") and x_.get("end_sec") is not None:
                    wins_all.append((float(x_["end_sec"]), float(x_["end_sec"]) + VETO_S))
                if x_["status"] == "RECOGNISED_UNVERIFIED":
                    r_ = interaction(L, *roles_of(inf_))["released"]
                    if r_ is not None:
                        wins_all.append((r_, r_ + POST_HOLD_S))
            if rs_all:
                prepare(video, L, rs_all, wins_all)
        except Exception:
            pass
    for s in model["spec"]["steps"]:
        sid = s["id"]
        inf = info.get(sid) or info.get(str(sid)) or {}
        x = res.get(sid)
        if not inf or x is None:
            continue
        tgt, dest = roles_of(inf)
        lc = interaction(L, tgt, dest)
        # (25 Nov) the frames the checks below look at - the last frame ('looks'), the second after a completion
        # (veto), the moment after a let-go (confirm) - not the whole video (a live check stalled 7 s embedding it)
        wins = [(t_end - 0.2, t_end)]
        if x["status"] in ("COMPLETED", "RECOGNISED_UNVERIFIED") and x.get("end_sec") is not None:
            wins.append((float(x["end_sec"]), float(x["end_sec"]) + VETO_S))
        if x["status"] == "RECOGNISED_UNVERIFIED" and lc["released"] is not None:
            wins.append((lc["released"], lc["released"] + POST_HOLD_S))
        V, ok = visual(video, L, inf, wins if VISUAL_WINDOWS else None)
        lc["looks"] = None
        if V is not None and ok[-1]:
            lc["looks"] = ("before" if V[-1, 0] > max(V[-1, 1], V[-1, 2]) + PRE_MARGIN else
                           "done" if V[-1, 3] > POST_MARGIN else "during")
        # VETO: confirmed, but its region still clearly looks like BEFORE the step - only WITHOUT physical proof
        # (25 Nov): a part taken from its place, not put back and let go AT its destination was placed, whatever the
        # region looks like. The learnt look comes from the good video's ORDER (the card goes into a box that already
        # holds the mouse); in a wrong-order video the same step looks different (an empty box) - and judging that
        # order is the point of the video. Box top wrong order: the card, put in first, was vetoed -> MISSING.
        # ... and still AT its destination over the veto window (a card let go over the box and lifted out again -
        # the confusion video - is no proof)
        placed = False
        if VETO_UNLESS_PLACED and lc["taken"] is not None and lc["put_back"] is None and lc["released"] is not None \
                and x.get("end_sec") is not None and dest:
            pass
            te_ = float(x["end_sec"])
            w_ = (t >= te_) & (t <= te_ + VETO_S)
            F_ = interaction_features_features(L, [tgt, dest])
            # the let-go may be registered later than the completion (hands near the box: box top wrong order, card
            # done 10.8 s, let go seen 12.0 s, at the box all the time) - its TIMING is the hand tracker's; the proof
            # is taken + not put back + let go AT the destination + staying there over the window
            placed = bool(w_.sum() >= 3 and float(F_[w_, 9].mean()) >= AT_DEST_SHARE)
        if V is not None and not placed and x["status"] in ("COMPLETED", "RECOGNISED_UNVERIFIED") \
                and x.get("end_sec") is not None:
            te = float(x["end_sec"])
            w = (t >= te) & (t <= te + VETO_S)
            if w.sum() >= 3 and ok[w].mean() >= VETO_SEEN:
                pre_like = (V[w, 0] > np.maximum(V[w, 1], V[w, 2]) + PRE_MARGIN)[ok[w]]
                if len(pre_like) and pre_like.mean() >= VETO_PRE:
                    x.update(status="MISSING_OR_UNCONFIRMED", start_sec=None, end_sec=None,
                             evidence=f"its {tgt} region still looks like BEFORE the step after {te:.1f}s "
                                      f"({pre_like.mean():.0%} of {VETO_S:g} s) - not yet done (lifecycle)")
                    lc["vetoed"] = te
        # CONFIRM: its action RECOGNISED but not verifiable (hidden: the lid / the hand), released at its destination,
        # and then its region looks DONE - never a step with no action evidence at all (a premature 'done' at 5 s)
        if V is not None and x["status"] == "RECOGNISED_UNVERIFIED" \
                and lc["released"] is not None and lc["put_back"] is None:
            tr_ = lc["released"]
            w = (t >= tr_) & (t <= tr_ + POST_HOLD_S)
            if t_end >= tr_ + POST_HOLD_S and w.sum() >= 3 and ok[w].mean() >= VETO_SEEN:
                done_like = (V[w, 3] > POST_MARGIN)[ok[w]]
                if len(done_like) and done_like.mean() >= POST_SHARE:
                    x.update(status="COMPLETED", start_sec=round(lc["taken"] or tr_, 2), end_sec=round(tr_, 2),
                             evidence=f"released AT its {dest} at {tr_:.1f}s and its region looks done "
                                      f"({done_like.mean():.0%} of {POST_HOLD_S:g} s) (lifecycle)")
                    lc["confirmed"] = tr_
        x["lifecycle"] = lc
        out[sid] = lc
    return out


def learn_open_when_taken(model, L, info):
    """TRAINING, on the good video: for each step, the EARLIER steps still not completed when its part was taken - a
    normal overlap (the pen body picked up before the spring step is done), never warned about. In place on info"""
    t = np.asarray(L["t"], float)
    ids = [s["id"] for s in model["spec"]["steps"]]
    for j in ids:
        inf = info.get(j)
        if not inf:
            continue
        tgt, dest = roles_of(inf)
        tk = interaction(L, tgt, dest)["taken"] if len(t) > 2 else None
        inf["taken_in_good_video"] = tk
        inf["open_when_taken"] = [i for i in ids[:ids.index(j)]
                                  if tk is not None and (info.get(i) or {}).get("end") is not None
                                  and float(info[i]["end"]) > tk]


def live_status(model, res, info, lifecycles):
    """LIVE: 'in progress' from the moment a step's part is taken; EARLY ORDER warnings. In place; -> [warnings]"""
    if not ENABLED:
        return []
    steps = [s["id"] for s in model["spec"]["steps"]]
    lab = {s["id"]: s.get("action_query") or s.get("text") for s in model["spec"]["steps"]}
    warn = []
    pass
    for j in steps:
        x, lc = res.get(j), lifecycles.get(j) or {}
        if x is None or x["status"] in DONE or lc.get("taken") is None or lc.get("put_back") is not None:
            continue
        # (26 Nov) its part is the SAME object an earlier step works on (or that object's other state: 'close box'
        # vs step 1's 'open box') - the box picked up to OPEN it is step 1's work, not 'close the box' in progress
        tgt_j, _ = roles_of(info.get(j) or info.get(str(j)) or {})
        if tgt_j and any(tgt_j == r or same_object(tgt_j, r) for i in steps[:steps.index(j)]
                         for r in ((info.get(i) or info.get(str(i)) or {}).get("roles") or [])):
            continue
        if x.get("provisional_t") is None:
            x["provisional_t"] = lc["taken"]
            x["live_state"] = "TAKEN"
        inf_j = info.get(j) or info.get(str(j)) or {}
        tgt, _ = roles_of(inf_j)
        normal_open = set(inf_j.get("open_when_taken") or [])   # learnt: still open when this part is taken (good video)
        for i in steps[:steps.index(j)]:
            xi, inf_i = res.get(i) or {}, info.get(i) or info.get(str(i)) or {}
            if xi.get("status") in DONE or tgt in (inf_i.get("roles") or []) or i in normal_open:
                continue                                 # done; the part needed by that step; or normal (good video)
            if (lifecycles.get(i) or {}).get("taken") is not None and (lifecycles.get(i) or {}).get("put_back") is None:
                continue                                 # its own part is in hand too: worked on together
            w = (f"step {j} ({lab[j]}): its {tgt} TAKEN at {lc['taken']:.1f}s while step {i} ({lab[i]}) is not "
                 f"done - out of order? (provisional)")
            x["order_warning"] = w
            warn.append(w)
            break
    return warn


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.lstr   (from sopmon\lstr.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""lstr.py - LSTR, the Long Short-Term Transformer for ONLINE action detection (Xu et al., NeurIPS 2021), compact.

The same interface as the TCN (temporal._net / object_actions._net): (B, C_in, T) -> (B, C_out, T), one score
vector per frame - so the training loops and the live predictor use either.

  short-term  the last SHORT frames (16 = ~5 s at 3 fps): causal self-attention (a frame sees itself and earlier ones)
  long-term   up to LONG earlier frames (128 = ~43 s): compressed by cross-attention into N_LATENT learnt queries
              (the 'long-term memory encoder'), which the short frames attend to (the 'decoder')
  causal      every frame's output uses only frames up to it - live inference reads it without waiting for the future

The sequence is processed in chunks of SHORT frames; each chunk's long memory is the LONG frames before it.

24 Nov: ALL CHUNKS IN ONE BATCH. A chunk reads only the INPUT frames before it (never another chunk's output), so the
chunks are independent: they are stacked into one batch (left-padded long memory + a padding mask) instead of a Python
loop of tiny GPU calls - the same result, ~30-50x faster to train (pen: 828 frames = 52 chunks were 896 ms per
iteration). And the windows can be given in SECONDS (for_fps): the action model runs at the video's frame rate (30 fps:
16 frames were only 0.5 s of short-term context, 128 only 4.3 s of memory).
"""
import torch
import torch.nn as nn

LONG, SHORT, N_LATENT = 128, 16, 16
SHORT_S, LONG_S = 5.0, 40.0          # for_fps(): the windows in seconds of video, at any frame rate


def for_fps(fps):
    """(long, short) in frames for a frame rate: ~LONG_S of memory, ~SHORT_S of short-term context"""
    fps = max(0.5, float(fps))
    return max(16, int(round(LONG_S * fps))), max(8, int(round(SHORT_S * fps)))


class LSTR(nn.Module):
    def __init__(self, d_in, n_out, h=64, heads=4, layers=2, dropout=0.2, long=LONG, short=SHORT, latent=N_LATENT):
        super().__init__()
        self.long, self.short = long, short
        self.inp = nn.Linear(d_in, h)
        self.pos_long = nn.Parameter(torch.zeros(long, h))
        self.pos_short = nn.Parameter(torch.zeros(short, h))
        nn.init.normal_(self.pos_long, std=0.02)
        nn.init.normal_(self.pos_short, std=0.02)
        self.latent = nn.Parameter(torch.randn(latent, h) * 0.02)
        mk = lambda: nn.TransformerDecoderLayer(h, heads, dim_feedforward=2 * h, dropout=dropout, batch_first=True)
        self.enc = nn.ModuleList([mk() for _ in range(1)])           # latents <- long memory
        self.dec = nn.ModuleList([mk() for _ in range(layers)])      # short frames <- themselves + the memory
        self.norm = nn.LayerNorm(h)
        self.out = nn.Linear(h, n_out)


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.workspots   (from sopmon\workspots.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""workspots.py — SOP steps as WORK SPOTS: where the wrench head (or the fingertips) stays, in the camera-corrected
view, learnt from ONE good video; a test video's visit order is checked against it. Its own program, separate from the
SOP Monitor (window: workspots_gui.py, run_workspots.bat).

  DETECT    the trained YOLO (weights/wrench_yolo11s.pt): hand, wrench, fitting_nut, bolt, band_clamp
  POINT     wrench held (touching a hand): its head = the end of its long side nearer a detected part (else the end
            farther from the hand); no wrench: the fingertips = the hand box side away from the image edge the arm
            enters from
  ALIGN     every frame -> the good video's REFERENCE frame by a SIFT homography (hands masked out); a frame too far
            from the reference is chained through the frame before it
  LEARN     the good video: where the point stays >= MIN_DWELL s within R -> a stay; stays closer than R -> one spot;
            spots numbered in the order they are first visited = the step order
  CHECK     a video: each frame -> the nearest spot (within 1.5 R), smoothed over 1 s; a spot with < MIN_VISIT s of visit
            is MISSING; spots visited out of the learnt order (not on the longest in-order run) are OUT OF ORDER

    python -m sopmon.workspots learn GOOD.mp4 [--name wrench]
    python -m sopmon.workspots check NAME VIDEO [VIDEO ...]
"""
import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

pass
workspots_ROOT = home()
workspots_OUT = workspots_ROOT / "outputs" / "_workspots"
WEIGHTS = workspots_ROOT / "weights" / "wrench_yolo11s.pt"
FPS = 5.0
REF_T = 0.3                    # the good video's reference frame (a clean view of the work place)
R_FRAC = 0.07                  # spot radius, share of the reference frame width
MIN_DWELL = 1.0                # s the point must stay to make a stay (learning)
MIN_VISIT = MIN_DWELL          # s of visit for a spot to count as done (checking): as long as a learnt stay - a tool
                               # passing through a spot's area on its way (~0.8 s) is not that step
SMOOTH = 1.0                   # s majority window over the per-frame spot labels
MIN_INLIERS = 40               # frame -> previous frame
REF_INLIERS = 200              # frame -> reference directly; weaker direct matches are wrong in close-ups (the scene
                               # is not flat), so those frames are chained through the frame before instead
HAND_WORDS = ("hand", "glove", "finger", "palm")
workspots_TOOL_WORDS = ("wrench", "spanner", "screwdriver", "driver", "ratchet", "tool", "drill", "gun", "plier", "hammer",
              "torque", "socket", "key", "scanner", "knife", "cutter")
workspots__M = {}


def app_name(prompt):
    """the work spots of an SOP are kept per prompt: each kind of video has its own"""
    import hashlib
    text = " ".join(str(prompt or "").split()).lower()
    return "sop_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


# ── pictures ──
SPOT_COLORS = [(232, 155, 76), (100, 211, 86), (65, 179, 227), (114, 123, 255), (255, 140, 188), (207, 197, 57),
               (186, 120, 247), (87, 166, 255)]


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.sop_events   (from sopmon\sop_events.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""sop_events.py — the OBJECT-ACTION-STATE EVENT layer (test version, not yet in the SOP Monitor's pipeline).
(Not events.py: that one is the live mode's ObjectEvents.)

Every SOP step gets an ACTION PRIMITIVE (from its verb); the evidence the pipeline already has is turned into ONE event
per step in a common schema, with what the evidence shows and how sure it is:

  PICK        the part leaves its learnt start place                              (places / pickups)
  PLACE       the part is seen at its destination after it was picked up         (lifecycle "on the product")
  INSTALL     = PLACE on the product
  MANIPULATE  the tool / hand stays at the step's learnt work spot AND moves there (work spots: where + motion)
  SCAN        the scanner is at the item                                          (needs a scanner class)
  REMOVE / INSPECT / OTHER: no rule yet -> UNCERTAIN

  status  VERIFIED   every completion condition of the primitive is seen
          OBSERVED   the action was seen, its completion is not provable here (a wrench turning: not the torque)
          UNCERTAIN  some evidence, not enough
          (no event) no evidence at all -> the step is MISSING
  verdict ANOMALY when a step is MISSING or OUT OF ORDER; REVIEW when a step is only UNCERTAIN; else NORMAL

    python -m sopmon.sop_events           the Dataset3 replay (saved runs) + the wrench videos, against their answers
"""
import json
import re
from pathlib import Path

import numpy as np

pass
sop_events_ROOT = home()
sop_events_TIE_S = 1.0                    # events closer than this keep the SOP order (the state machine's tie rule)
SLACK_S = 2.0                  # 'on the product' may be seen up to this long before the pickup is confirmed
MOTION_R = 0.10                # MANIPULATE: the work point must move by >= this share of the spot radius (median step)

VERBS = [  # first match wins: the more specific verbs first
    ("MANIPULATE", r"tighten|loosen|torque|screw|unscrew|fasten|ratchet|turn|rotate|wrench|drill|start the|by hand"),
    ("REMOVE", r"remove|detach|take off|disassembl|pull out|unplug"),
    ("SCAN", r"scan|barcode|qr\b|read the label"),
    ("INSPECT", r"inspect|check|look at|verify|examine"),
    ("INSTALL", r"mount|attach|install|fit\b|insert|assembl|connect|press|snap|plug|join|put on"),
    ("PLACE", r"place|put|set|lay|position|load"),
    ("PICK", r"pick|take|grab|lift|collect"),
]


def primitive(text):
    """the primitive of the step's FIRST action word ('mount the chassis with the screwdriver' -> INSTALL)"""
    t = str(text).lower()
    hits = [(m.start(), p) for p, rx in VERBS for m in [re.search(rx, t)] if m]
    return min(hits)[1] if hits else "OTHER"


# ── the tests ──
DATASET3 = {  # video stem -> (expected, the latest run with the lifecycle, 2026-11-12)
    "Dataset3_assemble_original": ("NORMAL", "2026-11-12/Dataset3_assemble_original/11-45-58_seq"),
    "Dataset3_assembly_edited": ("ANOMALY", "2026-11-12/Dataset3_assembly_edited/12-28-29_seq"),
    "Dataset3_assembly_edited2": ("ANOMALY", "2026-11-12/Dataset3_assembly_edited2/12-36-04_seq"),
    "Datset3_assembly_edited3": ("ANOMALY", "2026-11-12/Datset3_assembly_edited3/12-43-33_seq"),
}
WRENCH = {"Wrench_Train": "NORMAL", "Wrench_Test_Normal": "ANOMALY", "Wrench_Bolt_3_Skip": "ANOMALY"}
WRENCH_SOP = ["start the fitting nut by hand", "tighten the fitting nut with the wrench",
              "tighten the bolt with the wrench held upright", "tighten the right-hand fitting with the wrench"]


STATUS_COL = {"DONE": (90, 200, 110), "DONE (observed)": (60, 200, 235), "UNCERTAIN": (235, 160, 60),
              "OUT OF ORDER": (80, 90, 255), "MISSING": (80, 90, 255)}            # BGR: green, yellow, blue, red


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.objsteps   (from sopmon\objsteps.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""objsteps.py — SOP steps from WHERE THE HANDS WORK: the hands found by YOLOE ("hand"), the WORK POINT at the
fingertips (the part being worked on - the nut, the bolt, the clamp), what the hand holds (the tool), and where in the
scene this happens; each step learnt from the correct video as memories of these.

  VISION  frames at 2 fps -> DINOv3 (frozen, timm vit_small_patch16_dinov3) patch features per frame
          -> HANDS: YOLOE-11 seg prompted with "hand" (box + mask), at most 2 per frame
          -> the arm's ENTRY: where the moving region holding the hand (vs the background model, the median patch
             feature per location) meets the frame border
          -> WORK POINT: the hand's far end from the entry, a little beyond the fingertips; its disk of patches
          -> per hand: WORK feature (DINOv3 mean over the work disk = the part + the tool's tip), HAND feature (DINOv3
             mean over the hand box = the hand + the tool it holds), POSITION (the work point in the frame)
  STEPS   a correct video is split into its SOP steps in order from what it shows (least scatter of the per-frame
          hand descriptors - no labels); each step keeps its frames as MEMORIES (work, hand, position)
  SCORE   per frame, per step: the closest memories (work 0.5, hand-and-tool 0.3, position 0.2), the best hand, minus
          the mean of the other steps (relative); no hand = idle
  VERDICT step margins (score - the step's threshold) -> ordered alignment that drops idle clips and skips steps
          (two-sided Drop-DTW) -> a step matched long enough, in order = DONE; else searched over the whole video:
          found elsewhere = OUT OF ORDER, not found = MISSING
"""
import json
import re
from pathlib import Path

import numpy as np

pass
objsteps_ROOT = home()
objsteps_OUT = objsteps_ROOT / "outputs" / "_objsteps"
objsteps_WEIGHTS = objsteps_ROOT / "weights"
objsteps_FPS = 2.0
objsteps_LONG = 640                      # DINOv3 input: the long side, both sides a multiple of 16
DINO = "vit_small_patch16_dinov3.lvd1689m"
YOLOE_W = objsteps_WEIGHTS / "yoloe-11l-seg.pt"
HAND_PE = objsteps_WEIGHTS / "yoloe_hand_pe.pt"   # YOLOE's text prompt "hand", cached (needs mobileclip_blt.ts only once)
CACHE_V = "h3"              # the cache version (the hand pipeline)
BG_FRAMES = 120                 # frames the background model is the median of
HAND_CONF = 0.10
MAX_HANDS = 2
WORK_R = 2.5                    # the work disk's radius, in patches (16 px at 640)
W_WORK, W_HAND, W_POS = 0.5, 0.3, 0.2    # memory similarity: the work point, the hand + its tool, the position
POS_SIGMA = 0.08                # position: a fraction of the frame
TOPK = 3                        # a step's score = the mean of its closest memories
EXCL = 1.5                      # on the video a model learnt from, memories this near in time are not used (s)
SHRINK, PCT = 0.5, 50           # a step's threshold = SHRINK x this percentile of its score in its correct segment
STEP_DROP = 0.005
MIN_MATCH = 2                   # clips a step needs to count as matched
objsteps_KINDS = ("hand", "work point")
objsteps__M = {}


def hands_detector():
    if "yoloe" not in objsteps__M:
        import os
        import torch
        from ultralytics import YOLOE
        m = YOLOE(str(YOLOE_W))
        if HAND_PE.exists():
            pe = torch.load(HAND_PE)
        else:                                            # the text encoder (mobileclip_blt.ts) is in weights/
            cwd = os.getcwd()
            os.chdir(objsteps_WEIGHTS)
            try:
                pe = m.get_text_pe(["hand"])
            finally:
                os.chdir(cwd)
            torch.save(pe.cpu(), HAND_PE)
        m.set_classes(["hand"], pe)
        objsteps__M["yoloe"] = m
    return objsteps__M["yoloe"]


# ── steps ──
_NUM = re.compile(r"(?:^|\s)(\d{1,2})[.)]\s+")


# ── the output video ──
_COL = [(232, 155, 76), (100, 211, 86), (65, 179, 227), (114, 123, 255), (255, 140, 188), (207, 197, 57),
        (186, 120, 247), (87, 166, 255), (255, 214, 165), (135, 231, 126), (255, 168, 210), (93, 223, 255)]


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.hands   (from sopmon\hands.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""hands.py - where the operator's HANDS are, frame by frame, for the completion check (sopmon/temporal.py).

Two pretrained hand models, one interface (the hand masks on a small grid over the frame):

  egohos   EgoHOS two-hands segmentation (Zhang et al. 2022), in its own environment (C:/Users/sonar/Desktop/egohos_env,
           third_party/egohos_run.py). Good masks, but a separate process: run once per VIDEO, ahead (cached in
           outputs/_hands/<video key>), at HAND_FPS. A file only - not a live camera.
  yoloe    YOLOE-11L-seg with the text prompt "hand" (weights/yoloe_hand_pe.pt) in this process, per frame. Weaker
           masks, but it runs live (camera too).

  Hands(kind, video).masks(rgb, ts) -> [bool (GH, GW)] one per hand (EgoHOS: left / right; YOLOE: each instance)
"""
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

pass
hands_ROOT = home()
hands_OUT = hands_ROOT / "outputs" / "_hands"
EGOHOS_PY = Path(r"C:\Users\sonar\Desktop\egohos_env\Scripts\python.exe")
RUNNER = hands_ROOT / "third_party" / "egohos_run.py"
HAND_FPS = 3.0
GW, GH = 96, 54                 # the hand grid over the frame (normalised coordinates)
MIN_CELLS = 4                   # a hand smaller than this (grid cells) is noise
YOLOE_SZ, YOLOE_CONF = 1920, 0.2
hands_KINDS = ("off", "egohos", "yoloe")


def hands__key(video):
    p = Path(video)
    return f"{p.stem}_{p.stat().st_size}"


def egohos_available():
    return EGOHOS_PY.exists() and RUNNER.exists()


def egohos_run(video, log=print, check=None, fps=HAND_FPS):
    """-> the folder with pred_twohands/ + done.json for `video` (EgoHOS run once, then read back)"""
    d = hands_OUT / hands__key(video)
    if (d / "done.json").exists():
        return d
    if not egohos_available():
        raise RuntimeError(f"the EgoHOS environment is missing: {EGOHOS_PY}")
    log(f"[hands] {Path(video).name}: EgoHOS hand masks at {fps:g} fps (its own environment, once per video) ...")
    p = subprocess.Popen([str(EGOHOS_PY), str(RUNNER), str(video), str(d), str(fps), "pred_twohands"], cwd=str(hands_ROOT),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    tail = []
    for line in p.stdout:
        tail = (tail + [line.rstrip()])[-15:]
        if line.startswith("[egohos]"):
            log(f"[hands] {line.strip()}")
        if check is not None:
            try:
                check()
            except Exception:
                p.kill()
                raise
    if p.wait() != 0 or not (d / "done.json").exists():
        raise RuntimeError("EgoHOS failed:\n" + "\n".join(tail))
    return d


class Hands:
    def __init__(self, kind, video=None, log=print, check=None):
        self.kind, self.log = kind, log
        self.t, self.files, self.cache = None, None, {}
        self.model = None
        if kind == "egohos":
            v = str(video or "")
            if v.strip().isdigit() or "://" in v or not Path(v).is_file():
                raise RuntimeError("EgoHOS hands need a video FILE (it runs ahead, in its own environment) - use "
                                   "YOLOE hands for a camera")
            d = egohos_run(v, log, check)
            meta = json.load(open(d / "done.json"))
            self.t = np.asarray(meta["t"], float)
            self.files = [d / "pred_twohands" / f"{i:05d}.png" for i in range(len(self.t))]
            self.step = 1.0 / float(meta.get("fps") or HAND_FPS)
        elif kind == "yoloe":
            pass
            self.model = hands_detector()
        else:
            raise ValueError(kind)

    def _egohos(self, ts):
        if not len(self.t):
            return []
        i = int(np.argmin(np.abs(self.t - ts)))
        if abs(self.t[i] - ts) > self.step:              # past the masks' end
            return []
        if i not in self.cache:
            seg = cv2.imread(str(self.files[i]), cv2.IMREAD_GRAYSCALE)
            out = []
            if seg is not None:
                for lab in (1, 2):
                    m = cv2.resize((seg == lab).astype(np.uint8), (GW, GH), interpolation=cv2.INTER_AREA) > 0
                    if m.sum() >= MIN_CELLS:
                        out.append(m)
            self.cache = {i: out} if len(self.cache) > 64 else {**self.cache, i: out}
        return self.cache[i]

    def _yoloe(self, rgb):
        # far, small, gloved hands (a top camera over a bench) are found only at full resolution
        r = self.model.predict(np.ascontiguousarray(rgb[:, :, ::-1]), conf=YOLOE_CONF, verbose=False,
                               imgsz=YOLOE_SZ)[0]
        out = []
        if r.masks is None:
            return out
        for poly, c in zip(r.masks.xyn, r.boxes.conf.tolist()):
            if len(poly) < 3:
                continue
            m = np.zeros((GH, GW), np.uint8)
            cv2.fillPoly(m, [np.round(poly * [GW, GH]).astype(np.int32)], 1)
            if m.sum() >= MIN_CELLS:
                out.append(m > 0)
        return out[:4]

    def masks(self, rgb, ts):
        """-> [bool (GH, GW)] the hands in this frame"""
        return self._egohos(ts) if self.kind == "egohos" else self._yoloe(rgb)


def on_product(masks, P, W0, H0, grow=0.1):
    """the hands vs the product box P (frame pixels): -> (hands mostly on it, hand cells on it, hand cells)"""
    if not masks:
        return 0, 0, 0
    if P is None:
        return 0, 0, int(sum(m.sum() for m in masks))
    w, h = P[2] - P[0], P[3] - P[1]
    a = max(0, int((P[0] - grow * w) / W0 * GW)), max(0, int((P[1] - grow * h) / H0 * GH))
    b = min(GW, int(np.ceil((P[2] + grow * w) / W0 * GW))), min(GH, int(np.ceil((P[3] + grow * h) / H0 * GH)))
    on, cells, n = 0, 0, 0
    for m in masks:
        c = int(m[a[1]:b[1], a[0]:b[0]].sum())
        s = int(m.sum())
        on += c >= 0.5 * s
        cells += c
        n += s
    return on, cells, n


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.direction   (from sopmon\direction.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""direction.py — INSERT / REMOVE test: which way did a part go? Detectors only, no VLM, no training.

  ASSEMBLY BODY  the classes parts sit on (chassis, cab frame, phone ...); a class may be body AND part
                 (the cab frame sits on the chassis and carries the roof)
  ASSEMBLY       every sampled frame: the boxes of the body + part classes, grown by `grow`, joined when they touch;
                 the assembly = the biggest such cluster holding a body box (the chassis is often not detected
                 while it is covered by its parts - the cluster of cab frame + wheels + bucket is the truck then)
  PER PART       how many of it lie ON the assembly (in that cluster) and how many OFF it (elsewhere: on the table)
                 a frame without an assembly cluster keeps the last counts; a hand over a part only LOWERS a count,
                 so the counts take their max over `occl_s` before the median over `smooth_s`
  EVENTS         the ON count changes to a new level that holds for `hold_s`:
                   down = REMOVE (the part left the assembly)     up = ATTACH / INSERT (it arrived on it)
                 evidence: the OFF count moving the other way (the part now lies on the table / was taken from it)
                 a change undone within `flicker_s` with no table evidence either way is dropped as flicker
"""
import re

import cv2
import numpy as np

pass


# the direction of a step, from its own words: checked in this order (a remove step that ends 'and put them on the
# table' is still a remove; 'put the screwdriver down' moves nothing on the assembly)
_REMOVE = re.compile(r"\b(?:remov\w*|take[sn]?\s+(?:\w+\s+){0,4}?(?:off|out|apart)|took\s+(?:\w+\s+){0,4}?(?:off|out)|"
                     r"unscrew\w*|separat\w*|detach\w*|disassembl\w*|dismount\w*|uninstall\w*|unplug\w*|"
                     r"pull\w*\s+(?:\w+\s+){0,4}?(?:off|out)|extract\w*|lift\w*\s+(?:\w+\s+){0,4}?off|"
                     r"unclip\w*|unmount\w*|disconnect\w*)", re.I)
_ASIDE = re.compile(r"\b(?:put|place|set|lay|leave)\w*\s+(?:\w+\s+){0,4}?(?:down|aside|away)\b|"
                    r"\bon\s+the\s+(?:table|bench|desk|tray)\b", re.I)
# put / place / slide ... count only as the clause's verb ('put the phone into the box'), never as a noun ('the
# sticker place on the lid')
_ATTACH = re.compile(r"\b(?:attach\w*|install\w*|insert\w*|mount\w*|fit\w*|assembl\w*|reassembl\w*|plug\w*|connect\w*|"
                     r"fasten\w*|screw\w*\s+(?:\w+\s+){0,4}?(?:in|into|on|onto|down)|clip\w*|snap\w*)\b|"
                     r"(?:^|[,;:.]\s*|\b(?:and|then|to)\s+)(?:put|place|set|slide|seat)s?\s+(?:\w+\s+){0,6}?"
                     r"(?:back|onto|into|in|on)\b", re.I)


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.stepmodel   (from sopmon\stepmodel.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""stepmodel.py — EXPERIMENTAL action model: learn the task's actions from the pipeline's own finished runs.

  INPUT  (per time step of a video, ~3 per second)
           X-CLIP video features of a 1-second clip of the WORK AREA (motion: scanning, pressing, placing ...)
         + the detector's view, saved by every run in signals.npz: per object present / count / confidence /
           speed / appears / vanishes, and the motion energy of the picture
  LABELS (what it learns to output)
           the actions of the spec - 'install camera module', 'scan the spline', 'place plate' ... - with the times
           the run CONFIRMED (detector proposed + VLM said yes); everything else is background
  MODEL  ActionFormer-style: a temporal network that, for every time step, says which action is happening (or
         none) and how far that action's start and end are -> events {action, t_start, t_end, score}
  CHECK  several videos: leave-one-video-out. ONE video: train on the first 60 % of it, test on the rest (works for
         repeating actions - 30 plates, 5 scans; a task whose actions happen once each cannot be checked this way)

The model trained on everything is saved to outputs/_stepmodels/; ticking 'Use step model' makes a run use its
events for the step times. With few videos this is a feasibility check, not a production model.
"""
import glob
import hashlib
import json
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

pass
stepmodel_ROOT = home()
stepmodel_OUT = stepmodel_ROOT / "outputs"
FEAT_DIR = stepmodel_OUT / "_features"
MODEL_DIR = stepmodel_OUT / "_stepmodels"
XCLIP = "microsoft/xclip-base-patch32"
CLIP_FPS, CLIP_FRAMES = 8.0, 8          # one clip = 8 frames over 1 s
FOUND = ("OK", "OUT_OF_ORDER", "out_of_order (not flagged)")


# ── input: X-CLIP clip features of the work area + the run's detector signals ──
_XC = {}


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.seqdecode   (from sopmon\seqdecode.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""seqdecode.py — the SOP as a state machine, decoded over the detector evidence (profile-HMM / Viterbi alignment).

  1. evidence   per time step and per spec action: how much the RAW detector signals of its object show something
                happening (vanishing from the bench, appearing, moving, its count rising) - signals.npz of the run
  2. candidates the clear peaks of that evidence: candidate events {action, time, score} - noisy: some real ones are
                weak, some peaks are just flicker
  3. decoding   the spec's expected events (in order) are aligned to the candidates by dynamic programming, choosing
                the cheapest explanation of the whole video with these moves:
                  match    the expected step happened, at that candidate's time         cost 1 - score
                  missing  an expected step has no candidate                            cost MISS
                  extra    a candidate nothing expects (noise, or a repeat)             cost EXTRA * score
                  swap     two neighbouring steps done the other way round              cost of both matches + SWAP
                and afterwards a 'missing' step whose action is clearly seen elsewhere is a MOVED step (out of order)
  4. verdict    MISSING / OUT OF ORDER from those moves - generic for any spec, no training, no hand-tuned per-step rules

Preconditions (counts of what is already there) are states, not actions: they come from the run's own checks.
"""
import numpy as np

pass

MISS, EXTRA, SWAP = 1.0, 0.25, 0.2
seqdecode_TIE_S = 1.0                     # events closer than this have no meaningful order between them


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.videotext   (from sopmon\videotext.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""videotext.py — ZERO-SHOT actions: a video-text model (ViCLIP, InternVid / InternVideo family) scores short clips of
the video against the spec's own step texts. No training, any prompt: the steps ARE the action classes.

  CLIPS    3 s of video (8 frames), one every second, of the work area (or the whole picture)
  TEXTS    every step as an action sentence ('a person puts the phone into the outer box') + a few 'nothing is
           done' sentences (the background)
  OUTPUT   per step: its probability over time (softmax over all the texts, the model's own temperature)
"""
import re
import sys
import types
from collections import deque

import cv2
import numpy as np

MODEL_ID = "OpenGVLab/ViCLIP-B-16-hf"
videotext_FPS, FRAMES, WIN_S, HOP_S = 4.0, 8, 3.0, 1.0
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
BACKGROUND = ["a person's hands resting on the table, nothing is being done",
              "an empty work table with parts lying on it",
              "a person holds an object and looks at it"]

videotext__M = {}


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.vlm_client   (from sopmon\vlm_client.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""The app side of the warm VLM server: start it once, ask() from any thread (one at a time)."""
import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

pass

SERVER = Path(__file__).with_name("vlm_server.py")


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.action_model   (from sopmon\action_model.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""action_model.py - the ACTION TCN: learns from a correct video what a COMPLETED action looks like (a test; the app's
pickup TCN, temporal.pt, is not changed).

  LABELS    from the correct video, per step: START = its confirmed pickup (detector + places); COMPLETE = its end
            where two independent methods AGREE within AGREE_S - the change-point boundary (sopmon/boundaries.py) and
            TimeLens2 inside the bounded window (vtg_test 0b). A step they disagree on is left out of training.
  FEATURES  per part and frame, generic state only (no clock, no 'seconds since pickup' - the good video's timing is
            not a template): part at its place / seen anywhere / on the product, product visible, motion in the
            product's box, the product's change since the start and in the last second, another part just leaving
            its place, which detector classes are present
  MODEL     the original TCN backbone (5 dilated residual blocks, 64 channels, both directions) with its own input
            and a 3-class head: NOT STARTED / IN PROGRESS / COMPLETED; no augmentation
  TEST      per part, independently of the SOP order: completion = the first moment COMPLETED holds for HOLD_S ->
            the SOP verifier on the completion times

    trained/<model>/temporal_action.pt  (+ action_labels.json, action_ref.npz: each step's product state after it)
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

pass
action_model_ROOT = home()
action_model_CLASSES = ["NOT_STARTED", "IN_PROGRESS", "COMPLETED"]
AGREE_S = 2.0
action_model_HOLD_S = 2.0
# (no 'product changed since the start': it grows with every part fitted, so it tells HOW FAR into the SOP the video
#  is - an order / timing cue the model must not learn)
BASE = ["at_place", "seen", "on_product", "product_seen", "work", "changing", "other_left"]


# ── features ──
def action_model__ctr(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def action_model_features(video, L, classes, log=print):
    """-> X (U, T, F) for the place units of L (vlm_install_test.live_pass), t (T,), units"""
    pass
    S = signals(video, L, log)
    t = np.array(L["t"])[:len(S["t"])]
    T = len(t)
    units = [u for u in L["units"]]
    dt = float(np.median(np.diff(t))) if T > 1 else 1 / 3
    w1 = max(1, int(round(1.0 / dt)))
    m = np.convolve(S["motion"], np.ones(3) / 3, mode="same")
    work = np.clip(m / max(float(np.percentile(m, 95)), 1e-6), 0, 2)
    cr = S["crops"]
    changing = np.array([np.abs(cr[k] - cr[max(0, k - w1)]).mean() for k in range(T)]) / 20.0
    pv = np.array([L["P"][k] is not None for k in range(T)], float)
    cls_cnt = np.zeros((T, len(classes)), np.float32)
    for k in range(T):
        for n, _ in L["dets"][k]:
            if n in classes:
                cls_cnt[k, classes.index(n)] += 1
    cls_cnt = np.minimum(cls_cnt, 3) / 3.0
    at = []
    for u in units:
        b = np.array(u["box"])
        sz = max(b[2] - b[0], b[3] - b[1], 1.0)
        g = [b[0] - 0.3 * sz, b[1] - 0.3 * sz, b[2] + 0.3 * sz, b[3] + 0.3 * sz]
        a, s_, onp = np.zeros(T), np.zeros(T), np.zeros(T)
        for k in range(T):
            P = L["P"][k]
            for n, bb in L["dets"][k]:
                if n != u["family"]:
                    continue
                c = action_model__ctr(bb)
                s_[k] = 1
                if g[0] <= c[0] <= g[2] and g[1] <= c[1] <= g[3]:
                    a[k] = 1
                elif P is not None and P[0] <= c[0] <= P[2] and P[1] <= c[1] <= P[3]:
                    onp[k] = 1
        at.append((a, s_, onp))
    left = np.zeros((len(units), T))                       # another part's place emptied in the last second
    drops = [np.r_[0, np.maximum(0, a[:-1] - a[1:])] for a, _, _ in at]
    for i in range(len(units)):
        for j in range(len(units)):
            if j != i:
                left[i] = np.maximum(left[i], np.convolve(drops[j], np.ones(w1), mode="full")[:T])
    X = np.zeros((len(units), T, len(BASE) + len(classes)), np.float32)
    for i, (a, s_, onp) in enumerate(at):
        X[i, :, 0], X[i, :, 1], X[i, :, 2] = a, s_, onp
        X[i, :, 3], X[i, :, 4] = pv, work
        X[i, :, 5] = np.clip(changing, 0, 2)
        X[i, :, 6] = np.clip(left[i], 0, 1)
        X[i, :, len(BASE):] = cls_cnt
    return X, t, units


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.pack_actions   (from sopmon\pack_actions.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""pack_actions.py - SOP ACTIONS LEARNT FROM ONE CORRECT VIDEO WITH CONFIRMED TIMESTAMPS (test: pack_test.py).

The rule: learn the ACTION from the video, store the correct SEQUENCE from the SOP - never the one from the other.

  STORE (training)  per SOP step, from the human-confirmed START / END on the correct video:
                      action_labels.json      the confirmed times, the duration (a reference, not a timer)
                      action_references.npz   its PRE / ACTIVE / POST feature sequence: DINOv2 look of the work area
                                              (centred per video) + generic motion / change signals
                      signature               the detector change at its END (a class appears / disappears / changes
                                              count / enters another's box) - where to look for it in a new video
                      sop_graph.json          the steps and their order rules (from the SOP, not learnt)
                      training_metadata.json  versions, features, the reference video, who confirmed the labels
  CHECK (any video) every step INDEPENDENTLY: its signature's moments anywhere in the video (or, without one, the
                    whole video) -> each scored by subsequence DTW against its reference (any speed) -> the best:
                    its start and end; none good enough -> not seen. Then the SOP graph on the END times.
"""
import json
import time
from pathlib import Path

import numpy as np

pass
pack_actions_ROOT = home()
pack_actions_OUT = pack_actions_ROOT / "outputs" / "_packtest"
pack_actions_PRE_S = pack_actions_POST_S = 2.0
SNAP_S = 2.5                 # a confirmed END snaps to a detector change this close
MAX_COST = 0.9               # a DTW match costlier than this (mean per aligned frame) is not the action
GLOBAL = [2, 3, 4]           # ref_match state columns that do not depend on a part's place: product seen, work, change


def record(model, video, registry, log=print, check=None):
    """the model's live inference over a video, cached -> the run's record (temporal.LiveInference.record)"""
    pass
    v = Path(video)
    pack_actions_OUT.mkdir(parents=True, exist_ok=True)
    c = pack_actions_OUT / f"{v.stem}_{v.stat().st_size}_{model['name'] if 'name' in model else Path(model['places_file']).parent.name}.json"
    if c.exists():
        return json.loads(c.read_text(encoding="utf-8"))
    log(f"  detector + places over {v.name} (once; cached) ...")
    L = record_video(model, str(v), registry, log, check, out_dir=pack_actions_OUT / "run")
    c.write_text(json.dumps(L, default=float), encoding="utf-8")
    return L


# ── the trained temporal model (temporal_boundary.pt): the original TCN backbone, two heads per step ──
N_PCA = 16


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.semantic_events   (from sopmon\semantic_events.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""semantic_events.py - WHAT a detector change MEANS, and which SOP (sub-)action it can be evidence of (25 Nov, P0).

Training learns each step from ONE detector change in the good video (auto_labels). A change is not automatically a
completion: an object can vanish because the detector lost it, a hand covers it, it went INTO another object, it
turned into its other STATE (closed box -> open box), or it left the work area. Box top (25 Nov): 'close' was mapped
to 'disappears', so a 2-s detector gap while the box slid into its sleeve was learnt as step 5 - and in a test video
the box being OPENED (closed box -> open box) matched it.

  DECOMPOSE   an SOP step with several actions ('close the box and slide it into its sleeve') -> sub-actions, each
              with its action PRIMITIVE (close / open / insert / attach / remove / move / pick / place / scan) and
              whether the detector can OBSERVE it (its objects / destination are detector classes, or a state pair
              exists for open / close). The prompt is not changed.
  MEANING     every change of state_events.events: 'gap' (it disappears and is seen again at the same place - a
              detector gap or an occlusion), 'gap_end' (that reappearance), 'transform' (one object's state change:
              'closed box' <-> 'open box' in place), 'relation' (X enters / leaves Y), 'gone', 'appears', 'count'
  FIT         how well a change is evidence of a sub-action: gaps never; a transform only in the primitive's direction
              (close: -> closed state; open: -> open state); insert / attach: X enters the destination, or X gone
              while at the destination; ...
All by names, words and geometry - no product-specific rule.
"""
import re

import numpy as np

# action primitives: the FIRST pattern that matches a sub-action decides (insert before move: 'slide it into ...')
PRIMS = [(r"\b(close|closes|closed|shut|seal)\b", "close"),
         (r"\b(open|opens|unfold)\b", "open"),
         (r"\b(remove|take\s+\w*\s*out|pull\s+\w*\s*out|detach|unplug)\b", "remove"),
         (r"\b(insert|into|inside|plug)\b", "insert"),
         (r"\b(attach|fit|fix|snap|mount|onto|clip|screw|connect)\b", "attach"),
         (r"\b(scan)\b", "scan"),
         (r"\b(slide|push|move|turn|rotate|flip|carry)\b", "move"),
         (r"\b(pick|lift|grab|take)\b", "pick"),
         (r"\b(place|put|set|lay|leave|drop|position)\b", "place")]
semantic_events_VERBS = set("""put place insert install attach fit close open press slide push pull remove take pick mount screw
unscrew tighten loosen connect disconnect plug unplug add move lift turn rotate check inspect seal fold unfold apply
align hold drop pack unpack assemble cover secure fasten lay set get use grab return position load unload fill empty
wrap tape stick snap clip hang carry bring lower raise flip tilt fix glue join mark cut peel scan shut leave""".split())
STOP = {"the", "a", "an", "its", "it", "their", "them", "this", "that", "of", "to", "and", "with", "into", "onto",
        "in", "on", "inside", "from", "out", "up", "down", "all", "items", "then"}
OPEN_W = {"open", "opened", "unsealed", "unfolded", "empty"}
CLOSED_W = {"closed", "close", "sealed", "folded", "shut", "packed", "finished"}
REAPPEAR_S = 6.0         # seen again within this long ...
REAPPEAR_NEAR = 1.0      # ... within this x its size of where it vanished = a detector gap / an occlusion


def semantic_events__words(s):
    return re.findall(r"[a-z]+", str(s or "").lower().replace("_", " "))


def _head(c):
    w = semantic_events__words(c)
    return w[-1] if w else ""


def state_of(c):
    ws = set(semantic_events__words(c)[:-1])
    return "open" if ws & OPEN_W else "closed" if ws & CLOSED_W else None


# ── DECOMPOSE ────────────────────────────────────────────────────────────────────────────────────────────────────
def semantic_events_primitive(text, classes=()):
    """the action primitive of a (sub-)action's words - detector class names taken out first ('put the mouse into
    the OPEN BOX' is an insert, not an 'open'; 'slide the CLOSED BOX into its sleeve' is an insert)"""
    t = " " + " ".join(semantic_events__words(text)) + " "
    for c in sorted(classes, key=lambda c: -len(c)):
        cw = " ".join(semantic_events__words(c))
        if cw:
            t = t.replace(f" {cw} ", " ")
    for tt in (t.strip(), " ".join(semantic_events__words(text))):         # 'Open box' (all a class name): its own words
        for pat, p in PRIMS:
            if re.search(pat, tt):
                return p
    return None


def decompose(text):
    """a step's text -> [sub-action text] (split at 'and' / 'then' / ',' where both sides start with a verb)"""
    t = " ".join(str(text or "").replace(";", ",").split()).strip(" .")
    parts = [p.strip(" ,.") for p in re.split(r",?\s+\b(?:and then|then|and)\b\s+|,\s+(?=[a-z]+\b)", t, flags=re.I)]
    out = []
    for p in parts:
        if not p:
            continue
        first = (semantic_events__words(p) or [""])[0]
        if out and first not in semantic_events_VERBS:
            out[-1] = out[-1] + " and " + p        # 'the mouse and the card': one action
        else:
            out.append(p)
    return out or [t]


def _named(text, classes):
    """detector classes a text names (all of a class's words in it; 'box' alone names every '* box')"""
    ws = set(semantic_events__words(text))
    full = [c for c in classes if set(semantic_events__words(c)) <= ws]
    head = [c for c in classes if _head(c) in ws and c not in full]
    return full, head


def destination(text):
    m = re.search(r"\b(?:into|onto|inside|in|on|to)\s+((?:[a-z]+\s+){0,3}[a-z]+)", " ".join(semantic_events__words(text)))
    if not m:
        return None
    ws = [w for w in m.group(1).split() if w not in STOP]
    return " ".join(ws) or None


def sub_actions(step, classes, target):
    """-> [{"text", "prim", "observable", "why", "dest"}] for one SOP step (target = its main object class)"""
    text = step.get("text") or step.get("action_query") or ""
    subs = decompose(text)
    out = []
    for k, s in enumerate(subs):
        prim = semantic_events_primitive(s, classes) or (semantic_events_primitive(text, classes) if len(subs) == 1 else None)
        dest = destination(s) if prim in ("insert", "attach", "place") else None
        full, head = _named(s, classes)
        obs, why = True, ""
        if prim in ("open", "close"):
            h = _head(target) if target else ((full or head or [""])[0] and _head((full or head)[0]))
            pair = [c for c in classes if _head(c) == h and state_of(c) is not None]
            obs = bool({state_of(c) for c in pair} >= {"open", "closed"}) or bool(pair)
            why = "" if obs else f"no open / closed state of '{h}' among the detector's classes"
        elif dest:
            dn, dh = _named(dest, classes)
            dest_ws = [w for w in dest.split() if w not in STOP]
            if dn or dh:
                dest = (dn or dh)[0]
            elif dest_ws and dest_ws[-1] in {"table", "desk", "bench", "tray", "floor", "mat"}:
                dest = None                        # a surface: 'leave it on the table' = place, observable
            else:
                obs, why = False, f"'{dest}' is not one of the detector's classes - '{s}' cannot be observed"
        elif not (full or head) and not target:
            obs, why = False, "it names none of the detector's classes"
        out.append({"text": s, "prim": prim, "observable": obs, "why": why, "dest": dest if obs else None,
                    "index": k})
    return out


# ── MEANING ──────────────────────────────────────────────────────────────────────────────────────────────────────
def _box_near(L, t, c, t0, before):
    ks = [k for k in range(len(t)) if (t[k] <= t0 if before else t[k] >= t0)]
    for k in (reversed(ks) if before else ks):
        bs = [b for n, b in L["dets"][k] if n == c]
        if bs:
            return bs[0]
    return None


def classify(L, events):
    """in place: e["meaning"] for every change (state_events.events, with 'into' / 'from' marks)"""
    t = np.asarray(L["t"], float)
    for e in events:
        if e.get("other"):
            e["meaning"] = "relation"
        elif e["kind"] in ("up", "down"):
            e["meaning"] = "count"
        elif e.get("into") or e.get("from"):
            e["meaning"] = "transform"
        else:
            e["meaning"] = "gone" if e["kind"] == "disappears" else "appears"
    for e in events:
        if e["meaning"] != "gone":
            continue
        b0 = _box_near(L, t, e["cls"], e["t"], True)
        for f in events:
            if f["cls"] == e["cls"] and f["kind"] == "appears" and not f.get("other") \
                    and e["t"] < f["t"] <= e["t"] + REAPPEAR_S and f["meaning"] in ("appears",):
                b1 = _box_near(L, t, f["cls"], f["t"], False)
                if b0 is None or b1 is None:
                    continue
                size = max(1.0, float(np.hypot(b0[2] - b0[0], b0[3] - b0[1])))
                if np.hypot((b0[0] + b0[2] - b1[0] - b1[2]) / 2, (b0[1] + b0[3] - b1[1] - b1[3]) / 2) \
                        <= REAPPEAR_NEAR * size:
                    e["meaning"], f["meaning"] = "gap", "gap_end"
                    e["back_at"] = f["t"]
                    break
    return events


# ── FIT ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def fit(sub, e, target, roles):
    """how well change e is evidence of sub-action sub (None = no semantic preference: the old rule decides)"""
    m = e.get("meaning")
    if m in ("gap", "gap_end"):
        return 0.0                                  # a detector gap / an occlusion is never a completion
    p = sub.get("prim")
    if p in ("close", "open"):
        want = "closed" if p == "close" else "open"
        if m == "transform":
            to = e["cls"] if e["kind"] == "appears" else e.get("into")
            frm = e.get("from") if e["kind"] == "appears" else e["cls"]
            same = _head(to) == _head(target) if target else True
            st = state_of(to)
            if st is None and state_of(frm) in ("open", "closed"):
                # (26 Nov) a class with NO state word is its partner's OPPOSITE state: 'open box' -> 'box' = closed
                # (a detector naming the closed box just 'box' - packing, close step took the cover instead)
                st = "closed" if state_of(frm) == "open" else "open"
            return ((3.0 if e["kind"] == "appears" else 2.9) if st == want else 0.1) if same else 0.2
        if e["kind"] == "appears" and state_of(e["cls"]) == want and (not target or _head(e["cls"]) == _head(target)):
            return 1.6                              # the state appearing (its other state never seen)
        if e["kind"] == "disappears" and state_of(e["cls"]) not in (None, want) \
                and (not target or _head(e["cls"]) == _head(target)):
            return 1.2                              # the other state gone
        if m == "gone":
            return 0.3                              # gone is not 'closed'
        return None
    if p in ("insert", "attach"):
        if m == "relation":
            return None                             # the old rule: X enters Y, both named
        if m == "transform":
            return 0.2
        if m == "gone" and e["cls"] == target:
            return 1.4                              # gone (into something) - the step's own object
        return None
    if p == "remove":
        if m == "transform":
            return 0.2
        return None
    if m == "transform":
        return 0.3                                  # place / move / pick are not state changes
    return None


def step_fit(subs, e, target, roles, old):
    """the step's fit for change e: the best of its OBSERVABLE sub-actions (semantic fit where it has a view,
    else the old rule's) -> (fit, index of that sub-action)"""
    best, idx = 0.0, None
    for s in subs:
        if not s["observable"]:
            continue
        f = fit(s, e, target, roles)
        f = old if f is None else f
        if f > best:
            best, idx = f, s["index"]
    if idx is None and e.get("meaning") not in ("gap", "gap_end") and not any(s["observable"] for s in subs):
        return old, None
    return best, idx


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.boundary_learning   (from sopmon\boundary_learning.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""boundary_learning.py - TEMPORAL LANDMARKS of each step on the good video, for training (24 Nov, P0).

The automatic step labels (auto_labels.py) put a step's END at the detector's LASTING change - the moment the change
starts, but for a part that goes into the hand first and only later onto its destination, or a part pushed into another
under the hand, the physical completion and the lasting change are not the same event. Per step four landmarks:

  precondition   the step's object still in its initial state (the last moment before it is handled)
  start          the operator starts the manipulation (the label's start, or the object's first movement)
  transition     the detector's state change (the label's end: appears / disappears / enters ...)
  completion     the EARLIEST RELIABLE moment the postcondition holds - from the object's own evidence:
                   part GONE onto / into its destination   its last sighting AT the destination (it went there)
                   part PLACED in a destination             it came to rest in it / vanished inside it (relations.py)
                   anything else                            the transition
  interval       when the completion itself was hidden (last seen in the hand, not at the destination): [the last
                 sighting, the transition] - an uncertain boundary, NOT an artificially exact time

Offline the whole good video is used to validate a candidate (it must hold afterwards); the trained model still sees only
past and present frames at inference. The label's END becomes the completion; training spreads the 'action ends here'
target over the interval when there is one (object_actions.train).
"""
import numpy as np

AT_DEST = 0.5          # 'at the destination': the boxes overlap, or are apart by less than this x the smaller's size
MAX_SHIFT_S = 3.0      # the completion may be at most this much earlier than the transition (else: not this step's)


def _boxes(L, c, k):
    return [d[1] for d in L["dets"][k] if d[0] == c]


def boundary_learning__gap(a, b):
    return max(0.0, max(a[0], b[0]) - min(a[2], b[2]), max(a[1], b[1]) - min(a[3], b[3]))


def _diag(b):
    return max(1.0, ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5)


def _gone_completion(L, t, obj, dest, te, ts):
    """a part that DISAPPEARS (fitted onto / into its destination): its last sighting before te, and whether that
    sighting was at the destination -> (completion, interval or None, why)"""
    k_te = int(np.searchsorted(t, te))
    last = None
    for k in range(min(k_te, len(t) - 1), -1, -1):
        if t[k] < ts - 1.0:
            break
        if _boxes(L, obj, k):
            last = k
            break
    if last is None:
        return te, None, "not seen in the step - its transition"
    bo = _boxes(L, obj, last)[0]
    bd = None
    for dk in sorted(range(-15, 16), key=abs):
        j = last + dk
        if 0 <= j < len(t) and abs(t[j] - t[last]) <= 0.5 and dest and _boxes(L, dest, j):
            bd = _boxes(L, dest, j)[0]
            break
    tl = float(t[last])
    if bd is not None and boundary_learning__gap(bo, bd) <= AT_DEST * min(_diag(bo), _diag(bd)) and te - tl <= MAX_SHIFT_S:
        return tl, None, f"last seen AT the {dest} at {tl:.2f}s"
    # last seen somewhere else (in the hand): the completion happened hidden, between then and the transition
    return te, [round(tl, 2), round(float(te), 2)], f"last seen away from the {dest} at {tl:.2f}s - hidden completion"


def _placed_completion(L, t, obj, dest, te, ts, L_rel=None, t_next=None):
    """a part PUT INTO / ONTO a destination: the placement (relations.py) whose stay ends the step - when it came to
    rest in it, or vanished inside it -> (completion, interval or None, why)"""
    pass
    L = L_rel if L_rel is not None else L                    # with the hands (held / released evidence)
    try:
        F, pls = placements(L, obj, dest, live=False, hand=hand_class(L))
    except Exception as e:                                   # no hands etc.: the transition stays
        return te, None, f"placements not available ({type(e).__name__})"
    best = None
    hi = t_next if t_next is not None else te + 1.0
    for p in pls:
        # only a placement the placement rules ACCEPT (not one 'held when last seen, then handled again'); the FIRST
        # one from the step's start up to the next step's change - an insertion is complete when the part is fully in
        # and stays, which can be after its first entry (the refill: enters at 16.8 s, fully in at 19.9 s)
        if decide(p, False)[0] == "NOT_PLACED":
            continue
        if p.get("candidate") and ts - 1.0 <= p["entered"] and p["completion"] <= hi:
            best = p if best is None or p["completion"] < best["completion"] else best
    if best is None:
        return te, None, "no accepted placement in the step - its transition"
    tc = float(best["completion"])
    if tc > te:
        return tc, None, f"fully in the {dest} and stayed from {tc:.2f}s (accepted placement)"
    if best.get("vanished_inside") is not None and best["evidence"].get("rest_at") is None:
        # it vanished inside while held: the completion is hidden between its entry and its vanishing
        return tc, [round(float(best["entered"]), 2), round(tc, 2)], f"vanished inside the {dest} at {tc:.2f}s"
    if te - MAX_SHIFT_S <= tc <= te + 0.5:
        return min(tc, te), None, f"came to rest in the {dest} at {tc:.2f}s"
    return te, None, "its placement is far from the transition - the transition"


def landmarks(model, L, labels, info_roles, log=print, L_rel=None):
    """labels {step id: {"start", "end", "cls", "other", "kind"}} (auto_labels) + each step's roles -> the labels with
    the landmarks added and END moved to the earliest reliable completion (+ "interval" for a hidden one)"""
    t = np.asarray(L["t"], float)
    out = {}
    order = [s["id"] for s in model["spec"]["steps"] if labels.get(s["id"])]
    for s in model["spec"]["steps"]:
        sid = s["id"]
        lb = labels.get(sid)
        nx = [labels[o] for o in order[order.index(sid) + 1:]] if sid in order else []
        t_next = float(nx[0].get("transition", nx[0]["end"])) if nx else float(t[-1])
        if not lb or lb.get("kind") in (None, "start", "holds"):
            if lb:
                out[sid] = dict(lb)
            continue
        rs = info_roles.get(sid) or []
        ts, te = float(lb["start"]), float(lb["end"])
        obj = lb.get("cls") or (rs[0] if rs else None)
        dest = lb.get("other") or (rs[1] if len(rs) > 1 else None)
        if dest == obj:
            dest = None
        if lb["kind"] == "disappears" and obj and dest:
            tc, iv, why = _gone_completion(L, t, obj, dest, te, ts)
        elif lb["kind"] == "enters" and obj and dest:
            tc, iv, why = _placed_completion(L, t, obj, dest, te, ts, L_rel, t_next)
        else:
            tc, iv, why = te, None, "its own change (appears / leaves / count)"
        # PRECONDITION: the object's last sighting at rest where it lay, before the start
        pre = ts
        if obj:
            for k in range(int(np.searchsorted(t, ts)), -1, -1):
                if _boxes(L, obj, k):
                    pre = float(t[k])
                    break
        # never before the start; never after the transition - except an accepted placement (fully in) before the next
        # step's change
        tc = max(ts + 0.2, min(tc, t_next if lb["kind"] == "enters" else te))
        new = dict(lb, end=round(tc, 2), transition=round(te, 2), precondition=round(min(pre, ts), 2),
                   completion=round(tc, 2), completion_why=why)
        if iv:
            new["interval"] = iv
        out[sid] = new
        log(f"    step {sid}: completion {tc:.2f}s (transition {te:.2f}s)" + (f", hidden in {iv}" if iv else "")
            + f" - {why}")
    return out


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.auto_labels   (from sopmon\auto_labels.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""auto_labels.py - the CORRECT video's step timestamps written AUTOMATICALLY from the detector (no VLM, no marking).

  CHANGES   the detector's LASTING state changes (sopmon/state_events.py, held >= MIN_HELD; a class flickering for a
            second is not a change) + 'the video's start' (a step already done before the recording)
  ROLES     per step, the objects its SOP text names (object_actions.roles: target + others)
  MATCH     the correct video follows the SOP, so its steps take changes in increasing time - each change once, each
            step one change or none - with the most total fit: a change between BOTH of a step's objects (phone
            enters outer box) fits best, then its target appearing / disappearing, then any of its objects; 'the
            start' only for the first steps; a step none of whose objects change takes a remaining lasting change
            between its neighbours (the tape hides the sticker place)
  TIMES     END = the change; START = when the work on the product rises after the previous step's end
  -> {step id: {"start", "end", "change", "fit"}} - the same form as the user's marks (object_actions.train)
"""
import numpy as np

MIN_HELD = 2.5
SEMANTIC = True              # (25 Nov, P0) semantic event filtering + compound-action decomposition (semantic_events)
NONE_PENALTY = 1.0
MAX_ACTION_S = 8.0           # an automatic action starts at most this long before its change


import re

# the KIND of change a step's words describe (generic verbs, not one SOP's objects)
KIND_OF = [(r"\b(into|insert|inside|put\s+\w+\s+in)\b", "enters"),
           (r"\b(close|closed|seal|cover|covered|hide)\b", "disappears"),
           (r"\b(open|place|placed|put\s.*\bdown|set\s.*\bdown|lay|visible|appear)\b", "appears"),
           (r"\b(remove|take\s+out|pull\s+out)\b", "leaves")]


def expected_kinds(step):
    txt = " ".join(str(step.get(k) or "") for k in ("text", "action_query")).lower()
    return {kind for pat, kind in KIND_OF if re.search(pat, txt)}


def auto_labels__fit(step_roles, e, k_step, kinds=(), at_start=False):
    """how well a change fits a step (0 = not at all): its objects AND the kind of change its words describe"""
    if e["kind"] == "start":
        # only the SOP's first step can be 'done before the recording', and only if its object is there then
        return (1.8 if at_start else 0.6) if k_step == 0 else 0.0
    rs = step_roles
    if not rs:
        return 0.2
    tgt = rs[0]
    kind_ok = (not kinds) or e["kind"] in kinds
    if e.get("other"):                                        # a RELATION (one object enters / leaves another)
        # counts only for a step that names BOTH objects ('phone enters outer box' is the phone step's, not the
        # box's: the box step does not name the phone) - and of the kind its words say
        if e["cls"] in rs and e["other"] in rs:
            # in the step's direction first: ITS target enters the other ('phone enters outer box' for 'put the
            # phone into the outer box', not 'outer box enters phone' at the same moment)
            return (3.2 if e["cls"] == tgt else 3.0) if kind_ok else 0.5
        return 0.0
    if e["cls"] == tgt and e["kind"] in ("appears", "disappears"):
        return 2.0 if kind_ok else 0.8
    if e["cls"] in rs:
        return 1.5 if kind_ok else 0.6
    # a lasting change of some other object: a leftover (a step whose own object never changes visibly - the tape
    # strip: the sticker place it covers disappears) - one of the KIND its words describe first ('seal' -> covered)
    return 0.4 * min(e["held"], 20.0) / 20.0 * (1.5 if (kinds and e["kind"] in kinds) else 0.5 if kinds else 1.0)


def auto_labels(model, video, L, log=print, exclude=()):
    """exclude: {(step id, change time)} a step may NOT take - changes the self-check could not confirm for it
    (label_gui.auto_train retries without them: the pen cap step took 'pen enters pen cap' at 19.9 s, a box of
    the cap + the fingers; the real fitting is the cap disappearing at 24.4 s)"""
    pass
    steps = model["spec"]["steps"]
    classes = sorted({n for ds in L["dets"] for n, _ in ds})
    roles = {s["id"]: object_actions_roles(s, classes) for s in steps}
    t = np.array(L["t"], float)
    all_evs = events(L)
    if SEMANTIC:
        # (25 Nov, P0) what each change MEANS (a detector gap / occlusion, a state change, a relation ...) and each
        # step's sub-actions ('close the box AND slide it into its sleeve'): a change is matched to what it can be
        # evidence of - a detector gap never
        classify(L, all_evs)
        subs = {s["id"]: sub_actions(s, classes, (roles[s["id"]] or [None])[0]) for s in steps}
        for s in steps:
            if len(subs[s["id"]]) > 1 or not all(x["observable"] for x in subs[s["id"]]):
                log(f"    step {s['id']} sub-actions: " + "; ".join(
                    f"{chr(97 + x['index'])}) '{x['text']}' [{x['prim']}]"
                    + ("" if x["observable"] else f" NOT OBSERVABLE ({x['why']})") for x in subs[s["id"]]))
        gaps = [e for e in all_evs if e.get("meaning") == "gap"]
        if gaps:
            log("    detector gaps / occlusions (never a completion): " + ", ".join(
                f"{e['cls']} {e['t']:.1f}-{e['back_at']:.1f}s" for e in gaps))
    evs = [e for e in all_evs if e["held"] >= MIN_HELD]
    # one change per moment: of simultaneous changes keep the one that fits ANY step best (relation > own class)
    cand = [{"t": float(t[0]), "kind": "start", "cls": None, "held": 99.0}]
    for e in evs:
        cand.append(e)
    cand.sort(key=lambda e: e["t"])
    n, m = len(steps), len(cand)
    first = [k for k in range(len(t)) if t[k] <= t[0] + 2.0]
    seen0 = {c for k in first for c, _ in L["dets"][k]}
    kinds = {s["id"]: expected_kinds(s) for s in steps}
    F = np.array([[auto_labels__fit(roles[s["id"]], e, k, kinds[s["id"]], bool(roles[s["id"]]) and roles[s["id"]][0] in seen0)
                   for e in cand] for k, s in enumerate(steps)])
    sub_of = {}
    if SEMANTIC:
        # (25 Nov) a step whose observable sub-actions REPEAT the state change the step before made, on the same object
        # ('close the box' -> 'leave it closed on the table'): no change of its own exists - it is a STATE-HOLDS step
        # (_state_holds), never matched to a change (it took 'mouse disappears' under the lid)
        holds_ids = set()
        for i, s in enumerate(steps[1:], 1):
            obs = [x for x in subs[s["id"]] if x["observable"]]
            prev_obs = [x for x in subs[steps[i - 1]["id"]] if x["observable"]]
            tgt_, ptgt = (roles[s["id"]] or [None])[0], (roles[steps[i - 1]["id"]] or [None])[0]
            if obs and all(x["prim"] in STATE_PRIMS for x in obs) and tgt_ and tgt_ == ptgt \
                    and {x["prim"] for x in obs} <= {x["prim"] for x in prev_obs}:
                holds_ids.add(s["id"])
                F[i, :] = 0.0
                log(f"    step {s['id']}: repeats step {steps[i - 1]['id']}'s state ({', '.join(x['prim'] for x in obs)} "
                    f"{tgt_}) - a state that must HOLD, not a change of its own")
        for i, s in enumerate(steps):
            if s["id"] in holds_ids:
                continue
            for j, e in enumerate(cand):
                if e["kind"] == "start":
                    continue
                F[i, j], sub_of[(i, j)] = step_fit(subs[s["id"]], e, (roles[s["id"]] or [None])[0],
                                                      roles[s["id"]], float(F[i, j]))
    ex = {(int(sid), round(float(tt), 2)) for sid, tt in exclude}
    for i, s in enumerate(steps):
        for j, e in enumerate(cand):
            if (int(s["id"]), round(float(e["t"]), 2)) in ex:
                F[i, j] = 0.0
    # monotonic assignment: steps in order -> changes in increasing time, each used once, a step may take none
    NEG = -1e9
    D = np.full((n + 1, m + 1), NEG)
    D[0, :] = 0.0
    back = {}
    for i in range(1, n + 1):
        D[i, 0] = D[i - 1, 0] - NONE_PENALTY
        back[(i, 0)] = ("none", 0)
        for j in range(1, m + 1):
            opts = [(D[i, j - 1], ("skip", j - 1)), (D[i - 1, j] - NONE_PENALTY, ("none", j))]
            if F[i - 1, j - 1] > 0:
                opts.append((D[i - 1, j - 1] + F[i - 1, j - 1], ("take", j - 1)))
            D[i, j], back[(i, j)] = max(opts, key=lambda o: o[0])
    pick, i, j = {}, n, m
    while i > 0:
        how, jj = back[(i, j)]
        if how == "take":
            pick[steps[i - 1]["id"]] = cand[jj]
            i, j = i - 1, jj
        elif how == "none":
            i, j = i - 1, jj
        else:
            j = jj
    # START: the work on the product rising after the previous step's end
    S = signals(video, L, log)
    w = np.convolve(S["motion"], np.ones(3) / 3, mode="same")
    w = w / max(float(np.percentile(w, 95)), 1e-6)
    tw = S["t"]
    out, prev_end = {}, float(t[0])
    for s in steps:
        e = pick.get(s["id"])
        if e is None:
            log(f"    step {s['id']}: no detector change fits it - no automatic label")
            continue
        end = float(e["t"])
        if e["kind"] == "start":
            out[s["id"]] = {"start": float(t[0]), "end": float(t[0]) + 1.0, "change": "already there at the start",
                            "fit": 1.2, "cls": None, "other": None, "kind": "start"}
        else:
            # the LAST continuous burst of work on the product before the change (gaps up to 1 s bridged), not
            # earlier than the previous step's end
            k = int(np.searchsorted(tw, end - 0.3)) - 1
            while k > 0 and w[k] < 0.3 and tw[k] > prev_end and end - tw[k] < 6.0:
                k -= 1                                        # back to the burst's end
            gap = 0.0
            dt = float(np.median(np.diff(tw))) if len(tw) > 1 else 1 / 3
            while k > 0 and tw[k - 1] >= prev_end:
                gap = 0.0 if w[k - 1] >= 0.3 else gap + dt
                if gap > 1.0:
                    break
                k -= 1
            start = max(prev_end, float(tw[k]), end - MAX_ACTION_S)   # a long quiet gap is not one long action
            if end - start < 1.0:
                start = max(prev_end, end - 2.0)
            i_, j_ = steps.index(s), cand.index(e)
            out[s["id"]] = {"start": round(start, 2), "end": round(end, 2), "change": describe(e)
                            + (f" (from {e['from']})" if e.get("from") else "")
                            + (f" (became {e['into']})" if e.get("into") else ""),
                            "fit": round(float(F[i_, j_]), 2),
                            "cls": e["cls"], "other": e.get("other"), "kind": e["kind"], "into": e.get("into"),
                            "from": e.get("from"), "meaning": e.get("meaning")}
            if SEMANTIC:
                # the sub-actions: which one this change is evidence of, and the ones the detector cannot observe
                out[s["id"]]["subactions"] = [dict(x, matched=(x["index"] == sub_of.get((i_, j_))))
                                              for x in subs[s["id"]]]
        prev_end = out[s["id"]]["end"]
        log(f"    step {s['id']} {roles[s['id']]}: {out[s['id']]['start']:.1f}-{out[s['id']]['end']:.1f}s  "
            f"<- {out[s['id']]['change']}")
    if SEMANTIC:
        _state_holds(steps, roles, subs, out, L, t, log)
    return out


STATE_PRIMS = {"close", "open", "place"}
STATE_HOLD_S = 2.0           # (25 Nov) a 'state holds' step: its object seen steadily this long after the earlier step
STATE_GAP_S = 0.3            # gaps up to this long inside the hold are bridged (a hand passing over)


def _state_holds(steps, roles, subs, out, L, t, log):
    """(25 Nov, user's choice) a step that got NO change of its own and whose OBSERVABLE sub-actions are all STATES of
    its object ('leave it closed on the table') that an EARLIER step created (step 4 'close the box'): it is done when
    that state HOLDS afterwards - its object seen steadily STATE_HOLD_S after the earlier step's end. In place on out"""
    for k, s in enumerate(steps):
        sid = s["id"]
        if sid in out:
            continue
        obs = [x for x in subs.get(sid) or [] if x["observable"]]
        c = (roles.get(sid) or [None])[0]
        if not obs or any(x["prim"] not in STATE_PRIMS for x in obs) or not c:
            continue
        prev = [p["id"] for p in steps[:k] if p["id"] in out and out[p["id"]].get("kind") not in (None, "start")]
        if not prev:
            continue
        pv = out[prev[-1]]
        seen = np.array([any(n == c for n, _ in ds) for ds in L["dets"]], bool)
        th, k0, gap = None, None, 0.0
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
        for j in range(len(t)):
            if t[j] < float(pv["end"]):
                continue
            if seen[j]:
                k0 = j if k0 is None else k0
                gap = 0.0
                if t[j] - t[k0] >= STATE_HOLD_S:
                    th = float(t[j])
                    break
            elif k0 is not None:
                gap += dt
                if gap > STATE_GAP_S:
                    k0, gap = None, 0.0
        if th is None:
            log(f"    step {sid}: no change of its own, and its {c} is not seen held after step {prev[-1]} - no label")
            continue
        out[sid] = {"start": round(float(pv["end"]), 2), "end": round(th, 2),
                    "change": f"{c} still there {STATE_HOLD_S:g} s after step {prev[-1]}", "fit": 1.0,
                    "cls": c, "other": None, "kind": "holds", "after": prev[-1], "hold_s": STATE_HOLD_S,
                    "meaning": "holds",
                    "subactions": [dict(x, matched=x["observable"]) for x in subs.get(sid) or []]}
        log(f"    step {sid} {roles[sid]}: {out[sid]['start']:.1f}-{out[sid]['end']:.1f}s  <- {out[sid]['change']} "
            f"(its own change: none - a state an earlier step made, that must HOLD)")


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.ref_match   (from sopmon\ref_match.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""ref_match.py - ONE-VIDEO REFERENCE MATCHING (a test; temporal.pt and temporal_action.pt are not changed).

  REFERENCE  from the correct video only, per step: its feature sequence PRE (2 s before the start) / ACTIVE (start
             -> validated completion, trained/<model>/action_labels.json) / POST (2 s after)
  FEATURES   per frame: DINOv2-small (Apache 2.0, frozen) embedding of the product area, centred per video; the
             action model's state features (work on the product, its change, a part leaving, product seen; and for
             the step's own part: at its place, on the product); the part's distance to the product when it is seen
  MATCHING   subsequence DTW of the reference's ACTIVE+POST frames inside a test window (any speed, no
             augmentation): the completion = where the reference's last ACTIVE frame is aligned; the score = the
             path's mean cost. IDENTITY: every window scored against EVERY step's reference - the best should be its
             own step (actions recognised without being told the order)

    references(name, good)  -> saved trained/<model>/action_prototypes.npz
    match(name, video)      -> per step: window, completion, score; and the identity matrix
"""
import json
from pathlib import Path

import cv2
import numpy as np

pass
ref_match_ROOT = home()
ref_match_OUT = ref_match_ROOT / "outputs" / "_refmatch"
ref_match_PRE_S = ref_match_POST_S = 2.0
W_STATE = 0.5                 # the state features' weight next to the appearance distance
_DINO = {}


def ref_match__vkey(video):
    p = Path(video)
    return f"{p.stem}_{p.stat().st_size}"


def ref_match__dino():
    if "m" not in _DINO:
        import timm
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0,
                              dynamic_img_size=True).eval().to(dev)
        _DINO["m"] = (m.half() if dev == "cuda" else m, dev)
    return _DINO["m"]


def appearance(video, L, log=print):
    """DINOv2 embedding of the product area at every live frame (cached) -> (T, 384), centred per video, unit"""
    ref_match_OUT.mkdir(parents=True, exist_ok=True)
    pass
    c = ref_match_OUT / f"{ref_match__vkey(video)}_{rkey(L)}_dino.npy"
    E = np.load(c) if c.exists() else None               # keyed by the record's sampling too (boundaries.rkey)
    if E is None:
        import torch
        m, dev = ref_match__dino()
        t = np.array(L["t"])
        Ps = [p for p in L["P"] if p is not None]
        P = np.median(np.array(Ps), 0) if Ps else np.array([0, 0, L["W0"], L["H0"]], float)
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        s = (cap.get(cv2.CAP_PROP_FRAME_WIDTH) or L["W0"]) / L["W0"]
        w, h = P[2] - P[0], P[3] - P[1]
        x1, y1 = int(max(0, (P[0] - 0.3 * w) * s)), int(max(0, (P[1] - 0.3 * h) * s))
        x2, y2 = int((P[2] + 0.3 * w) * s), int((P[3] + 0.3 * h) * s)
        log(f"  [ref] DINOv2 of the product area of {Path(video).name} ({len(t)} frames) ...")
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        ims, fi, out = [], 0, []

        def flush():
            if ims:
                x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).to(dev)
                with torch.no_grad():
                    f = m(x.half() if dev == "cuda" else x).float().cpu().numpy()
                out.extend(f)
                ims.clear()
        for tk in t:
            want = int(round(tk * fps))
            while fi < want:
                if not cap.grab():
                    break
                fi += 1
            ok, fr = cap.read()
            fi += 1
            if not ok:
                break
            crop = cv2.resize(fr[y1:y2, x1:x2], (308, 224), interpolation=cv2.INTER_AREA)[:, :, ::-1]
            ims.append(((crop.astype(np.float32) / 255.0) - mean) / std)
            if len(ims) == 32:
                flush()
        flush()
        cap.release()
        E = np.array(out, np.float32)
        np.save(c, E)
    E = E - E.mean(0, keepdims=True)
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-6)


def frame_features(video, L, classes, log=print):
    """-> t, appearance (T, 384), per unit state (U, T, S), units"""
    pass
    X, t, units = action_model_features(video, L, classes, log)
    E = appearance(video, L, log)[:len(t)]
    # state features: 0 at_place, 2 on_product, 3 product_seen, 4 work, 5 changing, 6 other_left
    S = X[:, :, [0, 2, 3, 4, 5, 6]]
    return t, E, S, units


WIN_BEFORE = 1.5             # a step's window opens this long before its pickup


def references_from(d, good, L, lab, log=print):
    """the reference PRE / ACTIVE / POST sequences of every step with a validated label (lab: {step: {ok, start,
    end}}) from the correct video's recorded run L -> d/action_prototypes.npz + .json"""
    d = Path(d)
    classes = sorted({n for ds in L["dets"] for n, _ in ds})
    t, E, S, units = frame_features(good, L, classes, log)
    refs, meta = {}, {"classes": classes, "steps": {}}
    for k, u in enumerate(units):
        lb = lab.get(str(u["step"])) or lab.get(u["step"])
        if not lb or not lb.get("ok"):
            continue
        a, e = lb["start"], lb["end"]
        act = np.flatnonzero((t >= a) & (t <= e))
        post = np.flatnonzero((t > e) & (t <= e + ref_match_POST_S))
        pre = np.flatnonzero((t >= a - ref_match_PRE_S) & (t < a))
        idx = np.concatenate([act, post])
        refs[f"s{u['step']}_E"], refs[f"s{u['step']}_S"] = E[idx], S[k, idx]
        refs[f"s{u['step']}_preE"] = E[pre]
        meta["steps"][str(u["step"])] = {"n_active": int(len(act)), "n_post": int(len(post)), "start": a, "end": e}
    np.savez_compressed(d / "action_prototypes.npz", **refs)
    (d / "action_prototypes.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"  saved {len(meta['steps'])} reference actions -> {d / 'action_prototypes.npz'}")
    return refs, meta


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.completion   (from sopmon\completion.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""completion.py - WHEN each step was completed, learnt from the one correct video (app: Train model / Run inference).
Shown next to the verdict, not deciding it: the order is still judged on the confirmed pickups (temporal.py).

  TRAIN  the correct video through the model's live inference (detector + places + pickups) ->
         each step's action: START = its confirmed pickup; END = the change point where the work on the product goes
         from high to low, inside the window up to the next observed pickup (sopmon/boundaries.py);
         a label is TRUSTED only if the product's look (DINOv2, Apache 2.0) changed across the step by more than the
         video's median 2-second change - no grounding model, no labels from outside
         -> trained/<model>/action_labels.json, action_prototypes.npz/.json (each step's PRE/ACTIVE/POST sequence)
  CHECK  a test video's live run -> each step's completion = the reference action aligned by subsequence DTW inside
         its window (any speed) -> per step 'completed at X s', and the order on the completion times (a second
         opinion next to the pickups' verdict)
"""
import json
from pathlib import Path

import numpy as np

pass
completion_ROOT = home()
completion_OUT = completion_ROOT / "outputs" / "_completion"


def record_video(model, video, registry, log=print, check=None, out_dir=None):
    """the model's live inference over a video file, no window -> the run's record (temporal.LiveInference.record)"""
    pass
    pass
    models = models_for(model, registry)
    pl = Pipeline(None, models, Path(out_dir or completion_OUT / "run"), fps=float(model.get("fps") or 3.0),
                  algorithm_only=True, log=lambda s: None)
    # a RECORD for training / offline checks: only the step model's frames (detecting the ones between adds nothing)
    live = LiveInference(pl, model, str(video), fps=float(model.get("fps") or 3.0), log=lambda s: None,
                                  all_frames=False)
    n = 0
    while True:
        if check is not None and n % 50 == 0:
            check()
        if live.step() is None:
            break
        n += 1
    fin = live.finish()
    live.release()
    if hasattr(pl.bank, "release"):
        pl.bank.release()
    return live.record(fin)


def completion_labels(video, L, log=print):
    """change-point labels, each checked against the product's look before / after -> {step: {ok, start, end, why}}"""
    pass
    B = boundaries(video, L, log)
    E = appearance(video, L, log)
    t = np.array(L["t"])[:len(E)]
    mean = lambda a, b: E[(t >= a) & (t < b)].mean(0) if ((t >= a) & (t < b)).any() else None
    dist = lambda x, y: float(1 - (x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))
    null = [dist(mean(a, a + 2), mean(a + 2, a + 4)) for a in np.arange(t[0], t[-1] - 4, 1.0)
            if mean(a, a + 2) is not None and mean(a + 2, a + 4) is not None]
    thr = float(np.median(null)) if null else 0.0
    out = {}
    for sid, b in B.items():
        if b.get("end") is None:
            out[sid] = {"ok": False, "start": b["start"], "why": b.get("why") or "no end"}
            continue
        pre, post = mean(b["start"] - 2.0, b["start"]), mean(b["end"] + 0.3, b["end"] + 2.3)
        ch = dist(pre, post) if pre is not None and post is not None else None
        ok = ch is not None and ch >= thr and b["end"] - b["start"] >= 1.0
        out[sid] = {"ok": ok, "start": b["start"], "end": b["end"], "post_change": None if ch is None else round(ch, 3),
                    "threshold": round(thr, 3),
                    "why": f"change point {b['end']:.1f}s; product look changed {ch:.2f} "
                           f"({'>=' if ok else '<'} the video's median 2-s change {thr:.2f})" if ch is not None
                    else "no product view before / after"}
    return out


def completion_train(model, video, registry, log=print, check=None):
    """the correct video -> trusted labels + the reference actions in the model's folder -> summary lines"""
    pass
    d = Path(model["places_file"]).parent
    log("[completion] the correct video once more through the model (detector + places + pickups) ...")
    L = record_video(model, video, registry, log, check, out_dir=d / "_run")
    lab = completion_labels(video, L, log)
    (d / "action_labels.json").write_text(json.dumps({"video": Path(video).name, "source": "change points + "
                                                      "post-state check", "labels": lab}, indent=1, default=float),
                                          encoding="utf-8")
    references_from(d, video, L, lab, log)
    lines = []
    for s in model["spec"]["steps"]:
        lb = lab.get(s["id"])
        lines.append(f"  step {s['id']}: " + ("no confirmed pickup in the correct video" if not lb else
                                              ("TRUSTED  " if lb["ok"] else "left out  ") + lb["why"]))
    n = sum(1 for x in lab.values() if x["ok"])
    lines.append(f"  completion references: {n} / {len(model['spec']['steps'])} steps")
    return lines


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  MODULE: sopmon.label_gui   (from sopmon\label_gui.py)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""label_gui.py - LABEL STEPS (app: section 2 'Label steps...'): the selected model's training video, a viewer, a
timeline of the detector's changes, and one row per SOP step - Step | Start | End. The table is filled with the saved
labels, else the AUTOMATIC suggestions (sopmon/auto_labels.py: the detector's changes + the SOP's objects and verbs).
'Save labels & train' saves them (trained/<model>/marked_labels.json) and trains the object-conditioned action matcher
on them only (sopmon/object_actions.py -> action_matcher.pt). Run inference then uses it for that model."""
import json
import queue
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import ttk

import cv2

KIND_COL = {"appears": "#3fbf5f", "disappears": "#e05050", "enters": "#4a90e2", "leaves": "#9b6bd6", "up": "#e0a030",
            "down": "#e0a030"}
VW, VH = 880, 495


def suggested_path(model):
    return Path(model["places_file"]).parent / "suggested_labels.json"


def label_gui_record(model, video, registry, log=print, check=None):
    pass
    if "name" not in model:
        model = dict(model, name=Path(model["places_file"]).parent.name)
    return record(model, video, registry, log, check)


def suggest(model, video, registry, log=print, check=None, exclude=(), L=None):
    """the automatic labels of a model's training video -> suggested_labels.json (not used before confirmed)"""
    pass
    L = L or label_gui_record(model, video, registry, log, check)
    labs = auto_labels(model, video, L, log, exclude=exclude)
    if BOUNDARIES and labs:
        # (24 Nov, training P0) each step's END moved to its EARLIEST RELIABLE completion; a hidden one = an interval
        classes = sorted({n for ds in L["dets"] for n, _ in ds})
        roles = {s["id"]: object_actions_roles(s, classes) for s in model["spec"]["steps"]}
        log("  completion landmarks (boundary_learning):")
        try:
            L_rel = _with_hands(L, video, None, log)     # the hands, as the step check uses them
        except Exception:
            L_rel = None
        labs = landmarks(model, L, labs, roles, log=log, L_rel=L_rel)
    suggested_path(model).write_text(json.dumps({str(k): v for k, v in labs.items()}, indent=1), encoding="utf-8")
    return labs


BOUNDARIES = True        # (24 Nov) completion landmarks in the automatic labels - False = the lasting change as before
TIMING_J = True          # (24 Nov) self-correction keeps the labelling with the best timing score J, not the most steps
J_LATE, J_EARLY, J_FALSE, J_FALSE_S = 0.1, 0.2, 0.5, 2.0


def timing_score(labs, res):
    """J = correct rate - J_LATE x mean lateness - J_EARLY x mean earliness - J_FALSE x false rate, on the training
    video's self-check: a PREMATURE confirmation costs twice a late one; a confirmation more than J_FALSE_S off its
    label counts as false (another moment, not this step) -> (J, text)"""
    sids = [s for s in labs if labs[s].get("kind") not in (None, "start")]
    if not sids:
        return 0.0, "no timed steps"
    ok = late = early = false = 0.0
    for s in sids:
        x = res.get(s) or {}
        if x.get("status") != "COMPLETED" or x.get("end_sec") is None:
            continue
        d = float(x["end_sec"]) - float(labs[s]["end"])
        if abs(d) > J_FALSE_S:
            false += 1
            continue
        ok += 1
        late, early = late + max(0.0, d), early + max(0.0, -d)
    n = len(sids)
    J = ok / n - J_LATE * late / max(ok, 1) - J_EARLY * early / max(ok, 1) - J_FALSE * false / n
    return J, f"J {J:.2f} (correct {ok:.0f}/{n}, late {late / max(ok, 1):.2f}s, early {early / max(ok, 1):.2f}s, false {false:.0f})"


RETRIES = 3          # auto_train: how often an unconfirmed step may try its next-best change


def auto_train(model, video, registry, log=print, check=None, kind="tcn"):
    """TRAIN MODEL, automatically: the good video's step timestamps evaluated from the detector (auto_labels) ->
    the labels -> the object-conditioned action model trained on them -> action_matcher.pt; -> the lines to show.
    SELF-CORRECTING: the trained model is checked on the training video itself. A step it cannot confirm there
    was given the wrong change (the pen cap: 'pen enters pen cap' at 19.9 s - the cap box held the fingers too);
    that change is excluded for that step and the steps are timed again (next-best change, same SOP order),
    trained and checked again - up to RETRIES times. The labelling with the most confirmed steps is kept."""
    pass
    L = label_gui_record(model, video, registry, log, check)
    amb = ambiguous_steps(model["spec"], L)
    if not amb:
        # every step has its own place and its own pickup time (the phone): the pickups already time them
        return ["  every step has its own place and pickup time - the pickups time the steps; no action model "
                "needed (to train one anyway: 'Label steps…')"]
    log("  steps the pickups cannot time: " + "; ".join(f"step {k}: {v}" for k, v in sorted(amb.items())))
    labs = suggest(model, video, registry, log, check, L=L)
    if not labs:
        return ["  no step could be timed from the detector - no action model (use 'Label steps…')"]
    ok = (lambda r: timing_score(labs_now[0], r)[0]) if TIMING_J else \
        (lambda r: sum(1 for x in r.values() if x.get("status") == "COMPLETED"))
    labs_now = [labs]
    lines = train(model, video, L, labs, log, kind=kind)
    chk, res = _self_check(model, video, L, log, want_res=True)
    best, exclude, notes = (ok(res), labs, lines, chk, res), set(), []
    for attempt in range(RETRIES):
        bad = [sid for sid in labs if (res.get(sid) or {}).get("status") != "COMPLETED" and labs[sid].get("kind")
               not in (None, "start")]
        if not bad:
            break
        exclude |= {(sid, labs[sid].get("transition", labs[sid]["end"])) for sid in bad}   # the change itself
        msg = (f"  [self-correct {attempt + 1}] not confirmed on the training video: " + ", ".join(
            f"step {sid} ({labs[sid].get('change')} at {labs[sid]['end']:.1f}s)" for sid in bad)
            + " - timing again without those changes")
        log(msg)
        notes.append(msg)
        new = suggest(model, video, registry, log, check, exclude=exclude, L=L)
        if not new or all((new.get(sid) or {}).get("end") == (labs.get(sid) or {}).get("end") for sid in new) \
                and set(new) == set(labs):
            notes.append("    no other change fits those steps - kept the best labelling")
            break
        labs = new
        labs_now[0] = labs
        lines = train(model, video, L, labs, log, kind=kind)
        chk, res = _self_check(model, video, L, log, want_res=True)
        if ok(res) > best[0]:
            best = (ok(res), labs, lines, chk, res)
    if best[1] is not labs:                                  # the last try was worse: train the best one again
        labs = best[1]
        suggested_path(model).write_text(json.dumps({str(k): v for k, v in labs.items()}, indent=1),
                                         encoding="utf-8")
        lines = train(model, video, L, labs, log, kind=kind)
        chk = _self_check(model, video, L, log)
    else:
        lines, chk = best[2], best[3]
    if notes:
        notes.append("    kept: " + "; ".join(f"step {sid} {v['start']:.1f}-{v['end']:.1f}s <- {v.get('change')}"
                                              for sid, v in sorted(labs.items())))
    if TIMING_J:
        notes.append("  timing on the training video: " + timing_score(best[1], best[4])[1])
    return lines + notes + chk


def _self_check(model, video, L, log, want_res=False):
    pass
    res, vd = infer(model, video, L, log)
    out = ["  check on the training video itself:"]
    for s in model["spec"]["steps"]:
        x = res.get(s["id"]) or {}
        tt = f"{x['start_sec']}-{x['end_sec']}s" if x.get("start_sec") is not None else ""
        out.append(f"    step {s['id']}: {x.get('status', '-'):<23} {tt}")
    # kept with the model: the model summary shows THIS check (the action model decides the verdict)
    try:
        mj = Path(model["places_file"]).parent / "model.json"
        mm = json.loads(mj.read_text(encoding="utf-8"))
        mm.update(action_self_check=vd["status"],
                  action_self_check_steps={str(k): {"status": x["status"], "start": x.get("start_sec"),
                                                    "end": x.get("end_sec")} for k, x in res.items()})
        mj.write_text(json.dumps(mm, indent=1, default=str), encoding="utf-8")
    except (OSError, ValueError, KeyError):
        pass
    return (out, res) if want_res else out


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  API - one namespace per module, for train_sequence.py
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
import types as _types
models = _types.SimpleNamespace(own_labels=own_labels, specific_parts=specific_parts)
pipeline = _types.SimpleNamespace(Pipeline=Pipeline, rule_spec=rule_spec, Cancelled=Cancelled)
trained = _types.SimpleNamespace(default_name=default_name, model_dir=model_dir, save=save, update=update, load=trained_load, action_model=action_model, sequence_lines=sequence_lines)
temporal = _types.SimpleNamespace(train_model=train_model)
completion = _types.SimpleNamespace(train=completion_train)
label_gui = _types.SimpleNamespace(auto_train=auto_train)
object_actions = _types.SimpleNamespace(infer=infer)
Cancelled = Cancelled
