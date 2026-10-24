"""Check the model package against its own golden files on desktop ONNX Runtime.

Usage: python verify_golden.py <pkg_outputs dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

pkg = Path(sys.argv[1])
meta = json.loads((pkg / "surecure_model/meta.json").read_text())
labels = json.loads((pkg / "surecure_model/labels.json").read_text())
exp = json.loads((pkg / "golden/expected.json").read_text())

sess = ort.InferenceSession(str(pkg / "surecure_model/surecure.onnx"))
print("inputs ", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
print("outputs", [(o.name, o.shape, o.type) for o in sess.get_outputs()])
assert labels == exp["labels"], "labels.json and expected.json disagree"

T = meta["temperature"]
conf_thr = meta["confidence_threshold"]
guard_thr = meta["guard"]["threshold"]
fill = tuple(meta["fill_rgb"])


def run(x):
    out = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {"image": x})))
    return out["logits"][0], float(out["energy"].reshape(-1)[0])


def decide(logits, energy):
    z = logits / T
    p = np.exp(z - z.max()); p /= p.sum()
    conf = float(p.max())
    accept = energy <= guard_thr and conf >= conf_thr
    return int(p.argmax()), conf, accept


def letterbox(path, resample):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = img.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), fill)
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    canvas = canvas.resize((224, 224), resample)
    a = np.asarray(canvas, dtype=np.float32) / 255.0
    return a.transpose(2, 0, 1)[None].copy()


worst = 0.0
fails = 0
for im in exp["images"]:
    x = np.fromfile(pkg / "golden" / im["tensor_file"], dtype="<f4").reshape(1, 3, 224, 224)
    logits, energy = run(x)
    d = float(np.abs(logits - np.array(im["expected_logits"])).max())
    worst = max(worst, d)
    e_formula = float(-T * np.log(np.exp(logits / T - (logits / T).max()).sum()) - T * (logits / T).max())
    top1, conf, accept = decide(logits, energy)
    ok = (d < exp["tolerance_runtime_logits"] and labels[top1] == im["expected_top1"]
          and accept == im["expected_accept"] and abs(energy - im["expected_energy"]) < 1e-3)
    fails += not ok
    print(f"tensor {im['index']:02d} {im['true_label']:<10} top1={labels[top1]:<10} conf={conf:.4f} "
          f"energy={energy:.4f} (formula {e_formula:.4f}) accept={accept} maxdiff={d:.2e} {'OK' if ok else 'FAIL'}")
print(f"tensor check: {10 - fails}/10 pass, worst logit diff {worst:.2e}")

for name, rs in [("bilinear", Image.BILINEAR)]:
    print(f"--- raw images, PIL letterbox + {name}")
    for im in exp["images"]:
        raw = pkg / "golden" / im["raw_file"]
        x = letterbox(raw, rs)
        ref = np.fromfile(pkg / "golden" / im["tensor_file"], dtype="<f4").reshape(1, 3, 224, 224)
        logits, energy = run(x)
        top1, conf, accept = decide(logits, energy)
        d = float(np.abs(logits - np.array(im["expected_logits"])).max())
        print(f"raw {im['index']:02d} size={Image.open(raw).size} px_maxdiff={np.abs(x - ref).max():.4f} "
              f"top1={labels[top1]:<10} logit_maxdiff={d:.4f} accept={accept} "
              f"{'OK' if labels[top1] == im['expected_top1'] else 'TOP1 MISMATCH'}")
