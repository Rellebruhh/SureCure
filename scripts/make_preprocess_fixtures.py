"""Write PIL reference outputs for ImagePreprocessorTest.

Usage: python make_preprocess_fixtures.py <out dir>
Each case: <name>.in (raw RGB, stored orientation), <name>.out (expected RGB), index.txt line
"<name> <w> <h> <orientation> <kind>" with kind = letterbox (224x224 output) or orient (no resize).
"""
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

FILL = (124, 116, 104)
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)


def with_orientation(img, o):
    exif = Image.Exif()
    exif[0x0112] = o
    buf = io.BytesIO()
    img.save(buf, "PNG", exif=exif.tobytes())
    return ImageOps.exif_transpose(Image.open(io.BytesIO(buf.getvalue())))


def letterbox(img):
    w, h = img.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), FILL)
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas.resize((224, 224), Image.BILINEAR)


lines = []
for name, w, h, o in [("down_o6", 300, 135, 6), ("up_o1", 150, 61, 1), ("exact_o8", 100, 224, 8)]:
    raw = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    img = Image.fromarray(raw, "RGB")
    res = letterbox(with_orientation(img, o).convert("RGB"))
    (out / f"{name}.in").write_bytes(raw.tobytes())
    (out / f"{name}.out").write_bytes(np.asarray(res, dtype=np.uint8).tobytes())
    lines.append(f"{name} {w} {h} {o} letterbox")

for o in range(1, 9):
    raw = rng.integers(0, 256, (5, 7, 3), dtype=np.uint8)
    img = with_orientation(Image.fromarray(raw, "RGB"), o)
    name = f"orient_o{o}"
    (out / f"{name}.in").write_bytes(raw.tobytes())
    (out / f"{name}.out").write_bytes(np.asarray(img, dtype=np.uint8).tobytes())
    lines.append(f"{name} 7 5 {o} orient")

(out / "index.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
