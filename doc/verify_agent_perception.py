"""M0.5: real-page perception test against the configured endpoint.

For each page:
  A. free   - AI finds every text region by itself (bbox + transcription)
  B. som    - reference boxes drawn with number tags, AI transcribes by number
  C. zoom   - each reference box cropped and magnified, AI transcribes the crops
The reference is the existing *_translations.json (local OCR output, may itself be wrong).
Usage (from the project root):
    packaging/python/python.exe -X utf8 doc/verify_agent_perception.py . <chapter_dir> <out_dir> 003 006
"""
import base64
import difflib
import io
import json
import os
import re
import sys
import time
import unicodedata

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

ROOT, CHAPTER, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PAGES = sys.argv[4:]
os.makedirs(OUT, exist_ok=True)

env = {}
for line in open(os.path.join(ROOT, ".env"), encoding="utf-8"):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
KEY = re.split(r"[,\n;]", env["OPENAI_API_KEY"])[0].strip()
BASE = re.split(r"[,\n;]", env["OPENAI_API_BASE"])[0].strip()
MODEL = env["OPENAI_MODEL"]
client = OpenAI(api_key=KEY, base_url=BASE, timeout=300, max_retries=1)
FONT = ImageFont.truetype(os.path.join(ROOT, "fonts", "msgothic.ttc"), 28)


def data_url(img, fmt="JPEG"):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, fmt, quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def ask_json(content, label):
    t0 = time.time()
    r = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"})
    out = r.choices[0].message.content or "{}"
    m = re.search(r"\{.*\}", out, re.S)
    data = json.loads(m.group(0)) if m else {}
    usage = {"prompt": r.usage.prompt_tokens, "completion": r.usage.completion_tokens,
             "seconds": round(time.time() - t0, 1)}
    print(f"  {label}: {usage}", flush=True)
    return data, usage


def norm(s):
    s = unicodedata.normalize("NFKC", s or "")
    return re.sub(r"[\s・…\.。、，,!！?？「」『』()（）ー\-~〜]+", "", s)


def sim(a, b):
    a, b = norm(a), norm(b)
    if not a and not b:
        return 1.0
    return round(difflib.SequenceMatcher(None, a, b).ratio(), 3)


def bbox_of(region):
    pts = [p for quad in region["lines"] for p in quad]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    return ix / ua if ua > 0 else 0.0


def center_in(b, box):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


summary = {"model": MODEL, "pages": {}}
for page in PAGES:
    print(f"== page {page}", flush=True)
    img = Image.open(os.path.join(CHAPTER, f"{page}.jpg")).convert("RGB")
    W, H = img.size
    jd = json.load(open(os.path.join(CHAPTER, "manga_translator_work", "json", f"{page}_translations.json"), encoding="utf-8"))
    regions = list(jd.values())[0]["regions"]
    refs = [{"id": i + 1, "bbox": bbox_of(r), "text": r.get("text", ""), "direction": r.get("direction")}
            for i, r in enumerate(regions)]

    # A. free detection
    free, u_free = ask_json([
        {"type": "text", "text": (
            f"This is a Japanese manga page, {W}x{H} pixels. Find EVERY piece of text: speech bubbles, "
            "narration boxes, handwritten notes, signs, UI/status screens, and sound effects. "
            'Return JSON {"regions": [{"bbox": [x1,y1,x2,y2], "direction": "vertical|horizontal", '
            '"category": "dialogue|narration|note|sfx|sign|ui", "text": "exact Japanese transcription"}]} '
            "with pixel coordinates in this image. Bbox must tightly enclose the text itself, not the whole bubble. "
            "Transcribe exactly; do not translate or correct the author's wording.")},
        {"type": "image_url", "image_url": {"url": data_url(img), "detail": "high"}}], "free")
    free_regions = free.get("regions", [])

    # B. numbered reference boxes
    som = img.copy()
    d = ImageDraw.Draw(som)
    for r in refs:
        x1, y1, x2, y2 = r["bbox"]
        d.rectangle((x1 - 4, y1 - 4, x2 + 4, y2 + 4), outline=(255, 0, 0), width=3)
        tx, ty = max(0, x1 - 4), max(0, y1 - 36)
        d.rectangle((tx, ty, tx + 22 + 16 * (r["id"] >= 10), ty + 32), fill=(255, 0, 0))
        d.text((tx + 4, ty + 2), str(r["id"]), fill="white", font=FONT)
    som.save(os.path.join(OUT, f"{page}_som.jpg"), quality=90)
    som_ans, u_som = ask_json([
        {"type": "text", "text": (
            "Red boxes with number tags mark text regions on this manga page. For every number, transcribe the "
            'Japanese text inside that box exactly. Return JSON {"1": "...", "2": "...", ...}. '
            "Use an empty string if the box has no readable text.")},
        {"type": "image_url", "image_url": {"url": data_url(som), "detail": "high"}}], "som")

    # C. magnified crops
    parts = [{"type": "text", "text": (
        "Each image below is a magnified crop of one text region from a Japanese manga page, labeled with its "
        'number. Transcribe each exactly. Return JSON {"1": "...", "2": "...", ...}.')}]
    for r in refs:
        x1, y1, x2, y2 = r["bbox"]
        pad = 10
        crop = img.crop((max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)))
        scale = max(1.0, min(3.0, 320 / max(1, min(crop.size))))
        crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS)
        crop.save(os.path.join(OUT, f"{page}_r{r['id']:02d}.png"))
        parts.append({"type": "text", "text": f"Region {r['id']}:"})
        parts.append({"type": "image_url", "image_url": {"url": data_url(crop), "detail": "high"}})
    zoom_ans, u_zoom = ask_json(parts, "zoom")

    rows = []
    used_free = set()
    for r in refs:
        best, best_iou = None, 0.0
        for j, f in enumerate(free_regions):
            b = f.get("bbox") or [0, 0, 0, 0]
            v = iou(b, r["bbox"])
            if v > best_iou or (best is None and center_in(b, r["bbox"])):
                best, best_iou = j, v
        f = free_regions[best] if best is not None and (best_iou > 0.1 or center_in(free_regions[best].get("bbox", [0] * 4), r["bbox"])) else None
        if f is not None:
            used_free.add(best)
        rows.append({
            "id": r["id"], "ref_bbox": r["bbox"], "ref": r["text"],
            "free": f.get("text") if f else None, "free_bbox": f.get("bbox") if f else None,
            "free_iou": round(best_iou, 3) if f else 0.0,
            "som": som_ans.get(str(r["id"]), ""), "zoom": zoom_ans.get(str(r["id"]), ""),
            "sim_som_zoom": sim(som_ans.get(str(r["id"]), ""), zoom_ans.get(str(r["id"]), "")),
            "sim_ref_zoom": sim(r["text"], zoom_ans.get(str(r["id"]), "")),
        })
    extra = [f for j, f in enumerate(free_regions) if j not in used_free]
    summary["pages"][page] = {"size": [W, H], "rows": rows, "free_extra": extra,
                              "usage": {"free": u_free, "som": u_som, "zoom": u_zoom}}
    found = sum(1 for x in rows if x["free"] is not None)
    ious = [x["free_iou"] for x in rows if x["free"] is not None]
    print(f"  free found {found}/{len(rows)} ref regions, +{len(extra)} extra; "
          f"median IoU {sorted(ious)[len(ious)//2] if ious else 0}", flush=True)

with open(os.path.join(OUT, "m05_results.json"), "w", encoding="utf-8") as fh:
    json.dump(summary, fh, ensure_ascii=False, indent=1)
print("saved", os.path.join(OUT, "m05_results.json"))
