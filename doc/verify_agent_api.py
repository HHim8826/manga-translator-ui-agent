"""M0 API probe: can the configured endpoint do images + tool calling + multi-turn tool results?

Reads OPENAI_API_KEY / OPENAI_API_BASE / OPENAI_MODEL from the project's .env.
Never prints the key. Uses only synthetic images (no user content is sent).
Usage (from the project root):
    packaging/python/python.exe -X utf8 doc/verify_agent_api.py . <out_dir>
"""
import base64
import io
import json
import os
import re
import sys
import time
import traceback

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

ROOT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

env = {}
for line in open(os.path.join(ROOT, ".env"), encoding="utf-8"):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
KEY = re.split(r"[,\n;]", env["OPENAI_API_KEY"])[0].strip()
BASE = re.split(r"[,\n;]", env["OPENAI_API_BASE"])[0].strip()
MODEL = env.get("OPENAI_MODEL", "gpt-4o")
client = OpenAI(api_key=KEY, base_url=BASE, timeout=180, max_retries=0)


def scrub(text):
    text = str(text)
    if KEY:
        text = text.replace(KEY, "<key>")
    return text.replace(BASE, "<base>")


FONT = os.path.join(ROOT, "fonts", "msgothic.ttc")


def font(size):
    return ImageFont.truetype(FONT, size)


def to_data_url(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, fmt, quality=90) if fmt == "JPEG" else img.save(buf, fmt)
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def img_part(img, detail="high", fmt="PNG"):
    return {"type": "image_url", "image_url": {"url": to_data_url(img, fmt), "detail": detail}}


# --- synthetic images -------------------------------------------------------
text_img = Image.new("RGB", (800, 400), "white")
d = ImageDraw.Draw(text_img)
d.text((40, 60), "検証コード：4721", fill="black", font=font(48))
d.text((40, 180), "スキル　即死パンチ", fill="black", font=font(56))
text_img.save(os.path.join(OUT, "probe_text.png"))

SOM_BOXES = {1: ((60, 60, 420, 300), "りんご"), 2: ((560, 80, 940, 320), "電車"),
             3: ((80, 400, 440, 640), "ねこ"), 4: ((580, 420, 930, 650), "雨")}
som_img = Image.new("RGB", (1000, 700), "white")
d = ImageDraw.Draw(som_img)
for n, ((x1, y1, x2, y2), word) in SOM_BOXES.items():
    d.rectangle((x1, y1, x2, y2), outline="black", width=3)
    d.text(((x1 + x2) // 2 - 60, (y1 + y2) // 2 - 30), word, fill="black", font=font(56))
    d.rectangle((x1 - 2, y1 - 30, x1 + 40, y1 + 6), fill="red")
    d.text((x1 + 8, y1 - 28), str(n), fill="white", font=font(30))
som_img.save(os.path.join(OUT, "probe_som.png"))

page_like = Image.effect_noise((1351, 1920), 40).convert("RGB")  # same size as the user's pages

results = []


def run(name, fn):
    t0 = time.time()
    try:
        ok, detail, usage = fn()
    except Exception as exc:  # record, keep going
        ok, detail, usage = False, scrub(f"{type(exc).__name__}: {exc}")[:600], None
    dt = round(time.time() - t0, 1)
    results.append({"test": name, "ok": ok, "seconds": dt, "detail": detail, "usage": usage})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} ({dt}s): {str(detail)[:300]}", flush=True)


def usage_of(resp):
    u = getattr(resp, "usage", None)
    if not u:
        return None
    return {k: getattr(u, k, None) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}


def chat(messages, **kw):
    return client.chat.completions.create(model=MODEL, messages=messages, **kw)


ZOOM_TOOL = {
    "type": "function",
    "function": {
        "name": "zoom",
        "description": "Return a magnified crop of a numbered region on the current page.",
        "parameters": {
            "type": "object",
            "properties": {
                "region_id": {"type": "integer", "description": "Region number shown on the page"},
                "scale": {"type": "integer", "minimum": 1, "maximum": 6},
                "reason": {"type": "string", "description": "One sentence: why you need this"},
            },
            "required": ["region_id", "scale", "reason"],
            "additionalProperties": False,
        },
    },
}
SET_TEXT_TOOL = {
    "type": "function",
    "function": {
        "name": "set_text",
        "description": "Record the recognized source text of a region.",
        "parameters": {
            "type": "object",
            "properties": {
                "region_id": {"type": "integer"},
                "source_text": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["region_id", "source_text", "reason"],
            "additionalProperties": False,
        },
    },
}


# 1. plain text
def t_text():
    r = chat([{"role": "user", "content": "Reply with exactly: PONG"}])
    out = (r.choices[0].message.content or "").strip()
    return "PONG" in out, {"model_reported": getattr(r, "model", None), "reply": out}, usage_of(r)


# 2. image reading
def t_image():
    r = chat([{"role": "user", "content": [
        {"type": "text", "text": "Transcribe all text in this image exactly, one line per line. No commentary."},
        img_part(text_img)]}])
    out = r.choices[0].message.content or ""
    ok = "4721" in out and "即死パンチ" in out
    return ok, out.strip(), usage_of(r)


# 3. numbered-region reference (Set-of-Marks) + raw coordinate accuracy
def t_som():
    r = chat([{"role": "user", "content": [
        {"type": "text", "text": (
            "Each box has a red number tag. Answer in JSON only: "
            '{"neko": <number of the box containing ねこ>, "densha": <number of the box containing 電車>, '
            '"ame_bbox": [x1, y1, x2, y2] pixel coordinates of the box containing 雨 in this 1000x700 image}')},
        img_part(som_img)]}], response_format={"type": "json_object"})
    out = r.choices[0].message.content or ""
    data = json.loads(re.search(r"\{.*\}", out, re.S).group(0))
    ids_ok = data.get("neko") == 3 and data.get("densha") == 2
    bx = data.get("ame_bbox") or [0, 0, 0, 0]
    gx = SOM_BOXES[4][0]
    ix = max(0, min(bx[2], gx[2]) - max(bx[0], gx[0])) * max(0, min(bx[3], gx[3]) - max(bx[1], gx[1]))
    union = (bx[2] - bx[0]) * (bx[3] - bx[1]) + (gx[2] - gx[0]) * (gx[3] - gx[1]) - ix
    iou = round(ix / union, 3) if union > 0 else 0
    err = [abs(a - b) for a, b in zip(bx, gx)]
    return ids_ok, {"answer": data, "bbox_iou": iou, "bbox_abs_err_px": err}, usage_of(r)


# 4-6. tool calling, multi-turn, parallel
state = {}


def t_tool_call():
    msgs = [
        {"role": "system", "content": "You are a manga translation agent. Use tools; do not guess small text."},
        {"role": "user", "content": [
            {"type": "text", "text": "Region 3 on this page is too small to read. Zoom into it first."},
            img_part(som_img)]},
    ]
    r = chat(msgs, tools=[ZOOM_TOOL, SET_TEXT_TOOL], tool_choice="auto")
    m = r.choices[0].message
    calls = m.tool_calls or []
    state["msgs"] = msgs + [m.model_dump(exclude_none=True)]
    state["calls"] = calls
    if not calls:
        return False, {"content": m.content}, usage_of(r)
    args = json.loads(calls[0].function.arguments)
    ok = calls[0].function.name == "zoom" and args.get("region_id") == 3 and bool(args.get("reason"))
    return ok, {"tool": calls[0].function.name, "args": args, "finish_reason": r.choices[0].finish_reason}, usage_of(r)


def _crop_region3(scale=3):
    (x1, y1, x2, y2), _ = SOM_BOXES[3]
    c = som_img.crop((x1, y1, x2, y2))
    return c.resize((c.width * scale // 2, c.height * scale // 2))


def t_tool_result_text_then_image():
    """Tool result as text, image delivered in a following user message (works on any chat API)."""
    if not state.get("calls"):
        return False, "skipped: no tool call from previous step", None
    call = state["calls"][0]
    msgs = state["msgs"] + [
        {"role": "tool", "tool_call_id": call.id, "content": "Zoomed crop of region 3 is attached in the next message."},
        {"role": "user", "content": [{"type": "text", "text": "[zoom result for region 3]"}, img_part(_crop_region3())]},
    ]
    r = chat(msgs, tools=[ZOOM_TOOL, SET_TEXT_TOOL], tool_choice="auto")
    m = r.choices[0].message
    calls = m.tool_calls or []
    if calls and calls[0].function.name == "set_text":
        args = json.loads(calls[0].function.arguments)
        return "ねこ" in args.get("source_text", ""), {"tool": "set_text", "args": args}, usage_of(r)
    return "ねこ" in (m.content or ""), {"content": m.content, "tools": [c.function.name for c in calls]}, usage_of(r)


def t_image_inside_tool_message():
    """Some APIs accept image parts directly in the tool message; check if this endpoint does."""
    if not state.get("calls"):
        return False, "skipped: no tool call from previous step", None
    call = state["calls"][0]
    msgs = state["msgs"] + [
        {"role": "tool", "tool_call_id": call.id, "content": [
            {"type": "text", "text": "Zoomed crop of region 3:"}, img_part(_crop_region3())]},
    ]
    r = chat(msgs, tools=[ZOOM_TOOL, SET_TEXT_TOOL], tool_choice="auto")
    m = r.choices[0].message
    text = (m.content or "") + json.dumps([c.function.arguments for c in (m.tool_calls or [])], ensure_ascii=False)
    return "ねこ" in text, {"reply_or_args": text[:300]}, usage_of(r)


def t_parallel_and_required():
    r = chat([
        {"role": "user", "content": [
            {"type": "text", "text": "Record the text of regions 1, 2, 3 and 4 with set_text. Call the tool once per region."},
            img_part(som_img)]},
    ], tools=[SET_TEXT_TOOL], tool_choice="required")
    calls = r.choices[0].message.tool_calls or []
    got = {}
    for c in calls:
        a = json.loads(c.function.arguments)
        got[a.get("region_id")] = a.get("source_text")
    expect = {n: w for n, (_b, w) in SOM_BOXES.items()}
    ok = all(expect[k] in (got.get(k) or "") for k in expect)
    return ok, {"calls": len(calls), "got": got}, usage_of(r)


# 7. image token cost
def t_tokens():
    base = chat([{"role": "user", "content": "Reply OK."}])
    hi = chat([{"role": "user", "content": [{"type": "text", "text": "Reply OK."}, img_part(page_like, "high", "JPEG")]}])
    lo = chat([{"role": "user", "content": [{"type": "text", "text": "Reply OK."}, img_part(page_like, "low", "JPEG")]}])
    b, h, l = (usage_of(x) or {} for x in (base, hi, lo))
    info = {
        "baseline_prompt": b.get("prompt_tokens"),
        "page_1351x1920_high": (h.get("prompt_tokens") or 0) - (b.get("prompt_tokens") or 0),
        "page_1351x1920_low": (l.get("prompt_tokens") or 0) - (b.get("prompt_tokens") or 0),
    }
    return bool(b and h), info, None


# 8. Responses API (optional alternative)
def t_responses_api():
    r = client.responses.create(model=MODEL, input=[{"role": "user", "content": [
        {"type": "input_text", "text": "Which number tag is on the box containing ねこ? Call set_text for that region."},
        {"type": "input_image", "image_url": to_data_url(som_img)}]}],
        tools=[{"type": "function", "name": "set_text", "description": SET_TEXT_TOOL["function"]["description"],
                "parameters": SET_TEXT_TOOL["function"]["parameters"], "strict": True}])
    calls = [o for o in r.output if getattr(o, "type", "") == "function_call"]
    if not calls:
        return False, {"output_text": getattr(r, "output_text", "")[:200]}, None
    args = json.loads(calls[0].arguments)
    return args.get("region_id") == 3, {"args": args}, None


for name, fn in [
    ("text", t_text),
    ("image_input", t_image),
    ("numbered_regions_and_bbox", t_som),
    ("tool_call", t_tool_call),
    ("tool_result_then_image_in_user_msg", t_tool_result_text_then_image),
    ("image_inside_tool_message", t_image_inside_tool_message),
    ("parallel_calls_tool_choice_required", t_parallel_and_required),
    ("image_token_cost", t_tokens),
    ("responses_api", t_responses_api),
]:
    run(name, fn)

with open(os.path.join(OUT, "api_probe_results.json"), "w", encoding="utf-8") as f:
    json.dump({"model": MODEL, "results": results}, f, ensure_ascii=False, indent=2, default=str)
print("saved", os.path.join(OUT, "api_probe_results.json"))
