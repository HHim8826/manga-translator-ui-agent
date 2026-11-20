"""Fonts for the agent workflow (design doc, "字體").

``fix_missing_glyphs``: when a region's font has no glyph for a character, Qt draws that one
character from some other font, so the line mixes two designs (``A-OTF Shin Go Pr6N`` has no
值, 嗯, 啥 ... for Traditional Chinese). After the agents, every translated region is checked
against its font with the characters the renderer will draw (after the vertical-text
replacements). A region with missing glyphs gets the installed font that looks most like its
font, among the fonts that have all of them: the same characters are drawn in both fonts and
compared (shape and stroke weight), and fonts made for the target language come first.

``choose_fonts``: the translation agent sees a sheet of installed fonts that have every common
character of the target language and may give a region one of them when its original lettering
is clearly not normal dialogue (a narration box in a serif face, handwriting, a sign).

Every change is checked by rendering the region alone with the real renderer, once with the font
it had and once with the new one: the text has to stay in the same place at about the same size,
or the next candidate is tried (an agent's choice that fails goes back to the user's font).
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import sys
import threading
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from .store import write_json_atomic

logger = logging.getLogger("manga_translator.agent")

FONT_EXTS = (".ttf", ".otf", ".ttc", ".otc")
STYLE_SEP = "::"
"""The renderer's "Family::Style" form (``text_render._fonts.FONT_STYLE_SEPARATOR``)."""
RENDER_DEFAULT = "Microsoft YaHei UI"
"""What the renderer uses when a font is not installed (``text_render.DEFAULT_FONT_FAMILY``)."""
REGULAR_STYLES = {"regular", "normal", "book", "roman", "r", "plain", "standard"}
TARGET_LOCALES = {"CHT": "zh-TW", "CHS": "zh-CN", "JPN": "ja", "KOR": "ko"}
_LANGID_LOCALES = {0x0404: "zh-TW", 0x0C04: "zh-HK", 0x1404: "zh-HK", 0x0804: "zh-CN", 0x1004: "zh-CN",
                   0x0411: "ja", 0x0412: "ko"}
_CODEPAGE_LOCALES = {17: "ja", 18: "zh-CN", 19: "ko", 20: "zh-TW", 21: "ko"}
LOCALE_BONUS = 0.1
"""Added to the similarity of a font made for the target language, taken off one made for another."""
LOCALE_PREFS = {"CHT": {"zh-TW": LOCALE_BONUS, "zh-HK": LOCALE_BONUS / 2}, "CHS": {"zh-CN": LOCALE_BONUS},
                "JPN": {"ja": LOCALE_BONUS}, "KOR": {"ko": LOCALE_BONUS}}
SAMPLES = {"CHT": "「你好！這是什麼？」謝謝，我會記得的。",
           "CHS": "「你好！这是什么？」谢谢，我会记得的。",
           "JPN": "「こんにちは！これは何？」ありがとう、忘れないよ。",
           "KOR": "「안녕하세요! 이게 뭐예요?」 고마워요, 잊지 않을게요.",
           "": "\"Hello! What is this?\" Thanks, I won't forget it."}
CJK_PROBE = "人大中日月山口手田子女心水火木金土上下小本文目石竹米糸車雨青音書間開題聞語話時明東京"
LATIN_PROBE = "AaBbDdEeGgHhKkMmNnPpQqRrSsWwYy2357?"
SHEET_SIMILAR = 0.75
"""Fonts at least this similar to one already on the sheet are left off it."""
SHEET_LIMIT = 20
SHEET_PER_FAMILY = 2
"""A regular and a clearly heavier (or lighter) weight at most; hairline weights are left out."""
SHEET_MIN_WEIGHT = 300
SAME_DESIGN = 0.85
"""Glyph shapes this similar (weight aside) are one design, like Noto Sans TC and Noto Sans HK."""
CANDIDATES = 6
"""Similar fonts tried (each with a layout check) for a region with missing glyphs."""
PUNCT_MIN_SCORE = 0.3
"""When only punctuation is missing (a Ming face without vertical forms), a font must score at least
this in ``similar`` to replace it; otherwise the region keeps its font and borrows the punctuation."""


# --- languages ----------------------------------------------------------------
def _decode_rows(encoding: str, rows: range, cols: Sequence[int], last: Optional[int] = None) -> set:
    out = set()
    for hi in rows:
        for lo in cols:
            if last is not None and (hi << 8 | lo) > last:
                return out
            try:
                out.add(bytes([hi, lo]).decode(encoding))
            except UnicodeDecodeError:
                pass
    return out


@lru_cache(maxsize=8)
def language_chars(target_lang: str) -> FrozenSet[str]:
    """Common characters of the target language: Big5 level 1 for Traditional Chinese, GB2312
    level 1 for Simplified, JIS level 1 plus kana for Japanese, KS X 1001 Hangul for Korean."""
    target = (target_lang or "").upper()
    high = list(range(0xA1, 0xFF))
    if target == "CHT":
        chars = _decode_rows("big5", range(0xA4, 0xC7), list(range(0x40, 0x7F)) + high, last=0xC67E)
    elif target == "CHS":
        chars = _decode_rows("gb2312", range(0xB0, 0xD8), high)
    elif target == "JPN":
        chars = _decode_rows("euc_jp", range(0xB0, 0xD0), high)
        chars |= {chr(c) for c in range(0x3041, 0x3097)} | {chr(c) for c in range(0x30A1, 0x30FB)}
    elif target == "KOR":
        chars = _decode_rows("euc_kr", range(0xB0, 0xC9), high)
    else:
        chars = {chr(c) for c in range(0x21, 0x7F)}
    return frozenset(chars)


def probe_kind(target_lang: str) -> str:
    return "cjk" if (target_lang or "").upper() in TARGET_LOCALES else "latin"


def _significant(ch: str) -> bool:
    """Characters whose design shows: letters, digits, punctuation and CJK/full-width forms.
    Symbols (♥ ★ ♪) are drawn from another font anyway and never switch a region's font."""
    if unicodedata.category(ch)[0] in "LNP":
        return True
    cp = ord(ch)
    return 0x3000 <= cp <= 0x303F or 0xFE10 <= cp <= 0xFE4F or 0xFF00 <= cp <= 0xFFEF


_BREAKS = re.compile(r"\[BR\]|【BR】|<br\s*/?>", re.IGNORECASE)


@lru_cache(maxsize=1)
def _replacements():
    try:
        from ..rendering.text_replacements import load_replacements
        return load_replacements()
    except Exception:
        logger.warning("fonts: could not load the text replacement rules", exc_info=True)
        return None


def rendered_chars(text: str, direction: Any = "auto") -> str:
    """The significant characters the renderer will draw for ``text``, after the text replacements
    of the direction (vertical text turns 。 into ︒ ...); both directions when it is not fixed."""
    reps = _replacements()
    head = str(direction or "auto")[:1].lower()
    directions = (1,) if head == "v" else (0,) if head == "h" else (0, 1)
    out = set()
    for direction_code in directions:
        drawn = text
        if reps is not None:
            try:
                from ..rendering.text_replacements import apply_replacements
                drawn = apply_replacements(text, direction_code, reps)
            except Exception:
                drawn = text
        out |= {c for c in _BREAKS.sub("", drawn) if _significant(c)}
    return "".join(sorted(out))


# --- installed fonts ------------------------------------------------------------
@dataclass(eq=False)
class FontFace:
    path: str
    index: int
    family: str
    style: str
    weight: int
    names: Tuple[str, ...]
    locales: FrozenSet[str]
    bits: Any = field(repr=False, default=None)
    """Coverage of the Basic Multilingual Plane, ``np.packbits`` of 65536 flags."""
    extra: FrozenSet[int] = field(repr=False, default=frozenset())
    axes: Tuple[str, ...] = ()
    instance: Optional[Dict[str, float]] = None
    """Coordinates of a variable font's named instance."""
    value: str = ""
    """How the renderer addresses this face ("Family" or "Family::Style"); empty when it cannot."""
    duplicate: bool = False

    def has(self, ch: str) -> bool:
        cp = ord(ch)
        if cp < 0x10000:
            return self.bits is not None and bool(self.bits[cp >> 3] & (0x80 >> (cp & 7)))
        return cp in self.extra

    def missing(self, chars) -> str:
        return "".join(sorted(c for c in set(chars) if not self.has(c)))

    def coverage(self, chars) -> float:
        chars = list(chars)
        return sum(1 for c in chars if self.has(c)) / len(chars) if chars else 1.0

    @property
    def label(self) -> str:
        return self.family if STYLE_SEP not in self.value else f"{self.family} {self.style}"


def font_dirs() -> List[str]:
    from ..utils import BASE_PATH

    dirs = [os.path.join(BASE_PATH, "fonts")]
    if sys.platform == "win32":
        dirs.append(os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))
        if os.environ.get("LOCALAPPDATA"):
            dirs.append(os.path.join(os.environ["LOCALAPPDATA"], "Microsoft", "Windows", "Fonts"))
    elif sys.platform == "darwin":
        dirs += ["/System/Library/Fonts", "/Library/Fonts", os.path.expanduser("~/Library/Fonts")]
    else:
        dirs += ["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.local/share/fonts"),
                 os.path.expanduser("~/.fonts")]
    return [d for d in dirs if os.path.isdir(d)]


def _name_records(font) -> Dict[int, List[Tuple[int, str]]]:
    out: Dict[int, List[Tuple[int, str]]] = {}
    for rec in font["name"].names:
        if rec.nameID in (1, 2, 16, 17):
            try:
                value = rec.toUnicode().strip()
            except Exception:
                continue
            if value:
                out.setdefault(rec.nameID, []).append((rec.langID, value))
    return out


def _english(records, ids) -> str:
    for nid in ids:
        for lang, value in records.get(nid, []):
            if lang == 0x409:
                return value
    for nid in ids:
        if records.get(nid):
            return records[nid][0][1]
    return ""


def _locales(font, records, family: str) -> FrozenSet[str]:
    out = {_LANGID_LOCALES[lang] for nid in (1, 16) for lang, _ in records.get(nid, []) if lang in _LANGID_LOCALES}
    if not out:
        tokens = set(re.split(r"[\s_\-]+", family.upper()))
        if "TC" in tokens or family.upper().endswith("B5"):
            out.add("zh-TW")
        if "HK" in tokens:
            out.add("zh-HK")
        if tokens & {"SC", "CN", "GB", "GBK"} or any(t.endswith("GBK") for t in tokens):
            out.add("zh-CN")
        if "JP" in tokens:
            out.add("ja")
        if "KR" in tokens:
            out.add("ko")
    if not out and "OS/2" in font:
        bits = int(getattr(font["OS/2"], "ulCodePageRange1", 0) or 0)
        found = {loc for bit, loc in _CODEPAGE_LOCALES.items() if bits & (1 << bit)}
        if len(found) == 1:
            out = found
    return frozenset(out)


def _faces_of(font, path: str, index: int) -> List[FontFace]:
    cmap = font.getBestCmap() or {}
    if not cmap:
        return []
    records = _name_records(font)
    family = _english(records, (16, 1))
    if not family:
        return []
    style = _english(records, (17, 2)) or "Regular"
    names = tuple(dict.fromkeys([family] + [v for nid in (16, 1) for _, v in records.get(nid, [])]))
    weight = int(getattr(font["OS/2"], "usWeightClass", 400) or 400) if "OS/2" in font else 400
    flags = np.zeros(0x10000, dtype=bool)
    flags[np.fromiter((c for c in cmap if c < 0x10000), dtype=np.int64)] = True
    base = dict(path=path, index=index, family=family, names=names, locales=_locales(font, records, family),
                bits=np.packbits(flags), extra=frozenset(c for c in cmap if c >= 0x10000))
    if "fvar" in font:
        fvar = font["fvar"]
        axes = tuple(a.axisTag for a in fvar.axes)
        faces = []
        for inst in fvar.instances:
            sub = font["name"].getDebugName(inst.subfamilyNameID) or ""
            if sub:
                coords = dict(inst.coordinates)
                faces.append(FontFace(style=sub, weight=int(coords.get("wght", weight)), axes=axes,
                                      instance=coords, **base))
        if faces:
            return faces
    return [FontFace(style=style, weight=weight, **base)]


def scan_fonts(dirs: Optional[Sequence[str]] = None) -> List[FontFace]:
    """Every face of the font files in ``dirs`` (the renderer's font folders by default)."""
    from fontTools.ttLib import TTCollection, TTFont

    quiet = logging.getLogger("fontTools")
    level = quiet.level
    quiet.setLevel(logging.ERROR)
    faces: List[FontFace] = []
    try:
        for folder in dirs if dirs is not None else font_dirs():
            for root, subdirs, files in os.walk(folder):
                subdirs.sort()
                for name in sorted(files):
                    if not name.lower().endswith(FONT_EXTS):
                        continue
                    path = os.path.join(root, name)
                    try:
                        fonts = (TTCollection(path, lazy=True).fonts if name.lower().endswith((".ttc", ".otc"))
                                 else [TTFont(path, lazy=True)])
                    except Exception:
                        continue
                    for index, font in enumerate(fonts):
                        try:
                            faces.extend(_faces_of(font, path, index))
                        except Exception as exc:
                            logger.debug(f"fonts: skipping {path}#{index}: {exc}")
    finally:
        quiet.setLevel(level)
    return faces


def qt_resolver() -> Callable[[str], str]:
    """The renderer's own lookup: returns the spelling it uses for a "Family[::Style]", or "" when
    it would fall back to another font. Call from the thread that renders."""
    from PyQt6.QtGui import QFontDatabase

    from ..rendering.text_render import _fonts

    _fonts._ensure_qt_runtime()
    _fonts._register_project_fonts()
    _fonts._register_system_fonts()

    def resolve(value: str) -> str:
        family, style = _fonts._split_font_value(value)
        matched = _fonts._match_family(family)
        if not matched:
            return ""
        if not style:
            return matched
        styles = {s.casefold(): s for s in QFontDatabase.styles(matched)}
        found = styles.get(style.casefold())
        return f"{matched}{STYLE_SEP}{found}" if found else ""

    return resolve


def _split(value: str) -> Tuple[str, str]:
    family, sep, style = str(value or "").rpartition(STYLE_SEP)
    if sep and family and style:
        return family.strip(), style.strip()
    return str(value or "").strip(), ""


class FontCatalog:
    def __init__(self, faces: List[FontFace], resolve: Optional[Callable[[str], str]] = None):
        self.faces = faces
        self._by_name: Dict[str, List[FontFace]] = {}
        groups: Dict[str, List[FontFace]] = {}
        for face in faces:
            groups.setdefault(face.family.casefold(), []).append(face)
        for group in groups.values():
            seen: Dict[str, FontFace] = {}
            unique = []
            for face in group:  # the same family and style in two files: the first one counts
                key = face.style.casefold()
                if key in seen:
                    face.duplicate = True
                    continue
                seen[key] = face
                unique.append(face)
            regular = self._regular(unique)
            for face in unique:
                face.value = face.family if face is regular else f"{face.family}{STYLE_SEP}{face.style}"
                for name in face.names:
                    self._by_name.setdefault(name.casefold(), []).append(face)
        self.addressable = {id(f) for f in faces if f.value}
        families = {f.family.casefold() for f in faces}
        for face in faces:  # "Microsoft YaHei UI" has the glyphs of "Microsoft YaHei"
            if face.family.endswith(" UI") and face.family[:-3].casefold() in families:
                self.addressable.discard(id(face))
        if resolve is not None:
            for face in faces:
                if face.value:
                    resolved = resolve(face.value)
                    if resolved:
                        face.value = resolved
                    else:
                        self.addressable.discard(id(face))
        self._features: Dict[Tuple[int, str], Dict[str, Tuple[np.ndarray, float]]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _regular(group: List[FontFace]) -> FontFace:
        return min(group, key=lambda f: (f.style.casefold() not in REGULAR_STYLES, abs(f.weight - 400)))

    def usable(self, face: FontFace) -> bool:
        """A font the program may give a region. Variable fonts are left out: their glyphs are drawn
        from overlapping outlines, and the renderer's even-odd fill leaves holes where they cross."""
        return id(face) in self.addressable and not face.duplicate and face.instance is None

    def find(self, value: str) -> Optional[FontFace]:
        """The face the renderer uses for a font_family value (family names in any language)."""
        family, style = _split(value)
        group = [f for f in self._by_name.get(family.casefold(), []) if not f.duplicate]
        if not group:
            return None
        if style:
            for face in group:
                if face.style.casefold() == style.casefold():
                    return face
        return self._regular(group)

    # --- looks ----------------------------------------------------------------
    def features(self, face: FontFace, kind: str) -> Dict[str, Tuple[np.ndarray, float]]:
        """Each probe character drawn at 64 px: the ink cropped and scaled to 32x32 (shape) and the
        share of the em square it covers (weight)."""
        key = (id(face), kind)
        with self._lock:
            cached = self._features.get(key)
        if cached is not None:
            return cached
        out: Dict[str, Tuple[np.ndarray, float]] = {}
        try:
            from PIL import Image, ImageDraw, ImageFilter, ImageFont

            font = ImageFont.truetype(face.path, 64, index=face.index)
            if face.instance:
                font.set_variation_by_axes([float(face.instance.get(a, 0)) for a in face.axes])
            for ch in CJK_PROBE if kind == "cjk" else LATIN_PROBE:
                if not face.has(ch):
                    continue
                img = Image.new("L", (128, 128), 0)
                ImageDraw.Draw(img).text((32, 32), ch, font=font, fill=255)
                arr = np.asarray(img) > 127
                ys, xs = np.nonzero(arr)
                if len(xs) == 0:
                    continue
                crop = img.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
                side = max(crop.size)
                square = Image.new("L", (side, side), 0)
                square.paste(crop, ((side - crop.size[0]) // 2, (side - crop.size[1]) // 2))
                square = square.resize((32, 32), Image.LANCZOS).filter(ImageFilter.GaussianBlur(1))
                out[ch] = (np.asarray(square, dtype=np.float32) / 255.0, float(arr.sum()) / (64 * 64))
        except Exception as exc:
            logger.debug(f"fonts: cannot draw {face.path}#{face.index}: {exc}")
            out = {}
        with self._lock:
            self._features[key] = out
        return out

    def compare(self, a: FontFace, b: FontFace, kind: str) -> Optional[Tuple[float, float]]:
        """(shape, weight): 1 for the same glyph shapes, and the relative difference in ink."""
        fa, fb = self.features(a, kind), self.features(b, kind)
        common = [c for c in fa if c in fb]
        if len(common) < 8:
            return None
        shape = 1.0 - float(np.mean([np.abs(fa[c][0] - fb[c][0]).mean() for c in common])) / 0.25
        ink_a = float(np.mean([fa[c][1] for c in common]))
        ink_b = float(np.mean([fb[c][1] for c in common]))
        return shape, abs(ink_a - ink_b) / max(ink_a, 1e-6)

    def similarity(self, a: FontFace, b: FontFace, kind: str) -> Optional[float]:
        """1 for the same look; shape difference and relative weight difference count against."""
        found = self.compare(a, b, kind)
        return None if found is None else found[0] - found[1]

    def similar(self, ref: FontFace, chars: str, target_lang: str, limit: int = CANDIDATES
                ) -> List[Tuple[float, FontFace]]:
        """Usable faces that have every character in ``chars``, most like ``ref`` first."""
        kind = probe_kind(target_lang)
        ranked = []
        for face in self.faces:
            if face is ref or not self.usable(face) or face.missing(chars):
                continue
            score = self.similarity(ref, face, kind)
            if score is not None:
                ranked.append((score + locale_term(face, target_lang) + class_term(ref, face), face))
        ranked.sort(key=lambda item: -item[0])
        return ranked[:limit]

    def sheet_faces(self, target_lang: str, default: Optional[FontFace], limit: int = SHEET_LIMIT) -> List[FontFace]:
        """Distinct-looking usable faces with every common character of the language."""
        if probe_kind(target_lang) != "cjk":  # hundreds of Latin fonts and nothing to rank them by yet
            return []
        lang = language_chars(target_lang)
        kind = probe_kind(target_lang)
        pool = [f for f in self.faces if self.usable(f) and f is not default and f.coverage(lang) >= 0.999]
        pool.sort(key=lambda f: (-locale_term(f, target_lang), f.style.casefold() not in REGULAR_STYLES,
                                 abs(f.weight - 400), f.family.casefold()))
        chosen: List[FontFace] = []
        for face in pool:
            if face.weight < SHEET_MIN_WEIGHT or not self.features(face, kind):
                continue
            looks = [(other, self.compare(face, other, kind)) for other in chosen + ([default] if default else [])]
            if any(c is not None and c[0] - c[1] >= SHEET_SIMILAR for _, c in looks):
                continue
            # the same design (one family, or the same glyph shapes under another name): two weights at most
            stem = design_stem(face.family)
            same = [other for other, c in looks if other is not default and (
                design_stem(other.family) == stem or (c is not None and c[0] >= SAME_DESIGN))]
            if len(same) >= SHEET_PER_FAMILY or any(abs(f.weight - face.weight) < 200 for f in same):
                continue
            chosen.append(face)
            if len(chosen) >= limit:
                break
        return chosen


_STEM_DROP = {"tc", "hk", "sc", "jp", "kr", "cn", "tw", "gb", "gbk", "ui", "std", "pro", "pr6n", "pron", "pr6",
              "b5", "mo"}


def design_stem(family: str) -> str:
    """The family name without region and version words: Noto Sans TC / HK / SC are one design."""
    return " ".join(t for t in re.split(r"[\s_\-]+", family.casefold()) if len(t) > 1 and t not in _STEM_DROP)


# Kinds of typeface, from the font's names. Checked in this order: "FangSong" is not a Song (Ming)
# face, a rounded gothic is round, "Microsoft Sans Serif" is a sans. Short Latin words must be a
# whole word of the name ("Ming" in "PMingLiU", not in "Humming"); long ones may be part of a word.
DESIGN_CLASSES = (
    ("fangsong", ("fangsong", "仿宋"), ()),
    ("kai", ("kyokasho", "楷", "教科書"), ("kai",)),
    ("round", ("rounded", "圓", "圆", "丸"), ("yuan", "round", "maru")),
    ("clerical", ("lisu", "reisho", "隶", "隸"), ()),
    ("sans", ("sans", "gothic", "yahei", "jhenghei", "dengxian", "heiti", "meiryo", "malgun", "gulim", "dotum",
              "shin go", "黑", "ゴシック", "新ゴ", "고딕", "돋움", "굴림"), ("hei", "go")),
    ("serif", ("mincho", "serif", "simsun", "batang", "myeongjo", "明", "宋", "명조", "바탕"),
     ("ming", "song", "sung")),
)
CLASS_BONUS = 0.25
"""Same kind of typeface counts for, another kind against; more than the locale term, so a
Japanese Mincho beats a Chinese gothic for a Ming face (the shapes alone hardly tell them apart)."""


def _words(name: str) -> List[str]:
    """``PMingLiU`` -> p ming li u; ``FZLanTingHei-M-GBK`` -> fz lan ting hei m gbk."""
    return [w.casefold() for w in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", name)]


def design_class(face: Optional[FontFace]) -> Optional[str]:
    """``serif`` (Ming, Song, Mincho), ``sans``, ``kai``, ``fangsong``, ``round``, ``clerical`` or None."""
    if face is None:
        return None
    names = [face.family, *face.names]
    folded = [n.casefold() for n in names]
    words = {w for n in names for w in _words(n)}
    for kind, parts, whole in DESIGN_CLASSES:
        if any(p in n for p in parts for n in folded) or any(w in words for w in whole):
            return kind
    return None


def class_term(ref: FontFace, face: FontFace) -> float:
    a, b = design_class(ref), design_class(face)
    if a is None or b is None:
        return 0.0
    return CLASS_BONUS if a == b else -CLASS_BONUS


def locale_term(face: FontFace, target_lang: str) -> float:
    prefs = LOCALE_PREFS.get((target_lang or "").upper())
    if not prefs or not face.locales:
        return 0.0
    hits = [prefs[loc] for loc in face.locales if loc in prefs]
    return max(hits) if hits else -LOCALE_BONUS


def sheet_image(faces: Sequence[FontFace], labels: Sequence[str], target_lang: str):
    """One row per font: its number and name, then a sample line drawn in it."""
    from PIL import Image, ImageDraw, ImageFont

    from .images import _font

    sample = SAMPLES.get((target_lang or "").upper(), SAMPLES[""])
    row, width = 64, 1100
    img = Image.new("RGB", (width, row * len(faces) + 8), "white")
    draw = ImageDraw.Draw(img)
    label_font = _font(20)
    for i, (face, label) in enumerate(zip(faces, labels)):
        y = 4 + i * row
        if i:
            draw.line((0, y, width, y), fill=(210, 210, 210))
        draw.text((10, y + 20), label, font=label_font, fill=(40, 40, 160))
        try:
            font = ImageFont.truetype(face.path, 36, index=face.index)
            if face.instance:
                font.set_variation_by_axes([float(face.instance.get(a, 0)) for a in face.axes])
            draw.text((360, y + 10), sample, font=font, fill=(0, 0, 0))
        except Exception as exc:
            draw.text((360, y + 20), f"(cannot draw: {exc})", font=label_font, fill=(160, 0, 0))
    return img


# --- layout check -----------------------------------------------------------------
class LayoutUnknown(RuntimeError):
    """The region could not be rendered for the check; the font is then not changed."""


@dataclass
class Ink:
    box: Tuple[int, int, int, int]
    """x0, y0, x1, y1 of everything drawn (text and outline)."""
    outside: float = 0.0
    """Share of it outside the region's box grown by 15 %."""

    @property
    def size(self) -> Tuple[int, int]:
        return self.box[2] - self.box[0] + 1, self.box[3] - self.box[1] + 1

    @property
    def center(self) -> Tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2


def layout_ok(ref: Optional[Ink], new: Optional[Ink]) -> Tuple[bool, str]:
    """The new font keeps the text where it was: same place, about the same size, no more of it
    outside the box."""
    if new is None:
        return ref is None, "nothing was drawn"
    if ref is None:
        return True, ""
    (rw, rh), (nw, nh) = ref.size, new.size
    shift = float(np.hypot(new.center[0] - ref.center[0], new.center[1] - ref.center[1]))
    if shift > max(4.0, 0.08 * max(rw, rh)):
        return False, f"moved {shift:.0f}px"
    if nw > rw * 1.15 + 2 or nh > rh * 1.15 + 2:
        return False, f"grew from {rw}x{rh} to {nw}x{nh}px"
    if nw < rw * 0.8 - 2 or nh < rh * 0.8 - 2:  # also catches characters the renderer drops
        return False, f"shrank from {rw}x{rh} to {nw}x{nh}px"
    if new.outside > ref.outside + 0.03:
        return False, f"{new.outside:.0%} outside the box"
    return True, ""


def ink_of(rendered: np.ndarray, canvas_color: Sequence[int], region_box: Sequence[int]) -> Optional[Ink]:
    diff = np.abs(rendered.astype(np.int16) - np.asarray(canvas_color, dtype=np.int16)).max(axis=2) > 40
    ys, xs = np.nonzero(diff)
    if len(xs) == 0:
        return None
    x0, y0, x1, y1 = region_box
    gx, gy = (x1 - x0) * 0.15, (y1 - y0) * 0.15
    inside = (xs >= x0 - gx) & (xs <= x1 + gx) & (ys >= y0 - gy) & (ys <= y1 + gy)
    return Ink((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), float(1 - inside.mean()))


class PipelineFontBackend:
    """Renders one region alone with the translator's renderer and config, on a plain canvas,
    with the same bubble mask the real render uses."""

    CANVAS = (255, 0, 255)

    def __init__(self, translator, config):
        self.translator = translator
        self.config = config
        self._pages: Dict[str, tuple] = {}

    def _page(self, image_path: str):
        if image_path not in self._pages:
            from PIL import Image

            from ..utils import Context

            regions, _mask, _refined, skip_font_scaling, skip_replacements, failed = \
                self.translator._load_text_and_regions_from_file(image_path, self.config)
            if failed:  # region numbers would not match the JSON
                regions = None
            img = np.asarray(Image.open(image_path).convert("RGB"))
            ctx = Context()
            ctx.img_rgb = img
            ctx.bubble_mask = None
            try:
                self.translator._prime_bubble_detection_cache(self.config, ctx)
            except Exception:
                logger.warning("fonts: no bubble mask for the layout check", exc_info=True)
            self._pages = {image_path: (regions, img, ctx.bubble_mask, bool(skip_font_scaling),
                                        bool(skip_replacements))}  # one page at a time
        return self._pages[image_path]

    async def ink(self, image_path: str, index: int, font: str) -> Optional[Ink]:
        from ..rendering import dispatch as dispatch_rendering

        regions, img, bubble_mask, skip_scaling, skip_replacements = self._page(image_path)
        if regions is None or not 0 <= index < len(regions):
            raise LayoutUnknown(f"region {index + 1} could not be loaded for rendering")
        block = copy.deepcopy(regions[index])
        block.font_family = font
        canvas = np.empty_like(img)
        canvas[:] = self.CANVAS
        out = await dispatch_rendering(canvas, [block], self.config, img, skip_font_scaling=skip_scaling,
                                       skip_text_replacements=skip_replacements, bubble_mask=bubble_mask)
        if isinstance(out, tuple):
            out = out[0]
        pts = np.asarray(block.lines, dtype=np.float64).reshape(-1, 2)
        box = (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))
        return ink_of(np.asarray(out), self.CANVAS, box)

    def done(self, image_path: str) -> None:
        self._pages.pop(image_path, None)


# --- the service ------------------------------------------------------------------
_HUMAN = {
    "zh": "字型「{font}」沒有「{chars}」，也找不到有這些字、又能排在同一位置的相似字型",
    "zh_cn": "字体「{font}」没有「{chars}」，也找不到有这些字、又能排在同一位置的相似字体",
    "en": "The font {font} has no {chars}, and no similar font with them fits the same place",
}


def _human(target_lang: str, **kw) -> str:
    target = (target_lang or "").upper()
    key = "zh" if target == "CHT" else "zh_cn" if target == "CHS" else "en"
    return _HUMAN[key].format(**kw)


class FontService:
    def __init__(self, catalog: FontCatalog, backend, default_font: str, target_lang: str):
        self.catalog = catalog
        self.backend = backend
        self.default_font = default_font or RENDER_DEFAULT
        self.target_lang = target_lang
        self.choices: List[FontFace] = []
        self.sheet_url = ""

    def default_face(self) -> Optional[FontFace]:
        return self.catalog.find(self.default_font) or self.catalog.find(RENDER_DEFAULT)

    def prepare_sheet(self) -> None:
        """The fonts the agent may choose and their sample sheet (row 0 is the user's font)."""
        from . import images

        default = self.default_face()
        self.choices = self.catalog.sheet_faces(self.target_lang, default)
        if not self.choices:
            return
        faces = ([default] if default else []) + self.choices
        labels = ([f"0  {self.default_font} (default)"] if default else []) + \
                 [f"{i}  {face.label}" for i, face in enumerate(self.choices, start=1)]
        self.sheet_url = images.data_url(sheet_image(faces, labels, self.target_lang))

    def sheet_text(self) -> str:
        lines = [f"0: {self.default_font} (the user's font, for normal dialogue)"]
        lines += [f"{i}: {face.label}" for i, face in enumerate(self.choices, start=1)]
        return "\n".join(lines)

    def _same(self, a: str, b: str) -> bool:
        if (a or "").casefold() == (b or "").casefold():
            return True
        fa, fb = self.catalog.find(a), self.catalog.find(b)
        return fa is not None and fa is fb

    async def _fits(self, image_path: str, index: int, ref_font: str, new_font: str,
                    cache: Dict[str, Optional[Ink]]) -> Tuple[bool, str]:
        try:
            for font in (ref_font, new_font):
                if font not in cache:
                    cache[font] = await self.backend.ink(image_path, index, font)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"fonts: layout check of {os.path.basename(image_path)} #{index + 1} failed: {exc}")
            return False, f"layout check failed: {exc}"
        return layout_ok(cache[ref_font], cache[new_font])

    async def fix_page(self, image_path: str, store, events, fix_missing: bool = True) -> Dict[str, Any]:
        """Check every translated region's font and write the changes to the page's JSON."""
        from ..utils.path_manager import find_json_path

        json_path = find_json_path(image_path)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        if not json_path or not os.path.exists(json_path):
            return {"page": stem, "changed": 0, "unfixed": 0}
        with open(json_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        entry = raw[next(iter(raw))]
        regions = entry.get("regions", []) if isinstance(entry, dict) else entry
        changes: List[Dict[str, Any]] = []
        unfixed: List[Dict[str, Any]] = []
        kept: List[Dict[str, Any]] = []  # only punctuation missing and no close font: it is borrowed
        new_fonts: Dict[int, str] = {}
        for index, region in enumerate(regions):
            text = str(region.get("translation") or "")
            if not text.strip():
                continue
            original = str(region.get("font_family") or self.default_font)
            current = original
            cache: Dict[str, Optional[Ink]] = {}
            if not self._same(current, self.default_font):  # the agent's (or an earlier) choice
                ok, why = await self._fits(image_path, index, self.default_font, current, cache)
                if not ok:
                    changes.append({"region": index + 1, "from": current, "to": self.default_font,
                                    "reason": "layout", "detail": why})
                    current = self.default_font
            chars = rendered_chars(text, region.get("direction"))
            face = self.catalog.find(current) or self.catalog.find(RENDER_DEFAULT)
            missing = face.missing(chars) if face is not None else ""
            if missing and fix_missing:
                fixed = False
                tried = []
                ranked = await asyncio.to_thread(self.catalog.similar, face, chars, self.target_lang)
                punctuation = all(unicodedata.category(c)[0] in "PSZ" for c in missing)
                if punctuation:  # borrowed punctuation is a small blemish; a font of another kind is not
                    ranked = [(s, f) for s, f in ranked if s >= PUNCT_MIN_SCORE]
                for _score, candidate in ranked:
                    ok, why = await self._fits(image_path, index, current, candidate.value, cache)
                    if ok:
                        changes.append({"region": index + 1, "from": current, "to": candidate.value,
                                        "reason": "missing", "missing": missing})
                        current = candidate.value
                        fixed = True
                        break
                    tried.append(f"{candidate.label}: {why}")
                if not fixed:  # only missing letters go on the review list
                    (kept if punctuation else unfixed).append(
                        {"region": index + 1, "font": current, "missing": missing, "tried": tried})
            if current != original:
                new_fonts[index] = current
        if hasattr(self.backend, "done"):
            self.backend.done(image_path)
        if new_fonts:
            with open(json_path, "r", encoding="utf-8") as fh:  # re-read: write only the font names
                raw = json.load(fh)
            entry = raw[next(iter(raw))]
            saved = entry.get("regions", []) if isinstance(entry, dict) else entry
            for index, font in new_fonts.items():
                if index < len(saved):
                    saved[index]["font_family"] = font
            write_json_atomic(json_path, raw)
        if changes or unfixed or kept:
            side = store.load_sidecar(stem) or {}
            side.pop("version", None)
            flags = [{"region": u["region"], "type": "layout",
                      "reason": _human(self.target_lang, font=u["font"], chars=u["missing"])} for u in unfixed]
            side["review_flags"] = list(side.get("review_flags") or []) + flags
            side["fonts"] = {"changes": changes, "unfixed": unfixed, "kept": kept}
            store.save_sidecar(stem, side)
            for flag in flags:
                events.emit("flag", page=stem, agent="fonts", region=flag["region"], flag="layout",
                            reason=flag["reason"], source="check")
        if changes or unfixed:
            events.emit("fonts", page=stem, agent="fonts", changed=len(changes), unfixed=len(unfixed),
                        changes=[{k: c[k] for k in ("region", "from", "to", "reason", "missing") if k in c}
                                 for c in changes])
        return {"page": stem, "changed": len(changes), "unfixed": len(unfixed), "changes": changes}


async def build_font_service(translator, config, target_lang: str) -> FontService:
    """Scan the installed fonts (in a worker thread) and check their names with the renderer."""
    faces = await asyncio.to_thread(scan_fonts)
    catalog = FontCatalog(faces, qt_resolver())
    render = getattr(config, "render", None)
    default = getattr(render, "font_family", None) or getattr(translator, "font_family", None) or RENDER_DEFAULT
    return FontService(catalog, PipelineFontBackend(translator, config), default, target_lang)
