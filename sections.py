"""Heading-based section detection for Arabic legal books/commentaries.

Many Egyptian legal sources aren't bare codes but commentaries organised like:

    الصيغة رقم (٦)                         <- container (level 1), followed by a short title line
    إعلان موجه إلى هيئة أو مؤسسة عامة
      نصوص القانون :                      <- section type (level 2)
      المذكرة الإيضاحية :
      النصوص العربية المقابلة :
      الصيغة                              <- the form template itself
      آراء الشراح وأحكام القضاء :

Matching is fuzzy because OCR mangles headings (tatweel "الصيغـــة" often comes out as
"الصيغفة" / "الصيغغة"). Pass your own list with --headings FILE (one per line: `level|phrase|type`).
"""
from __future__ import annotations

import re
from pathlib import Path

# (level, phrase, type)
DEFAULT_HEADINGS = [
    (1, "الباب", "باب"),
    (1, "الفصل", "فصل"),
    (1, "الفرع", "فرع"),
    (1, "المبحث", "مبحث"),
    (1, "المطلب", "مطلب"),
    (1, "الصيغة رقم", "صيغة"),
    (2, "نصوص القانون", "نصوص القانون"),
    (2, "النص القانوني", "نصوص القانون"),
    (2, "المذكرة الإيضاحية", "المذكرة الإيضاحية"),
    (2, "النصوص العربية المقابلة", "النصوص العربية المقابلة"),
    (2, "آراء الشراح وأحكام القضاء", "آراء الشراح وأحكام القضاء"),
    (2, "آراء الشراح", "آراء الشراح وأحكام القضاء"),
    (2, "أحكام القضاء", "أحكام القضاء"),
    (2, "أحكام النقض", "أحكام القضاء"),
    (2, "المبادئ القانونية", "المبادئ القانونية"),
    (2, "التعليق", "التعليق"),
    (2, "الشرح", "الشرح"),
    (2, "الصيغة", "الصيغة (نموذج)"),
]


def _n(t: str) -> str:
    t = re.sub(r"[ً-ٰٟـ‎‏]", "", t)
    t = re.sub("[أإآٱ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def load_headings(path: str | None):
    if not path:
        return DEFAULT_HEADINGS
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            parts = [x.strip() for x in line.split("|")]
            lvl, phrase = int(parts[0]), parts[1]
            out.append((lvl, phrase, parts[2] if len(parts) > 2 else phrase))
    return out


def match_heading(line: str, headings, min_score: float = 85.0):
    """Return (level, type, matched_phrase) if `line` is a heading line, else None."""
    from compat import ratio
    ln = _n(line)
    if not ln or len(ln) > 70:
        return None
    best = None
    # longer phrases first so "الصيغة رقم" beats "الصيغة"
    for lvl, phrase, typ in sorted(headings, key=lambda h: -len(h[1])):
        ph = _n(phrase)
        if len(ln) > len(ph) + 28:          # headings are short lines
            continue
        head = ln[: len(ph) + 2]
        score = max(ratio(ph, head), ratio(ph, ln[: len(ph)]))
        if ln.split()[0][:2] != ph.split()[0][:2]:   # must start like the phrase
            continue
        if score >= min_score and (best is None or score > best[3]):
            best = (lvl, typ, phrase, score)
            if score >= 99:
                break
    return best[:3] if best else None


def detect_sections(doc: str, file: str, headings=DEFAULT_HEADINGS):
    """Return list of dicts: start, end, key, level, type, container, title."""
    lines, pos = [], 0
    for ln in doc.split("\n"):
        lines.append((pos, ln))
        pos += len(ln) + 1
    heads = []
    for i, (p, ln) in enumerate(lines):
        m = match_heading(ln, headings)
        if m:
            lvl, typ, _ = m
            title = ln.strip()
            if lvl == 1:   # container title = heading line + next 1-2 short non-heading lines
                extra = []
                for _, nxt in lines[i + 1:i + 4]:
                    if not nxt.strip():
                        continue
                    if len(nxt.strip()) > 70 or match_heading(nxt, headings) or len(extra) == 2:
                        break
                    extra.append(nxt.strip())
                title = " - ".join([title] + extra)
            heads.append((p, lvl, typ, title))
    spans, container = [], ""
    if not heads or heads[0][0] > 0:
        spans.append(dict(start=0, end=heads[0][0] if heads else len(doc), level=0, type="مقدمة",
                          container="", title=""))
    for j, (p, lvl, typ, title) in enumerate(heads):
        end = heads[j + 1][0] if j + 1 < len(heads) else len(doc)
        if lvl == 1:
            container = title
        spans.append(dict(start=p, end=end, level=lvl, type=typ, container=container, title=title))
    for k, sp in enumerate(spans):
        sp["key"] = f"{file}::S{k:03d}"
    return [sp for sp in spans if doc[sp["start"]:sp["end"]].strip()]
