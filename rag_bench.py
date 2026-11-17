#!/usr/bin/env python3
"""
rag_bench.py - a deliberately "vanilla" RAG test bench for Arabic legal PDFs (digital or scanned).

    PDF -> text (PyMuPDF, or OCR for scanned pages) -> chunks -> embeddings (Ollama) -> retrieval

Quick start
    python rag_bench.py ingest --pdf mybook.pdf --out idx_mybook --dump-text
    python rag_bench.py query  --index idx_mybook            # interactive: type questions

Optional tools: diagnose (embedding-space report), gen-queries + eval (precision/recall),
ocr-check (compare OCR engines), --rerank (second-stage reordering). See README.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Arabic text helpers
# --------------------------------------------------------------------------- #

AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
TASHKEEL = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
TATWEEL = "ـ"
ARABIC_CHAR = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")  # incl. presentation forms
LTR_RUN = re.compile(r"[0-9A-Za-z][0-9A-Za-z.,/:%\-]*[0-9A-Za-z]|[0-9A-Za-z]")

# Very frequent Arabic words; if their *reversed* spellings dominate, the PDF
# stores text in visual (right-to-left drawn) order and extraction is reversed.
COMMON_WORDS = ["في", "من", "على", "التي", "الذي", "هذا", "المادة", "إلى", "القانون", "أو", "مع", "عن"]

ORDINALS = {
    "الأولى": 1, "الاولى": 1, "الثانية": 2, "الثالثة": 3, "الرابعة": 4, "الخامسة": 5,
    "السادسة": 6, "السابعة": 7, "الثامنة": 8, "التاسعة": 9, "العاشرة": 10,
}

# Article headings: "مادة 5", "المادة (5)", "مادة ٥ مكرراً", "المادة 12 مكرر (أ)", "المادة الأولى"
ARTICLE_RE = re.compile(
    r"^[\s\-–—•*]*(?:ال)?مـ*اـ*دـ*[ةه]\s*[\(\[]?\s*"
    r"(?P<num>[0-9٠-٩۰-۹]+|" + "|".join(ORDINALS) + r")"
    r"\s*[\)\]]?[ \t]*(?P<bis>مكرر[اًا]?ً?(?:[ \t]*\([ \t]*[أ-ي][ \t]*\))?)?"
    # a heading is followed by end-of-line or ':' / '-' / '.'; this skips in-text
    # cross-references such as "...طبقاً للمادة 5 من هذا القانون" that wrap to a new line
    # what follows must look like a heading, not running text: a ':' within 60 chars
    # ("مادة ١٠ فقرة أولى مرافعات :", "مادة (3) من القانون رقم 47 لسنة 1973 :"), a dash/dot,
    # or a short tail to end-of-line ("مادة ١٧ بند (١) مرافعات"). This skips in-text cross-references
    # such as "...المادة ١١ من قانون المرافعات الحالى هى بذاتها" that happen to start a line.
    r"(?=[^\n:]{0,60}:|[ \t]*[\-–—.]|[^\n]{0,30}$)",
    re.MULTILINE,
)


def normalize_for_match(text: str) -> str:
    """Aggressive normalisation used ONLY for matching/labels, never for embedding
    unless --normalize is passed at ingest time."""
    t = unicodedata.normalize("NFKC", text)
    t = TASHKEEL.sub("", t).replace(TATWEEL, "")
    t = re.sub("[أإآٱ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    t = t.translate(AR_DIGITS)
    return re.sub(r"\s+", " ", t).strip()


def norm_words(text: str) -> str:
    """normalize_for_match + punctuation removed: used for phrase labels, robust to OCR punctuation noise."""
    return " ".join(re.findall(r"\w+", normalize_for_match(text)))


def light_normalize(text: str) -> str:
    """Optional light normalisation for embedded text: strip diacritics/tatweel."""
    return TASHKEEL.sub("", text).replace(TATWEEL, "")


def fix_visual_line(line: str) -> str:
    """Reverse a visually-ordered Arabic line back to logical order, keeping
    digit/Latin runs readable."""
    rev = line[::-1]
    return LTR_RUN.sub(lambda m: m.group(0)[::-1], rev)


def looks_visual_order(text: str) -> bool:
    words = re.findall(r"[؀-ۿ]+", text)
    if len(words) < 20:
        return False
    c = Counter(words)
    fwd = sum(c[w] for w in COMMON_WORDS)
    rev = sum(c[w[::-1]] for w in COMMON_WORDS)
    return rev > max(3, 2 * fwd)


def article_label(num: str, bis: str | None) -> str:
    n = str(ORDINALS[num]) if num in ORDINALS else num.translate(AR_DIGITS)
    if bis:
        letter = re.findall(r"[أ-ي]", bis.replace("مكرر", "").replace("ا", "", 1))
        n += " مكرر" + (f" {letter[0]}" if letter else "")
    return n


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

@dataclass
class Page:
    file: str
    page: int          # 1-based
    text: str
    flags: list = field(default_factory=list)
    ocr_conf: float | None = None      # Tesseract mean word confidence (0-100) when the page was OCR'd
    printed_page: int | None = None    # page number printed on the page, if detected
    info: dict = field(default_factory=dict)


BIGRAMS = ["هذا القانون", "لا يجوز", "يجب ان", "على ان", "في حاله", "من هذا", "وفقا لاحكام", "الجريده الرسميه",
           "اللائحه التنفيذيه", "بشان", "يعاقب بالحبس", "مع عدم الاخلال", "يجوز ل", "صاحب العمل", "رقم"]
LINE_END_PREP = re.compile(r"(?:^|\s)(?:في|من|على|إلى|الى|عن|أو|او|و)\s*$", re.MULTILINE)


def naturalness(text: str) -> int:
    """Rough score of how much `text` looks like logically-ordered Arabic."""
    t = normalize_for_match(text)
    fwd = sum(t.count(b) for b in BIGRAMS)
    rev = sum(t.count(" ".join(b.split()[::-1])) for b in BIGRAMS if " " in b)
    heads = len(ARTICLE_RE.findall(text))
    rev_heads = len(re.findall(r"[\(\[]?\s*[0-9٠-٩]+\s*[\)\]]?\s*(?:ال)?ماد[ةه]\s*$", text, re.MULTILINE))
    return 3 * (heads - rev_heads) + 2 * (fwd - rev) - len(LINE_END_PREP.findall(text)) // 3


def rtl_geometry_text(pg) -> str | None:
    """Rebuild text from word boxes, ordering Arabic lines right-to-left by x position."""
    words = pg.words()
    if not words:
        return None
    lines = defaultdict(list)
    for x0, y0, x1, y1, w, b, l in words:
        lines[(b, l)].append((x0, y0, w))
    out, prev_block = [], None
    for (b, l) in sorted(lines, key=lambda k: (round(min(t[1] for t in lines[k])), k)):
        ws = lines[(b, l)]
        arabic = sum(bool(ARABIC_CHAR.search(w)) for _, _, w in ws) >= len(ws) / 2
        ws = sorted(ws, key=lambda t: -t[0] if arabic else t[0])
        if prev_block is not None and b != prev_block:
            out.append("")
        out.append(" ".join(w for _, _, w in ws))
        prev_block = b
    return "\n".join(out)


def extract_pdf(path: Path, mode: str = "auto", ocr=None, ocr_mode: str = "auto") -> list[Page]:
    from compat import open_pdf

    pages = []
    with open_pdf(path) as doc:
        for i in range(len(doc)):
            pg = doc[i]
            flags = []
            raw = pg.text()
            if mode in ("auto", "rtl"):
                geo = rtl_geometry_text(pg) if raw.strip() else None
                if geo and mode == "rtl" or geo and naturalness(unicodedata.normalize("NFKC", geo)) > \
                        naturalness(unicodedata.normalize("NFKC", raw)):
                    raw = geo
                    flags.append("rtl_word_order_rebuilt")
            # NFKC folds Arabic presentation forms (ﻻ, ﺍﻟ…) back to normal letters.
            text = unicodedata.normalize("NFKC", raw)
            if raw != text:
                flags.append("presentation_forms_folded")
            if looks_visual_order(text):
                text = "\n".join(fix_visual_line(l) for l in text.split("\n"))
                flags.append("visual_order_fixed")
            if len(ARABIC_CHAR.findall(text)) < 20:
                if pg.has_images():
                    flags.append("likely_scanned_needs_ocr")
                elif not text.strip():
                    flags.append("empty")
            pages.append(Page(path.name, i + 1, _clean(text), flags))

    todo = [p.page for p in pages if ocr_mode == "force" or "likely_scanned_needs_ocr" in p.flags]
    if ocr is not None and ocr_mode != "off" and todo:
        def prog(i, n):
            print(f"\r    OCR {path.name}: {i}/{n} pages", end="", file=sys.stderr)
        results = ocr.ocr_pages(path, todo, workers=getattr(ocr, "workers", 2), progress=prog)
        print(file=sys.stderr)
        for p in pages:
            r = results.get(p.page)
            if r is None:
                continue
            p.text = _clean(unicodedata.normalize("NFKC", r.text))
            p.flags = [f for f in p.flags if f != "likely_scanned_needs_ocr"] + [f"ocr:{r.engine}"]
            if r.restored:
                p.flags.append("ocr_restored")
            p.flags += [f for f in r.flags if not f.startswith("deg:")]
            p.ocr_conf = r.conf
            for f in r.flags:
                if f.startswith("deg:"):
                    p.info["degradation"] = json.loads(f[4:])
            if r.agreement is not None:
                p.info["vlm_tesseract_agreement"] = r.agreement
            if len(ARABIC_CHAR.findall(p.text)) < 20:
                p.flags.append("ocr_blank")
            elif r.conf is not None and r.conf < 70:
                p.flags.append("ocr_low_confidence")
    from ocr import printed_page_number
    for p in pages:
        p.printed_page = printed_page_number(p.text)
    return pages


BIDI_MARKS = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\u061c]")


def _clean(text: str) -> str:
    text = BIDI_MARKS.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# --------------------------------------------------------------------------- #
# Article spans (used as ground-truth structure for diagnostics, and for
# the optional --chunking article mode)
# --------------------------------------------------------------------------- #

def article_spans(doc_text: str, file: str) -> list[tuple[int, int, str]]:
    """Return [(start, end, key)] covering doc_text. key = 'file::N' or 'file::preamble'."""
    heads = [(m.start(), article_label(m.group("num"), m.group("bis"))) for m in ARTICLE_RE.finditer(doc_text)]
    spans = []
    if not heads or heads[0][0] > 0:
        spans.append((0, heads[0][0] if heads else len(doc_text), f"{file}::preamble"))
    seen = Counter()
    for i, (s, lab) in enumerate(heads):
        e = heads[i + 1][0] if i + 1 < len(heads) else len(doc_text)
        seen[lab] += 1
        # same article number appearing twice in one file (e.g. issuance articles
        # + body, or two laws in one PDF) -> disambiguate
        key = f"{file}::{lab}" if seen[lab] == 1 else f"{file}::{lab}#{seen[lab]}"
        spans.append((s, e, key))
    return spans


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #

SEPARATORS = ["\n\n", "\n", ". ", "؛", "،", " ", ""]


def recursive_split(text: str, size: int, overlap: int, seps=SEPARATORS) -> list[tuple[int, int]]:
    """LangChain-style RecursiveCharacterTextSplitter, returning (start, end) offsets."""
    def split_rec(s: int, e: int, seps):
        if e - s <= size:
            return [(s, e)]
        sep = next((x for x in seps if x == "" or x in text[s:e]), "")
        rest = seps[seps.index(sep) + 1:] if sep else []
        if sep == "":
            return [(i, min(i + size, e)) for i in range(s, e, size)]
        pieces, i = [], s
        while i < e:
            j = text.find(sep, i, e)
            j = e if j == -1 else j + len(sep)
            pieces.append((i, j))
            i = j
        out = []
        for ps, pe in pieces:
            out.extend(split_rec(ps, pe, rest) if pe - ps > size else [(ps, pe)])
        return out

    atoms = split_rec(0, len(text), seps)
    chunks, cur = [], []
    for a in atoms:
        if cur and a[1] - cur[0][0] > size:
            chunks.append((cur[0][0], cur[-1][1]))
            end = cur[-1][1]
            cur = [x for x in cur if end - x[0] <= overlap]      # tail atoms -> overlap
            while cur and a[1] - cur[0][0] > size:
                cur.pop(0)
        cur.append(a)
    if cur:
        chunks.append((cur[0][0], cur[-1][1]))
    return [(s, e) for s, e in chunks if text[s:e].strip()]


@dataclass
class Chunk:
    id: int
    file: str
    pages: list
    text: str            # text that gets embedded
    raw: str             # original span text (for display/matching)
    articles: list       # article keys this chunk overlaps (file::N)
    primary_article: str
    n_chars: int


def build_document(pages: list[Page]) -> tuple[str, list[tuple[int, int, int]]]:
    parts, page_spans, pos = [], [], 0
    for p in pages:
        parts.append(p.text)
        page_spans.append((pos, pos + len(p.text), p.page))
        pos += len(p.text) + 2
    return "\n\n".join(parts), page_spans


def overlaps(s, e, spans):
    out = []
    for a, b, key in spans:
        ov = min(e, b) - max(s, a)
        if ov > 0:
            out.append((ov, key))
    return out


def chunk_file(pages: list[Page], args):
    """Returns (chunks, articles, sections) for one PDF."""
    from sections import detect_sections, load_headings
    file = pages[0].file
    conf_by_page = {p.page: p.ocr_conf for p in pages if p.ocr_conf is not None}
    doc, page_spans = build_document(pages)
    art = article_spans(doc, file)
    secs = detect_sections(doc, file, load_headings(getattr(args, "headings", None)))
    sec_spans = [(sp["start"], sp["end"], sp["key"]) for sp in secs]
    sec_by_key = {sp["key"]: sp for sp in secs}

    def split_units(spans):
        offs = []
        for s, e, _ in spans:
            if e - s <= args.chunk_size * 2:
                offs.append((s, e))
            else:
                offs += [(s + a, s + b) for a, b in recursive_split(doc[s:e], args.chunk_size, args.chunk_overlap)]
        return offs

    if args.chunking == "fixed" and args.chunk_scope == "page":
        offsets = []
        for (ps, pe, _) in page_spans:
            offsets += [(ps + s, ps + e) for s, e in recursive_split(doc[ps:pe], args.chunk_size, args.chunk_overlap)]
    elif args.chunking == "fixed":
        offsets = recursive_split(doc, args.chunk_size, args.chunk_overlap)
    elif args.chunking == "article":
        offsets = split_units(art)
    else:  # section
        offsets = split_units(sec_spans)

    out = []
    for s, e in offsets:
        raw = doc[s:e].strip()
        if len(raw) < 20:
            continue
        ov = sorted(overlaps(s, e, art), reverse=True)
        arts = [k for _, k in ov]
        prim = arts[0] if arts else f"{file}::preamble"
        sov = sorted(overlaps(s, e, sec_spans), reverse=True)
        sec = sec_by_key[sov[0][1]] if sov else None
        pgs = sorted({p for a, b, p in page_spans if min(e, b) - max(s, a) > 0})
        text = light_normalize(raw) if args.normalize else raw
        if args.contextual_header:
            label = prim.split("::", 1)[1].split("#")[0]
            parts = [Path(file).stem]
            if sec and sec["container"]:
                parts.append(_strip_marks(sec["container"]))
            if sec and sec["level"] == 2:
                parts.append(sec["type"])
            if label != "preamble":
                parts.append(f"المادة {label}")
            text = " | ".join(parts) + "\n" + text
        rec = dict(file=file, pages=pgs, text=text, raw=raw, articles=arts, primary_article=prim, n_chars=len(raw),
                   section=sec["key"] if sec else None, section_type=sec["type"] if sec else None)
        if any(p in conf_by_page for p in pgs):
            rec["ocr_conf"] = round(min(conf_by_page[p] for p in pgs if p in conf_by_page), 1)
        out.append(rec)
    articles = [dict(key=k, file=file, article=k.split("::", 1)[1], text=doc[s:e].strip(), pages=_pages_of(s, e, page_spans))
                for s, e, k in art if not k.endswith("::preamble") and doc[s:e].strip()]
    sections = [dict(key=sp["key"], file=file, level=sp["level"], type=sp["type"],
                     container=_strip_marks(sp["container"]), title=_strip_marks(sp["title"]),
                     pages=_pages_of(sp["start"], sp["end"], page_spans), text=doc[sp["start"]:sp["end"]].strip())
                for sp in secs]
    return out, articles, sections


def _pages_of(s, e, page_spans):
    return sorted({p for a, b, p in page_spans if min(e, b) - max(s, a) > 0})


def _strip_marks(t: str) -> str:
    return re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]", "", t or "").strip()


# --------------------------------------------------------------------------- #
# Backends (see backends.py)
# --------------------------------------------------------------------------- #

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backends import ApiError, Embedder, Reranker, ollama_chat, ollama_host, parse_json_loose  # noqa: E402


def make_embedder(spec: str, index_dir: Path, args, meta: dict | None = None) -> Embedder:
    meta = meta or {}
    return Embedder(spec, index_dir, device=getattr(args, "device", None), host=getattr(args, "ollama_host", None),
                    num_ctx=meta.get("num_ctx") or getattr(args, "num_ctx", 8192),
                    query_instruction=meta.get("query_instruction", getattr(args, "query_instruction", None)),
                    batch_size=getattr(args, "batch_size", 16), normalize_fn=normalize_for_match)


# --------------------------------------------------------------------------- #
# Index I/O
# --------------------------------------------------------------------------- #

def save_index(out: Path, chunks, emb, meta):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    np.save(out / "embeddings.npy", emb.astype(np.float32))
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index(idx: Path):
    chunks = [json.loads(l) for l in open(idx / "chunks.jsonl", encoding="utf-8")]
    emb = np.load(idx / "embeddings.npy")
    meta = json.loads((idx / "meta.json").read_text(encoding="utf-8"))
    return chunks, emb, meta


def search(emb, qv, k):
    scores = emb @ qv.T  # (n, q)
    k = min(k, emb.shape[0])
    top = np.argsort(-scores, axis=0)[:k].T
    return top, np.take_along_axis(scores.T, top, axis=1)


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #

def cmd_ingest(args):
    pdfs = [Path(x) for x in (args.pdf or [])]
    if args.pdf_dir:
        pdfs += sorted(Path(args.pdf_dir).glob("**/*.pdf"))
    if not pdfs:
        sys.exit("Give --pdf FILE [FILE ...] and/or --pdf-dir DIR")
    missing = [str(x) for x in pdfs if not x.exists()]
    if missing:
        sys.exit(f"Not found: {', '.join(missing)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ocr_engine = None
    if args.ocr != "off":
        import ocr as ocr_mod
        from ocr import PageOCR
        tcmd = ocr_mod.find_tesseract(args.tesseract)
        langs, terr = ocr_mod.tesseract_langs(tcmd) if tcmd else ([], None)
        missing = [l for l in args.ocr_lang.split("+") if l not in langs]
        if not tcmd:
            print("NOTE: Tesseract not found, so scanned pages will be SKIPPED. Install it, then re-run:\n"
                  "  Windows: UB-Mannheim installer (tick 'Arabic' under Additional language data); it is found\n"
                  "           automatically in C:\\Program Files\\Tesseract-OCR, or pass --tesseract PATH\\tesseract.exe\n"
                  "  Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-ara   macOS: brew install tesseract tesseract-lang",
                  file=sys.stderr)
        elif terr:
            sys.exit(f"Tesseract was found at {tcmd} but could not run:\n  {terr}\n"
                     "If this says it was blocked by policy, see README > Windows.")
        elif missing and args.ocr_engine == "tesseract":
            sys.exit(f"Tesseract at {tcmd} has no '{'+'.join(missing)}' language data (installed: {', '.join(langs) or 'none'}).\n"
                     "Re-run the installer and tick Arabic, or copy ara.traineddata (github.com/tesseract-ocr/tessdata_best)\n"
                     "into the tessdata folder next to tesseract.exe.")
        if tcmd and not terr and not (missing and args.ocr_engine == "tesseract"):
            ocr_mod.TESSERACT_CMD = tcmd
            print(f"OCR: {args.ocr_engine} (tesseract at {tcmd})", file=sys.stderr)
            ocr_engine = PageOCR(args.ocr_engine, args.ocr_restore, args.ocr_lang, args.ocr_dpi, args.ocr_cache,
                                 host=args.ollama_host)
            ocr_engine.workers = args.ocr_workers
    from compat import pdf_backend
    print(f"PDF reader: {pdf_backend()}", file=sys.stderr)

    from ocr import page_number_check
    all_chunks, all_articles, all_sections, extraction, page_rows = [], [], [], [], []
    for p in pdfs:
        pages = extract_pdf(p, args.extract, ocr_engine, args.ocr)
        for pg in pages:
            page_rows.append(dict(file=pg.file, page=pg.page, chars=len(pg.text), flags=pg.flags,
                                  ocr_conf=None if pg.ocr_conf is None else round(pg.ocr_conf, 1),
                                  printed_page=pg.printed_page, **pg.info))
        confs = [pg.ocr_conf for pg in pages if pg.ocr_conf is not None]
        ocr_stats = dict(ocr_pages=len(confs), ocr_mean_conf=round(float(np.mean(confs)), 1) if confs else None,
                         page_numbers=page_number_check({pg.page: pg.printed_page for pg in pages})) if confs else {}
        flags = Counter(f for pg in pages for f in pg.flags)
        extraction.append(dict(file=p.name, pages=len(pages), chars=sum(len(pg.text) for pg in pages),
                               articles_detected=len(ARTICLE_RE.findall("\n".join(pg.text for pg in pages))),
                               **ocr_stats, **{f"pages_{k}": v for k, v in flags.items()}))
        if args.dump_text:
            (out / "text").mkdir(exist_ok=True)
            (out / "text" / f"{p.stem}.txt").write_text(
                "\n\n".join(f"===== page {pg.page} {pg.flags} =====\n{pg.text}" for pg in pages), encoding="utf-8")
        if any(pg.text for pg in pages):
            ch, arts, secs = chunk_file(pages, args)
            all_chunks += ch
            all_articles += arts
            all_sections += secs
        print(f"  {p.name}: {len(pages)} pages, flags={dict(flags) or '-'}")

    for i, c in enumerate(all_chunks):
        c["id"] = i
    with open(out / "pages.jsonl", "w", encoding="utf-8") as f:
        for r in page_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for name, rows in (("articles.jsonl", all_articles), ("sections.jsonl", all_sections)):
        with open(out / name, "w", encoding="utf-8") as f:
            for a in rows:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
    print(f"{len(all_chunks)} chunks ({len(all_articles)} articles, {len(all_sections)} sections) from {len(pdfs)} PDFs; embedding with {args.model} ...")

    em = make_embedder(args.model, out, args)
    texts = [c["text"] for c in all_chunks]
    emb = em.encode_passages(texts, fit=True)
    tokens = em.token_counts(texts)
    if tokens:
        for c, t in zip(all_chunks, tokens):
            c["n_tokens"] = t
    for i in em.truncated:
        all_chunks[i]["truncated"] = True

    total_chars = sum(len(em.p_prefix + t) for t in texts)
    meta = dict(model=args.model, chunking=args.chunking, chunk_scope=args.chunk_scope,
                chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, normalize=args.normalize,
                contextual_header=args.contextual_header, max_tokens=em.max_tokens, num_ctx=args.num_ctx,
                query_instruction=args.query_instruction, query_prefix=em.q_prefix,
                truncated_chunks=len(em.truncated) if em.kind == "ollama" else None,
                chars_per_token=(total_chars / em.prompt_tokens) if em.prompt_tokens else None,
                extraction=extraction, n_chunks=len(all_chunks), dim=int(emb.shape[1]))
    save_index(out, all_chunks, emb, meta)
    if tokens and em.max_tokens:
        trunc = sum(t > em.max_tokens for t in tokens)
        if trunc:
            print(f"WARNING: {trunc} chunks exceed the model's {em.max_tokens}-token window and were truncated.")
    if em.truncated:
        print(f"WARNING: {len(em.truncated)} chunks exceeded num_ctx={args.num_ctx} and were truncated by Ollama.")
    print(f"Index written to {out}")


# --------------------------------------------------------------------------- #
# query
# --------------------------------------------------------------------------- #

def retrieve(emb, chunks, em, queries, k, reranker=None, depth=30):
    """Dense top-k, optionally reranked. Returns (dense_rows, dense_scores, rerank_rows|None, rerank_scores|None)."""
    top, sc = search(emb, em.encode_queries(queries), max(k, depth if reranker else k))
    if not reranker:
        return top, sc, None, None
    rr_rows, rr_scores = [], []
    for qi, q in enumerate(queries):
        cand = list(top[qi][:depth])
        s = reranker.score(q, [chunks[i]["text"] for i in cand])
        order = np.argsort(-np.asarray(s), kind="stable")
        rr_rows.append([cand[o] for o in order])
        rr_scores.append([s[o] for o in order])
        if len(queries) > 1:
            print(f"\r  reranked {qi + 1}/{len(queries)}", end="", file=sys.stderr)
    if len(queries) > 1:
        print(file=sys.stderr)
    return top, sc, rr_rows, rr_scores


def cmd_query(args):
    idx = Path(args.index)
    chunks, emb, meta = load_index(idx)
    em = make_embedder(meta["model"], idx, args, meta)
    rr = Reranker(args.rerank, args.ollama_host, args.device) if args.rerank else None

    log = idx / "query_log.md"

    def ask(q):
        top, sc, rtop, rsc = retrieve(emb, chunks, em, [q], args.k, rr, args.rerank_depth)
        lines = [f"\n## {q}\n"]
        rows, scores = (rtop[0], rsc[0]) if rr else (top[0], sc[0])
        dense_rank = {int(i): r for r, i in enumerate(top[0], 1)}
        for r, (i, s) in enumerate(zip(rows[:args.k], scores[:args.k]), 1):
            c = chunks[i]
            where = f"{c['file']}  page {', '.join(map(str, c['pages']))}"
            if c.get("section_type"):
                where += f"  [{c['section_type']}]"
            if c.get("ocr_conf") is not None:
                where += f"  ocr-conf {c['ocr_conf']:.0f}"
            extra = f"  (dense rank {dense_rank.get(int(i))})" if rr else ""
            print(f"\n#{r}  score {s:.3f}{extra}  {where}")
            body = c["raw"] if args.chars <= 0 else c["raw"][:args.chars] + ("…" if len(c["raw"]) > args.chars else "")
            print("    " + body.replace("\n", "\n    "))
            lines.append(f"**#{r}** score {s:.3f}{extra} · {where}\n\n" + "\n".join("> " + l for l in c["raw"].splitlines()) + "\n")
        with open(log, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))

    if args.text:
        ask(" ".join(args.text))
        return
    print(f"Index {idx}: {len(chunks)} chunks, model {meta['model']}. Type a question (empty line or Ctrl-D/Ctrl-Z to quit).\n"
          f"Full results are also appended to {log} (open it in VS Code or Notepad to read Arabic properly).")
    while True:
        try:
            q = input("\nسؤال> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        ask(q)


# --------------------------------------------------------------------------- #
# diagnose
# --------------------------------------------------------------------------- #

def _pair_stats(S, mask):
    v = S[mask]
    return (float(v.mean()), float(np.median(v)), int(v.size)) if v.size else (float("nan"), float("nan"), 0)


def cmd_diagnose(args):
    idx = Path(args.index)
    chunks, emb, meta = load_index(idx)
    rep = idx / "report"
    rep.mkdir(exist_ok=True)
    n = len(chunks)
    L = []  # markdown lines

    L += ["# Embedding-space diagnostics", "",
          f"- index: `{idx}`", f"- model: `{meta['model']}` (dim {meta['dim']}, max tokens {meta['max_tokens']})",
          f"- chunking: `{meta['chunking']}` scope=`{meta['chunk_scope']}` size={meta['chunk_size']} overlap={meta['chunk_overlap']}"
          f" normalize={meta['normalize']} contextual_header={meta['contextual_header']}",
          f"- chunks: {n}", ""]

    # ---- extraction
    L += ["## 1. Extraction", "", "| file | pages | chars | article headings | flags |", "|---|---|---|---|---|"]
    for e in meta["extraction"]:
        fl = ", ".join(f"{k[6:]}={v}" for k, v in e.items() if k.startswith("pages_")) or "-"
        L.append(f"| {e['file']} | {e['pages']} | {e['chars']} | {e['articles_detected']} | {fl} |")
    L += ["", "> `likely_scanned_needs_ocr` pages contribute nothing to the index. `rtl_word_order_rebuilt` / "
              "`visual_order_fixed` mean the PDF stored Arabic in drawing order and the text was reordered; always eyeball "
              "a few pages in `text/` (ingest with `--dump-text`) before trusting anything below.", ""]

    # ---- OCR quality
    prow_path = idx / "pages.jsonl"
    prow = [json.loads(l) for l in open(prow_path, encoding="utf-8")] if prow_path.exists() else []
    ocrd = [r for r in prow if r.get("ocr_conf") is not None]
    if ocrd:
        L += ["### OCR", "", "| file | OCR'd pages | mean conf | conf < 70 | printed page numbers read | of those, correct |",
              "|---|---|---|---|---|---|"]
        for e in meta["extraction"]:
            if not e.get("ocr_pages"):
                continue
            pn = e.get("page_numbers") or {}
            low = sum(1 for r in ocrd if r["file"] == e["file"] and r["ocr_conf"] < 70)
            cons = "-" if pn.get("consistent") is None else f"{100 * pn['consistent']:.0f}%"
            L.append(f"| {e['file']} | {e['ocr_pages']} | {e['ocr_mean_conf']} | {low} | "
                     f"{pn.get('pages_with_number', 0)}/{e['ocr_pages']} | {cons} |")
        L += ["", "> Printed page numbers should run in step with the PDF page index, so the last two columns are a "
                  "free test of how well the OCR reads digits. Read few or got them wrong ⇒ article numbers "
                  "(`مادة ٥`) are unreliable too, and so are article labels and any `المادة N` lookup.", ""]
        worst = sorted(ocrd, key=lambda r: r["ocr_conf"])[:args.top]
        L += ["Lowest-confidence pages (look at these first):", "", "| file | page | conf | restored | stroke px | specks | flags |",
              "|---|---|---|---|---|---|---|"]
        for r in worst:
            dg = r.get("degradation") or {}
            fl = ", ".join(f for f in r["flags"] if not f.startswith("ocr:") and f != "ocr_restored") or "-"
            L.append(f"| {r['file']} | {r['page']} | {r['ocr_conf']} | {'yes' if 'ocr_restored' in r['flags'] else 'no'} | "
                     f"{dg.get('stroke', '-')} | {dg.get('specks', '-')} | {fl} |")
        dis = [r for r in ocrd if "vlm_tesseract_disagree" in r["flags"]]
        if dis:
            L += ["", f"**{len(dis)} pages where the vision model and Tesseract disagree** (agreement < 0.8): "
                      + ", ".join(f"{r['file']} p{r['page']}" for r in dis[:15])
                      + ". Check them by eye: the vision model may have invented text."]
        if "ocr_conf" in chunks[0] or any("ocr_conf" in c for c in chunks):
            cc = np.array([c.get("ocr_conf", np.nan) for c in chunks], dtype=float)
            ok = ~np.isnan(cc)
            if ok.sum() > 10:
                Sx = emb @ emb.T
                np.fill_diagonal(Sx, -np.inf)
                nnsim = Sx.max(axis=1)
                r = np.corrcoef(cc[ok], nnsim[ok])[0, 1]
                L += ["", f"- correlation between a chunk's OCR confidence and its nearest-neighbour similarity: {r:+.2f} "
                          "(clearly positive ⇒ noisy OCR chunks drift away from everything else in embedding space "
                          "and become hard to retrieve)"]
        L.append("")

    # ---- chunk/structure alignment
    lens = np.array([c["n_chars"] for c in chunks])
    art_to_chunks = defaultdict(set)
    for c in chunks:
        for a in c["articles"]:
            art_to_chunks[a].add(c["id"])
    real_arts = [a for a in art_to_chunks if not a.endswith("::preamble")]
    split_arts = [a for a in real_arts if len(art_to_chunks[a]) > 1]
    multi = [c for c in chunks if len([a for a in c["articles"] if not a.endswith("preamble")]) > 1]
    L += ["## 2. Chunk ↔ article alignment", "",
          f"- chunk length (chars): min {lens.min()}, median {int(np.median(lens))}, max {lens.max()}",
          f"- articles detected: {len(real_arts)}",
          f"- articles split across >1 chunk: {len(split_arts)} ({100*len(split_arts)/max(1,len(real_arts)):.0f}%)",
          f"- chunks mixing >1 article: {len(multi)} ({100*len(multi)/n:.0f}%)"]
    if "n_tokens" in chunks[0] and meta["max_tokens"]:
        toks = np.array([c["n_tokens"] for c in chunks])
        L.append(f"- tokens/chunk: median {int(np.median(toks))}, max {toks.max()}; "
                 f"truncated (> {meta['max_tokens']}): {(toks > meta['max_tokens']).sum()}; "
                 f"chars/token ≈ {lens.sum()/toks.sum():.2f}")
    if meta.get("chars_per_token"):
        L.append(f"- Ollama: ≈ {meta['chars_per_token']:.2f} chars/token overall, so a {meta['chunk_size']}-char chunk "
                 f"≈ {meta['chunk_size'] / meta['chars_per_token']:.0f} tokens; chunks truncated at "
                 f"num_ctx={meta['num_ctx']}: {meta.get('truncated_chunks') or 0}")
    if any(c.get("section_type") for c in chunks):
        st = Counter(c.get("section_type") or "-" for c in chunks)
        L.append("- chunks by section type: " + ", ".join(f"{k} {v}" for k, v in st.most_common()))
        secs_per_chunk = [c for c in chunks if c.get("section")]
        L.append(f"- sections detected: {len({c['section'] for c in secs_per_chunk})}")
    L += ["", "> High split/mix rates mean one retrieved chunk ≠ one legal unit: answers can be cut mid-article, "
              "or an article's meaning gets averaged with its neighbour's.", ""]

    # ---- geometry
    S = emb @ emb.T
    np.fill_diagonal(S, np.nan)
    art_ok, art_why = articles_trustworthy(chunks)
    has_sec = any(c.get("section") for c in chunks)
    use_sec = args.unit == "section" or (args.unit == "auto" and has_sec and not art_ok)
    unit = "section" if use_sec else "article"
    prim = np.array([(c.get("section") or "none") if use_sec else c["primary_article"] for c in chunks])
    files = np.array([c["file"] for c in chunks])
    same_art = (prim[:, None] == prim[None, :])
    same_file = (files[:, None] == files[None, :])
    offdiag = ~np.eye(n, dtype=bool)
    valid = ~np.isnan(S)
    groups = {
        f"same {unit}": same_art & offdiag,
        f"same file, different {unit}": same_file & ~same_art & offdiag,
        "different file": ~same_file,
        "all pairs": offdiag,
    }
    L += ["## 3. Geometry", "",
          f"Structural unit used below: **{unit}** ({art_why}"
          + ("; using heading-based sections instead)" if use_sec and args.unit == "auto" else ")"), "",
          "| pair type | mean cos | median cos | pairs |", "|---|---|---|---|"]
    gstats = {}
    for g, m in groups.items():
        mu, med, cnt = _pair_stats(S, m & valid)
        gstats[g] = mu
        L.append(f"| {g} | {mu:.3f} | {med:.3f} | {cnt} |")
    L += ["", f"- **anisotropy baseline** (mean cos of random pairs) = {gstats['all pairs']:.3f}. "
              "Absolute scores mean little; only the gaps between rows matter.",
          f"- gap same-{unit} vs same-file-other-{unit} = "
          f"{gstats[f'same {unit}'] - gstats[f'same file, different {unit}']:+.3f} "
          f"(near 0 ⇒ the model barely separates {unit}s inside a document)",
          f"- gap same-file vs different-file = "
          f"{gstats[f'same file, different {unit}'] - gstats['different file']:+.3f} "
          f"(large ⇒ embeddings cluster by *law/document* more than by topic)", ""]

    # nearest neighbour agreement
    Sn = np.where(np.isnan(S), -np.inf, S)
    nn = Sn.argmax(axis=1)
    prim_count = Counter(prim)
    elig = [i for i in range(n) if prim_count[prim[i]] > 1 and not prim[i].endswith("preamble")]
    nn_same_art = np.mean([prim[i] == prim[nn[i]] for i in elig]) if elig else float("nan")
    nn_same_file = float(np.mean(files == files[nn]))
    L += [f"- for chunks whose {unit} spans ≥2 chunks ({len(elig)}), nearest neighbour is a sibling chunk "
          f"of the same {unit}: {100*nn_same_art:.0f}%",
          f"- nearest neighbour is from the same file: {100*nn_same_file:.0f}%", ""]

    # ---- hubs
    k_occ = min(10, n - 1)
    knn = np.argsort(-Sn, axis=1)[:, :k_occ]
    occ = np.bincount(knn.ravel(), minlength=n)
    expected = k_occ
    hubs = np.argsort(-occ)[:args.top]
    L += ["## 4. Hubs (chunks that show up in everyone's top-10)", "",
          f"Expected k-occurrence ≈ {expected}. Hubs win retrieval for many unrelated queries — often "
          "boilerplate like publication/enforcement clauses (يُنشر هذا القانون في الجريدة الرسمية…).", "",
          *(["_Corpus too small (<30 chunks) for hubness to mean anything._", ""] if n < 30 else []),
          f"| chunk | k-occ | file | {unit} | text |", "|---|---|---|---|---|"]
    for i in hubs:
        c = chunks[i]
        L.append(f"| {i} | {occ[i]} | {c['file']} | {_unit_label(c, unit)} | "
                 f"{_snip(c['raw'], 90)} |")
    L.append("")

    # ---- near duplicates across articles
    iu = np.triu_indices(n, 1)
    sv = S[iu]
    order = np.argsort(-np.nan_to_num(sv, nan=-1))
    dups = []
    for o in order:
        if sv[o] < args.dup_threshold:
            break
        i, j = iu[0][o], iu[1][o]
        if prim[i] != prim[j]:
            dups.append((i, j, sv[o]))
        if len(dups) >= args.top:
            break
    L += ["## 5. Near-duplicates across different articles", "",
          f"Pairs with cos ≥ {args.dup_threshold} that belong to different articles (confusable at retrieval time — "
          "amendments, repeated formulas, same article number in different laws).", ""]
    if dups:
        L += ["| cos | chunk A | chunk B |", "|---|---|---|"]
        for i, j, s in dups:
            L.append(f"| {s:.3f} | {_unit_label(chunks[i], unit)}: {_snip(chunks[i]['raw'], 60)} | "
                     f"{_unit_label(chunks[j], unit)}: {_snip(chunks[j]['raw'], 60)} |")
    else:
        L.append("_none_")
    L.append("")

    # ---- article-number probe
    em = None
    if real_arts:
        em = make_embedder(meta["model"], idx, args, meta)
        labels = sorted({a.split("::", 1)[1].split("#")[0] for a in real_arts},
                        key=lambda x: (len(x), x))[: args.probe_articles]
        qs = [f"المادة {lab}" for lab in labels]
        top, _ = search(emb, em.encode_queries(qs), 5)
        hits1 = hits5 = 0
        for lab, row in zip(labels, top):
            ok = [any(a.split("::", 1)[1].split("#")[0] == lab for a in chunks[i]["articles"]) for i in row]
            hits1 += ok[0]
            hits5 += any(ok)
        L += ["## 6. Article-number probe", "",
              f"Query `المادة N` for {len(labels)} article numbers; does a chunk containing that article come back?", "",
              f"- hit@1 = {hits1/len(labels):.2f}, hit@5 = {hits5/len(labels):.2f}", "",
              "> Dense embeddings usually encode numbers weakly. If this is low, users asking “ما نص المادة 15” "
              "need metadata filtering or a lexical (BM25) leg, not a better embedding.", ""]

    # ---- 2-D map
    if not args.no_map:
        _write_map(rep / "embedding_map.html", emb, chunks)
        L += ["## 7. Map", "", "Open `embedding_map.html`: UMAP projection, coloured by file and by article. "
              "Hover to read each chunk. Look for: tight per-law blobs (document dominates topic), "
              "articles on the same subject from different laws landing together (good), "
              "or isolated junk (headers/footers, tables of contents).", ""]

    # ---- CSV
    with open(rep / "chunks.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "file", "pages", "primary_article", "articles", "n_chars", "n_tokens",
                    "k_occurrence", "nn_id", "nn_cos", "nn_article", "text"])
        for c in chunks:
            i = c["id"]
            w.writerow([i, c["file"], " ".join(map(str, c["pages"])), c["primary_article"], " | ".join(c["articles"]),
                        c["n_chars"], c.get("n_tokens", ""), occ[i], nn[i], f"{Sn[i, nn[i]]:.4f}",
                        chunks[nn[i]]["primary_article"], c["raw"]])

    (rep / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nReport: {rep/'report.md'}\nCSV:    {rep/'chunks.csv'}" + ("" if args.no_map else f"\nMap:    {rep/'embedding_map.html'}"))


def articles_trustworthy(chunks) -> tuple[bool, str]:
    """Article labels are usable only if articles cover most chunks AND the text isn't mostly OCR
    (OCR misreads Arabic-Indic digits, so 'مادة ١٠' may become 'مادة ٠١' or 'مادة ه')."""
    if not chunks:
        return False, "no chunks"
    cov = float(np.mean([not c["primary_article"].endswith("::preamble") for c in chunks]))
    ocr_share = float(np.mean([c.get("ocr_conf") is not None for c in chunks]))
    if ocr_share > 0.5:
        return False, f"{100 * ocr_share:.0f}% of chunks come from OCR, whose article numbers are unreliable"
    if cov < 0.5:
        return False, f"articles cover only {100 * cov:.0f}% of chunks"
    return True, f"articles cover {100 * cov:.0f}% of chunks"


def _unit_label(c, unit):
    if unit == "section":
        return f"{(c.get('section') or '-').split('::')[-1]} {c.get('section_type') or ''}".strip()
    return c["primary_article"].split("::", 1)[1]


def _snip(t, k):
    t = re.sub(r"\s+", " ", t).replace("|", "/")
    return t[:k] + ("…" if len(t) > k else "")


def _write_map(path: Path, emb, chunks):
    import plotly.graph_objects as go

    n = len(chunks)
    if n >= 15:
        import umap
        xy = umap.UMAP(n_neighbors=min(15, n - 1), min_dist=0.1, metric="cosine", random_state=42).fit_transform(emb)
        method = "UMAP"
    else:
        from sklearn.decomposition import PCA
        xy = PCA(2).fit_transform(emb)
        method = "PCA"

    hover = []
    for c in chunks:
        body = "<br>".join(re.findall(r".{1,70}(?:\s|$)", re.sub(r"\s+", " ", c["raw"][:420])))
        hover.append(f"<b>#{c['id']}</b> {c['file']} p{c['pages']}<br>article: {c['primary_article'].split('::',1)[1]}"
                     f"<br><br>{body}")

    def fig_by(key_fn, title, max_groups=25):
        keys = [key_fn(c) for c in chunks]
        top = [k for k, _ in Counter(keys).most_common(max_groups)]
        fig = go.Figure()
        for k in top + ["(other)"]:
            idx = [i for i, kk in enumerate(keys) if (kk == k if k != "(other)" else kk not in top)]
            if not idx:
                continue
            fig.add_trace(go.Scattergl(x=xy[idx, 0], y=xy[idx, 1], mode="markers", name=str(k)[:40],
                                       text=[hover[i] for i in idx], hoverinfo="text",
                                       marker=dict(size=8, opacity=0.8, color="#bbb" if k == "(other)" else None)))
        fig.update_layout(title=f"{title} ({method}, cosine)", height=700, template="plotly_white",
                          hoverlabel=dict(align="right", font=dict(family="Tahoma, Arial", size=13)))
        return fig

    f1 = fig_by(lambda c: c["file"], "Chunks coloured by file")
    f2 = fig_by(lambda c: c["primary_article"], "Chunks coloured by primary article (top 25)")
    html = ("<html><head><meta charset='utf-8'><title>Embedding map</title></head><body>"
            + f1.to_html(full_html=False, include_plotlyjs=True)
            + f2.to_html(full_html=False, include_plotlyjs=False) + "</body></html>")
    path.write_text(html, encoding="utf-8")


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #

def chunk_units(c, rel):
    """Which relevance units (from one query's labels) does this chunk satisfy?"""
    hit = set()
    norm = None
    for u_i, r in enumerate(rel):
        if "article" in r:
            want = str(r["article"]).translate(AR_DIGITS).strip()
            for a in c["articles"]:
                f, lab = a.split("::", 1)
                if lab.split("#")[0] == want and ("file" not in r or r["file"] == f):
                    hit.add(u_i)
        if "text" in r and ("file" not in r or r["file"] == c["file"]):
            norm = norm or norm_words(c["raw"])
            if norm_words(r["text"]) in norm:
                hit.add(u_i)
        if "page" in r and r.get("file") == c["file"] and r["page"] in c["pages"]:
            hit.add(u_i)
        if "pages" in r and r.get("file") == c["file"] and set(r["pages"]) & set(c["pages"]) and "text" not in r:
            hit.add(u_i)
        if "section" in r and c.get("section") == r["section"]:
            hit.add(u_i)
    return hit


def ranking_metrics(chunks, row, rel, K, n_rel_total):
    units = [chunk_units(chunks[i], rel) for i in row]
    first = next((r for r, u in enumerate(units, 1) if u), None)
    res = {}
    for k in K:
        got = set().union(*units[:k]) if units[:k] else set()
        res[f"hit@{k}"] = float(bool(got))
        res[f"recall@{k}"] = len(got) / len(rel)
        res[f"precision@{k}"] = sum(bool(u) for u in units[:k]) / k
    res["mrr"] = 1 / first if first else 0.0
    # nDCG@10, binary gain per chunk; ideal = all relevant chunks in the index ranked first
    dcg = sum(1 / math.log2(r + 1) for r, u in enumerate(units[:10], 1) if u)
    ideal = sum(1 / math.log2(r + 1) for r in range(1, min(10, n_rel_total) + 1))
    res["ndcg@10"] = dcg / ideal if ideal else 0.0
    return res, units, first


def cmd_eval(args):
    reranker = Reranker(args.rerank, args.ollama_host, args.device) if args.rerank else None
    summaries = {}
    for ix in args.index:
        summaries.update(eval_one(Path(ix), args, reranker))
    if len(summaries) > 1:
        ms = list(next(iter(summaries.values())).keys())
        w = max(len(n) for n in summaries) + 2
        print("\n=== comparison ===\n" + "metric".ljust(14) + "".join(n.rjust(w) for n in summaries))
        for m in ms:
            print(m.ljust(14) + "".join(f"{summaries[n][m]:.3f}".rjust(w) for n in summaries))


def eval_one(idx: Path, args, reranker=None):
    chunks, emb, meta = load_index(idx)
    qs = [json.loads(l) for l in open(args.queries, encoding="utf-8") if l.strip() and not l.lstrip().startswith("//")]
    em = make_embedder(meta["model"], idx, args, meta)
    K = sorted(set(args.k))
    kmax = max(K + [10])
    top, _, rtop, _ = retrieve(emb, chunks, em, [q["query"] for q in qs], kmax, reranker, args.rerank_depth)

    systems = {idx.name: [list(r[:kmax]) for r in top]}
    if reranker:
        systems[f"{idx.name}+rerank"] = [r[:kmax] for r in rtop]
    agg = {s: defaultdict(list) for s in systems}
    ceiling = []
    rows = []
    for qi, q in enumerate(qs):
        rel = q["relevant"]
        n_rel_total = sum(bool(chunk_units(c, rel)) for c in chunks)
        detail = dict(query=q["query"], relevant=rel)
        if n_rel_total == 0:
            detail["warning"] = "no chunk in the index matches this query's labels - check the label"
        for s, ranked in systems.items():
            res, units, first = ranking_metrics(chunks, ranked[qi], rel, K, n_rel_total)
            for m, v in res.items():
                agg[s][m].append(v)
            detail[s] = dict(first_hit_rank=first, **{m: round(v, 4) for m, v in res.items()},
                             top=[dict(rank=r, file=chunks[i]["file"],
                                       article=chunks[i]["primary_article"].split("::", 1)[1],
                                       relevant=bool(u), ocr_conf=chunks[i].get("ocr_conf"),
                                       text=_snip(chunks[i]["raw"], 120))
                                  for r, (i, u) in enumerate(zip(ranked[qi][:5], units[:5]), 1)])
        if reranker:
            got = set().union(*[chunk_units(chunks[i], rel) for i in top[qi][:args.rerank_depth]])
            ceiling.append(len(got) / len(rel))
        rows.append(detail)

    print(f"\nIndex: {idx}  model={meta['model']}  chunking={meta['chunking']}  queries={len(qs)}"
          + (f"  rerank={reranker.spec} (depth {args.rerank_depth})" if reranker else ""))
    names = list(systems)
    print("\n  " + "metric".ljust(14) + "".join(n[-22:].rjust(24) for n in names))
    summary = {s: {m: float(np.mean(v)) for m, v in agg[s].items()} for s in systems}
    for m in summary[names[0]]:
        print("  " + m.ljust(14) + "".join(f"{summary[s][m]:.3f}".rjust(24) for s in names))
    if reranker:
        print(f"\n  recall@{args.rerank_depth} of the dense stage (ceiling for the reranker): {np.mean(ceiling):.3f}")

    main_sys = names[-1]
    fails = [r for r in rows if not r[main_sys]["first_hit_rank"] or r[main_sys]["first_hit_rank"] > max(K)]
    if fails:
        print(f"\n{len(fails)} queries with no relevant chunk in top-{max(K)} ({main_sys}):")
        for r in fails[: args.show_failures]:
            print(f"\n  Q: {r['query']}   [want: {', '.join(str(x.get('article', x.get('text', x.get('page')))) for x in r['relevant'])}]")
            for t in r[main_sys]["top"][:3]:
                oc = f" [ocr {t['ocr_conf']}]" if t.get("ocr_conf") is not None else ""
                print(f"     #{t['rank']} {t['file']} م{t['article']}{oc}: {t['text'][:90]}")
    bad = [r for r in rows if "warning" in r]
    if bad:
        print(f"\nWARNING: {len(bad)} queries have labels that match no chunk at all (e.g. {bad[0]['query'][:60]!r}).")
    out = idx / ("eval_results_rerank.json" if reranker else "eval_results.json")
    out.write_text(json.dumps(dict(summary=summary, per_query=rows), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nPer-query details: {out}")
    return summary


# --------------------------------------------------------------------------- #
# gen-queries: synthetic, auto-labelled evaluation questions from an LLM
# --------------------------------------------------------------------------- #

GEN_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {
        "type": "object",
        "properties": {"question": {"type": "string"}, "style": {"type": "string", "enum": ["everyday", "legal"]}},
        "required": ["question", "style"]}}},
    "required": ["questions"],
}

GEN_PROMPT = """Below is a passage from an Egyptian legal source (source file: {file}). It may be a law article,
an explanatory memorandum, legal commentary or a summary of court rulings. Its text may contain OCR errors.

Write {k} different questions in Arabic that this passage answers.
Rules:
- Ask the way a real person would: a citizen, a lawyer or a law student. Use your own words.
  Do NOT copy phrases from the article, and do NOT mention the article number or the law's name or number.
- Each question must be answerable mainly from THIS passage. Ignore OCR noise; don't ask about garbled words or numbers.
- Mix styles: "everyday" = a plain question in Modern Standard Arabic or Egyptian colloquial about a real situation;
  "legal" = a precise question using legal terminology.
Return JSON: {{"questions": [{{"question": "...", "style": "everyday" | "legal"}}]}}

Passage:
\"\"\"
{text}
\"\"\""""


def _word_overlap(q, text):
    qw = {w for w in normalize_for_match(q).split() if len(w) > 2}
    tw = set(normalize_for_match(text).split())
    return len(qw & tw) / max(1, len(qw))


def _anchor_phrase(text: str, corpus_norm: str, n_words: int = 7) -> str | None:
    """A short digit-free phrase from the middle of `text` that occurs exactly once in the corpus.
    Used as the relevance label when article numbers can't be trusted (OCR)."""
    words = norm_words(text).split()
    if len(words) < n_words + 4:
        return None
    mid = len(words) // 2
    for off in [0, 5, -5, 10, -10, 15, -15, 20, -20]:
        i = mid + off
        if i < 0 or i + n_words > len(words):
            continue
        w = words[i:i + n_words]
        if any(ch.isdigit() for t in w for ch in t) or sum(len(t) > 2 for t in w) < 4:
            continue
        phrase = " ".join(w)
        if corpus_norm.count(phrase) == 1:
            return phrase
    return None


def cmd_gen_queries(args):
    import random

    idx = Path(args.index)
    load = lambda n: [json.loads(l) for l in open(idx / n, encoding="utf-8")] if (idx / n).exists() else []  # noqa: E731
    arts, secs = load("articles.jsonl"), load("sections.jsonl")
    chunks, _, _ = load_index(idx)
    art_ok, art_why = articles_trustworthy(chunks)
    units = args.units
    if units == "auto":
        units = "articles" if (art_ok or not secs) else "sections"
        print(f"units=auto -> {units} ({art_why})", file=sys.stderr)
    if units == "sections":
        # split long sections into ~1500-char passages so each question has a focused source
        pool = []
        for sct in secs:
            if sct["level"] == 1 or sct["type"] == "الصيغة (نموذج)":
                continue   # container title lines and blank form templates make poor question sources
            t = sct["text"]
            for a, b in recursive_split(t, 1500, 0):
                pool.append(dict(key=sct["key"], file=sct["file"], text=t[a:b].strip(), pages=sct["pages"],
                                 label=f"{sct['container'][:40]} › {sct['type']}"))
        arts = pool
    print(f"Question sources: {units} ({len(arts)} candidates)", file=sys.stderr)
    corpus_norm = " ".join(norm_words(c["raw"]) for c in chunks)
    arts = [a for a in arts if len(a["text"]) >= args.min_chars]
    random.Random(args.seed).shuffle(arts)
    arts = arts[: args.n_articles]
    model = args.llm.split(":", 1)[1] if args.llm.startswith("ollama:") else args.llm
    host = ollama_host(args.ollama_host)
    n_written = n_dropped = 0
    with open(args.out, "a" if args.append else "w", encoding="utf-8") as f:
        for n, a in enumerate(arts, 1):
            msgs = [{"role": "system", "content": "You write evaluation questions for a search system over Egyptian legislation."},
                    {"role": "user", "content": GEN_PROMPT.format(file=a["file"], k=args.per_article, text=a["text"][:4000])}]
            try:
                data = parse_json_loose(ollama_chat(host, model, msgs, GEN_SCHEMA, temperature=0.7, num_ctx=args.num_ctx))
            except ApiError as e:
                if e.status in (0, 404):
                    sys.exit(f"LLM call failed: {e}" + (f"\nRun `ollama pull {model}`." if e.status == 404 else ""))
                print(f"\n  ! skipped {a['key']}: {e}", file=sys.stderr)
                continue
            if "article" in a:
                label = a["article"].split("#")[0]
                relevant = [dict(file=a["file"], article=label)]
            else:
                label = None
                anchor = _anchor_phrase(a["text"], corpus_norm)
                relevant = [dict(file=a["file"], text=anchor)] if anchor else [dict(file=a["file"], pages=a["pages"])]
            for q in (data.get("questions") or [])[: args.per_article]:
                text = str(q.get("question", "")).strip()
                leak = label and re.search(rf"ماد[ةه]\s*\(?\s*{re.escape(label.split()[0])}\b", normalize_for_match(text))
                if len(text) < 8 or leak:
                    n_dropped += 1
                    continue
                rec = dict(query=text, relevant=relevant, style=q.get("style"),
                           overlap=round(_word_overlap(text, a["text"]), 3), source=f"synthetic:{model}",
                           unit=a.get("label") or a.get("key"))
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1
            print(f"\r  {n}/{len(arts)} sources, {n_written} questions", end="", file=sys.stderr)
    print(f"\nWrote {n_written} questions to {args.out}" + (f" ({n_dropped} dropped: empty or leaked the article number)" if n_dropped else ""))
    print("Skim them before trusting the numbers: delete vague ones, and where another passage also answers a "
          "question, add it to 'relevant'.")


# --------------------------------------------------------------------------- #
# ocr-check: compare OCR engines on a few pages before committing to one
# --------------------------------------------------------------------------- #

def _parse_pages(spec: str, n: int) -> list[int]:
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        elif part.strip():
            out.append(int(part))
    return [p for p in out if 1 <= p <= n]


def cmd_ocr_check(args):
    import time

    import ocr as ocr_mod
    from compat import norm_distance, open_pdf
    from ocr import PageOCR, _norm, page_number_check, printed_page_number

    ocr_mod.TESSERACT_CMD = ocr_mod.find_tesseract(args.tesseract)
    if not ocr_mod.TESSERACT_CMD:
        sys.exit("Tesseract not found (needed by every engine for cross-checking). Install it or pass --tesseract PATH.")
    pdf = Path(args.pdf)
    with open_pdf(pdf) as d:
        pages = _parse_pages(args.pages, len(d))
    gold = {}
    if args.gold_dir:
        for g in Path(args.gold_dir).glob("*.txt"):
            m = re.search(r"p(\d+)\.txt$", g.name)
            if m and int(m.group(1)) in pages:
                gold[int(m.group(1))] = g.read_text(encoding="utf-8")
    outdir = Path(args.out)
    rows = []
    for eng in args.engines:
        o = PageOCR(eng, args.ocr_restore, args.ocr_lang, args.ocr_dpi, args.ocr_cache, host=args.ollama_host)
        t0 = time.time()
        res = o.ocr_pages(pdf, pages, workers=args.ocr_workers,
                          progress=lambda i, n: print(f"\r  {eng}: {i}/{n}", end="", file=sys.stderr))
        print(file=sys.stderr)
        dt = (time.time() - t0) / len(pages)
        labels = {p: printed_page_number(_clean(r.text)) for p, r in res.items()}
        pn = page_number_check(labels)
        if args.page_offset is not None:
            right = sum(labels[p] == p + args.page_offset for p in pages)
        else:
            right = round((pn.get("consistent") or 0) * pn.get("pages_with_number", 0))
        cers = {p: norm_distance(_norm(res[p].text), _norm(g)) for p, g in gold.items()}
        confs = [r.conf for r in res.values() if r.conf is not None]
        agr = [r.agreement for r in res.values() if r.agreement is not None]
        flagged = sum(any(f.startswith("vlm_") for f in r.flags) for r in res.values())
        rows.append(dict(engine=eng, sec_per_page=dt, page_numbers=f"{right}/{len(pages)}",
                         cer=np.mean(list(cers.values())) if cers else None,
                         conf=np.mean(confs) if confs else None, agreement=np.mean(agr) if agr else None,
                         flagged=flagged, restored=sum(r.restored for r in res.values())))
        d = outdir / re.sub(r"[^\w.-]", "_", eng)
        d.mkdir(parents=True, exist_ok=True)
        for p, r in res.items():
            (d / f"p{p:04d}.txt").write_text(r.text, encoding="utf-8")
    print(f"\n{pdf.name}: {len(pages)} pages" + (f", {len(gold)} with ground truth" if gold else "")
          + (f", printed page = pdf page + {args.page_offset}" if args.page_offset is not None else ""))
    print(f"\n{'engine':<30}{'page nos right':>15}{'CER vs gold':>13}{'tess conf':>11}{'agree w/ tess':>15}"
          f"{'flagged':>9}{'restored':>10}{'s/page':>8}")
    fmt = lambda v, f: "-" if v is None else format(v, f)  # noqa: E731
    for r in rows:
        print(f"{r['engine']:<30}{r['page_numbers']:>15}{fmt(r['cer'], '.3f'):>13}{fmt(r['conf'], '.1f'):>11}"
              f"{fmt(r['agreement'], '.3f'):>15}{r['flagged']:>9}{r['restored']:>10}{r['sec_per_page']:>8.1f}")
    print(f"\nPer-page text for side-by-side reading: {outdir}/<engine>/pNNNN.txt"
          "\n'page nos right' is a free test of digit reading: article numbers (مادة ٥) fail the same way."
          "\ns/page is near 0 for cached pages.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            if not stream.isatty():
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default=None, help="sentence-transformers device (cpu, cuda, mps)")
    common.add_argument("--ollama-host", default=None, help="default $OLLAMA_HOST or http://localhost:11434")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", parents=[common], help="extract, chunk, embed")
    p.add_argument("--pdf", nargs="+", help="one or more PDF files")
    p.add_argument("--pdf-dir", help="a folder of PDFs (searched recursively)")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="ollama:bge-m3",
                   help="ollama:<name> (default ollama:bge-m3; also ollama:qwen3-embedding:0.6b), "
                        "a sentence-transformers id such as BAAI/bge-m3, or 'tfidf' (offline lexical baseline)")
    p.add_argument("--num-ctx", type=int, default=8192, help="Ollama context window for embedding (tokens)")
    p.add_argument("--query-instruction", default=None,
                   help="override the query instruction (qwen3-embedding gets an Egyptian-law instruction by default; "
                        "pass '' to disable)")
    p.add_argument("--chunking", choices=["fixed", "article", "section"], default="fixed",
                   help="fixed = vanilla recursive character splitter (default); article = one chunk per مادة; "
                        "section = one chunk per heading-delimited section (see sections.py), long ones sub-split")
    p.add_argument("--headings", default=None,
                   help="custom heading list for section detection: lines of 'level|phrase|type' (see sections.py)")
    p.add_argument("--chunk-scope", choices=["page", "document"], default="page",
                   help="fixed chunking per page (like PyPDFLoader+splitter, default) or across the whole file")
    p.add_argument("--chunk-size", type=int, default=1000, help="characters")
    p.add_argument("--chunk-overlap", type=int, default=150, help="characters")
    p.add_argument("--normalize", action="store_true", help="strip tashkeel/tatweel before embedding")
    p.add_argument("--contextual-header", action="store_true",
                   help="prepend 'file | container | section type | المادة N' to each chunk before embedding")
    p.add_argument("--extract", choices=["auto", "plain", "rtl"], default="auto",
                   help="plain = PyMuPDF text as stored; rtl = rebuild lines right-to-left from word positions; "
                        "auto (default) = per page, whichever reads more like natural Arabic")
    p.add_argument("--dump-text", action="store_true", help="write extracted text per PDF to OUT/text/ for inspection")
    p.add_argument("--ocr", choices=["auto", "force", "off"], default="auto",
                   help="auto (default) = OCR pages without a text layer; force = OCR every page; off")
    p.add_argument("--ocr-engine", default="tesseract",
                   help="tesseract (default) or ollama:<vision model>, e.g. ollama:qwen2.5vl:7b (cross-checked with Tesseract)")
    p.add_argument("--ocr-restore", choices=["auto", "on", "off"], default="auto",
                   help="stroke restoration for faded print; auto = OCR raw and restored, keep the more confident")
    p.add_argument("--ocr-lang", default="ara", help="Tesseract languages, e.g. ara or ara+eng")
    p.add_argument("--ocr-dpi", type=int, default=300)
    p.add_argument("--ocr-workers", type=int, default=2, help="parallel Tesseract pages")
    p.add_argument("--ocr-cache", default=".ocr_cache", help="OCR results are cached here and reused across ingests")
    p.add_argument("--tesseract", default=None, help="path to tesseract(.exe) if it isn't found automatically")
    p.add_argument("--batch-size", type=int, default=16)
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("query", parents=[common], help="search the index")
    p.add_argument("--index", required=True)
    p.add_argument("text", nargs="*", help="question; omit it to get an interactive prompt")
    p.add_argument("-k", type=int, default=5, help="number of chunks to return")
    p.add_argument("--chars", type=int, default=600, help="characters of each chunk to print (0 = all)")
    p.add_argument("--rerank", default=None, help="reranker spec, see backends.py (e.g. http://localhost:8081/v1/rerank)")
    p.add_argument("--rerank-depth", type=int, default=30, help="dense candidates passed to the reranker")
    p.set_defaults(fn=cmd_query)

    p = sub.add_parser("diagnose", parents=[common], help="embedding-space report")
    p.add_argument("--index", required=True)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--dup-threshold", type=float, default=0.95)
    p.add_argument("--probe-articles", type=int, default=50)
    p.add_argument("--unit", choices=["auto", "article", "section"], default="auto",
                   help="structural unit for the geometry stats (auto: sections when articles cover <50%% of chunks)")
    p.add_argument("--no-map", action="store_true")
    p.set_defaults(fn=cmd_diagnose)

    p = sub.add_parser("eval", parents=[common], help="precision/recall against labelled queries")
    p.add_argument("--index", required=True, nargs="+", help="one or more index dirs to compare")
    p.add_argument("--queries", required=True, help="JSONL; see tests/queries.sample.jsonl")
    p.add_argument("-k", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument("--rerank", default=None, help="also score dense+rerank; reranker spec, see backends.py")
    p.add_argument("--rerank-depth", type=int, default=30)
    p.add_argument("--show-failures", type=int, default=10)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("gen-queries", parents=[common], help="generate labelled test questions with a local LLM")
    p.add_argument("--index", required=True, help="any index built from your PDFs (uses its articles.jsonl)")
    p.add_argument("--llm", required=True, help="Ollama chat model, e.g. qwen2.5:7b")
    p.add_argument("--out", default="queries.synthetic.jsonl")
    p.add_argument("--n-articles", type=int, default=50)
    p.add_argument("--per-article", type=int, default=2)
    p.add_argument("--min-chars", type=int, default=120, help="skip sources shorter than this")
    p.add_argument("--units", choices=["auto", "articles", "sections"], default="auto",
                   help="what to write questions about; auto = sections when articles cover <50%% of chunks "
                        "(typical for OCR'd books whose article numbers are unreliable)")
    p.add_argument("--num-ctx", type=int, default=8192)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--append", action="store_true")
    p.set_defaults(fn=cmd_gen_queries)

    p = sub.add_parser("ocr-check", parents=[common], help="compare OCR engines on sample pages")
    p.add_argument("--pdf", required=True)
    p.add_argument("--pages", default="1-15", help="e.g. 3-20 or 3,10,25-30 (1-based)")
    p.add_argument("--engines", nargs="+", default=["tesseract"],
                   help="e.g. tesseract ollama:qwen2.5vl:7b ollama:qwen3-vl:8b")
    p.add_argument("--gold-dir", default=None, help="hand-checked transcriptions named *pN.txt (N = pdf page)")
    p.add_argument("--page-offset", type=int, default=None,
                   help="printed page number minus pdf page, if known (otherwise inferred)")
    p.add_argument("--out", default="ocr_check")
    p.add_argument("--ocr-restore", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--ocr-lang", default="ara")
    p.add_argument("--ocr-dpi", type=int, default=300)
    p.add_argument("--ocr-workers", type=int, default=2)
    p.add_argument("--ocr-cache", default=".ocr_cache")
    p.add_argument("--tesseract", default=None, help="path to tesseract(.exe)")
    p.set_defaults(fn=cmd_ocr_check)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
