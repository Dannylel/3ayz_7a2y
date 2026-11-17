"""OCR stage for scanned Arabic PDFs.

Engines
    tesseract            local Tesseract (`ara`), fast. Good body text on clean print, but it often misreads
                         Arabic-Indic digits (٥ looks like ه), so article and page numbers are unreliable.
    ollama:<vision-llm>  a vision model served by Ollama (e.g. qwen2.5vl:7b, qwen3-vl:8b). Usually better on
                         digits and degraded print, but it can hallucinate fluent text. It is always
                         cross-checked against Tesseract, and pages where the two disagree are flagged.

Restoration (for faded / partially erased print)
    Binarise (Otsu, or CLAHE + adaptive threshold for real grayscale scans), close small gaps in
    strokes, thicken thin strokes, drop speckle. In `auto` mode both the raw and the restored image
    are OCR'd and the version with the higher Tesseract confidence wins.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
import re
import subprocess
import tempfile
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PAGE_NUM_RE = re.compile(r"^[\s\-–—_ـ.]*([0-9٠-٩]{1,4})[\s\-–—_ـ.]*$")
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

VLM_PROMPT = (
    "Transcribe ALL the text on this scanned page of an Arabic legal book exactly as printed.\n"
    "- Keep the original wording, spelling and punctuation. Do not correct, summarise, translate or complete anything.\n"
    "- Keep numbers exactly as printed, including Arabic-Indic digits (٠١٢٣٤٥٦٧٨٩).\n"
    "- One output line per printed line, top to bottom. Include the page number, headings and footnotes.\n"
    "- If a word is too faded or damaged to read, write [؟] in its place. Never guess missing words.\n"
    "Output only the transcription."
)


@dataclass
class OcrResult:
    text: str
    engine: str
    conf: float | None = None          # mean Tesseract word confidence 0-100 (None for VLM-only)
    restored: bool = False
    agreement: float | None = None     # VLM vs Tesseract character agreement 0-1
    flags: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# image helpers
# --------------------------------------------------------------------------- #

def restore(gray: np.ndarray) -> np.ndarray:
    import compat as cx
    mid = np.mean((gray > 40) & (gray < 215))
    ink = cx.adaptive_ink(gray) if mid > 0.05 else cx.otsu_ink(gray)   # real grayscale scan: even out fading
    ink = cx.close(ink, 3)          # bridge broken strokes
    ink = cx.dilate(ink, 2)         # thicken thin strokes
    ink = cx.remove_specks(ink, 4)
    return np.where(ink == 1, 0, 255).astype(np.uint8)


def degradation(gray: np.ndarray) -> dict:
    """Cheap page-quality signals: ink coverage, median stroke width, share of speck components."""
    import compat as cx
    ink = cx.otsu_ink(gray)
    cover = float(ink.mean())
    if cover < 0.003:
        return dict(ink=round(cover, 4), stroke=None, specks=None)
    stroke, specks = cx.stroke_width_and_specks(ink)
    return dict(ink=round(cover, 4), stroke=None if stroke is None else round(stroke, 2),
                specks=None if specks is None else round(specks, 3))


# --------------------------------------------------------------------------- #
# engines
# --------------------------------------------------------------------------- #

TESSERACT_CMD: str | None = None
WINDOWS_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
]


def find_tesseract(explicit: str | None = None) -> str | None:
    """--tesseract PATH, then $TESSERACT_CMD, then PATH, then the usual Windows install folders."""
    import shutil
    for cand in [explicit, os.environ.get("TESSERACT_CMD")]:
        if cand and Path(cand).is_file():
            return str(cand)
    found = shutil.which("tesseract")
    if found:
        return found
    for cand in WINDOWS_TESSERACT_PATHS:
        if cand and Path(cand).is_file():
            return cand
    return None


def tesseract_langs(cmd: str) -> tuple[list[str], str | None]:
    """(languages, error). error is set if the binary can't even run (e.g. blocked by policy)."""
    try:
        r = subprocess.run([cmd, "--list-langs"], capture_output=True, text=True, timeout=60)
    except OSError as e:
        return [], f"{type(e).__name__}: {e}"
    if r.returncode != 0 and not r.stdout:
        return [], (r.stderr or "").strip()[:300]
    lines = (r.stdout + "\n" + r.stderr).splitlines()
    return [l.strip() for l in lines if l.strip() and " " not in l.strip() and not l.lower().startswith("list")], None


def tesseract(gray: np.ndarray, lang: str = "ara", psm: int = 4) -> tuple[str, float, int]:
    import compat as cx
    cmd = TESSERACT_CMD or find_tesseract() or "tesseract"
    with tempfile.TemporaryDirectory() as td:
        img = Path(td) / "p.png"
        cx.write_image(img, gray)
        base = Path(td) / "out"
        r = subprocess.run([cmd, str(img), str(base), "-l", lang, "--psm", str(psm), "--dpi", "300",
                            "txt", "tsv"], capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            raise RuntimeError(f"tesseract failed: {r.stderr.strip()[:300]}")
        text = (base.with_suffix(".txt")).read_text(encoding="utf-8")
        confs = []
        with open(base.with_suffix(".tsv"), encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                try:
                    c = float(row.get("conf", -1))
                except (TypeError, ValueError):
                    continue
                if (row.get("text") or "").strip() and c >= 0:
                    confs.append(c)
    return text, (float(np.mean(confs)) if confs else 0.0), len(confs)


def vlm(gray: np.ndarray, model: str, host: str, max_side: int = 1800, num_ctx: int = 16384) -> str:
    import compat as cx
    from backends import post_json
    h, w = gray.shape
    img = cx.downscale(gray, min(1.0, max_side / max(h, w)))
    payload = dict(model=model, stream=False, options=dict(temperature=0, num_ctx=num_ctx),
                   messages=[{"role": "user", "content": VLM_PROMPT,
                              "images": [base64.b64encode(cx.png_bytes(img)).decode()]}])
    if re.search(r"qwen3", model, re.I):
        payload["think"] = False
    r = post_json(f"{host}/api/chat", payload, timeout=1800)
    return re.sub(r"<think>.*?</think>", "", r["message"]["content"], flags=re.S).strip()


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"[ً-ٰٟـ]", "", t)
    t = re.sub("[أإآٱ]", "ا", t).replace("ى", "ي").replace("ة", "ه").translate(AR_DIGITS)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def agreement(a: str, b: str) -> float:
    from compat import norm_distance
    return 1.0 - norm_distance(_norm(a), _norm(b))


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

class PageOCR:
    def __init__(self, engine="tesseract", restore_mode="auto", lang="ara", dpi=300,
                 cache_dir: str | Path = ".ocr_cache", host=None, min_agreement=0.8):
        self.engine, self.restore_mode, self.lang, self.dpi = engine, restore_mode, lang, dpi
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.host = host
        self.min_agreement = min_agreement

    def _key(self, pdf: Path, page_no: int) -> Path:
        st = pdf.stat()
        h = hashlib.sha1(f"{pdf.name}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:12]
        tag = re.sub(r"[^\w.-]", "_", f"{self.engine}_{self.restore_mode}_{self.lang}_{self.dpi}")
        return self.cache / h / f"p{page_no:04d}_{tag}.json"

    def ocr_page(self, pdf: Path, page_no: int) -> OcrResult:
        key = self._key(pdf, page_no)
        if key.exists():
            return OcrResult(**json.loads(key.read_text(encoding="utf-8")))
        from compat import open_pdf
        with open_pdf(pdf) as doc:
            gray = doc[page_no - 1].render_gray(self.dpi)
        if gray is None:
            return OcrResult("", self.engine, 0.0, False, flags=["no_page_image"])
        res = self._run(gray)
        res.flags.append("deg:" + json.dumps(degradation(gray)))
        key.parent.mkdir(parents=True, exist_ok=True)
        key.write_text(json.dumps(res.__dict__, ensure_ascii=False), encoding="utf-8")
        return res

    def _tess_best(self, gray) -> OcrResult:
        cands = []
        if self.restore_mode in ("auto", "off"):
            t, c, n = tesseract(gray, self.lang)
            cands.append(OcrResult(t, "tesseract", c, False))
        if self.restore_mode in ("auto", "on"):
            t, c, n = tesseract(restore(gray), self.lang)
            cands.append(OcrResult(t, "tesseract", c, True))
        return max(cands, key=lambda r: r.conf or 0)

    def _run(self, gray) -> OcrResult:
        tess = self._tess_best(gray)
        if self.engine == "tesseract":
            return tess
        model = self.engine.split(":", 1)[1]
        from backends import ollama_host
        img = restore(gray) if tess.restored else gray
        text = vlm(img, model, ollama_host(self.host))
        res = OcrResult(text, self.engine, tess.conf, tess.restored)
        if len(_norm(tess.text)) > 40:
            res.agreement = round(agreement(text, tess.text), 3)
            if res.agreement < self.min_agreement:
                res.flags.append("vlm_tesseract_disagree")
            ratio = len(_norm(text)) / max(1, len(_norm(tess.text)))
            if ratio > 1.25:
                res.flags.append("vlm_longer_possible_hallucination")
            elif ratio < 0.75:
                res.flags.append("vlm_shorter_possible_omission")
        return res

    def ocr_pages(self, pdf: Path, page_nos: list[int], workers: int = 2, progress=None) -> dict[int, OcrResult]:
        out = {}
        w = workers if self.engine == "tesseract" else 1   # don't queue parallel requests on one local GPU
        with ThreadPoolExecutor(max_workers=max(1, w)) as ex:
            futs = {ex.submit(self.ocr_page, pdf, p): p for p in page_nos}
            for i, fut in enumerate(futs, 1):
                out[futs[fut]] = fut.result()
                if progress:
                    progress(i, len(page_nos))
        return out


def printed_page_number(text: str) -> int | None:
    lines = [l for l in text.splitlines() if l.strip()]
    for l in lines[:3] + lines[-3:]:
        m = PAGE_NUM_RE.match(l)
        if m:
            return int(m.group(1).translate(AR_DIGITS))
    return None


def page_number_check(labels: dict[int, int | None]) -> dict:
    """Printed page numbers should equal pdf page + constant offset. The share that do is a free
    estimate of how reliably the OCR reads numbers."""
    found = {p: n for p, n in labels.items() if n is not None}
    if len(found) < 3:
        return dict(pages_with_number=len(found), consistent=None, offset=None)
    offs = [n - p for p, n in found.items()]
    off = max(set(offs), key=offs.count)
    ok = sum(o == off for o in offs)
    return dict(pages_with_number=len(found), offset=off, consistent=round(ok / len(found), 3),
                bad_pages=[p for p, n in found.items() if n - p != off][:20])
