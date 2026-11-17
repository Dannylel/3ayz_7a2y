"""Compatibility layer: every compiled dependency gets a fallback.

Some Windows machines (Smart App Control, or a company WDAC policy) refuse to load certain compiled
Python extensions ("An Application Control policy has blocked this file"). Instead of crashing, the bench
picks whatever works:

    PDF reading     PyMuPDF  ->  pypdfium2  ->  pypdf (+ Pillow for scanned pages)
    image ops       OpenCV   ->  pure NumPy
    image files     OpenCV   ->  Pillow  ->  built-in PNG/PGM writer
    fuzzy matching  rapidfuzz ->  difflib (standard library)

Force a PDF backend with the environment variable RAG_PDF_BACKEND=pymupdf|pdfium|pypdf.
`python check_env.py` shows what is usable on your machine.
"""
from __future__ import annotations

import difflib
import os
import struct
import zlib
from pathlib import Path

import numpy as np

LOAD_ERRORS: dict[str, str] = {}


def _try(name: str):
    try:
        return __import__(name)
    except Exception as e:  # ImportError, OSError (blocked DLL), ...
        LOAD_ERRORS[name] = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
        return None


# --------------------------------------------------------------------------- #
# PDF backends
# --------------------------------------------------------------------------- #

class _MuPage:
    def __init__(self, pg):
        self.pg = pg

    def text(self):
        return self.pg.get_text("text", sort=True)

    def words(self):
        return [(w[0], w[1], w[2], w[3], w[4], w[5], w[6]) for w in self.pg.get_text("words", sort=False)]

    def has_images(self):
        return bool(self.pg.get_images())

    def render_gray(self, dpi):
        import pymupdf
        pix = self.pg.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
        return np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w).copy()


class _MuDoc:
    def __init__(self, path):
        import pymupdf
        self.doc = pymupdf.open(str(path))

    def __len__(self):
        return len(self.doc)

    def __getitem__(self, i):
        return _MuPage(self.doc[i])

    def close(self):
        self.doc.close()


class _PdfiumPage:
    def __init__(self, pg):
        self.pg = pg
        self._tp = None

    @property
    def tp(self):
        if self._tp is None:
            self._tp = self.pg.get_textpage()
        return self._tp

    def text(self):
        return self.tp.get_text_bounded().replace("\r\n", "\n")

    def words(self):
        """Group characters into words and lines from their boxes (pdfium has no word API)."""
        tp = self.tp
        n = tp.count_chars() if hasattr(tp, "count_chars") else len(tp.get_text_range())
        H = self.pg.get_height()
        chars = []
        for i in range(n):
            ch = tp.get_text_range(i, 1)
            if not ch:
                continue
            l, b, r, t = tp.get_charbox(i)
            chars.append((ch, l, H - t, r, H - b))
        lines = []   # [y_center, height, [chars]]
        for c in chars:
            if c[0].isspace():
                if lines:
                    lines[-1][2].append(c)
                continue
            yc, h = (c[2] + c[4]) / 2, max(1.0, c[4] - c[2])
            for ln in lines[-3:]:
                if abs(ln[0] - yc) < 0.5 * max(h, ln[1]):
                    ln[2].append(c)
                    break
            else:
                lines.append([yc, h, [c]])
        out = []
        for li, (_, _, cs) in enumerate(lines):
            word, box = "", None
            for c in cs + [(" ", 0, 0, 0, 0)]:
                if c[0].isspace():
                    if word:
                        out.append((box[0], box[1], box[2], box[3], word, 0, li))
                    word, box = "", None
                else:
                    word += c[0]
                    box = list(c[1:]) if box is None else [min(box[0], c[1]), min(box[1], c[2]),
                                                           max(box[2], c[3]), max(box[3], c[4])]
        return out

    def has_images(self):
        import pypdfium2.raw as c
        return any(True for _ in self.pg.get_objects(filter=[c.FPDF_PAGEOBJ_IMAGE], max_depth=2))

    def render_gray(self, dpi):
        a = self.pg.render(scale=dpi / 72, grayscale=True).to_numpy()
        return np.ascontiguousarray(a[..., 0] if a.ndim == 3 else a)


class _PdfiumDoc:
    def __init__(self, path):
        import pypdfium2
        self.doc = pypdfium2.PdfDocument(str(path))

    def __len__(self):
        return len(self.doc)

    def __getitem__(self, i):
        return _PdfiumPage(self.doc[i])

    def close(self):
        self.doc.close()


class _PypdfPage:
    def __init__(self, pg):
        self.pg = pg

    def text(self):
        try:
            return self.pg.extract_text() or ""
        except Exception:
            return ""

    def words(self):
        return None

    def has_images(self):
        try:
            return len(self.pg.images) > 0
        except Exception:
            return False

    def render_gray(self, dpi):
        """pypdf can't render; for scanned pages the page *is* its largest embedded image."""
        try:
            imgs = list(self.pg.images)
        except Exception as e:
            raise RuntimeError(f"pypdf could not extract page images (needs Pillow): {e}") from None
        if not imgs:
            return None
        best = max(imgs, key=lambda im: im.image.size[0] * im.image.size[1])
        return np.asarray(best.image.convert("L"), dtype=np.uint8).copy()


class _PypdfDoc:
    def __init__(self, path):
        from pypdf import PdfReader
        self.doc = PdfReader(str(path))

    def __len__(self):
        return len(self.doc.pages)

    def __getitem__(self, i):
        return _PypdfPage(self.doc.pages[i])

    def close(self):
        pass


_BACKENDS = {"pymupdf": ("pymupdf", _MuDoc), "pdfium": ("pypdfium2", _PdfiumDoc), "pypdf": ("pypdf", _PypdfDoc)}
_chosen: str | None = None


def pdf_backend() -> str:
    global _chosen
    if _chosen:
        return _chosen
    want = os.environ.get("RAG_PDF_BACKEND", "auto").lower()
    order = [want] if want in _BACKENDS else ["pymupdf", "pdfium", "pypdf"]
    for name in order:
        if _try(_BACKENDS[name][0]) is not None:
            _chosen = name
            return name
    details = "\n".join(f"  {k}: {v}" for k, v in LOAD_ERRORS.items())
    raise SystemExit("No PDF library could be loaded:\n" + details +
                     "\nRun `python check_env.py`; if a file is 'blocked by Application Control', see README > Windows.")


class open_pdf:
    """with open_pdf(path) as doc: doc[i].text() / .words() / .has_images() / .render_gray(dpi)"""

    def __init__(self, path):
        self.doc = _BACKENDS[pdf_backend()][1](Path(path))

    def __enter__(self):
        return self.doc

    def __exit__(self, *a):
        self.doc.close()


# --------------------------------------------------------------------------- #
# image operations (OpenCV if allowed, else NumPy)
# --------------------------------------------------------------------------- #

cv2 = _try("cv2")


def otsu_ink(gray: np.ndarray) -> np.ndarray:
    """Binary ink mask (1 = ink) using Otsu's threshold."""
    if cv2 is not None:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return (bw == 0).astype(np.uint8)
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    p = hist / hist.sum()
    w0 = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    between = (mu[-1] * w0 - mu) ** 2 / np.clip(w0 * (1 - w0), 1e-12, None)
    return (gray <= int(np.argmax(between))).astype(np.uint8)


def adaptive_ink(gray: np.ndarray) -> np.ndarray:
    """For real grayscale scans with uneven fading: CLAHE + local threshold (OpenCV), else Otsu."""
    if cv2 is None:
        return otsu_ink(gray)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15)
    return (bw == 0).astype(np.uint8)


def _shift_op(m: np.ndarray, k: int, op) -> np.ndarray:
    h, w = m.shape
    pad = np.pad(m, ((k // 2, k - 1 - k // 2), (k // 2, k - 1 - k // 2)))
    out = pad[0:h, 0:w].copy()
    for dy in range(k):
        for dx in range(k):
            if dy or dx:
                out = op(out, pad[dy:dy + h, dx:dx + w])
    return out


def dilate(m, k):
    if cv2 is not None:
        return cv2.dilate(m, np.ones((k, k), np.uint8))
    return _shift_op(m, k, np.maximum)


def erode(m, k):
    if cv2 is not None:
        return cv2.erode(m, np.ones((k, k), np.uint8))
    return _shift_op(m, k, np.minimum)


def close(m, k):
    if cv2 is not None:
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    return erode(dilate(m, k), k)


def remove_specks(m, min_px):
    if cv2 is not None:
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        keep = st[:, 4] >= min_px
        keep[0] = False
        return keep[lab].astype(np.uint8)
    return m   # skipped without OpenCV (minor effect)


def stroke_width_and_specks(ink):
    if cv2 is None:
        return None, None
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    stroke = float(2 * np.median(dist[dist > 0])) if (dist > 0).any() else None
    n, _, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    specks = float((st[1:, 4] < 12).mean()) if n > 1 else 0.0
    return stroke, specks


def downscale(gray, s):
    if s >= 1:
        return gray
    if cv2 is not None:
        return cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    f = max(1, int(round(1 / s)))
    h, w = (gray.shape[0] // f) * f, (gray.shape[1] // f) * f
    return gray[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3)).astype(np.uint8)


def png_bytes(gray: np.ndarray) -> bytes:
    if cv2 is not None:
        return cv2.imencode(".png", gray)[1].tobytes()
    raw = b"".join(b"\x00" + row.tobytes() for row in np.ascontiguousarray(gray, dtype=np.uint8))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)  # noqa: E731
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", gray.shape[1], gray.shape[0], 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def write_image(path: Path, gray: np.ndarray) -> None:
    Path(path).write_bytes(png_bytes(gray))


# --------------------------------------------------------------------------- #
# fuzzy string matching
# --------------------------------------------------------------------------- #

_rf = _try("rapidfuzz")


def ratio(a: str, b: str) -> float:
    """0-100 similarity."""
    if _rf is not None:
        from rapidfuzz import fuzz
        return fuzz.ratio(a, b)
    return 100.0 * difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def norm_distance(a: str, b: str) -> float:
    """Normalised edit distance 0-1 (difflib approximation when rapidfuzz is unavailable)."""
    if _rf is not None:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.normalized_distance(a, b)
    if not a and not b:
        return 0.0
    return 1.0 - difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
