#!/usr/bin/env python3
"""Check what the bench can use on this machine. Standard library only, so it runs even if
compiled packages are blocked. Usage:  python check_env.py  [--tesseract PATH]"""
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

PKGS = [  # (import name, pip name, needed for)
    ("numpy", "numpy", "everything"),
    ("pymupdf", "pymupdf", "PDF reading (1st choice)"),
    ("pypdfium2", "pypdfium2", "PDF reading (2nd choice)"),
    ("pypdf", "pypdf", "PDF reading (3rd choice; scanned pages also need Pillow)"),
    ("PIL", "pillow", "pypdf image extraction"),
    ("cv2", "opencv-python-headless", "OCR image clean-up (optional, NumPy fallback)"),
    ("rapidfuzz", "rapidfuzz", "fuzzy matching (optional, difflib fallback)"),
    ("sklearn", "scikit-learn", "tfidf model, diagnose"),
    ("umap", "umap-learn", "diagnose map (optional)"),
    ("plotly", "plotly", "diagnose map (optional)"),
]
WIN_TESS = [r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe")]


def check_import(mod):
    r = subprocess.run([sys.executable, "-c", f"import {mod}"], capture_output=True, text=True, timeout=180)
    if r.returncode == 0:
        return "OK", ""
    err = (r.stderr.strip().splitlines() or ["?"])
    text = r.stderr
    if "Application Control" in text or "blocked" in text.lower():
        return "BLOCKED", err[-1]
    if f"No module named '{mod}'" in text:
        return "MISSING", ""
    return "ERROR", err[-1]


def main():
    explicit = sys.argv[sys.argv.index("--tesseract") + 1] if "--tesseract" in sys.argv else None
    print(f"Python {sys.version.split()[0]} on {sys.platform}  ({sys.executable})\n")
    status = {}
    for mod, pipname, why in PKGS:
        st, detail = check_import(mod)
        status[mod] = st
        print(f"  {st:8} {mod:11} {why}" + (f"\n           -> {detail}" if detail else "")
              + (f"\n           -> pip install {pipname}" if st == "MISSING" else ""))

    pdf = next((m for m in ("pymupdf", "pypdfium2", "pypdf") if status.get(m) == "OK"), None)
    print(f"\nPDF reader that will be used: {pdf or 'NONE'}")

    tess = explicit or os.environ.get("TESSERACT_CMD") or shutil.which("tesseract") \
        or next((p for p in WIN_TESS if Path(p).is_file()), None)
    tess_ok = False
    if not tess:
        print("Tesseract: NOT FOUND (scanned pages can't be OCR'd)")
    else:
        try:
            r = subprocess.run([tess, "--list-langs"], capture_output=True, text=True, timeout=60)
            langs = [l.strip() for l in (r.stdout + r.stderr).splitlines() if l.strip() and " " not in l.strip()]
            tess_ok = "ara" in langs
            print(f"Tesseract: {tess}\n  languages: {', '.join(langs) or '(none)'}"
                  + ("" if tess_ok else "\n  -> Arabic ('ara') missing: re-run the installer and tick Arabic"))
        except OSError as e:
            print(f"Tesseract: found at {tess} but it can't run: {e}")

    host = (os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
    host = host if host.startswith("http") else "http://" + host
    ollama_ok = False
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"{host}/api/tags", timeout=10) as r:
            models = [m["name"] for m in json.loads(r.read())["models"]]
        ollama_ok = any(m.startswith("bge-m3") for m in models)
        print(f"Ollama at {host}: {len(models)} models" + ("" if ollama_ok else "  -> `ollama pull bge-m3`"))
    except Exception as e:
        print(f"Ollama at {host}: not reachable ({e}) -> start Ollama")

    np_ok = status.get("numpy") == "OK"
    scanned_ok = np_ok and tess_ok and (pdf in ("pymupdf", "pypdfium2") or (pdf == "pypdf" and status.get("PIL") == "OK"))
    print("\nCan I ...")
    print(f"  read digital PDFs        {'yes' if np_ok and pdf else 'NO'}")
    print(f"  OCR scanned PDFs         {'yes' if scanned_ok else 'NO'}")
    print(f"  embed with Ollama        {'yes' if ollama_ok else 'NO'}")
    print(f"  diagnose map             {'yes' if all(status.get(m) == 'OK' for m in ('sklearn', 'umap', 'plotly')) else 'no (use --no-map)'}")
    if "BLOCKED" in status.values():
        print("\nSome packages are BLOCKED by Windows Application Control (Smart App Control or a company policy).\n"
              "The bench falls back to other packages where it can. See README > Windows for how to unblock.")


if __name__ == "__main__":
    main()
