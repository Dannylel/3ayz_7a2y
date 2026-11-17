# Arabic legal RAG test bench

A plain, framework-free RAG pipeline for Arabic legal PDFs, digital or scanned:

```
PDF ──► text ──► chunks ──► embeddings ──► retrieve top-k for a question
        (PyMuPDF, or OCR       (Ollama:
         for scanned pages)     bge-m3)
```

Use it to see what out-of-the-box retrieval gets right and wrong on your documents before deciding how to
structure them.

---

## Quick start

### 1. Install (once)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For scanned PDFs, install Tesseract with Arabic:

| OS | Command |
|---|---|
| Ubuntu / Debian | `sudo apt install tesseract-ocr tesseract-ocr-ara` |
| macOS | `brew install tesseract tesseract-lang` |
| Windows | UB-Mannheim Tesseract installer, tick *Arabic* under additional languages, and add it to `PATH` |

The installer puts it in `C:\Program Files\Tesseract-OCR\`, and the bench finds it there automatically, so you
don't need to edit `PATH`. Elsewhere, pass `--tesseract "D:\path\to\tesseract.exe"`.

Make sure Ollama is running and has the embedding model (you already have it).

Then check everything in one go:

```bash
python check_env.py
```

It lists each package as OK / MISSING / BLOCKED, shows which PDF reader will be used, whether Tesseract has
Arabic, and whether Ollama has `bge-m3`. It ends with a yes/no for "read digital PDFs", "OCR scanned PDFs" and
"embed with Ollama". If anything says BLOCKED, see [Windows: "blocked by Application Control"](#windows-blocked-by-application-control).

### 2. Transcribe, chunk and embed a PDF

```bash
python rag_bench.py ingest --pdf "path/to/your.pdf" --out idx_mybook --dump-text
```

This does, in one run:
1. **Extract text.** Pages with a text layer are read directly. Scanned pages are OCR'd with Tesseract, and each
   faded page is OCR'd both as-is and with repaired strokes, keeping the better result.
2. **Chunk.** About 1000 characters per chunk, 150 overlap, split per page (the typical out-of-the-box setting).
3. **Embed** each chunk with `bge-m3` through Ollama.
4. **Save** everything to `idx_mybook/`.

Expect roughly 1–2 seconds per scanned page for OCR. Results are cached in `.ocr_cache/`, so re-running (e.g. with
different chunk settings) skips OCR.

**Read the transcription before anything else:** `idx_mybook/text/<your file>.txt`. Each page is marked with
flags such as `ocr:tesseract`, `ocr_restored` or `ocr_low_confidence`. If the text is bad, retrieval will be bad,
whatever you do later.

### 3. Ask questions

```bash
python rag_bench.py query --index idx_mybook
```

```
سؤال> ما هي البيانات التي يجب أن تشتمل عليها ورقة الإعلان؟

#1  score 0.672  your.pdf  page 23  [نصوص القانون]  ocr-conf 88
    مادة ٩: يجب أن تشتمل الأوراق التى يقوم المحضرون باعلانها على البيانات الآتية ...
```

Each result shows its page, the detected section type (when headings are found) and the OCR confidence of that
page. Every query's full results are also appended to `idx_mybook/query_log.md`, which is easier to read than the
Windows console. Empty line or Ctrl-D quits. For a single question: `python rag_bench.py query --index idx_mybook "سؤالك هنا"`.
Options: `-k 10` for more results, `--chars 0` to print whole chunks.

That's the whole pipeline. Everything below is optional.

---

## Things to know about scanned Arabic

Measured on a 140-page scanned legal book (أوراق المحضرين, 1-bit scans) against a hand-checked page:

| | Result with Tesseract |
|---|---|
| Body text, clean page | ~2.5% character error, ~7% word error |
| Simulated moderate fading, without → with stroke repair | 9.0% → 2.3% character error |
| Simulated heavy fading | 41.7% → 12.1% |
| Printed page numbers read correctly | **3 of 137** |

**Words come out well, but numbers don't.** In common typefaces `٥` looks like `ه`, and multi-digit numbers get
scrambled. So retrieving by meaning works, but anything number-based (`المادة ١٥`, Cassation case numbers) is
unreliable. If that matters, try a vision model for OCR (see *Choosing an OCR engine*).

Other OCR leftovers: page headers (`- ٢٣ -` often comes out as junk like `-١٠١85`) and footnotes stay inline in
the text.

---

## Optional: experiment with the setup

Rebuild the index with one change and compare answers. OCR is cached, so this is fast.

```bash
# chunk by detected headings instead of every 1000 chars; prefix each chunk with "book | form | section type"
python rag_bench.py ingest --pdf your.pdf --out idx_sec --chunking section --contextual-header

# one chunk per article (مادة), for clean digital legal codes
python rag_bench.py ingest --pdf your.pdf --out idx_art --chunking article --contextual-header

# a different embedding model
python rag_bench.py ingest --pdf your.pdf --out idx_qwen --model ollama:qwen3-embedding:0.6b
```

| Option | Values | Default |
|---|---|---|
| `--model` | `ollama:bge-m3`, `ollama:qwen3-embedding:0.6b`, `tfidf` (offline, no model) | `ollama:bge-m3` |
| `--chunking` | `fixed`, `section`, `article` | `fixed` |
| `--chunk-size` / `--chunk-overlap` | characters | 1000 / 150 |
| `--chunk-scope` | `page` (never crosses a page break), `document` | `page` |
| `--contextual-header` | prefix chunks with book / section / article | off |
| `--normalize` | strip tashkeel and tatweel before embedding | off |
| `--ocr` | `auto` (scanned pages only), `force` (every page), `off` | `auto` |
| `--ocr-engine` | `tesseract`, `ollama:<vision model>` | `tesseract` |
| `--ocr-restore` | `auto` (try both, keep better), `on`, `off` | `auto` |

**Section detection** (`sections.py`) fuzzy-matches heading lines common in Egyptian legal books: `الباب`, `الفصل`,
`الصيغة رقم`, `نصوص القانون`, `المذكرة الإيضاحية`, `النصوص العربية المقابلة`, `آراء الشراح وأحكام القضاء`, … It tolerates
OCR damage (`الصيغفة` → `الصيغة`). For different headings, pass `--headings myheadings.txt` with lines of
`level|phrase|type` (level 1 = container such as a chapter or form, level 2 = section inside it).

## Optional: choosing an OCR engine

A vision model through Ollama usually reads digits and damaged print better, but it can invent fluent text that
isn't on the page. So the bench always cross-checks it against Tesseract and flags pages where they disagree.
Compare engines on a sample of your pages first:

```bash
ollama pull qwen2.5vl:7b          # or qwen3-vl:8b; `ollama show <model>` must list "vision"
python rag_bench.py ocr-check --pdf your.pdf --pages 3-30 --engines tesseract ollama:qwen2.5vl:7b --page-offset 20
```

`--page-offset` is the printed page number minus the PDF page number (open the PDF and check one page). With it,
every page's number becomes a free test of digit reading. The output table shows page numbers read correctly,
confidence, agreement and seconds per page. Each engine's text is saved to `ocr_check/<engine>/` for reading side
by side. If the vision model wins, ingest with `--ocr-engine ollama:qwen2.5vl:7b` (expect tens of seconds per page).

## Optional: measure retrieval quality

1. **Get test questions.** Generate them with a local LLM:
   ```bash
   python rag_bench.py gen-queries --index idx_mybook --llm qwen3.8:27b --n-articles 60 --out questions.jsonl
   ```
   Each question is labelled with a unique phrase from the passage it came from. Skim the file: delete vague
   questions, and add a second label where another passage also answers. Better still, add 20–30 questions
   written by people:
   ```json
   {"query": "متى لا يجوز إجراء الإعلان؟", "relevant": [{"file": "your.pdf", "text": "قبل الساعة السابعة صباحا"}]}
   ```
   A `text` label is any distinctive 5–8 word phrase from the right passage. Matching ignores punctuation,
   diacritics and alef/ya/ta-marbuta variants, so it survives OCR noise. You can also label by
   `{"file", "page": 23}` or, for digital codes, `{"file", "article": "15"}`.

2. **Score one or more indexes side by side:**
   ```bash
   python rag_bench.py eval --index idx_mybook idx_sec idx_qwen --queries questions.jsonl
   ```
   This reports hit@k, recall@k, precision@k, MRR and nDCG@10, and lists failed questions with what came back.

## Optional: look at the embedding space

```bash
python rag_bench.py diagnose --index idx_mybook
```

Writes `report/report.md`, `report/embedding_map.html` (interactive map; hover to read chunks) and
`report/chunks.csv`. The report covers:
- **OCR quality:** confidence per file, the worst pages, and the page-number digit test.
- **Structure:** how often chunks cut through articles or sections, and how chunks split across section types.
  This shows how much of a book is law text versus commentary, rulings or blank form templates.
- **Geometry:** whether chunks of the same section sit closer together than chunks of different sections.
- **Hubs:** chunks that come back for almost any question, often boilerplate.
- **Near-duplicates** across sections.
- **Article-number probe:** does `المادة N` find article N?

## Optional: reranking

A reranker re-orders the top ~30 retrieved chunks with a slower, more accurate model. It does **not** affect
transcription. Try it only once the retrieval numbers are in: it tells you whether misses are ranking problems
(the right chunk was in the top 30) or extraction/chunking problems (it wasn't).
Ollama can't serve rerankers, so serve your `bge-reranker-v2-m3` with llama.cpp:

```bash
ollama show --modelfile qllama/bge-reranker-v2-m3 | grep ^FROM     # path of the model file
llama-server -m <that path> --reranking --port 8081
python rag_bench.py query --index idx_mybook --rerank http://localhost:8081/v1/rerank
```

Or use an LLM as the reranker (slow): `--rerank llm:qwen2.5:7b`.

---

## Files

| File | What it does |
|---|---|
| `rag_bench.py` | commands: `ingest`, `query`, `diagnose`, `gen-queries`, `eval`, `ocr-check` |
| `check_env.py` | checks packages, PDF reader, Tesseract + Arabic, Ollama (standard library only) |
| `compat.py` | fallbacks for blocked/missing compiled packages |
| `ocr.py` | rendering, stroke restoration, Tesseract / vision-model OCR, caching, page-number check |
| `sections.py` | fuzzy heading detection for books and commentaries |
| `backends.py` | Ollama / sentence-transformers / TF-IDF embeddings, rerankers, LLM calls |
| `tests/` | synthetic sample PDFs, a hand-checked gold page, a mock Ollama server for testing without models |

An index folder holds `chunks.jsonl` (text + metadata per chunk), `embeddings.npy`, `meta.json`, `pages.jsonl`
(per-page OCR info), `sections.jsonl`, `articles.jsonl` and, with `--dump-text`, `text/` (full transcription).

## Windows: "blocked by Application Control"

`ImportError: DLL load failed … An Application Control policy has blocked this file` means Windows refused to load a
compiled part of a Python package. It's either **Smart App Control** (Windows 11, personal PCs) or a **company
WDAC policy** (managed PCs). It isn't a bug in the package or in your install.

**You may not need to change anything.** The bench has a fallback for each compiled dependency (`compat.py`):

| Needed for | 1st choice | Fallbacks |
|---|---|---|
| reading PDFs / rendering scanned pages | PyMuPDF | pypdfium2 → pypdf (+ Pillow) |
| OCR image clean-up | OpenCV | pure NumPy |
| fuzzy heading matching | rapidfuzz | Python's difflib |

Run `python check_env.py`. If it says "OCR scanned PDFs: yes", just run `ingest` again: it prints
`PDF reader: pdfium` (or `pypdf`) and carries on. With PyMuPDF, OpenCV and rapidfuzz all blocked, the fallbacks
were tested on the reference book: OCR accuracy matched the normal setup (≈2% character error on the hand-checked
page). To force a reader: `set RAG_PDF_BACKEND=pdfium` (or `pypdf`).

If `numpy` itself is blocked, or Tesseract can't run, nothing can work around it:
- **Work / managed PC:** ask IT to allow Python packages in your project's `.venv` folder (or Tesseract). Don't try
  to get around the policy.
- **Personal PC with Smart App Control:** Windows Security → App & browser control → Smart App Control settings →
  Off (needs admin). Updates since April 2026 let you switch it back on afterwards, but the October 2026 update
  (KB5127648) has a known bug that greys out the switch. If you're on that update, you may not be able to turn it
  back on until Microsoft fixes it.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `NOTE: Tesseract not found … scanned pages will be SKIPPED` | install Tesseract + Arabic (step 1), or pass `--tesseract PATH\tesseract.exe` |
| `Tesseract … has no 'ara' language data` | re-run the installer and tick Arabic, or put `ara.traineddata` into Tesseract's `tessdata` folder |
| `DLL load failed … blocked this file` | see the Windows section above; run `python check_env.py` |
| Arabic looks reversed or broken in the terminal | Windows' console doesn't lay out Arabic properly. Use Windows Terminal, or read `idx_mybook\query_log.md` (every query's full results) in VS Code or Notepad |
| `Ollama embedding failed: … cannot connect` | start Ollama (`ollama serve`) or set `OLLAMA_HOST` / `--ollama-host` |
| `model 'bge-m3' not found` | `ollama pull bge-m3` |
| `chunks exceeded num_ctx … truncated` | lower `--chunk-size`, or raise `--num-ctx` (bge-m3 max 8192) |
| transcription is garbage on some pages | check `ocr-conf` and flags in `text/`; try `--ocr-restore on` or a vision model |
| digital PDF reads with reversed word order | already handled by `--extract auto`; if not, force `--extract rtl` |
