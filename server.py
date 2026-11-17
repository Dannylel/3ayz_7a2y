#!/usr/bin/env python3
"""
server.py - a small password-protected web UI for testing retrieval and chat over an index.

    python server.py --index idx_mybook --pdf "D:/Legal Assistant/01 - أوراق المحضرين.pdf"

Then open http://localhost:8000 (user name: anything, password: printed at start-up).
Standard library only (plus the bench's own modules); no web framework needed.

Endpoints
    GET  /                 the UI (ui.html)
    GET  /api/info         index + model info
    POST /api/search       {"query", "k"}                     -> retrieved chunks
    POST /api/chat         {"question", "history", "k", "llm"} -> server-sent events: sources, tokens, done
    GET  /api/page         ?file=&page=                        -> PNG of the original PDF page
    POST /api/feedback     ratings; marked-relevant sources also go to <index>/queries.from_ui.jsonl
"""
from __future__ import annotations

import argparse
import base64
import hmac
import json
import re
import secrets
import socket
import sys
import threading
import time
import urllib.parse
import urllib.request
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rag_bench as rb  # noqa: E402
from backends import Reranker, ollama_host  # noqa: E402

SYSTEM_PROMPT = """أنت مساعد قانوني متخصص في القانون المصري. تجيب عن سؤال المستخدم اعتمادًا فقط على المقتطفات المرقمة المرفقة.
القواعد:
- لا تستخدم أي معلومة غير موجودة في المقتطفات. إذا لم تكفِ المقتطفات للإجابة فقل ذلك صراحة.
- استشهد بالمصدر بعد كل معلومة برقمه بين قوسين مربعين مثل [1] أو [2][3].
- النصوص مستخرجة آليًا (OCR) من مستندات ممسوحة ضوئيًا، وقد تحتوي على أخطاء، خاصة في الأرقام (أرقام المواد والطعون والسنوات). إذا ذكرت رقمًا فنبّه إلى ضرورة التحقق منه من الصفحة الأصلية.
- أجب بالعربية الفصحى بإيجاز ووضوح."""

STATE = SimpleNamespace()
LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# retrieval / generation
# --------------------------------------------------------------------------- #

def search(query: str, k: int):
    top, sc, rtop, rsc = rb.retrieve(STATE.emb, STATE.chunks, STATE.embedder, [query], k,
                                     STATE.reranker, STATE.rerank_depth)
    rows, scores = (rtop[0], rsc[0]) if STATE.reranker else (top[0], sc[0])
    out = []
    for r, (i, s) in enumerate(zip(rows[:k], scores[:k]), 1):
        c = STATE.chunks[int(i)]
        out.append(dict(rank=r, id=int(i), score=round(float(s), 4), file=c["file"], pages=c["pages"],
                        section_type=c.get("section_type"), ocr_conf=c.get("ocr_conf"), text=c["raw"],
                        page_image=c["file"] in STATE.pdfs))
    return out


def build_messages(question: str, sources: list, history: list):
    ctx = []
    for s in sources:
        where = f"{s['file']} - صفحة {', '.join(map(str, s['pages']))}"
        if s.get("section_type"):
            where += f" - {s['section_type']}"
        ctx.append(f"[{s['rank']}] ({where})\n{s['text']}")
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-4:]:      # short memory for follow-up questions
        if h.get("role") in ("user", "assistant") and h.get("content"):
            msgs.append({"role": h["role"], "content": str(h["content"])[:2000]})
    msgs.append({"role": "user", "content": "المقتطفات:\n\n" + "\n\n---\n\n".join(ctx) + f"\n\nالسؤال: {question}"})
    return msgs


def stream_llm(model: str, messages: list):
    """Yield text pieces from Ollama's streaming chat API."""
    payload = dict(model=model, messages=messages, stream=True,
                   options=dict(temperature=0.1, num_ctx=STATE.num_ctx))
    if re.search(r"qwen3|deepseek-r1|gpt-oss", model, re.I):
        payload["think"] = False
    req = urllib.request.Request(f"{STATE.host}/api/chat", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    in_think = False
    with opener.open(req, timeout=1800) as r:
        for line in r:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("error"):
                raise RuntimeError(d["error"])
            piece = (d.get("message") or {}).get("content", "")
            if "<think>" in piece:
                in_think = True
            if not in_think and piece:
                yield piece
            if "</think>" in piece:
                in_think = False
            if d.get("done"):
                break


def list_llms():
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"{STATE.host}/api/tags", timeout=5) as r:
            names = [m["name"] for m in json.loads(r.read())["models"]]
    except Exception:
        return [STATE.llm]
    chat = [n for n in names if not re.search(r"embed|bge|rerank|minilm|nomic", n, re.I)]
    return sorted(set(chat + [STATE.llm]), key=lambda n: (n != STATE.llm, n))


@lru_cache(maxsize=64)
def page_png(file: str, page: int) -> bytes:
    from compat import open_pdf, png_bytes
    with open_pdf(STATE.pdfs[file]) as doc:
        gray = doc[page - 1].render_gray(STATE.page_dpi)
    return png_bytes(gray)


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

class Handler(BaseHTTPRequestHandler):
    server_version = "rag-bench/1.0"

    def log_message(self, fmt, *args):
        if STATE.verbose:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # ---- helpers
    def _authed(self) -> bool:
        if not STATE.password:
            return True
        h = self.headers.get("Authorization", "")
        if h.startswith("Basic "):
            try:
                _, pw = base64.b64decode(h[6:]).decode("utf-8").split(":", 1)
                return hmac.compare_digest(pw.encode(), STATE.password.encode())
            except Exception:
                return False
        return False

    def _deny(self):
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Legal RAG test", charset="UTF-8"')
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write("Password required".encode())

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 1_000_000:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(n) or b"{}")

    # ---- routes
    def do_GET(self):
        if not self._authed():
            return self._deny()
        u = urllib.parse.urlparse(self.path)
        if u.path in ("/", "/index.html"):
            b = (HERE / "ui.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            return self.wfile.write(b)
        if u.path == "/api/info":
            return self._json(dict(index=STATE.index.name, chunks=len(STATE.chunks), embed_model=STATE.meta["model"],
                                   chunking=STATE.meta.get("chunking"), files=sorted({c["file"] for c in STATE.chunks}),
                                   llm=STATE.llm, llms=list_llms(), k=STATE.k,
                                   rerank=STATE.reranker.spec if STATE.reranker else None,
                                   page_images=bool(STATE.pdfs)))
        if u.path == "/api/page":
            q = urllib.parse.parse_qs(u.query)
            file, page = q.get("file", [""])[0], int(q.get("page", ["0"])[0] or 0)
            if file not in STATE.pdfs or page < 1:
                return self._json({"error": "page image not available (start the server with --pdf)"}, 404)
            try:
                b = page_png(file, page)
            except Exception as e:
                return self._json({"error": str(e)}, 500)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "private, max-age=3600")
            self.end_headers()
            return self.wfile.write(b)
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        if not self._authed():
            return self._deny()
        try:
            body = self._body()
        except Exception as e:
            return self._json({"error": f"bad request: {e}"}, 400)
        path = urllib.parse.urlparse(self.path).path
        k = max(1, min(int(body.get("k") or STATE.k), 20))

        if path == "/api/search":
            q = str(body.get("query", "")).strip()
            if not q:
                return self._json({"error": "empty query"}, 400)
            try:
                t = time.time()
                res = search(q, k)
                return self._json(dict(results=res, ms=int(1000 * (time.time() - t))))
            except BaseException as e:   # SystemExit from backends on connection errors
                return self._json({"error": str(e)}, 502)

        if path == "/api/chat":
            q = str(body.get("question", "")).strip()
            llm = str(body.get("llm") or STATE.llm)
            if not q:
                return self._json({"error": "empty question"}, 400)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            def send(obj):
                self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8"))
                self.wfile.flush()
            try:
                sources = search(q, k)
                send(dict(type="sources", sources=sources))
                for piece in stream_llm(llm, build_messages(q, sources, body.get("history") or [])):
                    send(dict(type="token", text=piece))
                send(dict(type="done"))
            except (BrokenPipeError, ConnectionResetError):
                pass
            except BaseException as e:
                try:
                    send(dict(type="error", error=str(e)))
                except Exception:
                    pass
            return

        if path == "/api/feedback":
            rec = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), **{k_: body.get(k_) for k_ in
                       ("kind", "question", "answer", "rating", "comment", "chunk_id", "relevant", "llm")})
            with LOCK:
                with open(STATE.index / "feedback.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                # a source marked relevant becomes an eval question for `rag_bench.py eval`
                if body.get("kind") == "source" and body.get("relevant") and body.get("chunk_id") is not None:
                    c = STATE.chunks[int(body["chunk_id"])]
                    anchor = rb._anchor_phrase(c["raw"], STATE.corpus_norm, 7)
                    label = dict(file=c["file"], text=anchor) if anchor else dict(file=c["file"], pages=c["pages"])
                    with open(STATE.index / "queries.from_ui.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps(dict(query=body.get("question"), relevant=[label], source="ui"),
                                           ensure_ascii=False) + "\n")
            return self._json({"ok": True})

        self._json({"error": "not found"}, 404)


def lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="Web UI for testing retrieval and chat over a rag_bench index")
    ap.add_argument("--index", required=True, help="index folder built by `rag_bench.py ingest`")
    ap.add_argument("--pdf", nargs="*", default=[], help="the PDF file(s) of the index, to show original pages")
    ap.add_argument("--pdf-dir", default=None, help="or a folder containing them")
    ap.add_argument("--llm", default="qwen2.5:7b", help="Ollama chat model for answers (switchable in the UI)")
    ap.add_argument("--k", type=int, default=5, help="default number of retrieved chunks")
    ap.add_argument("--rerank", default=None, help="optional reranker spec (see backends.py)")
    ap.add_argument("--rerank-depth", type=int, default=30)
    ap.add_argument("--num-ctx", type=int, default=8192, help="LLM context window (tokens)")
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = reachable from your network; 127.0.0.1 = this PC only")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--password", default=None, help="access password (default: a random one, printed at start)")
    ap.add_argument("--no-password", action="store_true", help="disable the password (only with --host 127.0.0.1)")
    ap.add_argument("--ollama-host", default=None)
    ap.add_argument("--page-dpi", type=int, default=120)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    idx = Path(a.index)
    if not (idx / "chunks.jsonl").exists():
        sys.exit(f"{idx} is not an index folder (no chunks.jsonl). Build one with rag_bench.py ingest.")
    if a.no_password and a.host != "127.0.0.1":
        sys.exit("--no-password is only allowed together with --host 127.0.0.1")

    chunks, emb, meta = rb.load_index(idx)
    args_ns = SimpleNamespace(device=None, ollama_host=a.ollama_host, num_ctx=8192, query_instruction=None, batch_size=16)
    pdfs = {}
    for p in a.pdf:
        if Path(p).is_file():
            pdfs[Path(p).name] = Path(p)
    if a.pdf_dir:
        for p in Path(a.pdf_dir).glob("**/*.pdf"):
            pdfs.setdefault(p.name, p)
    wanted = {c["file"] for c in chunks}
    missing = wanted - set(pdfs)
    STATE.__dict__.update(
        index=idx, chunks=chunks, emb=emb, meta=meta, embedder=rb.make_embedder(meta["model"], idx, args_ns, meta),
        reranker=Reranker(a.rerank, a.ollama_host) if a.rerank else None, rerank_depth=a.rerank_depth,
        host=ollama_host(a.ollama_host), llm=a.llm, k=a.k, num_ctx=a.num_ctx, verbose=a.verbose,
        pdfs={k: v for k, v in pdfs.items() if k in wanted}, page_dpi=a.page_dpi,
        password=None if a.no_password else (a.password or secrets.token_urlsafe(9)),
        corpus_norm=" ".join(rb.norm_words(c["raw"]) for c in chunks),
    )
    # warm-up: fail early if Ollama / the embedding model isn't reachable
    try:
        STATE.embedder.encode_queries(["اختبار"])
    except BaseException as e:
        sys.exit(f"Embedding model not reachable ({meta['model']}): {e}")

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"\nIndex: {idx}  ({len(chunks)} chunks, {meta['model']})   answers: {a.llm} via {STATE.host}")
    if missing:
        print(f"Original-page view off for: {', '.join(sorted(missing))}  (pass --pdf FILE to enable)")
    print(f"\n  This PC:        http://localhost:{a.port}")
    if a.host == "0.0.0.0":
        print(f"  Same network:   http://{lan_ip()}:{a.port}   (allow Python through Windows Firewall if asked)")
    if STATE.password:
        print(f"\n  Password: {STATE.password}     (user name: anything)")
    print("\n  Online for a colleague (second terminal):  cloudflared tunnel --url http://localhost:%d" % a.port)
    print("  Ctrl+C to stop.\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
