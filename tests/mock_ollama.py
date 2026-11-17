"""Tiny stand-in for an Ollama server (+ a llama.cpp-style /v1/rerank), for testing
rag_bench without real models. Embeddings are hashed character trigrams, so they're
crude but meaningful. Usage: python tests/mock_ollama.py 11999"""
import hashlib
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

MODELS = {"bge-m3", "qwen3-embedding:0.6b", "qwen2.5:7b", "qwen3.8:27b", "qwen2.5vl:7b"}
GOLD = open(__import__("pathlib").Path(__file__).parent / "gold" / "ref_p3.txt", encoding="utf-8").read()


def vec(t, dim=256):
    t = re.sub(r"[ً-ٟ]", "", t)
    v = np.zeros(dim)
    for i in range(len(t) - 2):
        v[int(hashlib.md5(t[i:i + 3].encode()).hexdigest(), 16) % dim] += 1
    return (v / (np.linalg.norm(v) or 1)).tolist()


def overlap(q, d):
    qw = {w for w in re.findall(r"\w+", q) if len(w) > 2}
    return len(qw & set(re.findall(r"\w+", d))) / max(1, len(qw))


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path in ("/api/embed", "/api/chat", "/api/rerank") and body.get("model") not in MODELS:
            return self.send(404, {"error": f"model '{body.get('model')}' not found"})
        if self.path == "/api/embed":
            inp = body["input"] if isinstance(body["input"], list) else [body["input"]]
            limit = body.get("options", {}).get("num_ctx", 2048) * 2  # pretend 2 chars/token
            if not body.get("truncate", True) and any(len(t) > limit for t in inp):
                return self.send(400, {"error": "the input length exceeds the context length"})
            return self.send(200, {"model": body["model"], "embeddings": [vec(t[:limit]) for t in inp],
                                   "prompt_eval_count": sum(min(len(t), limit) // 2 for t in inp)})
        if self.path == "/api/chat":
            if body.get("think") is False and body["model"] == "qwen2.5:7b":
                return self.send(400, {"error": '"qwen2.5:7b" does not support thinking'})
            user = body["messages"][-1]["content"]
            if body["messages"][-1].get("images"):   # fake vision model: always "reads" the gold page
                return self.send(200, {"message": {"role": "assistant", "content": GOLD}})
            if "questions in Arabic" in user:
                art = user.split('"""')[1]
                words = [w for w in re.findall(r"[؀-ۿ]+", art) if len(w) > 3]
                qs = [{"question": "ما حكم " + " ".join(words[2:6]) + "؟", "style": "everyday"},
                      {"question": "متى " + " ".join(words[5:9]) + "؟", "style": "legal"}]
                out = {"questions": qs}
            else:
                q = user.split("Question:\n")[1].split("\n\nPassage:")[0]
                d = user.split("Passage:\n")[1]
                out = {"score": round(10 * overlap(q, d))}
            return self.send(200, {"message": {"role": "assistant", "content": json.dumps(out, ensure_ascii=False)}})
        if self.path == "/v1/rerank":
            res = [{"index": i, "relevance_score": overlap(body["query"], d)} for i, d in enumerate(body["documents"])]
            return self.send(200, {"results": sorted(res, key=lambda r: -r["relevance_score"])})
        self.send(404, {"error": "404 page not found"})


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 11999), H).serve_forever()
