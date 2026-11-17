"""Embedding / reranking / LLM backends for rag_bench.

Model specs
    ollama:bge-m3                 embeddings via a local Ollama server (/api/embed)
    ollama:qwen3-embedding:0.6b   (query instruction added automatically)
    BAAI/bge-m3  or  st:BAAI/bge-m3   sentence-transformers (downloads from Hugging Face)
    tfidf                         offline char-n-gram baseline, no model needed

Reranker specs
    http://localhost:8081/v1/rerank   any Cohere/Jina-style rerank endpoint (llama.cpp llama-server, TEI, Infinity)
    ollama:<model>                    Ollama /api/rerank, only if your Ollama build has it
    st:BAAI/bge-reranker-v2-m3        sentence-transformers CrossEncoder (Hugging Face download)
    llm:qwen2.5:7b                    an Ollama chat model scores each passage 0-10 (slow, but works with what you have)
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

DEFAULT_QWEN_INSTRUCTION = "Given a question about Egyptian law, retrieve the legal articles that answer it"


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

class ApiError(Exception):
    def __init__(self, status: int, body: str, url: str):
        super().__init__(f"HTTP {status} from {url}: {body[:300]}")
        self.status, self.body, self.url = status, body, url


def ollama_host(host: str | None = None) -> str:
    h = host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
    if not h.startswith("http"):
        h = "http://" + h
    return h.rstrip("/")


def post_json(url: str, payload: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    local = re.match(r"https?://(localhost|127\.|0\.0\.0\.0|\[::1\])", url)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if local else urllib.request.build_opener()
    try:
        with opener.open(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise ApiError(e.code, e.read().decode("utf-8", "replace"), url) from None
    except urllib.error.URLError as e:
        raise ApiError(0, f"cannot connect ({e.reason}). Is the server running?", url) from None


def ollama_chat(host: str, model: str, messages: list, schema: dict | None = None,
                temperature: float = 0.0, num_ctx: int = 8192) -> str:
    payload = dict(model=model, messages=messages, stream=False,
                   options=dict(temperature=temperature, num_ctx=num_ctx))
    if schema:
        payload["format"] = schema
    if re.search(r"qwen3|deepseek-r1|gpt-oss|thinking", model, re.I):
        payload["think"] = False
    try:
        r = post_json(f"{host}/api/chat", payload)
    except ApiError as e:
        if "think" in payload and e.status == 400 and "think" in e.body.lower():
            payload.pop("think")
            r = post_json(f"{host}/api/chat", payload)
        else:
            raise
    text = r["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def parse_json_loose(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        return json.loads(m.group(0)) if m else {}


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #

def parse_spec(spec: str) -> tuple[str, str]:
    if spec == "tfidf":
        return "tfidf", ""
    for k in ("ollama", "st"):
        if spec.startswith(k + ":"):
            return k, spec[len(k) + 1:]
    return "st", spec


def _l2(X):
    X = np.asarray(X, dtype=np.float32)
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


class Embedder:
    """Uniform encode_passages / encode_queries over tfidf, sentence-transformers and Ollama."""

    def __init__(self, spec: str, index_dir: Path, device=None, host=None, num_ctx: int = 8192,
                 query_instruction: str | None = None, batch_size: int = 16, normalize_fn=None):
        self.spec = spec
        self.kind, self.name = parse_spec(spec)
        self.index_dir = Path(index_dir)
        self.host = ollama_host(host)
        self.num_ctx = num_ctx
        self.batch_size = batch_size
        self.normalize_fn = normalize_fn
        self.max_tokens = None
        self.truncated: list[int] = []
        self.prompt_tokens = 0

        low = self.name.lower()
        self.q_prefix = self.p_prefix = ""
        if "e5" in low and "qwen" not in low:
            self.q_prefix, self.p_prefix = "query: ", "passage: "
        if "qwen3-embedding" in low or "qwen3_embedding" in low:
            instr = DEFAULT_QWEN_INSTRUCTION if query_instruction is None else query_instruction
            self.q_prefix = f"Instruct: {instr}\nQuery:" if instr else ""
        elif query_instruction:
            self.q_prefix = query_instruction
        self.query_instruction = query_instruction

        if self.kind == "st":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.name, device=device)
            self.max_tokens = self.model.max_seq_length
        elif self.kind == "ollama":
            self.max_tokens = num_ctx

    # ---- public
    def encode_passages(self, texts: list[str], fit: bool = False):
        if self.kind == "tfidf":
            return self._tfidf(texts, fit=fit)
        return self._encode([self.p_prefix + t for t in texts], progress=True)

    def encode_queries(self, qs: list[str]):
        if self.kind == "tfidf":
            return self._tfidf(qs, fit=False)
        return self._encode([self.q_prefix + q for q in qs], progress=False)

    def token_counts(self, texts):
        if self.kind != "st":
            return None
        tok = self.model.tokenizer
        return [len(tok(self.p_prefix + t, add_special_tokens=True)["input_ids"]) for t in texts]

    # ---- backends
    def _encode(self, texts, progress):
        if self.kind == "st":
            return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                     show_progress_bar=progress, convert_to_numpy=True)
        out = []
        for b in range(0, len(texts), self.batch_size):
            batch = texts[b:b + self.batch_size]
            out += self._ollama_batch(batch, offset=b)
            if progress:
                print(f"\r  embedded {min(b + self.batch_size, len(texts))}/{len(texts)}", end="", file=sys.stderr)
        if progress:
            print(file=sys.stderr)
        return _l2(out)

    def _ollama_embed(self, inputs, truncate: bool):
        r = post_json(f"{self.host}/api/embed", dict(model=self.name, input=inputs, truncate=truncate,
                                                      options=dict(num_ctx=self.num_ctx)))
        self.prompt_tokens += int(r.get("prompt_eval_count") or 0)
        return r["embeddings"]

    def _ollama_batch(self, batch, offset):
        try:
            return self._ollama_embed(batch, truncate=False)
        except ApiError as e:
            if e.status in (0, 404):
                hint = f" Run `ollama pull {self.name}`." if e.status == 404 else ""
                raise SystemExit(f"Ollama embedding failed: {e}.{hint}")
        # something in the batch is too long: go one by one to find it
        res = []
        for j, t in enumerate(batch):
            try:
                res += self._ollama_embed([t], truncate=False)
            except ApiError:
                res += self._ollama_embed([t], truncate=True)
                self.truncated.append(offset + j)
        return res

    def _tfidf(self, texts, fit):
        path = self.index_dir / "tfidf.pkl"
        if fit:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), preprocessor=self.normalize_fn,
                                  sublinear_tf=True)
            X = vec.fit_transform(texts)
            k = max(2, min(256, X.shape[0] - 1, X.shape[1] - 1))
            svd = TruncatedSVD(k, random_state=0).fit(X)
            with open(path, "wb") as f:
                pickle.dump((vec, svd), f)
            self._tf = (vec, svd)
        elif not hasattr(self, "_tf"):
            with open(path, "rb") as f:
                self._tf = pickle.load(f)
        vec, svd = self._tf
        return _l2(svd.transform(vec.transform(texts)))


# --------------------------------------------------------------------------- #
# Rerankers
# --------------------------------------------------------------------------- #

LLM_RERANK_SCHEMA = {"type": "object", "properties": {"score": {"type": "integer", "minimum": 0, "maximum": 10}},
                     "required": ["score"]}


class Reranker:
    def __init__(self, spec: str, host=None, device=None, num_ctx: int = 8192):
        self.spec = spec
        self.host = ollama_host(host)
        self.num_ctx = num_ctx
        if spec.startswith(("http://", "https://")):
            self.kind, self.name = "http", spec
        elif ":" in spec and spec.split(":", 1)[0] in ("ollama", "st", "llm"):
            self.kind, self.name = spec.split(":", 1)
        else:
            raise SystemExit(f"Unknown reranker spec '{spec}'. Use http://…/v1/rerank, ollama:<m>, st:<m> or llm:<m>.")
        if self.kind == "st":
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.name, device=device, max_length=1024)

    def score(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        if self.kind == "st":
            return [float(s) for s in self.model.predict([(query, d) for d in docs])]
        if self.kind == "llm":
            return [self._llm_score(query, d) for d in docs]
        url = self.name if self.kind == "http" else f"{self.host}/api/rerank"
        payload = dict(query=query, documents=docs, top_n=len(docs))
        if self.kind == "ollama":
            payload["model"] = self.name
        try:
            r = post_json(url, payload)
        except ApiError as e:
            if self.kind == "ollama" and e.status in (404, 405):
                raise SystemExit(
                    "This Ollama server has no /api/rerank endpoint (Ollama doesn't natively serve rerankers).\n"
                    "Serve the same GGUF with llama.cpp instead:\n"
                    f"  ollama show --modelfile {self.name} | grep ^FROM        # path of the GGUF blob\n"
                    "  llama-server -m <that path> --reranking --port 8081\n"
                    "  ... --rerank http://localhost:8081/v1/rerank\n"
                    "or use --rerank llm:qwen2.5:7b, or --rerank st:BAAI/bge-reranker-v2-m3.") from None
            raise SystemExit(f"Rerank request failed: {e}") from None
        results = r.get("results") or r.get("data") or []
        scores = [float("-inf")] * len(docs)
        for item in results:
            scores[int(item["index"])] = float(item.get("relevance_score", item.get("score", 0.0)))
        return scores

    def _llm_score(self, query, doc):
        msgs = [
            {"role": "system", "content": "You judge search results for a legal retrieval system over Egyptian "
                                          "legislation. Rate how well the passage answers the question, from 0 "
                                          "(irrelevant) to 10 (directly and fully answers it). Reply with JSON only."},
            {"role": "user", "content": f"Question:\n{query}\n\nPassage:\n{doc[:3000]}\n\nReturn {{\"score\": <0-10>}}."},
        ]
        try:
            out = parse_json_loose(ollama_chat(self.host, self.name, msgs, LLM_RERANK_SCHEMA, num_ctx=self.num_ctx))
            return float(out.get("score", 0))
        except ApiError as e:
            raise SystemExit(f"LLM rerank failed: {e}") from None
        except (ValueError, TypeError):
            return 0.0
