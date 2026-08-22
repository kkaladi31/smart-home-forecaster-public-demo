"""Cross-encoder reranking — the fix for "confidently cites the wrong passage".

The retrieval threshold was never trustworthy. Measured on this corpus, asking
"Am I allowed to keep a pet tiger in my backyard?" scored **0.409** against the
HOA landscaping section — above the 0.35 grounding threshold — purely because
both mention a backyard. The passage does not address the question at all, so
the only thing standing between that and an invented rule was the prompt telling
the model to refuse.

The cause is structural: a bi-encoder embeds the query and the passage
*separately*, so the score measures topical neighbourhood, not whether the
passage answers the question. A cross-encoder reads the pair together and scores
them jointly, which separates the two cases by a wide margin:

    query                                    dense    cross-encoder
    "replace my backyard grass with stones"  0.498        -1.14      (real match)
    "keep a pet tiger in my backyard"        0.409        -8.65      (not addressed)

0.09 apart before, 7.5 apart after. That is what makes a threshold meaningful.

Runs on onnxruntime + tokenizers, both of which Chroma already installs, so this
adds **no new Python dependency** — deliberately, because the alternative
(sentence-transformers) would pull in PyTorch and roughly 2 GB, and the project
is meant to stay clonable and runnable by anyone.

Everything here degrades to None rather than raising. If the weights cannot be
downloaded, retrieval silently falls back to hybrid fusion order, which is
exactly the behaviour that existed before this module.
"""
from __future__ import annotations

import threading
from pathlib import Path

import telemetry

MODEL_DIR = Path(__file__).resolve().parent / "models" / "ms-marco-MiniLM-L-6-v2"
_HF_BASE = "https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2/resolve/main"
_FILES = [("tokenizer.json", "tokenizer.json"), ("onnx/model_quantized.onnx", "model.onnx")]

# Hugging Face's CDN closes the connection on a default python-requests
# User-Agent, and the TLS-inspecting proxy on some networks makes the transfer
# flaky on top of that — so send a browser UA and retry. Same lesson as fetching
# floor plans from Wikimedia (see SOURCES.md).
_UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "*/*",
}
_DOWNLOAD_ATTEMPTS = 4

# Score below which a passage is treated as not actually answering the question.
#
# Calibrated by measurement, not taste (`python -m memory.rerank` reproduces it):
#     +7.31  "Do I need a permit to build a deck?"
#     +6.31  "How often should I change my HVAC filter?"
#     +1.72  "Can I list my house on Airbnb?"
#     -1.31  "Can I replace my backyard grass with stones?"
#     ------ decision boundary ------
#     -6.40  "Can I park a submarine in my driveway?"
#     -8.53  "Am I allowed to keep a pet tiger in my backyard?"
#    -11.07  "What is the capital of France?"
# -4.0 sits in the empty band with ~2.7 of margin below the weakest true match
# and ~2.4 above the strongest false one. Compare the dense scores for the same
# queries, where the true and false cases were 0.09 apart and straddled the
# threshold — that is the whole reason this module exists.
#
# These are raw logits and are only meaningful for this specific model; re-measure
# if the model changes.
MIN_RERANK_SCORE = -4.0

_LOCK = threading.Lock()
_RUNTIME: tuple | None = None      # (tokenizer, session, input_names)
_UNAVAILABLE = False               # sticky: don't retry a failed download every query


def _download() -> bool:
    """Fetch the weights once into MODEL_DIR. True if they are present afterwards."""
    import time

    from tools.http import SESSION

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for remote, local in _FILES:
        target = MODEL_DIR / local
        if target.exists() and target.stat().st_size > 0:
            continue
        for attempt in range(_DOWNLOAD_ATTEMPTS):
            try:
                resp = SESSION.get(f"{_HF_BASE}/{remote}", headers=_UA,
                                   timeout=180, stream=True)
                resp.raise_for_status()
                # Write to a .part file first so an interrupted download can never
                # leave a truncated model that loads and then scores nonsense.
                part = target.with_suffix(target.suffix + ".part")
                with open(part, "wb") as fh:
                    for chunk in resp.iter_content(1 << 20):
                        fh.write(chunk)
                part.replace(target)
                break
            except Exception as exc:
                if attempt == _DOWNLOAD_ATTEMPTS - 1:
                    telemetry.record(
                        "rag", "rerank.download_failed",
                        f"Could not fetch reranker weights ({local}): {type(exc).__name__}",
                        level="warn", data={"error": str(exc)[:300]},
                    )
                    return False
                time.sleep(1.5 * (attempt + 1))
    return True


def _runtime():
    """Load tokenizer + ONNX session once. Returns None if unavailable."""
    global _RUNTIME, _UNAVAILABLE
    with _LOCK:
        if _RUNTIME is not None:
            return _RUNTIME
        if _UNAVAILABLE:
            return None
        try:
            if not _download():
                _UNAVAILABLE = True
                return None
            import onnxruntime as ort
            from tokenizers import Tokenizer

            with telemetry.span("rag", "rerank.load", "Loading the reranking model"):
                tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
                tokenizer.enable_truncation(max_length=512)
                tokenizer.enable_padding()
                session = ort.InferenceSession(
                    str(MODEL_DIR / "model.onnx"), providers=["CPUExecutionProvider"]
                )
            names = {i.name for i in session.get_inputs()}
            _RUNTIME = (tokenizer, session, names)
            return _RUNTIME
        except Exception as exc:
            telemetry.record("rag", "rerank.unavailable",
                             f"Reranker disabled: {type(exc).__name__}: {exc}",
                             level="warn")
            _UNAVAILABLE = True
            return None


def available() -> bool:
    """True if reranking can run (loads the model on first call)."""
    return _runtime() is not None


def score(query: str, passages: list[str]) -> list[float] | None:
    """Relevance logit for each (query, passage) pair, or None if unavailable.

    Higher is more relevant. Scores are raw logits, comparable only within one
    call and only for this model — see MIN_RERANK_SCORE.
    """
    if not passages:
        return []
    runtime = _runtime()
    if runtime is None:
        return None

    tokenizer, session, names = runtime
    try:
        import numpy as np

        with telemetry.span("rag", "rerank.score",
                            f"Reranking {len(passages)} passages") as s:
            encodings = tokenizer.encode_batch([(query, p) for p in passages])
            feed = {
                "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encodings],
                                           dtype=np.int64),
            }
            if "token_type_ids" in names:
                feed["token_type_ids"] = np.array([e.type_ids for e in encodings],
                                                  dtype=np.int64)
            logits = session.run(None, {k: v for k, v in feed.items() if k in names})[0]
            scores = [float(x) for x in logits.reshape(-1)]
            s["top_score"] = round(max(scores), 2) if scores else None
        return scores
    except Exception as exc:
        # Never let a reranking failure cost the user their answer.
        telemetry.record("rag", "rerank.error",
                         f"Reranking failed: {type(exc).__name__}: {exc}", level="warn")
        return None


if __name__ == "__main__":
    # Re-calibrate MIN_RERANK_SCORE: the first group should score well above it,
    # the second well below.
    from memory.rag_store import search_policies

    should_ground = [
        "Can I replace my backyard grass with stones?",
        "Do I need a permit to build a deck?",
        "Can I list my house on Airbnb?",
        "How often should I change my HVAC filter?",
    ]
    should_refuse = [
        "Am I allowed to keep a pet tiger in my backyard?",
        "What is the capital of France?",
        "Can I park a submarine in my driveway?",
    ]
    print(f"threshold = {MIN_RERANK_SCORE}\n")
    for label, queries in (("SHOULD GROUND", should_ground), ("SHOULD REFUSE", should_refuse)):
        print(f"--- {label} ---")
        for q in queries:
            hits = search_policies(q, k=4, rerank=False)
            scores = score(q, [h["text"] for h in hits]) or []
            best = max(scores) if scores else float("-inf")
            verdict = "grounded" if best >= MIN_RERANK_SCORE else "refused"
            print(f"  {best:8.2f}  {verdict:9} {q}")
        print()
