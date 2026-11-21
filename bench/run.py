"""Benchmark: retrieval hit rate and citation correctness on synthetic PDFs.

python -m bench.run --stub            offline, no model
python -m bench.run --ollama          real local models (only if Ollama answers on localhost)
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import httpx

from bench.synth import QA, build_corpus
from pdfqa.embed import Embedder, HashEmbedder, OllamaEmbedder
from pdfqa.llm import LLM, OllamaLLM, StubLLM
from pdfqa.qa import ask
from pdfqa.retrieve import Retriever
from pdfqa.store import Index


def ollama_up(base: str = "http://localhost:11434") -> bool:
    try:
        return httpx.get(f"{base}/api/tags", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


def evaluate(
    qas: list[QA], docs: Path, embedder: Embedder, llm: LLM, k: int = 5
) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        idx = Index(Path(tmp) / "bench.db")
        t0 = time.time()
        idx.index_folder(docs, embedder)
        chunks, vecs = idx.load()
        idx.close()
        index_s = time.time() - t0
    retriever = Retriever(chunks, vecs, embedder)
    answerable = [q for q in qas if q.answer is not None]
    refusals = [q for q in qas if q.answer is None]
    hit1 = hitk = ans_ok = cite_ok = false_refusal = 0
    t1 = time.time()
    for q in answerable:
        result = ask(q.question, retriever, llm, k)
        gold = (q.file, q.page)
        locs = [(h.chunk.file, h.chunk.page) for h in result.hits]
        hit1 += bool(locs and locs[0] == gold)
        hitk += gold in locs
        false_refusal += not result.found
        assert q.answer is not None
        ans_ok += result.found and q.answer.lower() in result.text.lower()
        cite_ok += result.found and gold in result.citations
    refused_ok = sum(not ask(q.question, retriever, llm, k).found for q in refusals)
    ask_s = time.time() - t1
    n = len(answerable)
    return {
        "embedder": embedder.name,
        "llm": llm.name,
        "documents": len({c.file for c in chunks}),
        "chunks": len(chunks),
        "answerable_questions": n,
        "unanswerable_questions": len(refusals),
        f"retrieval_hit_at_{k}": round(hitk / n, 3),
        "retrieval_hit_at_1": round(hit1 / n, 3),
        "answer_contains_fact": round(ans_ok / n, 3),
        "citation_correct": round(cite_ok / n, 3),
        "false_refusal_rate": round(false_refusal / n, 3),
        "correct_refusal_rate": round(refused_ok / len(refusals), 3),
        "index_seconds": round(index_s, 1),
        "ask_seconds_per_question": round(ask_s / (n + len(refusals)), 2),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stub", action="store_true")
    p.add_argument("--ollama", action="store_true")
    p.add_argument(
        "--seed", type=int, default=2, help="corpus seed (thresholds were tuned on seed 1)"
    )
    p.add_argument("--limit", type=int, default=0, help="use N evenly spaced answerable questions")
    p.add_argument("--out", type=Path, default=None, help="write JSON results here")
    args = p.parse_args()
    if not (args.stub or args.ollama):
        p.error("choose --stub and/or --ollama")
    work = Path(__file__).parent / "data" / f"seed{args.seed}"
    qas = build_corpus(work, seed=args.seed)
    if args.limit:
        ans = [q for q in qas if q.answer is not None]
        step = max(len(ans) // args.limit, 1)
        qas = ans[::step][: args.limit] + [q for q in qas if q.answer is None]
    results = []
    if args.stub:
        results.append(evaluate(qas, work, HashEmbedder(), StubLLM()))
    if args.ollama:
        if not ollama_up():
            print("Ollama is not answering on localhost:11434, skipping the real run.")
        else:
            results.append(evaluate(qas, work, OllamaEmbedder(), OllamaLLM()))
    for r in results:
        print(json.dumps(r, indent=2))
    if args.out and results:
        args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
