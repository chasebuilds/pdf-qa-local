"""Command line: pdfqa index DIR, pdfqa ask QUESTION."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .embed import Embedder, HashEmbedder, OllamaEmbedder
from .llm import LLM, OllamaLLM, StubLLM
from .qa import Answer, ask
from .retrieve import Retriever
from .store import Index

DEFAULT_DB = Path(".pdfqa") / "index.db"


def make_embedder(stub: bool) -> Embedder:
    return HashEmbedder() if stub else OllamaEmbedder()


def make_llm(stub: bool) -> LLM:
    return StubLLM() if stub else OllamaLLM()


def open_retriever(db: Path, embedder: Embedder) -> Retriever:
    index = Index(db)
    try:
        have = index.embedder_name()
        if have is not None and have != embedder.name:
            raise ValueError(f"Index was built with {have}, not {embedder.name}.")
        chunks, vectors = index.load()
    finally:
        index.close()
    return Retriever(chunks, vectors, embedder)


def format_answer(ans: Answer) -> str:
    lines = [ans.text]
    if ans.found:
        shown = ans.citations or [(h.chunk.file, h.chunk.page) for h in ans.hits[:1]]
        label = "Sources" if ans.citations else "Closest source (answer had no valid citation)"
        lines.append(f"\n{label}: " + ", ".join(f"{f} p.{p}" for f, p in shown))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pdfqa", description=__doc__)
    p.add_argument(
        "--db", type=Path, default=DEFAULT_DB, help="index file (default .pdfqa/index.db)"
    )
    p.add_argument(
        "--stub", action="store_true", help="offline stub model and embedder (demo/tests)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("index", help="index every PDF under a folder")
    pi.add_argument("folder", type=Path)
    pa = sub.add_parser("ask", help="ask a question")
    pa.add_argument("question")
    pa.add_argument("-k", type=int, default=5, help="chunks to retrieve")
    args = p.parse_args(argv)

    try:
        embedder = make_embedder(args.stub)
        if args.cmd == "index":
            if not args.folder.is_dir():
                print(f"not a folder: {args.folder}", file=sys.stderr)
                return 2
            index = Index(args.db)
            try:
                s = index.index_folder(args.folder, embedder)
            finally:
                index.close()
            print(
                f"indexed {s.files_indexed} files ({s.chunks_added} chunks), "
                f"skipped {s.files_skipped} unchanged, removed {s.files_removed}, "
                f"{s.empty_pages} pages had no text"
            )
            return 0
        retriever = open_retriever(args.db, embedder)
        print(format_answer(ask(args.question, retriever, make_llm(args.stub), args.k)))
        return 0
    except (ValueError, RuntimeError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # network and HTTP errors from the local server
        print(
            f"error: {type(e).__name__}: {e}\nIs Ollama running on localhost:11434?",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
