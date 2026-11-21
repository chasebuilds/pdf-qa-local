"""Tests run with the stub embedder and stub model. No Ollama needed."""

from __future__ import annotations

from pathlib import Path

import pytest
from bench.synth import QA, build_corpus

from pdfqa.cli import main
from pdfqa.embed import HashEmbedder, OllamaEmbedder, ensure_local
from pdfqa.llm import NOT_FOUND, StubLLM, build_prompt
from pdfqa.qa import ask
from pdfqa.retrieve import Retriever
from pdfqa.store import Index
from pdfqa.text import chunk_pages, split_page, tokenize


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[QA]]:
    d = tmp_path_factory.mktemp("docs")
    return d, build_corpus(d, seed=1, n_docs=3, pages_per_doc=3)


def make_retriever(docs: Path, db: Path) -> Retriever:
    emb = HashEmbedder()
    idx = Index(db)
    idx.index_folder(docs, emb)
    chunks, vecs = idx.load()
    idx.close()
    return Retriever(chunks, vecs, emb)


def test_tokenize_drops_stopwords_and_plurals() -> None:
    assert tokenize("The pumps are in the Zindle 400 unit.") == ["pump", "zindle", "400", "unit"]


def test_chunks_keep_page_numbers() -> None:
    chunks = chunk_pages("a.pdf", ["First page. " * 120, "", "Third page text."])
    assert {c.page for c in chunks} == {1, 3}
    assert all(len(c.text) <= 700 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_split_page_short_and_empty() -> None:
    assert split_page("   ") == []
    assert split_page("One sentence.") == ["One sentence."]


def test_answer_cites_file_and_page(corpus: tuple[Path, list[QA]], tmp_path: Path) -> None:
    docs, qas = corpus
    r = make_retriever(docs, tmp_path / "i.db")
    q = next(x for x in qas if x.answer)
    res = ask(q.question, r, StubLLM())
    assert res.found
    assert (q.file, q.page) in res.citations
    assert q.answer is not None and q.answer.lower() in res.text.lower()


def test_unrelated_question_is_refused(corpus: tuple[Path, list[QA]], tmp_path: Path) -> None:
    docs, _ = corpus
    r = make_retriever(docs, tmp_path / "i.db")
    res = ask("Who won the 2019 world chess championship?", r, StubLLM())
    assert not res.found
    assert res.text == NOT_FOUND
    assert res.citations == []


def test_missing_entity_is_refused(corpus: tuple[Path, list[QA]], tmp_path: Path) -> None:
    docs, _ = corpus
    r = make_retriever(docs, tmp_path / "i.db")
    res = ask("What is the maximum pressure of the Quorvex 901 unit?", r, StubLLM())
    assert not res.found


class FabricatingLLM:
    name = "fab"

    def answer(self, question: str, hits: list) -> str:  # type: ignore[type-arg]
        return "It is 5 bar [ghost.pdf p.99]."


def test_invented_citations_are_dropped(corpus: tuple[Path, list[QA]], tmp_path: Path) -> None:
    docs, qas = corpus
    r = make_retriever(docs, tmp_path / "i.db")
    res = ask(next(x for x in qas if x.answer).question, r, FabricatingLLM())
    assert res.found and res.citations == []


class RefusingLLM:
    name = "refuse"

    def answer(self, question: str, hits: list) -> str:  # type: ignore[type-arg]
        return NOT_FOUND


def test_model_refusal_passes_through(corpus: tuple[Path, list[QA]], tmp_path: Path) -> None:
    docs, qas = corpus
    r = make_retriever(docs, tmp_path / "i.db")
    res = ask(next(x for x in qas if x.answer).question, r, RefusingLLM())
    assert not res.found and res.text == NOT_FOUND


def test_reindex_skips_unchanged_and_removes_deleted(tmp_path: Path) -> None:
    docs = tmp_path / "d"
    build_corpus(docs, seed=3, n_docs=2, pages_per_doc=2)
    idx = Index(tmp_path / "i.db")
    emb = HashEmbedder()
    assert idx.index_folder(docs, emb).files_indexed == 2
    again = idx.index_folder(docs, emb)
    assert again.files_skipped == 2 and again.files_indexed == 0
    (docs / "manual_02.pdf").unlink()
    assert idx.index_folder(docs, emb).files_removed == 1
    assert {c.file for c in idx.load()[0]} == {"manual_01.pdf"}
    idx.close()


def test_index_refuses_other_embedder(tmp_path: Path) -> None:
    docs = tmp_path / "d"
    build_corpus(docs, seed=3, n_docs=1, pages_per_doc=1)
    idx = Index(tmp_path / "i.db")
    idx.index_folder(docs, HashEmbedder())
    with pytest.raises(ValueError, match="built with"):
        idx.index_folder(docs, HashEmbedder(dim=64))
    idx.close()


def test_remote_urls_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PDFQA_ALLOW_REMOTE", raising=False)
    assert ensure_local("http://localhost:11434/") == "http://localhost:11434"
    with pytest.raises(ValueError, match="not a local address"):
        ensure_local("https://api.example.com")
    with pytest.raises(ValueError):
        OllamaEmbedder(base_url="http://10.0.0.5:11434")


def test_prompt_lists_sources_with_citations(corpus: tuple[Path, list[QA]], tmp_path: Path) -> None:
    docs, qas = corpus
    r = make_retriever(docs, tmp_path / "i.db")
    q = next(x for x in qas if x.answer)
    prompt = build_prompt(q.question, r.search(q.question, 2))
    assert "[manual_" in prompt and "Question:" in prompt


def test_cli_end_to_end(
    corpus: tuple[Path, list[QA]], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    docs, qas = corpus
    db = str(tmp_path / "cli.db")
    assert main(["--stub", "--db", db, "index", str(docs)]) == 0
    q = next(x for x in qas if x.answer)
    assert main(["--stub", "--db", db, "ask", q.question]) == 0
    out = capsys.readouterr().out
    assert f"{q.file} p.{q.page}" in out
    assert main(["--stub", "--db", db, "ask", "Who won the 2019 world chess championship?"]) == 0
    assert NOT_FOUND in capsys.readouterr().out


def test_cli_bad_folder(tmp_path: Path) -> None:
    assert main(["--stub", "--db", str(tmp_path / "x.db"), "index", str(tmp_path / "nope")]) == 2
