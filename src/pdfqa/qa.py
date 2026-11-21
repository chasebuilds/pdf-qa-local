"""Question answering: retrieve, check strength, generate, validate citations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .llm import LLM, NOT_FOUND
from .retrieve import Hit, Retriever

_CITE = re.compile(r"\[([^\[\]]+?) p\.(\d+)\]")


@dataclass
class Answer:
    text: str
    found: bool
    citations: list[tuple[str, int]] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)


def ask(question: str, retriever: Retriever, llm: LLM, k: int = 5) -> Answer:
    hits = retriever.search(question, k)
    if not hits or not retriever.is_answerable(hits):
        return Answer(NOT_FOUND, False, [], hits)
    text = llm.answer(question, hits)
    if NOT_FOUND.rstrip(".").lower() in text.lower():
        return Answer(NOT_FOUND, False, [], hits)
    allowed = {(h.chunk.file, h.chunk.page) for h in hits}
    cites: list[tuple[str, int]] = []
    for m in _CITE.finditer(text):
        ref = (m.group(1).strip(), int(m.group(2)))
        if ref in allowed and ref not in cites:  # drop citations the model made up
            cites.append(ref)
    return Answer(text, True, cites, hits)
