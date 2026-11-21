"""Synthetic PDFs with invented content and known question/answer pairs."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

SYLLABLES = ["zin", "dle", "quor", "vex", "bram", "tal", "orin", "pex", "lum", "kest", "fay", "nid"]
CITIES = ["Marrowick", "Tellbrook", "Oxenfeld", "Harrowgate", "Pellmoor", "Dunvale", "Sorrel Bay"]
FILLER = [
    "Operators should read the full manual before first use.",
    "All figures were measured under standard laboratory conditions.",
    "Contact the regional office for replacement parts and scheduling.",
    "This section was last reviewed by the quality committee.",
    "Unauthorized modification voids every stated guarantee.",
    "Storage should be dry, shaded and away from open flame.",
    "Training sessions are held quarterly at the main depot.",
    "Records must be kept for audit purposes for the stated period.",
]
# (fact template, question template, answer template, value maker)
ATTRS = [
    (
        "The {p} has a maximum pressure of {v} bar.",
        "What is the maximum pressure of the {p}?",
        "{v} bar",
    ),
    (
        "The warranty for the {p} lasts {v} months.",
        "How long is the warranty for the {p}?",
        "{v} months",
    ),
    ("The {p} is assembled at the {v} plant.", "Where is the {p} assembled?", "{v} plant"),
    (
        "The service interval of the {p} is {v} hours.",
        "What is the service interval of the {p}?",
        "{v} hours",
    ),
    ("The {p} weighs {v} kilograms.", "How heavy is the {p}?", "{v} kilograms"),
    ("The {p} runs on a {v} volt supply.", "What voltage does the {p} need?", "{v} volt"),
]
UNRELATED = [
    "Who won the 2019 world chess championship?",
    "What is the boiling point of mercury?",
    "How do I bake sourdough bread?",
    "What is the capital of Mongolia?",
]


@dataclass(frozen=True)
class QA:
    question: str
    answer: str | None  # None means the documents do not contain it
    file: str | None
    page: int | None


def _name(rng: random.Random, used: set[str]) -> str:
    while True:
        n = (
            rng.choice(SYLLABLES) + rng.choice(SYLLABLES)
        ).capitalize() + f" {rng.randint(2, 9) * 100}"
        if n not in used:
            used.add(n)
            return n


def build_corpus(out_dir: Path, seed: int = 1, n_docs: int = 6, pages_per_doc: int = 5) -> list[QA]:
    """Write PDFs into out_dir and return the question set."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    used: set[str] = set()
    qas: list[QA] = []
    for d in range(n_docs):
        fname = f"manual_{d + 1:02d}.pdf"
        story = []
        for page in range(1, pages_per_doc + 1):
            story.append(Paragraph(f"Section {page}", styles["Heading2"]))
            sentences: list[str] = []
            for attr_i in rng.sample(range(len(ATTRS)), 3):
                fact_t, q_t, a_t = ATTRS[attr_i]
                prod = _name(rng, used) + " unit"
                value = (
                    rng.choice(CITIES)
                    if attr_i == 2
                    else str(rng.choice([12, 18, 24, 36, 48, 60, 87, 110, 150, 230, 400]))
                )
                sentences.append(fact_t.format(p=prod, v=value))
                qas.append(QA(q_t.format(p=prod), a_t.format(v=value), fname, page))
            sentences += rng.sample(FILLER, 3)
            rng.shuffle(sentences)
            story.append(Paragraph(" ".join(sentences), styles["BodyText"]))
            story.append(Spacer(1, 12))
            if page < pages_per_doc:
                story.append(PageBreak())
        SimpleDocTemplate(str(out_dir / fname), pagesize=A4).build(story)
    # Not in the documents: real attribute questions about products that do not exist
    for _ in range(8):
        attr_i = rng.randrange(len(ATTRS))
        qas.append(QA(ATTRS[attr_i][1].format(p=_name(rng, used) + " unit"), None, None, None))
    qas += [QA(q, None, None, None) for q in UNRELATED]
    return qas
