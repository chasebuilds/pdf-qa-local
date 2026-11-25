"""Write the synthetic corpus and its question list as JSON for the browser benchmark.

python -m bench.export_questions --seed 2
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from bench.synth import build_corpus


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=2)
    args = p.parse_args()
    work = Path(__file__).parent / "data" / f"seed{args.seed}"
    qas = build_corpus(work, seed=args.seed)
    out = work / "questions.json"
    out.write_text(json.dumps([asdict(q) for q in qas], indent=1) + "\n", encoding="utf-8")
    print(f"{len(qas)} questions -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
