"""Run the summary step with each available engine on one speaker transcript.

Not part of the pipeline: prints each engine's time and its four lists, so the
default engine is chosen against a hand-written reference rather than guessed.

Usage: python scripts/benchmark_summary.py output/simsamu_douleur_thoracique.txt [qwen mistral7b groq mistral]
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from speech_to_notes.summarize import ApiEngine, LocalEngine, build_prompt, parse_summary, summarize

LLM_DIR = Path.home() / ".cache/speech-to-notes/llm"
ENGINES = {
    "qwen": lambda: LocalEngine(LLM_DIR / "Qwen3-4B-Instruct-2507.Q4_K_M.gguf"),
    "mistral7b": lambda: LocalEngine(LLM_DIR / "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"),
    "groq": lambda: ApiEngine("groq"),
    "mistral": lambda: ApiEngine("mistral"),
}


def main() -> None:
    transcript = Path(sys.argv[1]).read_text(encoding="utf-8")
    names = sys.argv[2:] or list(ENGINES)
    print(f"transcript: {sys.argv[1]} ({len(transcript.split())} words, prompt ~{len(build_prompt(transcript)) // 4} tokens)\n")

    for name in names:
        try:
            engine = ENGINES[name]()
        except Exception as e:
            print(f"=== {name}: skipped ({e})\n")
            continue
        t0 = time.perf_counter()
        raw = engine.complete(build_prompt(transcript))
        first = time.perf_counter() - t0
        try:
            parse_summary(raw)
            status = "valid JSON on first try"
        except ValueError as e:
            status = f"FIRST REPLY REJECTED ({e})"
        summary = summarize(transcript, engine)
        total = time.perf_counter() - t0
        print(f"=== {name} ({engine.name}): first reply {first:.0f} s, {status}; total {total:.0f} s")
        if summary is None:
            print("  -> FAILED after retry; raw reply:\n" + raw[:600])
        else:
            for key in ("topics", "decisions", "action_items", "open_questions"):
                print(f"  {key}:")
                for item in getattr(summary, key):
                    print(f"    - {item}")
        print()


if __name__ == "__main__":
    main()
