"""How many transcribed words got the right speaker, against a reference .rttm.

Not part of the pipeline. DER measures the diarization model alone; this measures
what the reader actually sees: for each word in output/<name>.json, the reference
speaker at the word's midpoint vs the speaker our alignment assigned.

Usage: python scripts/word_attribution_accuracy.py samples/simsamu_douleur_thoracique
       (reads samples/<name>.rttm and output/<name>.json)
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from speech_to_notes.align import assign_speakers
from speech_to_notes.diarize import Turn
from speech_to_notes.transcribe import Word


def load_rttm(path: Path) -> list[Turn]:
    turns = []
    for line in path.read_text().splitlines():
        p = line.split()
        if p and p[0] == "SPEAKER":
            turns.append(Turn(float(p[3]), float(p[3]) + float(p[4]), p[7]))
    return turns


def reference_speaker(t: float, ref: list[Turn]) -> str | None:
    hits = [r for r in ref if r.start <= t <= r.end]
    return hits[0].speaker if hits else None


def main() -> None:
    stem = Path(sys.argv[1])
    ref = load_rttm(stem.with_suffix(".rttm"))
    data = json.loads((Path("output") / f"{stem.name}.json").read_text())
    words = [Word(**w) for s in data["segments"] for w in s["words"]]
    turns = [Turn(**t) for t in data["turns"]]

    labeled = assign_speakers(words, turns)

    # Map our cluster ids to reference names by majority vote (ids are arbitrary).
    votes: dict[str, Counter] = {}
    for w, spk in labeled:
        r = reference_speaker((w.start + w.end) / 2, ref)
        if r:
            votes.setdefault(spk, Counter())[r] += 1
    mapping = {spk: c.most_common(1)[0][0] for spk, c in votes.items()}
    print("cluster -> reference:", mapping)

    correct = wrong = no_ref = 0
    errors = []
    for w, spk in labeled:
        r = reference_speaker((w.start + w.end) / 2, ref)
        if r is None:
            no_ref += 1
        elif mapping.get(spk) == r:
            correct += 1
        else:
            wrong += 1
            errors.append(f"{w.start:6.2f} {w.text:12s} got {mapping.get(spk, spk):8s} ref {r}")

    total = correct + wrong
    print(f"words: {len(words)} | with reference: {total} | correct: {correct} ({100 * correct / total:.1f} %) | wrong: {wrong} | in silence per reference: {no_ref}")
    print("\nwrong words:")
    print("\n".join(errors))


if __name__ == "__main__":
    main()
