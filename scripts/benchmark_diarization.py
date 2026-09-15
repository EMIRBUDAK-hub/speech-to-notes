"""Run pyannote speaker diarization on one file and compare with a reference .rttm.

Not part of the pipeline: a one-off tool to look at what the model produces
(who speaks when), how long it takes on CPU, and how far it is from a
hand-annotated reference (DER, diarization error rate).

Usage: python scripts/benchmark_diarization.py samples/simsamu_douleur_thoracique.wav
       (the .rttm next to the audio file is used as reference if present)
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

from speech_to_notes.audio import SAMPLE_RATE, load_audio

PIPELINE = "pyannote/speaker-diarization-community-1"


def load_rttm(path: Path) -> Annotation:
    """Read a reference .rttm: 'SPEAKER file 1 start duration <NA> <NA> label <NA> <NA>'."""
    ref = Annotation()
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == "SPEAKER":
            start, dur, label = float(parts[3]), float(parts[4]), parts[7]
            ref[Segment(start, start + dur)] = label
    return ref


def main() -> None:
    audio_file = Path(sys.argv[1])
    audio = load_audio(str(audio_file))
    duration = len(audio) / SAMPLE_RATE
    print(f"file: {audio_file} ({duration:.1f} s)")

    pipeline = Pipeline.from_pretrained(PIPELINE, token=os.environ["HF_TOKEN"])

    t0 = time.perf_counter()
    # pyannote wants a (channels, samples) float tensor
    waveform = torch.from_numpy(audio).unsqueeze(0)
    output = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE})
    elapsed = time.perf_counter() - t0
    hyp = getattr(output, "speaker_diarization", output)  # pyannote 4 wraps the Annotation

    print(f"diarization: {elapsed:.1f} s (RTF {elapsed / duration:.2f})")
    print(f"speakers found: {len(hyp.labels())} -> {hyp.labels()}\n")

    print("first 12 turns (model):")
    for i, (turn, _, speaker) in enumerate(hyp.itertracks(yield_label=True)):
        if i >= 12:
            break
        print(f"  {turn.start:6.2f} -> {turn.end:6.2f}  {speaker}")

    rttm = audio_file.with_suffix(".rttm")
    if rttm.exists():
        ref = load_rttm(rttm)
        print("\nfirst 12 turns (reference .rttm):")
        for i, (turn, _, speaker) in enumerate(ref.itertracks(yield_label=True)):
            if i >= 12:
                break
            print(f"  {turn.start:6.2f} -> {turn.end:6.2f}  {speaker}")
        der = DiarizationErrorRate()
        print(f"\nDER: {100 * der(ref, hyp):.1f} %")


if __name__ == "__main__":
    main()
