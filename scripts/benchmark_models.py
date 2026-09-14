"""Measure CPU speed and output quality of several Whisper sizes on one file.

Not part of the pipeline: a one-off tool to choose the model size with real
numbers (compute time, real-time factor) instead of guesses.

Usage: python scripts/benchmark_models.py samples/sample_fr.wav tiny base small
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faster_whisper import WhisperModel

from speech_to_notes.audio import SAMPLE_RATE, load_audio


def main() -> None:
    audio_file, *sizes = sys.argv[1:]
    audio = load_audio(audio_file)
    duration = len(audio) / SAMPLE_RATE
    print(f"file: {audio_file} ({duration:.1f} s)\n")

    for size in sizes:
        t0 = time.perf_counter()
        model = WhisperModel(size, device="cpu", compute_type="int8")
        load_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        segments, info = model.transcribe(audio, beam_size=5)
        text = " ".join(s.text.strip() for s in segments)  # generator: this runs the model
        run_s = time.perf_counter() - t0

        print(f"=== {size} ===")
        print(f"language: {info.language} (p={info.language_probability:.2f})")
        print(f"model load: {load_s:.1f} s | transcribe: {run_s:.1f} s | RTF: {run_s / duration:.2f}")
        print(f"text: {text}\n")


if __name__ == "__main__":
    main()
