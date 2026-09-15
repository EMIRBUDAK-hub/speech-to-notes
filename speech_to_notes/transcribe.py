"""Transcription: 16 kHz audio samples in, timestamped segments and words out."""

from dataclasses import dataclass, field

import numpy as np
from faster_whisper import WhisperModel

DEFAULT_MODEL_SIZE = "small"  # see README, "Model size" for the measured rationale


@dataclass
class Word:
    """One recognised word and the time span (seconds) it was spoken in."""

    start: float
    end: float
    text: str


@dataclass
class Segment:
    """A phrase-sized chunk of transcript with its words."""

    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


def transcribe(
    audio: np.ndarray,
    model_size: str = DEFAULT_MODEL_SIZE,
    language: str | None = None,
) -> list[Segment]:
    """Run Whisper on 16 kHz mono samples and return timestamped segments.

    ``language`` is a two-letter code ("fr", "en"); None lets the model detect it.
    """
    model = WhisperModel(model_size, device="cpu", compute_type="int8")                      # 1. load the model
    raw_segments, info = model.transcribe(audio, beam_size=5, word_timestamps=True, language=language)     # 2. run Whisper
    segments = []
    for s in raw_segments:                                                                       # 3. convert to our own Segment / Word
        words = [Word(start=w.start, end=w.end, text=w.word.strip()) for w in s.words]
        segments.append(Segment(start=s.start, end=s.end, text=s.text.strip(), words=words))
    return segments                                # 4.Done
