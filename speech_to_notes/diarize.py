"""Speaker diarization: 16 kHz samples in, list of (start, end, speaker) turns out."""

import os
from dataclasses import dataclass

import numpy as np
import torch
from pyannote.audio import Pipeline

from speech_to_notes.audio import SAMPLE_RATE

PIPELINE = "pyannote/speaker-diarization-community-1"


@dataclass
class Turn:
    """One stretch of speech attributed to one speaker by the diarization model.

    ``speaker`` is an arbitrary cluster id ("SPEAKER_00", ...), not an identity.
    Turns of different speakers may overlap when the model detects overlapped speech.
    """

    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def diarize(audio: np.ndarray) -> list[Turn]:
    """Run pyannote on 16 kHz mono samples and return the raw turns, in time order.

    Needs a Hugging Face token in HF_TOKEN (the pipeline's models are gated).
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set; see README, 'Setup'")

    pipeline = Pipeline.from_pretrained(PIPELINE, token=token)
    waveform = torch.from_numpy(audio).unsqueeze(0)  # pyannote wants (channels, samples)
    output = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE})
    annotation = getattr(output, "speaker_diarization", output)

    turns = [
        Turn(start=segment.start, end=segment.end, speaker=label)
        for segment, _, label in annotation.itertracks(yield_label=True)
    ]
    return sorted(turns, key=lambda t: t.start)
