"""Audio loading: turn any supported audio file into 16 kHz mono samples."""

from pathlib import Path

import numpy as np
from faster_whisper import decode_audio

# Whisper models are trained on 16 kHz audio; anything else must be resampled.
SAMPLE_RATE = 16_000
SUPPORTED_EXTENSIONS = {".mp3", ".wav"}


def load_audio(path: str) -> np.ndarray:
    """Decode an audio file into a 1-D float32 array of samples at 16 kHz, mono.

    Raises FileNotFoundError if the file does not exist and ValueError if the
    extension is not supported.
    """
    audio_path = Path(path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if audio_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported audio format '{audio_path.suffix}'. Supported: {supported}"
        )

    # decode_audio handles the container/codec (mp3, wav, ...) via PyAV and
    # resamples to SAMPLE_RATE, mono, float32 in [-1, 1].
    return decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
