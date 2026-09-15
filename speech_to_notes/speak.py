"""Voice output: read a Summary aloud with a local Piper voice, into a .wav file."""

import wave
from pathlib import Path

from speech_to_notes.summarize import SUMMARY_KEYS, Summary

VOICE_DIR = Path.home() / ".cache/speech-to-notes/tts"
VOICES = {"fr": "fr_FR-siwis-medium", "en": "en_US-lessac-medium"}

# What the voice says for each heading, and for an empty list.
SPOKEN = {
    "fr": {
        "intro": "Résumé de la conversation.",
        "topics": "Sujets abordés",
        "decisions": "Décisions",
        "action_items": "Actions à faire",
        "open_questions": "Questions ouvertes",
        "empty": "aucune",
    },
    "en": {
        "intro": "Summary of the conversation.",
        "topics": "Topics",
        "decisions": "Decisions",
        "action_items": "Action items",
        "open_questions": "Open questions",
        "empty": "none",
    },
}


def summary_to_speech(summary: Summary, language: str = "fr") -> str:
    """Turn the four lists into one text a voice can read naturally.

    Each heading is announced as a word, each item becomes a sentence, an
    empty list is read as "aucune"/"none". Sentences end with a full stop:
    that is where the voice pauses.
    """
    words = SPOKEN[language] if language in SPOKEN else SPOKEN["en"]
    parts = [words["intro"]]
    for key in SUMMARY_KEYS:
        parts.append(f"{words[key]}.")  # announce the heading; the full stop makes the voice pause
        items = getattr(summary, key)  # summary.topics, summary.decisions, ... by name
        if not items:
            parts.append(f"{words['empty']}.")
        for item in items:
            parts.append(item if item.endswith((".", "?", "!")) else item + ".")
    return " ".join(parts)


def synthesize(text: str, language: str, out_path: Path) -> Path:
    """Render ``text`` to a 16-bit mono .wav with the Piper voice for ``language``."""
    from piper import PiperVoice  # imported here: the rest of the pipeline never needs it

    name = VOICES.get(language, VOICES["en"])
    model = VOICE_DIR / f"{name}.onnx"
    if not model.exists():
        raise FileNotFoundError(f"Piper voice not found: {model} (see README, 'Setup')")
    voice = PiperVoice.load(str(model))
    with wave.open(str(out_path), "wb") as wav:
        voice.synthesize_wav(text, wav)
    return out_path
