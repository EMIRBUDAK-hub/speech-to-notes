"""Write transcripts to disk: a readable .txt for people, a .json cache for the pipeline."""

import json
from dataclasses import asdict
from pathlib import Path

from speech_to_notes.transcribe import Segment


def format_timestamp(seconds: float) -> str:
    """Turn a time in seconds into "MM:SS" (e.g. 75.3 -> "01:15")."""
    minutes = int(seconds // 60)        # 1 : division entière par 60, converti en entier
    secs = int(seconds % 60)             # 2 : reste de la division par 60, converti en entier
    return f"{minutes:02d}:{secs:02d}"           # 3 : la chaîne "MM:SS"

def save_text(segments: list[Segment], path: str) -> None:
    """Write one line per segment: "[MM:SS] text"."""
    lines = [f"[{format_timestamp(s.start)}] {s.text}" for s in segments]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_json(segments: list[Segment], path: str) -> None:
    """Write every segment and word with timestamps, so nothing is lost."""
    data = [asdict(s) for s in segments]  # dataclass -> dict, recursively (words included)
    for seg in data:  # round to 10 ms: enough for alignment, avoids 1.2399999999999998
        seg["start"], seg["end"] = round(seg["start"], 2), round(seg["end"], 2)
        for w in seg["words"]:
            w["start"], w["end"] = round(w["start"], 2), round(w["end"], 2)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
