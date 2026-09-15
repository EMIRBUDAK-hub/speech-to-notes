"""Write transcripts to disk: a readable .txt for people, a .json cache for the pipeline."""

import json
from dataclasses import asdict
from pathlib import Path

from speech_to_notes.align import Utterance
from speech_to_notes.diarize import Turn
from speech_to_notes.transcribe import Segment


def format_timestamp(seconds: float) -> str:
    """Turn a time in seconds into "MM:SS" (e.g. 75.3 -> "01:15")."""
    minutes = int(seconds // 60)  # 1: whole minutes (integer division)
    secs = int(seconds % 60)  # 2: what is left after the minutes (remainder)
    return f"{minutes:02d}:{secs:02d}"  # 3: two digits each, zero-padded


def save_text(segments: list[Segment], path: str) -> None:
    """Write one line per segment: "[MM:SS] text"."""
    lines = [f"[{format_timestamp(s.start)}] {s.text}" for s in segments]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_speaker_text(utterances: list[Utterance], path: str) -> None:
    """Write one line per utterance: "[MM:SS] SPEAKER_00: text", plus overlap markers."""
    lines = []
    for u in utterances:
        lines.append(f"[{format_timestamp(u.start)}] {u.speaker}: {u.text}")
        for o in u.overlaps:
            lines.append(f"        [{o.speaker} speaks at the same time, {o.start:.1f}-{o.end:.1f} s]")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rounded(d: dict) -> dict:
    """Round every start/end to 10 ms: enough for alignment, avoids 1.2399999999999998."""
    for k in ("start", "end"):
        if k in d:
            d[k] = round(d[k], 2)
    for w in d.get("words", []):
        _rounded(w)
    return d


def save_json(segments: list[Segment], path: str, turns: list[Turn] | None = None) -> None:
    """Write every segment and word with timestamps (and the diarization turns, if
    any), so later stages can reuse the results instead of recomputing them."""
    data = {
        "segments": [_rounded(asdict(s)) for s in segments],
        "turns": [_rounded(asdict(t)) for t in turns] if turns is not None else None,
    }
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
