"""Attach a speaker to every transcribed word, then group words into utterances.

Rules (decided on real data, see notes and README):
  1. Diarization turns shorter than MIN_TURN_DURATION are dropped (boundary artifacts).
  2. Each word goes to the turn with the largest overlap with [word.start, word.end];
     ties go to the longer turn (in overlapped speech Whisper transcribes the dominant
     voice, so giving its words to the interrupter is the costlier mistake).
  3. A word overlapping no turn goes to the nearest turn if it is at most MAX_GAP away,
     otherwise to UNKNOWN.
  4. Words may have start == end (Whisper artifact); the rules above must still work.
  5. Consecutive words with the same speaker form one utterance; its timestamp is the
     start of its first word.
  6. A kept turn fully inside a longer turn of another speaker is reported as an
     overlap marker after the utterance it falls in.
"""

from dataclasses import dataclass

from speech_to_notes.diarize import Turn
from speech_to_notes.transcribe import Word

MIN_TURN_DURATION = 0.15  # seconds; shorter turns are segmentation noise
MAX_GAP = 1.0  # seconds; a word further than this from any turn is UNKNOWN
UNKNOWN = "UNKNOWN"


@dataclass
class Utterance:
    """One line of the final transcript: who said what, from when to when."""

    start: float
    end: float
    speaker: str
    text: str


def overlap(word: Word, turn: Turn) -> float:
    """Length in seconds of the intersection of the word and the turn (0 if none)."""
    start = max(word.start, turn.start)  # the common part begins at the later of the two starts
    end = min(word.end, turn.end)  # ...and ends at the earlier of the two ends
    return max(0.0, end - start)  # negative means no common part at all -> 0


def distance(word: Word, turn: Turn) -> float:
    """Gap in seconds between the word and the turn (0 if they touch or overlap)."""
    # Only one of the two differences can be positive: word before the turn,
    # or word after the turn. Both negative means they overlap -> 0.
    return max(0.0, turn.start - word.end, word.start - turn.end)


def assign_speakers(words: list[Word], turns: list[Turn]) -> list[tuple[Word, str]]:
    """Rules 1-4: return (word, speaker) pairs, in the words' original order."""
    # Rule 1: drop the tiny turns the segmentation model emits around speaker
    # changes (0.02-0.12 s on Simsamu) -- nobody says a syllable in 20 ms.
    kept = []
    for t in turns:
        if t.duration >= MIN_TURN_DURATION:
            kept.append(t)

    result = []
    for word in words:
        # Rule 2: largest overlap wins; ties go to the longer turn. `max` compares
        # the (overlap, duration) tuples element by element, so duration only
        # matters when overlaps are equal -- that is the tie-break rule.
        best = max(kept, key=lambda t: (overlap(word, t), t.duration))
        if overlap(word, best) > 0:
            speaker = best.speaker
        else:
            # Rule 3: no turn touches the word -> nearest turn, if close enough.
            # Rule 4 lands here too: a zero-duration word (start == end, a Whisper
            # artifact) has overlap 0 with every turn, but its distance to the
            # turn that contains it is 0, so the nearest turn still claims it.
            nearest = min(kept, key=lambda t: distance(word, t))
            if distance(word, nearest) <= MAX_GAP:
                speaker = nearest.speaker
            else:
                speaker = UNKNOWN
        result.append((word, speaker))
    return result


def group_utterances(labeled: list[tuple[Word, str]]) -> list[Utterance]:
    """Rule 5: merge consecutive words of the same speaker into utterances."""
    raise NotImplementedError
