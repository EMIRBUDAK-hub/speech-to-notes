"""Tests for the pure functions of the alignment step: no models, no audio."""

import pytest

from speech_to_notes.align import Utterance, distance, overlap, smooth
from speech_to_notes.diarize import Turn
from speech_to_notes.transcribe import Word

TURN = Turn(4.49, 10.53, "SPEAKER_01")


@pytest.mark.parametrize(
    "word, expected",
    [
        (Word(4.70, 4.95, "bonjour"), 0.25),  # fully inside the turn
        (Word(4.30, 4.60, "oui"), 0.11),  # straddles the start
        (Word(10.40, 10.80, "ça"), 0.13),  # straddles the end
        (Word(11.00, 11.30, "non"), 0.0),  # after the turn
        (Word(3.00, 3.20, "allô"), 0.0),  # before the turn
        (Word(5.00, 5.00, "l"), 0.0),  # zero-duration word (Whisper artifact)
        (Word(4.00, 12.00, "long"), 6.04),  # word longer than the turn
    ],
)
def test_overlap(word, expected):
    assert overlap(word, TURN) == pytest.approx(expected)


# --- your turn: same shape as test_overlap, for distance ---------------------
# Cases to cover: word before the turn (gap = turn.start - word.end), word after
# (gap = word.start - turn.end), word inside (0), word straddling an edge (0),
# zero-duration word inside (0).
@pytest.mark.parametrize(
    "word, expected",
    [
        (Word(3.00, 3.20, "allô"), 1.29),   # before: 4.49 - 3.20
        (Word(11.00, 11.30, "non"), 0.47),   # after: word.start - 10.53
        (Word(4.70, 4.95, "bonjour"), 0.0), # inside
        (Word(4.30, 4.60, "oui"), 0.0),     # straddles the start
        (Word(5.00, 5.00, "l"), 0.0),       # zero-duration, inside
    ],
)
def test_distance(word, expected):
    assert distance(word,TURN)==pytest.approx(expected)


def test_smooth_absorbs_a_tiny_sandwiched_line():
    lines = [
        Utterance(37.9, 39.8, "SPEAKER_01", "c'était là-bas."),
        Utterance(40.8, 41.2, "SPEAKER_00", "Vous"),  # 0.4 s, wedged between two SPEAKER_01 lines
        Utterance(41.2, 42.4, "SPEAKER_01", "êtes avec elle."),
    ]
    out = smooth(lines)
    assert [u.speaker for u in out] == ["SPEAKER_01"]
    assert out[0].text == "c'était là-bas. Vous êtes avec elle."


def test_smooth_keeps_a_real_short_reply():
    lines = [
        Utterance(2.0, 10.5, "SPEAKER_01", "Vous appelez pour votre mère ?"),
        Utterance(11.2, 13.0, "SPEAKER_00", "Oui, c'est ça."),  # 1.8 s: a real answer, not noise
        Utterance(13.5, 15.8, "SPEAKER_01", "C'était la première fois ?"),
    ]
    assert [u.speaker for u in smooth(lines)] == ["SPEAKER_01", "SPEAKER_00", "SPEAKER_01"]
