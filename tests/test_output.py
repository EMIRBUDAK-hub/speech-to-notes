import pytest

from speech_to_notes.output import format_timestamp


@pytest.mark.parametrize(
    "seconds, expected",
    [(0, "00:00"), (1.24, "00:01"), (59.9, "00:59"), (60, "01:00"), (75.3, "01:15"), (3599.5, "59:59"), (3600, "60:00")],
)
def test_format_timestamp(seconds, expected):
    assert format_timestamp(seconds) == expected
