"""parse_summary must accept exactly the four-list form and name what is wrong otherwise."""

import pytest

from speech_to_notes.summarize import Summary, parse_summary, summarize

GOOD = (
    '{"topics": ["douleur thoracique"], "decisions": ["hôpital en voiture"], '
    '"action_items": ["la fille emmène sa mère"], "open_questions": []}'
)


def test_valid_form_is_parsed():
    s = parse_summary(GOOD)
    assert s.decisions == ["hôpital en voiture"]
    assert s.open_questions == []  # an empty list is valid


@pytest.mark.parametrize(
    "reply, message",
    [
        ("Voici le résumé : " + GOOD, "not valid JSON"),  # chatter around the JSON
        ('{"topics": [], "decisions": [], "action_items": []}', "keys must be exactly"),  # missing key
        (GOOD[:-1] + ', "notes": "x"}', "keys must be exactly"),  # unexpected key
        ('{"topics": "douleur", "decisions": [], "action_items": [], "open_questions": []}', "must be a list"),
        ('{"topics": [1, 2], "decisions": [], "action_items": [], "open_questions": []}', "must be a string"),
        ("[1, 2, 3]", "must be an object"),  # valid JSON, wrong shape
    ],
)
def test_bad_replies_are_rejected_with_a_reason(reply, message):
    with pytest.raises(ValueError, match=message):
        parse_summary(reply)


class FakeEngine:
    """Returns scripted replies one by one and records the prompts it received."""

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def test_summarize_retries_once_with_the_error_then_succeeds():
    engine = FakeEngine(["not json at all", GOOD])
    result = summarize("transcript", engine)
    assert isinstance(result, Summary)
    assert len(engine.prompts) == 2
    assert "rejected" in engine.prompts[1] and "not valid JSON" in engine.prompts[1]


def test_summarize_gives_up_after_the_retry():
    engine = FakeEngine(["nope", "nope", "nope"])
    assert summarize("transcript", engine) is None
    assert len(engine.prompts) == 2  # one attempt + one retry, never more
