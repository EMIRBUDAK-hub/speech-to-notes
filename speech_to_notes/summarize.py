"""Structured summary: speaker transcript in, four fixed lists out.

Two interchangeable engines (local llama.cpp model, or the Mistral API) behind
one interface: they take a prompt and return text. Everything else -- building
the prompt, reading the reply, validating its shape, retrying, falling back --
is written once and does not care which engine answered.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

SUMMARY_KEYS = ("topics", "decisions", "action_items", "open_questions")

# Chosen over Qwen3-4B on a real transcript: slower, but it found the decision,
# the action and the open question a human noted; Qwen missed the decision.
DEFAULT_LOCAL_MODEL = Path.home() / ".cache/speech-to-notes/llm/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"


@dataclass
class Summary:
    """The four lists a reader wants from a meeting. Empty lists are valid:
    a model that invents items to fill a box is worse than an empty box."""

    topics: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)


PROMPT = """You are taking notes on a recorded conversation. Below is its transcript,
with speaker labels (SPEAKER_00, SPEAKER_01, ...) and timestamps. The transcript
comes from automatic speech recognition and may contain errors.

Fill in the following JSON form and return ONLY the JSON, nothing else:

{{
  "topics": ["short phrase per topic discussed"],
  "decisions": ["each decision that was actually taken"],
  "action_items": ["who does what, one per item"],
  "open_questions": ["each question raised and not settled"]
}}

Rules: write every item in {language}; keep each item to one short sentence;
leave a list empty ([]) if the transcript contains nothing for it; do not
invent anything that is not in the transcript.

Transcript:
{transcript}

{reminder}
"""

# Small local models follow the language the instructions are written in more
# than an instruction *about* language, so the last line speaks the target one.
REMINDERS = {
    "fr": "IMPORTANT : rédige chaque élément du JSON en français, pas en anglais.",
    "en": "IMPORTANT: write every item of the JSON in English.",
}


LANGUAGE_NAMES = {"fr": "French", "en": "English"}


def build_prompt(transcript: str, language: str = "fr") -> str:
    """``language`` is the two-letter code Whisper detected; small local models
    ignore an indirect "same language as the transcript", so we name it."""
    return PROMPT.format(
        transcript=transcript,
        language=LANGUAGE_NAMES.get(language, language),
        reminder=REMINDERS.get(language, f"IMPORTANT: write every item in {language}."),
    )


# ---------------------------------------------------------------- engines


class LocalEngine:
    """A quantised GGUF model run on CPU through llama.cpp, inside this process."""

    def __init__(self, model_path: Path = DEFAULT_LOCAL_MODEL, context: int = 8192):
        from llama_cpp import Llama  # imported here: the API path never needs it

        if not model_path.exists():
            raise FileNotFoundError(f"Local model not found: {model_path}")
        self.name = model_path.stem
        self._llm = Llama(model_path=str(model_path), n_ctx=context, verbose=False)

    def complete(self, prompt: str) -> str:
        out = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},  # constrain output to JSON
            temperature=0.0,
            max_tokens=1024,
        )
        self.last_reply = out["choices"][0]["message"]["content"]
        return self.last_reply


class ApiEngine:
    """A hosted model behind an OpenAI-compatible chat endpoint (Groq, Mistral,
    ...). The provider is a setting, not a class: URL, model, and the name of
    the environment variable that holds the key."""

    PROVIDERS = {
        "groq": ("https://api.groq.com/openai/v1/chat/completions", "openai/gpt-oss-120b", "GROQ_API_KEY"),
        "mistral": ("https://api.mistral.ai/v1/chat/completions", "mistral-small-latest", "MISTRAL_API_KEY"),
    }

    def __init__(self, provider: str = "groq", model: str | None = None):
        url, default_model, key_env = self.PROVIDERS[provider]
        self._url = url
        self.name = f"{provider}:{model or default_model}"
        self._model = model or default_model
        self._key = os.environ.get(key_env)
        if not self._key:
            raise RuntimeError(f"{key_env} is not set; see README, 'Setup'")

    def complete(self, prompt: str) -> str:
        import requests  # imported here: the local path never needs it

        r = requests.post(
            self._url,
            headers={"Authorization": f"Bearer {self._key}"},
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
                "temperature": 0.0,
            },
            timeout=120,
        )
        r.raise_for_status()
        self.last_reply = r.json()["choices"][0]["message"]["content"]
        return self.last_reply


# ---------------------------------------------------------------- your part


def parse_summary(text: str) -> Summary:
    """Read the model's reply and check its shape.

    Must raise ValueError with a message that says what is wrong (this message
    is sent back to the model on retry): not JSON, missing key, unexpected key,
    a value that is not a list, a list item that is not a string.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"not valid JSON: {e}")
    if not isinstance(data, dict):
        raise ValueError("the JSON must be an object with four keys, not a list or a value")
    if set(data) != set(SUMMARY_KEYS):
        raise ValueError(f"keys must be exactly {list(SUMMARY_KEYS)}, got {list(data)}")
    for key in SUMMARY_KEYS:
        if not isinstance(data[key], list):
            raise ValueError(f"{key} must be a list")
        for item in data[key]:
            if not isinstance(item, str):
                raise ValueError(f"every item of {key} must be a string, got {item!r}")
    return Summary(**data)


def summarize(transcript: str, engine, language: str = "fr", retries: int = 1) -> Summary | None:
    """Ask the engine, parse the reply; on a bad reply, retry once with the
    error message appended to the prompt; if it still fails, return None."""
    prompt = build_prompt(transcript, language)
    for attempt in range(retries + 1):
        text = engine.complete(prompt)
        try:
            return parse_summary(text)
        except ValueError as error:
            # feed the mistake back so the next attempt can fix it
            prompt = build_prompt(transcript, language) + (
                f"\n\nYour previous reply was rejected: {error}\n"
                f"Previous reply:\n{text}\n\nReturn only the corrected JSON."
            )
    return None
