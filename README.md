# speech-to-notes

A command-line tool that turns a recording of a conversation (meeting, interview,
lecture) into structured, readable notes.

Input: an audio file (mp3, wav). Output: a timestamped transcript, then speaker
labels, then a structured summary. Everything runs locally on a CPU-only laptop.
French and English are supported.

## Status

Work in progress. Milestones, in order:

1. Transcription — audio in, timestamped transcript out (in progress)
2. Speaker diarization — who spoke when
3. Structured summary — topics, decisions, action items, open questions
4. Voice output — read the summary aloud with a local TTS engine
5. Polish — clean CLI, examples, known limitations

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Design decisions

### Transcription engine: faster-whisper

Whisper (OpenAI, 2022) is the reference open-weights multilingual model for
speech recognition. Among the implementations that run it on CPU:

- **faster-whisper** (chosen): same weights as the original, ~4x faster on CPU
  thanks to CTranslate2 and int8 quantization, reliable word-level timestamps,
  clean Python API. Word-level timestamps are required to align the transcript
  with speaker diarization in milestone 2.
- **whisper.cpp**: same weights, even faster, but the Python bindings are less
  mature and word-level timestamps are less reliable.
- **Vosk**: much smaller and faster, but one model per language (no language
  detection) and a clearly higher word error rate in French.

## Known limitations

To be written honestly as the project progresses.
