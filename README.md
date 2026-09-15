# speech-to-notes

A command-line tool that turns a recording of a conversation (meeting, interview,
lecture) into structured, readable notes.

Input: an audio file (mp3, wav, m4a). Output: a timestamped transcript, then speaker
labels, then a structured summary. Everything runs locally on a CPU-only laptop.
French and English are supported.

## Status

Work in progress. Milestones, in order:

1. Transcription — audio in, timestamped transcript out (done)
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

## Usage

```bash
python -m speech_to_notes recording.mp3
```

Writes two files to `output/`:

- `recording.txt` — one line per segment, `[MM:SS] text`, for people to read
- `recording.json` — every segment and word with start/end times, kept so
  later stages (speaker diarization) can reuse the transcript instead of
  recomputing it

Options: `--model tiny|base|small|medium` (default `small`), `--language fr`
(default: auto-detect), `--output-dir`.

Example on a 12 s French recording:

```
$ python -m speech_to_notes samples/sample_fr.wav
Loaded samples/sample_fr.wav (11.7 s). Transcribing with 'small'...
Done in 4.1 s (RTF 0.35). 1 segment(s) -> output/sample_fr.txt / .json

$ cat output/sample_fr.txt
[00:01] Rendez-vous le 14 à 15h30 avec monsieur Lefebvre, budget 2300€.
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

### Model size: small

Measured on a 12 s French recording (a sentence with a date, a surname and an
amount), CPU only, int8 (`scripts/benchmark_models.py`):

| model | size   | RTF  | output quality                                   |
|-------|--------|------|--------------------------------------------------|
| tiny  | 75 MB  | 0.06 | common words fine; surname and amount wrong      |
| base  | 145 MB | 0.10 | amount right, surname still wrong                |
| small | 480 MB | 0.29 | everything right                                 |

RTF (real-time factor) is compute time divided by audio duration: at 0.29,
one hour of audio takes about 17 minutes to transcribe, 5x more than `tiny`.
That cost is acceptable because the tool runs offline on a finished recording
(archiving, notes read later), so nobody waits in front of the screen. What
breaks first in smaller models is exactly what matters in meeting notes: proper
nouns, numbers, less frequent vocabulary — the words that are rare in the
training data. One 12 s file is not enough to settle this for good; the
benchmark will be re-run on longer, more varied recordings (other voices,
English) before the choice is final.

Whisper models are pre-trained by OpenAI; this project only runs inference.

## Known limitations

- **Quiet or phone-quality audio.** On a real 8 kHz emergency-call recording
  (Simsamu corpus), the first version produced only empty segments: Whisper
  conditions each 30 s window on the previous output, and a silent, very quiet
  first window sent it into a loop of "." segments. Two fixes are in place:
  peak normalisation in `load_audio` and Silero VAD (`vad_filter=True`) in
  `transcribe`. Normalisation is naive: a single loud transient (a door slam)
  would set the peak and leave speech quiet.
- **Rare words on degraded audio.** On the same phone-quality recording,
  "docteur Damani" comes out as "deux pères de vanille". Proper nouns are the
  first casualty; see the model-size section.
- **Not bit-for-bit reproducible across exports.** The same recording exported
  as .wav and as .m4a gave two transcripts with the same meaning but different
  wording in places, and one of them dropped a quiet 45 s passage (VAD
  threshold). Tiny numeric differences in the input can flip beam search and
  VAD decisions. Judging quality requires a reference transcript, not a diff
  between two runs.
