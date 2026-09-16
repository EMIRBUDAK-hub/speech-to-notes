# speech-to-notes

A command-line tool that turns a recording of a conversation (meeting, interview,
lecture) into structured, readable notes.

Input: an audio file (mp3, wav, m4a). Output: a timestamped transcript, then speaker
labels, then a structured summary. Everything runs locally on a CPU-only laptop.
French and English are supported.

## Status

All five milestones are in place:

1. Transcription — audio in, timestamped transcript out
2. Speaker diarization — who spoke when
3. Structured summary — topics, decisions, action items, open questions
4. Voice output — the summary read aloud by a local TTS engine
5. Polish — CLI, tests on what actually breaks, this README

Every design decision below was measured before it was taken; the numbers are
from one laptop (8-core CPU, no GPU) and one annotated French recording, and
the [Known limitations](#known-limitations) say what has *not* been checked.

Layout: `speech_to_notes/` is the pipeline (one module per stage: `audio`,
`transcribe`, `diarize`, `align`, `summarize`, `speak`, `output`, `cli`);
`scripts/` holds the one-off benchmarks behind the tables below; `tests/`
covers the pure logic.

## Setup

Python 3.12 (PyTorch does not fully support 3.14 yet). On Windows, use WSL
(Ubuntu): Smart App Control blocks the unsigned native libraries that PyTorch
and PyAV ship, and it cannot be re-enabled once turned off, so a Linux
environment is the safer choice.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio torchcodec
pip install -r requirements.txt
```

The structured summary (`--summarize local`) needs a GGUF model on disk:

```bash
mkdir -p ~/.cache/speech-to-notes/llm && cd ~/.cache/speech-to-notes/llm
curl -LO https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
```

`--summarize api` calls a hosted model instead, through any OpenAI-compatible
endpoint. Default provider is Groq (free tier, no card): create a key on
console.groq.com and export it as `GROQ_API_KEY`. `--api-provider mistral`
uses Mistral's API with `MISTRAL_API_KEY` (their free plan was not available
when this was written, so that path is implemented but untested).

Voice output (`--speak`) needs a Piper voice per language (60 MB each):

```bash
mkdir -p ~/.cache/speech-to-notes/tts && cd ~/.cache/speech-to-notes/tts
base=https://huggingface.co/rhasspy/piper-voices/resolve/main
curl -LO $base/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx -LO $base/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx.json
curl -LO $base/en/en_US/lessac/medium/en_US-lessac-medium.onnx -LO $base/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

Speaker diarization (`--diarize`) uses pyannote models gated on Hugging Face:
accept the terms of `pyannote/speaker-diarization-community-1` (and, for
older pyannote versions, `speaker-diarization-3.1` and `segmentation-3.0`),
create a read token, and export it as `HF_TOKEN`.

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
(default: auto-detect), `--diarize` (who speaks when; slower, needs `HF_TOKEN`),
`--summarize local|api`, `--speak` (read the summary aloud), `--output-dir`.

The whole pipeline, audio in and audio out:

```bash
python -m speech_to_notes meeting.mp3 --diarize --summarize local --speak
```

With `--diarize`, the `.txt` has one line per speaker turn, and a marker
whenever the diarization model heard someone else speak *during* a line —
those words are not in the transcript (Whisper writes the dominant voice), but
the reader knows where to listen:

```
[00:02] SPEAKER_01: Oui, bonjour madame. Je suis [...] le médecin régulateur, [...]
        [SPEAKER_00 speaks at the same time, 4.6-5.0 s]
[00:11] SPEAKER_00: Oui, c'est ça.
```

Speaker labels are cluster ids, not identities: the model cannot know who is
the doctor and who is the patient, only that there are two voices.

With `--summarize local` (or `api`), a third file `<name>.summary.md` holds
four fixed lists — topics, decisions, action items, open questions — with
`(none)` where the conversation had nothing to offer. On the same call:

```
## Decisions
- l'individu doit aller à l'hôpital
## Action items
- la fille doit conduire la personne malade à l'hôpital
## Open questions
- est-ce que la personne a déjà eu des problèmes de santé cardiaques?
```

Example on a 12 s French recording, summary through the API and read aloud:

```
$ python -m speech_to_notes samples/sample_fr.wav --summarize api --speak
Loaded samples/sample_fr.wav (11.7 s). Transcribing with 'small'...
  transcription: 3.2 s (RTF 0.27), 1 segment(s), language 'fr'
Summarizing with 'api'...
  summary: 1.1 s with groq:openai/gpt-oss-120b, ok
  speech: 1.7 s -> output/sample_fr.summary.wav
Done -> output/sample_fr.txt / .json / .summary.md / .summary.wav

$ cat output/sample_fr.txt
[00:01] Rendez-vous le 14 à 15h30 avec Monsieur Lefebvre, budget 2300€
```

## Tests

```bash
python -m pytest tests
```

Thirty tests, all on pure logic that a wrong sign or a changed model reply
would break silently: the overlap/distance maths and the smoothing rule of
the alignment, the summary parser and its retry loop, timestamp formatting.
Nothing tests the models themselves — slow, non-deterministic, and not our
code; the `scripts/` benchmarks are how those are judged.

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

### Speaker diarization: pyannote.audio, then our own word alignment

Diarization answers "who speaks when" without looking at the words. The
pipeline is: voice activity detection, short windows, a speaker-embedding
model that turns each window into a vector summarising the voice's timbre
(fundamental frequency and harmonics — everything Whisper is trained to
ignore), and clustering of those vectors. `pyannote.audio`
(`speaker-diarization-community-1`, 32 MB of models) is the reference
implementation; building the same from parts (Silero VAD + ECAPA embeddings +
scikit-learn clustering) would have cost sessions for a worse result, and
`sherpa-onnx` is faster on CPU but thinly documented.

On a 4 min 20 Simsamu call (French, phone quality, two speakers, with a
hand-annotated reference): 2 speakers found, **DER 14.6 %**, RTF 0.5–1.2 on
this laptop depending on load (Whisper is at 0.05 on the same file — the
bottleneck is diarization, which is where any future speed work goes).

Whisper and pyannote each produce their own timeline. Attaching a speaker to
every transcribed word is our code (`speech_to_notes/align.py`), with rules
decided on real output rather than guessed:

1. drop diarization turns shorter than 0.15 s (boundary artifacts of the
   segmentation model; nobody says a syllable in 20 ms);
2. a word goes to the turn with the largest overlap; ties go to the longer
   turn — in overlapped speech Whisper transcribes the dominant voice, so
   giving its words to the interrupter is the costlier mistake;
3. a word touching no turn goes to the nearest one within 1 s, else `UNKNOWN`;
4. zero-duration words (a Whisper artifact) fall through rule 3 unharmed;
5. consecutive words of one speaker form a line, timestamped at its first word;
6. a turn of another speaker fully inside a line is reported as a marker;
7. a line under 0.5 s wedged between two lines of the same other speaker is
   absorbed.

Measured with `scripts/word_attribution_accuracy.py` against the reference:
**95.4 % of words get the right speaker**; every error sits at a turn boundary
or inside overlapped speech. Rule 7 does not change that figure (it fixed one
word and broke one); it removes one-word lines that were almost always
boundary errors (15 → 9 lines) and is kept for readability only.

### Structured summary: a JSON form, two engines, and no trust in the reply

"Structured" means four short lists in fixed boxes, never prose. The model is
shown the empty JSON form and asked to fill it; empty lists are valid, because
a model that invents items to fill a box is worse than an empty box.

Two engines sit behind one interface (prompt in, text out), so everything
else — prompt, parsing, validation, retry — is written once:

- **local** (default): a 4-bit GGUF model run on CPU through `llama-cpp-python`,
  inside the process, no server to install. Chosen over Ollama (a separate
  service) and `transformers` (slower and hungrier on CPU for the same model).
- **api**: a hosted model behind an OpenAI-compatible endpoint (the provider is
  a setting, not code; Groq by default). Faster and stronger, but the
  transcript leaves the machine and free tiers typically let the provider
  train on it — so local stays the default and the API is opt-in, chosen by
  the user according to how confidential the meeting is. Mistral was the first
  choice (French company, free tier); its free plan had gone by the time the
  key was created, so the provider was switched rather than adding a card.

The reply is never trusted as is: it is parsed as JSON, checked key by key
(exactly the four keys, each a list of strings), and on failure the error
message is sent back to the model once; if it still fails, the raw reply is
kept in the summary file and nothing is faked. Both engines use the JSON
output mode, and on every run so far the first reply was already valid.

Local model chosen on one real transcript (Simsamu, 277 words), against notes
written by hand first:

| model | size | time | found the decision | found the action | open question |
|-------|------|------|--------------------|------------------|---------------|
| Qwen3-4B-Instruct-2507 (local) | 2.5 GB | 47 s | no (filed as an action) | roughly | one of two |
| Mistral-7B-Instruct-v0.3 (local) | 4.4 GB | 49–76 s | yes | yes, precisely | one of two |
| gpt-oss-120b via Groq (API) | — | 3 s | yes | yes | both |

Mistral-7B is the local default: in offline use the extra 30 s cost nothing, a
missed decision costs everything. The API is 20x faster and slightly more
complete, which is exactly the trade-off the user makes when choosing it over
confidentiality. One transcript is a thin basis; the comparison will be re-run
when a second annotated recording is available.

A lesson learnt on the way: Mistral-7B kept answering in English to an
English prompt that said "write in the language of the transcript", and
even to "write every item in French". Small models follow the language the
instructions are *written in* more than instructions *about* language, so the
prompt now ends with a reminder written in the target language.

### Voice output: Piper, and punctuation for the ear

The summary is read by Piper, a small ONNX text-to-speech engine that runs on
CPU with no server and no PyTorch, one 60 MB voice per language (French and
English here). Kokoro sounds better but needs `espeak-ng` installed
system-wide; `espeak-ng` alone sounds robotic. Piper was the "no dependencies"
choice.

What the voice says is built from the four lists, not from the Markdown: each
heading is announced as a word, each item becomes one sentence, an empty
list is read as "aucune"/"none" rather than skipped in silence. Every heading
ends with a full stop because that is where the voice pauses — punctuation
for the ear. 19 s of French audio takes about 1.4 s to generate.

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
- **Overlapped speech.** Two voices at once are a sum the transcription cannot
  undo: Whisper writes one of them. The other speaker is *marked*, not
  transcribed. This is where most of the remaining attribution errors are.
- **Turn boundaries.** Whisper's word timestamps are only accurate to about
  0.1 s and tend to start the first word of a segment early; the first or last
  word of a phrase can land on the wrong speaker. Rule 7 hides the one-word
  cases; it does not fix them.
- **Speaker labels are anonymous** (`SPEAKER_00`, ...), and may swap between
  runs. Naming speakers needs either user input or reading the content.
- **Diarization speed varies** with machine load: RTF 0.5 to 1.2 measured on
  the same file, so a one-hour meeting can take from 30 to 70 minutes.
- **Tested on two speakers only** so far. Three or more voices, and long
  meetings, have not been checked.
- **The summary can slip in inferences.** On a one-line test ("rendez-vous
  avec Monsieur Lefebvre"), the local model wrote "Monsieur Lefebvre must go to
  the meeting" — plausible, not in the text. Topics can also be merged into one
  long item. The summary is a reading aid, not a record.
- **Summary quality is bounded by the transcript.** "docteur Damani" became
  "deux pères de la vanille" upstream; no summary step recovers that.
- **Not bit-for-bit reproducible across exports.** The same recording exported
  as .wav and as .m4a gave two transcripts with the same meaning but different
  wording in places, and one of them dropped a quiet 45 s passage (VAD
  threshold). Tiny numeric differences in the input can flip beam search and
  VAD decisions. Judging quality requires a reference transcript, not a diff
  between two runs.
