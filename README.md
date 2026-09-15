# speech-to-notes

A command-line tool that turns a recording of a conversation (meeting, interview,
lecture) into structured, readable notes.

Input: an audio file (mp3, wav, m4a). Output: a timestamped transcript, then speaker
labels, then a structured summary. Everything runs locally on a CPU-only laptop.
French and English are supported.

## Status

Work in progress. Milestones, in order:

1. Transcription — audio in, timestamped transcript out (done)
2. Speaker diarization — who spoke when (done)
3. Structured summary — topics, decisions, action items, open questions
4. Voice output — read the summary aloud with a local TTS engine
5. Polish — clean CLI, examples, known limitations

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
`--output-dir`.

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
- **Not bit-for-bit reproducible across exports.** The same recording exported
  as .wav and as .m4a gave two transcripts with the same meaning but different
  wording in places, and one of them dropped a quiet 45 s passage (VAD
  threshold). Tiny numeric differences in the input can flip beam search and
  VAD decisions. Judging quality requires a reference transcript, not a diff
  between two runs.
