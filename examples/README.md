# Examples

Real inputs and the files the tool produced from them, committed so the result
can be read without installing anything. Nothing here was edited by hand.

## `sample_fr.*` — a 12 s sentence (author's voice)

```bash
python -m speech_to_notes examples/sample_fr.wav --summarize api --speak
```

- `sample_fr.wav` — input, one speaker, laptop microphone
- `sample_fr.txt` — the transcript
- `sample_fr.json` — segments and words with timestamps, detected language, summary
- `sample_fr.summary.md` — the structured summary (Groq, gpt-oss-120b)
- `sample_fr.summary.wav` — the summary read aloud (Piper, fr_FR-siwis)

## `simsamu_douleur_thoracique.*` — a 4 min 20 simulated emergency call, two speakers

```bash
python -m speech_to_notes examples/simsamu_douleur_thoracique.m4a --language fr --diarize --summarize api --speak
```

- `simsamu_douleur_thoracique.m4a` — input, 8 kHz telephone quality. From the
  [Simsamu](https://huggingface.co/datasets/medkit/simsamu) corpus (medkit,
  MIT license): interns in emergency medicine playing a caller and a
  regulating doctor.
- `simsamu_douleur_thoracique.rttm` — the hand-annotated reference (who speaks
  when) shipped with the corpus; what `scripts/benchmark_diarization.py` and
  `scripts/word_attribution_accuracy.py` measure against
- `simsamu_douleur_thoracique.txt` — the speaker-labelled transcript with
  overlap markers
- `simsamu_douleur_thoracique.json` — segments, words, diarization turns, summary
- `simsamu_douleur_thoracique.summary.md` / `.summary.wav` — the summary, as
  text and as speech

Things worth noticing in these files, all discussed in the main README:

- "docteur Damani" is transcribed as "deux pères de la vanille": phone-quality
  audio, rare proper noun. The summary cannot recover it.
- Speaker labels are cluster ids. In this run the doctor is `SPEAKER_00`; in
  an earlier run on the same file it was `SPEAKER_01`. Same speakers, same
  turns, different numbering.
- Short answers ("Oui", "Non") sometimes end up inside the other speaker's
  line: Whisper's word timestamps are ±0.1 s and the turn boundaries are where
  attribution errors concentrate (95.4 % of words correct against the .rttm).
- The overlap markers show where the diarization model heard both voices;
  those words are not in the transcript, by construction.
