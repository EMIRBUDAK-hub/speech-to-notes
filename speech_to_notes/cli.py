"""Command-line entry point: audio file in, transcript files out."""

import argparse
import time
from pathlib import Path

from speech_to_notes.audio import SAMPLE_RATE, load_audio
from speech_to_notes.output import save_json, save_speaker_text, save_text
from speech_to_notes.transcribe import DEFAULT_MODEL_SIZE, transcribe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="speech_to_notes",
        description="Turn a recorded conversation into a timestamped transcript.",
    )
    parser.add_argument("audio", help="path to an .mp3, .wav or .m4a file")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_SIZE,
        help=f"Whisper size: tiny, base, small, medium (default: {DEFAULT_MODEL_SIZE})",
    )
    parser.add_argument(
        "--language", default=None,
        help="two-letter language code, e.g. fr or en (default: auto-detect)",
    )
    parser.add_argument(
        "--diarize", action="store_true",
        help="also identify who speaks when (slower; needs HF_TOKEN, see README)",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="folder where <audio name>.txt and .json are written (default: output/)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    audio = load_audio(args.audio)
    duration = len(audio) / SAMPLE_RATE
    print(f"Loaded {args.audio} ({duration:.1f} s). Transcribing with '{args.model}'...")

    t0 = time.perf_counter()
    segments = transcribe(audio, model_size=args.model, language=args.language)
    t_asr = time.perf_counter() - t0
    print(f"  transcription: {t_asr:.1f} s (RTF {t_asr / duration:.2f}), {len(segments)} segment(s)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.audio).stem
    turns = None

    if args.diarize:
        # imported here so the transcription-only path never loads torch/pyannote
        from speech_to_notes.align import assign_speakers, group_utterances, mark_overlaps, smooth
        from speech_to_notes.diarize import diarize

        print("Diarizing...")
        t0 = time.perf_counter()
        turns = diarize(audio)
        t_dia = time.perf_counter() - t0
        speakers = sorted({t.speaker for t in turns})
        print(f"  diarization: {t_dia:.1f} s (RTF {t_dia / duration:.2f}), {len(speakers)} speaker(s)")

        words = [w for s in segments for w in s.words]
        utterances = smooth(group_utterances(assign_speakers(words, turns)))
        mark_overlaps(utterances, turns)
        save_speaker_text(utterances, out_dir / f"{stem}.txt")
    else:
        save_text(segments, out_dir / f"{stem}.txt")

    save_json(segments, out_dir / f"{stem}.json", turns=turns)
    print(f"Done -> {out_dir / stem}.txt / .json")


if __name__ == "__main__":
    main()
