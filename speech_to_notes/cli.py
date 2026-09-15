"""Command-line entry point: audio file in, transcript files out."""

import argparse
import time
from pathlib import Path

from speech_to_notes.audio import SAMPLE_RATE, load_audio
from speech_to_notes.output import save_json, save_text
from speech_to_notes.transcribe import DEFAULT_MODEL_SIZE, transcribe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="speech_to_notes",
        description="Turn a recorded conversation into a timestamped transcript.",
    )
    parser.add_argument("audio", help="path to an .mp3 or .wav file")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_SIZE,
        help=f"Whisper size: tiny, base, small, medium (default: {DEFAULT_MODEL_SIZE})",
    )
    parser.add_argument(
        "--language", default=None,
        help="two-letter language code, e.g. fr or en (default: auto-detect)",
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
    elapsed = time.perf_counter() - t0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.audio).stem
    save_text(segments, out_dir / f"{stem}.txt")
    save_json(segments, out_dir / f"{stem}.json")

    print(f"Done in {elapsed:.1f} s (RTF {elapsed / duration:.2f}). "
          f"{len(segments)} segment(s) -> {out_dir / stem}.txt / .json")


if __name__ == "__main__":
    main()
