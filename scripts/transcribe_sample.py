#!/usr/bin/env python3
"""Transcribe one sample audio file with OpenAI Whisper (Phase 1 smoke test)."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO = ROOT / "data" / "samples" / "jfk.flac"
SAMPLE_URL = "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac"


def ensure_sample(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    print(f"Downloading sample audio to {path} ...")
    urllib.request.urlretrieve(SAMPLE_URL, path)
    return path


def load_audio_16k(path: Path):
    """Load mono 16 kHz float32 audio without requiring ffmpeg on PATH."""
    import librosa
    import numpy as np

    audio, _ = librosa.load(path, sr=16000, mono=True)
    return np.asarray(audio, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe one audio file with Whisper.")
    parser.add_argument(
        "--audio",
        type=Path,
        default=DEFAULT_AUDIO,
        help="Path to audio (wav/mp3/flac). Default: bundled JFK sample.",
    )
    parser.add_argument(
        "--model",
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (tiny is fastest for smoke tests).",
    )
    args = parser.parse_args()

    import torch
    import whisper

    audio_path = ensure_sample(args.audio)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading whisper-{args.model} ...")
    model = whisper.load_model(args.model, device=device)

    print(f"Transcribing: {audio_path}")
    audio = load_audio_16k(audio_path)
    result = model.transcribe(audio, language="en", fp16=(device == "cuda"))
    text = result["text"].strip()

    print("\n--- Transcription ---")
    print(text)
    print("---------------------")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
