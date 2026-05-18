#!/usr/bin/env python3
"""
Create placeholder WAV files for any .txt annotation missing a matching .wav.

Use ONLY when automated Google Drive download failed and you need to test the
pipeline before manually copying audio from the browser. Replace with real
audio before training or evaluation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autolyrics.nus48e import find_corpus_root, load_phoneme_segments

DEFAULT_RAW = ROOT / "data" / "raw" / "nus48e"
SR = 16000


def duration_from_txt(txt_path: Path) -> float:
    segments = load_phoneme_segments(txt_path)
    if not segments:
        return 1.0
    return max(seg.end for seg in segments) + 0.05


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    args = parser.parse_args()

    root = find_corpus_root(args.raw_dir)
    created = 0
    for txt_path in root.rglob("*.txt"):
        wav_path = txt_path.with_suffix(".wav")
        if wav_path.exists():
            continue
        duration = duration_from_txt(txt_path)
        samples = int(duration * SR)
        audio = np.zeros(samples, dtype=np.float32)
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(wav_path, audio, SR)
        created += 1
        print(f"Created placeholder: {wav_path.relative_to(root)}")

    print(f"\nCreated {created} placeholder wav file(s).")
    print("Replace these with real audio from Google Drive before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
