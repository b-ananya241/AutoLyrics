#!/usr/bin/env python3
"""Check that NUS-48E is fully downloaded before building the dataset."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autolyrics.nus48e import SPEAKERS, find_corpus_root, iter_clips

DEFAULT_RAW = ROOT / "data" / "raw" / "nus48e"


def main() -> int:
    raw_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RAW
    try:
        root = find_corpus_root(raw_dir)
    except FileNotFoundError as exc:
        print(f"FAIL: {exc}")
        print(
            "\nDownload the corpus manually from:\n"
            "  https://drive.google.com/drive/folders/12pP9uUl0HTVANU3IPLnumTJiRjPtVUMx\n"
            "Extract so that ADIZ/sing/01.wav exists under data/raw/nus48e/"
        )
        return 1

    missing_wav: list[str] = []
    missing_txt: list[str] = []
    for spk in SPEAKERS:
        for modality in ("sing", "read"):
            mod_dir = root / spk / modality
            if not mod_dir.is_dir():
                missing_wav.append(f"{spk}/{modality}/ (folder)")
                continue
            for wav in mod_dir.glob("*.wav"):
                if not wav.with_suffix(".txt").exists():
                    missing_txt.append(str(wav.relative_to(root)))

    clips = list(iter_clips(root))

    print(f"Corpus root: {root}")
    print(f"Indexed clips (wav+txt pairs): {len(clips)}")

    if not clips:
        print("\nFAIL: No wav+txt pairs found. Audio files may be missing.")
        print("Only annotations downloaded — re-download the full corpus from Google Drive.")
        return 1

    if len(clips) < 200:
        print(f"\nWARN: Expected ~240 singing clips; found {len(clips)}. Download may be incomplete.")

    print("\nOK: Corpus ready for scripts/build_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
