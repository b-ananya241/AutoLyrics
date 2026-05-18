#!/usr/bin/env python3
"""Download NUS-48E from Google Drive (official mirror used by Amphion)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "raw" / "nus48e"

# Official folder: https://drive.google.com/drive/folders/12pP9uUl0HTVANU3IPLnumTJiRjPtVUMx
GDRIVE_FOLDER_ID = "12pP9uUl0HTVANU3IPLnumTJiRjPtVUMx"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download NUS-48E corpus.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Directory to store the extracted corpus.",
    )
    parser.add_argument(
        "--skip-if-found",
        action="store_true",
        default=True,
        help="Skip download when corpus layout is already present.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from autolyrics.nus48e import find_corpus_root

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_if_found:
        try:
            root = find_corpus_root(args.output_dir)
            print(f"Corpus already present at: {root}")
            return 0
        except FileNotFoundError:
            pass

    try:
        import gdown
    except ImportError:
        print("Install gdown first: pip install gdown")
        return 1

    print("Downloading NUS-48E from Google Drive (this may take several minutes) ...")
    print(f"Folder ID: {GDRIVE_FOLDER_ID}")
    print(f"Output:    {args.output_dir}")

    try:
        gdown.download_folder(
            id=GDRIVE_FOLDER_ID,
            output=str(args.output_dir),
            quiet=False,
        )
    except Exception as exc:
        print(f"\nAutomated download failed: {exc}")
        print(
            "\nManual download (recommended):\n"
            "  1. Open https://drive.google.com/drive/folders/12pP9uUl0HTVANU3IPLnumTJiRjPtVUMx\n"
            "  2. Download the full folder (or each speaker folder).\n"
            f"  3. Extract so ADIZ/sing/01.wav exists under:\n"
            f"     {args.output_dir.resolve()}\n"
            "  4. Run: python scripts/verify_nus48e.py\n"
        )
        return 1

    root = find_corpus_root(args.output_dir)
    print(f"Download complete. Corpus root: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
