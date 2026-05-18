#!/usr/bin/env python3
"""Build train/val/test HuggingFace datasets from NUS-48E."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autolyrics.dataset import DEFAULT_OUTPUT_DIR, DEFAULT_RAW_DIR, build_nus48e_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Curate NUS-48E into HuggingFace datasets.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--include-spoken",
        action="store_true",
        help="Include read/speech clips in addition to singing (default: singing only).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Building NUS-48E dataset ...")
    print(f"  Raw:     {args.raw_dir}")
    print(f"  Output:  {args.output_dir}")
    print(f"  Singing only: {not args.include_spoken}")

    ds = build_nus48e_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        singing_only=not args.include_spoken,
        seed=args.seed,
    )

    print("\n--- Dataset summary ---")
    for split, subset in ds.items():
        print(f"{split:12} clips={len(subset):4}  duration={sum(subset['duration_s']):.1f}s")
    print(f"\nSaved to: {args.output_dir / 'hf_dataset'}")
    print(f"Stats:    {args.output_dir / 'dataset_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
