"""Build HuggingFace Dataset objects from NUS-48E."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, Features, Value

from autolyrics.nus48e import (
    ALL_SONG_IDS,
    assign_splits,
    build_phone_to_word_map,
    annotation_to_text,
    find_corpus_root,
    iter_clips,
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "nus48e"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "nus48e"


def build_nus48e_dataset(
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    singing_only: bool = True,
    sample_rate: int = 16000,
    seed: int = 42,
) -> DatasetDict:
    """Parse NUS-48E, normalize lyrics, split by song, save to disk."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    corpus_root = find_corpus_root(raw_dir)

    modalities = ("sing",) if singing_only else ("sing", "read")
    phone_to_word = build_phone_to_word_map()
    split_map = assign_splits(ALL_SONG_IDS, seed=seed)

    records: list[dict[str, Any]] = []
    for clip in iter_clips(corpus_root, modalities=modalities):
        text, phoneme_text = annotation_to_text(clip.annotation_path, phone_to_word)
        if not text:
            continue
        records.append(
            {
                "audio_path": str(clip.audio_path),
                "text": text,
                "phoneme_text": phoneme_text,
                "speaker_id": clip.speaker_id,
                "song_id": clip.song_id,
                "song_name": clip.song_name,
                "modality": clip.modality,
                "is_singing": clip.modality == "sing",
                "split": split_map.get(clip.song_id, "train"),
                "duration_s": _audio_duration(clip.audio_path),
            }
        )

    if not records:
        raise RuntimeError(f"No clips found under {corpus_root}")

    features = Features(
        {
            "audio_path": Value("string"),
            "text": Value("string"),
            "phoneme_text": Value("string"),
            "speaker_id": Value("string"),
            "song_id": Value("string"),
            "song_name": Value("string"),
            "modality": Value("string"),
            "is_singing": Value("bool"),
            "split": Value("string"),
            "duration_s": Value("float32"),
        }
    )

    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for row in records:
        by_split[row["split"]].append(row)

    dataset_dict = DatasetDict(
        {
            split: Dataset.from_list(rows, features=features)
            for split, rows in by_split.items()
            if rows
        }
    )
    _ = sample_rate  # reserved for training-time resampling

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir / "hf_dataset"))

    stats = _summarize(dataset_dict, corpus_root, split_map, singing_only)
    stats_path = output_dir / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return dataset_dict


def _audio_duration(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return float(info.duration)


def _summarize(
    dataset_dict: DatasetDict,
    corpus_root: Path,
    split_map: dict[str, str],
    singing_only: bool,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "corpus_root": str(corpus_root),
        "singing_only": singing_only,
        "song_splits": split_map,
        "splits": {},
    }
    for name, ds in dataset_dict.items():
        summary["splits"][name] = {
            "num_clips": len(ds),
            "total_duration_s": round(sum(ds["duration_s"]), 2),
            "unique_speakers": sorted(set(ds["speaker_id"])),
            "unique_songs": sorted(set(ds["song_id"])),
        }
    return summary


def load_dataset_from_disk(path: Path | str = DEFAULT_OUTPUT_DIR / "hf_dataset") -> DatasetDict:
    from datasets import load_from_disk

    return load_from_disk(str(path))
