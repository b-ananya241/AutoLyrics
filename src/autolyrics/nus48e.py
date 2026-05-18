"""NUS-48E corpus discovery, parsing, and split logic."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

Modality = Literal["sing", "read"]

SPEAKERS = [
    "ADIZ",
    "JLEE",
    "JTAN",
    "KENN",
    "MCUR",
    "MPOL",
    "MPUR",
    "NJAT",
    "PMAR",
    "SAMF",
    "VKOW",
    "ZHIY",
]

ALL_SONG_IDS = [f"{i:02d}" for i in range(1, 21)]

SONG_NAMES: dict[str, str] = {
    "01": "edelweiss",
    "02": "wonderful_tonight",
    "03": "love_me_tender",
    "04": "rhythm_of_the_rain",
    "05": "jingle_bells",
    "06": "proud_of_you",
    "07": "moon_river",
    "08": "lemon_tree",
    "09": "twinkle_twinkle_little_star",
    "10": "i_have_a_dream",
    "11": "do_re_mi",
    "12": "can_you_feel_the_love_tonight",
    "13": "silent_night",
    "14": "a_little_love",
    "15": "you_are_my_sunshine",
    "16": "far_away_from_home",
    "17": "happy_birthday",
    "18": "you_raise_me_up",
    "19": "my_heart_will_go_on",
    "20": "yesterday_once_more",
}

_SIL_PHONES = frozenset({"sil", "sp", "silence", "spn"})


@dataclass(frozen=True)
class PhonemeSegment:
    start: float
    end: float
    phone: str


@dataclass(frozen=True)
class NusClip:
    audio_path: Path
    annotation_path: Path
    speaker_id: str
    song_id: str
    modality: Modality
    song_name: str


def find_corpus_root(raw_dir: Path) -> Path:
    """Locate NUS-48E root (folder that contains speaker subdirs with sing/read)."""
    raw_dir = raw_dir.resolve()
    candidates = [raw_dir]
    for name in ("nus-smc-corpus_48", "NUS_48E", "NUS-48E", "nus48e"):
        candidates.append(raw_dir / name)

    for root in candidates:
        if _looks_like_nus_root(root):
            return root

    for child in raw_dir.iterdir() if raw_dir.is_dir() else []:
        if child.is_dir() and _looks_like_nus_root(child):
            return child

    raise FileNotFoundError(
        f"Could not find NUS-48E layout under {raw_dir}. "
        "Expected {SpeakerID}/sing|read/{{SongID}}.wav"
    )


def _looks_like_nus_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    for spk in SPEAKERS[:3]:
        if (path / spk / "sing").is_dir() and (path / spk / "read").is_dir():
            return True
    return False


def parse_phoneme_line(line: str) -> PhonemeSegment | None:
    """Parse one HTK-style line: start end PHONE."""
    line = line.strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) < 3:
        return None
    try:
        start = float(parts[0])
        end = float(parts[1])
    except ValueError:
        return None
    phone = parts[2].strip()
    return PhonemeSegment(start=start, end=end, phone=phone)


def load_phoneme_segments(path: Path) -> list[PhonemeSegment]:
    segments: list[PhonemeSegment] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            seg = parse_phoneme_line(line)
            if seg is not None:
                segments.append(seg)
    return segments


def group_phonemes_by_word(segments: list[PhonemeSegment]) -> list[list[str]]:
    """Split phoneme stream into words using SIL/SP boundary markers."""
    words: list[list[str]] = []
    current: list[str] = []
    for seg in segments:
        if seg.phone.lower() in _SIL_PHONES:
            if current:
                words.append(current)
                current = []
            continue
        phone = re.sub(r"\d+$", "", seg.phone).upper()
        current.append(phone)
    if current:
        words.append(current)
    return words


def segments_to_phoneme_text(segments: list[PhonemeSegment]) -> str:
    phones = []
    for seg in segments:
        if seg.phone.lower() not in _SIL_PHONES:
            phones.append(re.sub(r"\d+$", "", seg.phone).upper())
    return " ".join(phones)


def build_phone_to_word_map() -> dict[tuple[str, ...], str]:
    """Reverse CMUdict (ARPAbet without stress) -> word."""
    import nltk
    from nltk.corpus import cmudict as cmu_corpus

    try:
        cmu = cmu_corpus.dict()
    except LookupError:
        nltk.download("cmudict", quiet=True)
        cmu = cmu_corpus.dict()

    mapping: dict[tuple[str, ...], str] = {}
    for word, pronunciations in cmu.items():
        for pron in pronunciations:
            key = tuple(re.sub(r"\d", "", p) for p in pron)
            mapping.setdefault(key, word)
    return mapping


def phones_to_word(phones: list[str], phone_to_word: dict[tuple[str, ...], str]) -> str:
    key = tuple(phones)
    if key in phone_to_word:
        return phone_to_word[key]
    return "".join(p.lower() for p in phones)


def annotation_to_text(
    annotation_path: Path,
    phone_to_word: dict[tuple[str, ...], str],
) -> tuple[str, str]:
    """Return (word-level lyrics, phoneme-level transcript)."""
    segments = load_phoneme_segments(annotation_path)
    word_groups = group_phonemes_by_word(segments)
    words = [phones_to_word(g, phone_to_word) for g in word_groups]
    from autolyrics.text_norm import normalize_lyrics

    word_text = normalize_lyrics(" ".join(words))
    phoneme_text = segments_to_phoneme_text(segments)
    return word_text, phoneme_text


def iter_clips(corpus_root: Path, modalities: tuple[Modality, ...] = ("sing", "read")) -> Iterator[NusClip]:
    corpus_root = find_corpus_root(corpus_root)
    for speaker_id in sorted(SPEAKERS):
        for modality in modalities:
            mod_dir = corpus_root / speaker_id / modality
            if not mod_dir.is_dir():
                continue
            for wav_path in sorted(mod_dir.glob("*.wav")):
                song_id = wav_path.stem
                txt_path = wav_path.with_suffix(".txt")
                if not txt_path.exists():
                    continue
                yield NusClip(
                    audio_path=wav_path.resolve(),
                    annotation_path=txt_path.resolve(),
                    speaker_id=speaker_id,
                    song_id=song_id,
                    modality=modality,
                    song_name=SONG_NAMES.get(song_id, song_id),
                )


def assign_splits(
    song_ids: list[str],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, str]:
    """Song-level split so the same song never appears in two splits."""
    import random

    ids = sorted(set(song_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio)) if n > 2 else 0
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)
    split_map: dict[str, str] = {}
    for sid in ids[:n_train]:
        split_map[sid] = "train"
    for sid in ids[n_train : n_train + n_val]:
        split_map[sid] = "validation"
    for sid in ids[n_train + n_val :]:
        split_map[sid] = "test"
    assert len(split_map) == n
    return split_map
