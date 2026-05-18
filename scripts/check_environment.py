#!/usr/bin/env python3
"""Verify Python, GPU/CUDA, and core AutoLyrics dependencies."""

from __future__ import annotations

import platform
import sys


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def check_python() -> None:
    section("Python")
    print(f"Version:     {sys.version.split()[0]} ({platform.platform()})")
    print(f"Executable:  {sys.executable}")


def check_torch() -> bool:
    section("PyTorch & CUDA")
    try:
        import torch
    except ImportError:
        print("FAIL: torch not installed.")
        print("  CPU:  pip install torch torchaudio")
        print("  CUDA: https://pytorch.org/get-started/locally/")
        return False

    print(f"PyTorch:     {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA avail:  {cuda_ok}")
    if cuda_ok:
        print(f"CUDA ver:    {torch.version.cuda}")
        print(f"cuDNN:       {torch.backends.cudnn.version()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024**3)
            print(f"GPU {i}:       {props.name} ({vram_gb:.1f} GB)")
    else:
        print("GPU:         none detected (CPU inference is OK for Phase 1)")
    return True


def check_packages() -> None:
    section("Dependencies")
    packages = [
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("peft", "peft"),
        ("jiwer", "jiwer"),
        ("gradio", "gradio"),
        ("whisper", "openai-whisper"),
        ("torchaudio", "torchaudio"),
    ]
    for import_name, label in packages:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "ok")
            print(f"  {label:20} {ver}")
        except ImportError:
            print(f"  {label:20} MISSING")


def check_whisper_load(device: str) -> bool:
    section("Whisper load test")
    try:
        import whisper
    except ImportError:
        print("FAIL: openai-whisper not installed.")
        return False

    model_name = "tiny"
    print(f"Loading whisper-{model_name} on {device} ...")
    model = whisper.load_model(model_name, device=device)
    print(f"OK: whisper-{model_name} loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    return True


def main() -> int:
    print("AutoLyrics — environment check")
    check_python()
    torch_ok = check_torch()
    check_packages()
    if not torch_ok:
        return 1

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not check_whisper_load(device):
        return 1

    section("Summary")
    if device == "cuda":
        print("Ready: GPU + CUDA detected. You can fine-tune with LoRA.")
    else:
        print("Ready (CPU). Install NVIDIA drivers + CUDA PyTorch for GPU training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
