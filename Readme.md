# AutoLyrics — LoRA Fine-Tuning Whisper for Sung Lyrics Transcription

## Overview

Automatic speech recognition (ASR) systems perform well on spoken language, but sung lyrics remain challenging due to elongated vowels, pitch variation, repeated words, overlapping instruments, and unclear segmentation.

This project improves sung-lyrics transcription by fine-tuning OpenAI Whisper using LoRA adapters on the Jam-ALT dataset.

The baseline `openai/whisper-small` model showed poor zero-shot performance on lyric transcription. By applying lightweight parameter-efficient fine-tuning (LoRA), the model achieved a significant reduction in transcription error while remaining practical to train on a single Colab GPU.

---

# Problem Statement

Whisper models are primarily trained on spoken audio. When applied to singing vocals, performance degrades because:

* Words are stretched across notes
* Lyrics are repeated or blended
* Instrumentals interfere with vocals
* Singing rhythm differs from speech rhythm
* Lyric boundaries are difficult to segment

As a result, zero-shot Whisper produced high error rates on the Jam-ALT dataset.

---

# Solution

We fine-tuned `openai/whisper-small` using **LoRA (Low-Rank Adaptation)** on line-level lyric segments extracted from the `jamendolyrics/jam-alt` dataset.

The pipeline:

1. Load Jam-ALT dataset
2. Segment songs into timestamped lyric lines
3. Evaluate zero-shot Whisper baseline
4. Fine-tune Whisper-small with LoRA adapters
5. Evaluate tuned model on held-out test split
6. Compare WER/CER and runtime metrics

---

# Dataset

Dataset used:

* `jamendolyrics/jam-alt`

The dataset contains aligned lyrics with timestamps, enabling extraction of lyric-line training samples.

---

# Model

Base model:

* `openai/whisper-small`

Fine-tuning method:

* LoRA adapters applied to Whisper decoder

Frameworks:

* Hugging Face Transformers
* PEFT (LoRA)
* PyTorch

---

# Evaluation Metrics

We evaluate transcription quality using:

* **WER (Word Error Rate)**
* **CER (Character Error Rate)**

Lower values indicate better transcription quality.

---

# Results

## Baseline Whisper (Zero-Shot)

| Metric | Value  |
| ------ | ------ |
| WER    | 1.5018 |
| CER    | 0.9021 |

---

## LoRA-Tuned Whisper

| Metric | Value  |
| ------ | ------ |
| WER    | 0.8984 |
| CER    | 0.6979 |

---

# Improvement

* Relative WER improvement: **40.18%**
* Project target: **15% WER reduction**
* Target achieved: **Yes**

The LoRA-tuned model significantly outperformed the baseline and comfortably exceeded the project goal.

---

# Performance Tradeoffs

## Latency

| Model            | Latency                 |
| ---------------- | ----------------------- |
| Baseline Whisper | 1.96 sec / audio minute |
| LoRA Whisper     | 3.36 sec / audio minute |

## VRAM Usage

| Model            | VRAM    |
| ---------------- | ------- |
| Baseline Whisper | 0.72 GB |
| LoRA Whisper     | 1.71 GB |

The LoRA model improves transcription quality at the cost of higher memory usage and slower inference.

---

# Conclusion

This project demonstrates that lightweight LoRA fine-tuning can substantially improve sung-lyrics transcription quality for Whisper models.

Key outcomes:

* Reduced WER from **1.5018 → 0.8984**
* Achieved **40.18% relative improvement**
* Exceeded the original project target
* Maintained practical training and evaluation costs on Google Colab GPUs

LoRA proved to be an efficient and scalable method for adapting speech recognition models to music-domain transcription tasks.

# Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* PEFT (LoRA)
* OpenAI Whisper
* Google Colab

---

# References

* OpenAI Whisper
* Hugging Face Transformers
* PEFT Library
* Jam-ALT Dataset
