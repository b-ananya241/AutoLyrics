import sys
sys.path.append("src")
from autolyrics.audio_preprocess import preprocess_audio


import json
import time
import torch
import whisper
from datasets import load_from_disk
from jiwer import wer, cer
from pathlib import Path

def evaluate_baseline():
    dataset_path = "data/processed/nus48e/hf_dataset"
    results_path = Path("results")
    results_path.mkdir(exist_ok=True)

    print("Loading dataset...")
    dataset = load_from_disk(dataset_path)
    test_split = dataset["test"]
    print(f"Test clips: {len(test_split)}")

    print("Loading Whisper-small...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = whisper.load_model("small", device=device)

    predictions = []
    references = []
    latencies = []

    for i, clip in enumerate(test_split):
        audio_path = clip["audio_path"]
        reference_text = clip["text"]

        print(f"[{i+1}/{len(test_split)}] Transcribing: {audio_path}")

        start = time.time()
        result = model.transcribe(audio_path, language="en")
        elapsed = time.time() - start

        prediction = result["text"].strip().lower()
        predictions.append(prediction)
        references.append(reference_text)
        latencies.append(elapsed)

        print(f"  REF : {reference_text}")
        print(f"  PRED: {prediction}")
        print(f"  Time: {elapsed:.2f}s")

    word_error_rate = wer(references, predictions)
    char_error_rate = cer(references, predictions)
    avg_latency = sum(latencies) / len(latencies)

    results = {
        "model": "whisper-small",
        "mode": "zero-shot baseline",
        "num_clips": len(test_split),
        "WER": round(word_error_rate, 4),
        "CER": round(char_error_rate, 4),
        "avg_latency_s": round(avg_latency, 3),
        "per_clip": [
            {
                "audio_path": test_split[i]["audio_path"],
                "reference": references[i],
                "prediction": predictions[i],
                "latency_s": round(latencies[i], 3)
            }
            for i in range(len(predictions))
        ]
    }

    output_file = results_path / "baseline.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n--- Baseline Results ---")
    print(f"WER : {word_error_rate:.4f} ({word_error_rate*100:.1f}%)")
    print(f"CER : {char_error_rate:.4f} ({char_error_rate*100:.1f}%)")
    print(f"Avg latency: {avg_latency:.3f}s per clip")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    evaluate_baseline()