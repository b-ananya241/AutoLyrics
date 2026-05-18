import json
import time
import torch
import whisper
import numpy as np
from pathlib import Path
from datasets import load_dataset, Audio
from jiwer import wer, cer

RESULTS_FILE = "results/baseline.json"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading Whisper-small...")
    model = whisper.load_model("small", device=device)

    print("Loading jam-alt test data...")
    ds    = load_dataset("audioshake/jam-alt", "en")
    split = ds["test"].train_test_split(test_size=0.2, seed=42)
    test  = split["test"].cast_column("audio", Audio(sampling_rate=16000))
    print(f"Test clips: {len(test)}")

    predictions = []
    references  = []
    latencies   = []

    for i, sample in enumerate(test):
        audio_array = sample["audio"]["array"].astype(np.float32)
        reference   = sample["text"].strip().lower()

        start   = time.time()
        result  = model.transcribe(audio_array, language="en", fp16=torch.cuda.is_available())
        elapsed = time.time() - start

        prediction = result["text"].strip().lower()
        predictions.append(prediction)
        references.append(reference)
        latencies.append(elapsed)

        print(f"[{i+1}/{len(test)}] REF : {reference[:60]}")
        print(f"         PRED: {prediction[:60]}")
        print(f"         Time: {elapsed:.2f}s")

    word_error_rate = wer(references, predictions)
    char_error_rate = cer(references, predictions)
    avg_latency     = sum(latencies) / len(latencies)

    results = {
        "model":         "whisper-small",
        "mode":          "zero-shot baseline",
        "dataset":       "audioshake/jam-alt (en)",
        "num_clips":     len(test),
        "WER":           round(word_error_rate, 4),
        "CER":           round(char_error_rate, 4),
        "avg_latency_s": round(avg_latency, 3),
        "per_clip": [
            {
                "reference":  references[i],
                "prediction": predictions[i],
                "latency_s":  round(latencies[i], 3)
            }
            for i in range(len(predictions))
        ]
    }

    Path("results").mkdir(exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n--- Baseline Results ---")
    print(f"WER         : {word_error_rate:.4f} ({word_error_rate*100:.1f}%)")
    print(f"CER         : {char_error_rate:.4f} ({char_error_rate*100:.1f}%)")
    print(f"Avg latency : {avg_latency:.3f}s")
    print(f"Saved to    : {RESULTS_FILE}")

if __name__ == "__main__":
    main()