import json
import time
import torch
from pathlib import Path
from datasets import load_from_disk
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from jiwer import wer, cer
import librosa
import numpy as np

DATASET_PATH  = "data/processed/nus48e/hf_dataset"
ADAPTER_PATH  = "models/lora_decoder"
RESULTS_FILE  = "results/lora_decoder_eval.json"
SAMPLE_RATE   = 16000
MAX_DURATION_S = 30

def evaluate_lora():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading LoRA model...")
    processor  = WhisperProcessor.from_pretrained(ADAPTER_PATH)
    base_model = WhisperForConditionalGeneration.from_pretrained(
                     "openai/whisper-small")
    model      = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.to(device)
    model.eval()

    print("Loading test split...")
    ds         = load_from_disk(DATASET_PATH)
    test_split = ds["test"]
    print(f"Test clips: {len(test_split)}")

    predictions, references, latencies = [], [], []

    for i, clip in enumerate(test_split):
        audio, _ = librosa.load(clip["audio_path"],
                                 sr=SAMPLE_RATE, mono=True)
        audio = audio[:MAX_DURATION_S * SAMPLE_RATE]

        inputs = processor(audio, sampling_rate=SAMPLE_RATE,
                           return_tensors="pt")
        input_features = inputs.input_features.to(device)

        start = time.time()
        with torch.no_grad():
            predicted_ids = model.generate(
    input_features=input_features,
    language="en",
    task="transcribe",
    num_beams=5,
    temperature=0.0
)
        elapsed = time.time() - start

        prediction = processor.batch_decode(
                         predicted_ids, skip_special_tokens=True)[0]
        prediction = prediction.strip().lower()
        reference  = clip["text"]

        predictions.append(prediction)
        references.append(reference)
        latencies.append(elapsed)

        print(f"[{i+1}/{len(test_split)}]")
        print(f"  REF : {reference}")
        print(f"  PRED: {prediction}")
        print(f"  Time: {elapsed:.2f}s")

    word_error_rate = wer(references, predictions)
    char_error_rate = cer(references, predictions)
    avg_latency     = sum(latencies) / len(latencies)

    # load baseline to compare
    baseline_wer = None
    try:
        with open("results/baseline.json") as f:
            baseline_wer = json.load(f)["WER"]
        relative_improvement = (baseline_wer - word_error_rate) / baseline_wer * 100
    except:
        relative_improvement = None

    results = {
        "model":        "whisper-small + LoRA decoder",
        "num_clips":    len(test_split),
        "WER":          round(word_error_rate, 4),
        "CER":          round(char_error_rate, 4),
        "avg_latency_s": round(avg_latency, 3),
        "baseline_WER": baseline_wer,
        "relative_WER_improvement_pct": round(relative_improvement, 2)
                                         if relative_improvement else None
    }

    Path("results").mkdir(exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("\n--- LoRA Decoder Results ---")
    print(f"WER : {word_error_rate:.4f} ({word_error_rate*100:.1f}%)")
    print(f"CER : {char_error_rate:.4f} ({char_error_rate*100:.1f}%)")
    print(f"Avg latency : {avg_latency:.3f}s")
    if relative_improvement:
        print(f"Relative WER improvement: {relative_improvement:.1f}%")
    print(f"Saved to: {RESULTS_FILE}")

if __name__ == "__main__":
    evaluate_lora()