import torch
import json
import time
from pathlib import Path
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import evaluate

MODEL_NAME   = "openai/whisper-small"
ADAPTER_PATH = "models/lora_decoder"
RESULTS_FILE = "results/lora_decoder_eval.json"
LANGUAGE     = "english"
TASK         = "transcribe"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading model...")
    processor  = WhisperProcessor.from_pretrained(ADAPTER_PATH, language=LANGUAGE, task=TASK)
    base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model      = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.to(device).eval()

    print("Loading test data...")
    ds    = load_dataset("audioshake/jam-alt", "en", trust_remote_code=True)
    split = ds["test"].train_test_split(test_size=0.2, seed=42)
    test  = split["test"].cast_column("audio", Audio(sampling_rate=16000))
    print(f"Test clips: {len(test)}")

    wer_metric  = evaluate.load("wer")
    predictions = []
    references  = []
    latencies   = []

    for i, sample in enumerate(test):
        inputs = processor(
            sample["audio"]["array"],
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(device)

        start = time.time()
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features=inputs,
                forced_decoder_ids=processor.get_decoder_prompt_ids(
                    language=LANGUAGE, task=TASK
                )
            )
        latencies.append(time.time() - start)

        pred = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip().lower()
        ref  = sample["text"].strip().lower()

        predictions.append(pred)
        references.append(ref)
        print(f"[{i+1}/{len(test)}] REF : {ref[:60]}")
        print(f"        PRED: {pred[:60]}")

    wer = wer_metric.compute(predictions=predictions, references=references)

    # load baseline
    try:
        baseline_wer = json.load(open("results/baseline.json"))["WER"]
        improvement  = (baseline_wer - wer) / baseline_wer * 100
    except:
        baseline_wer, improvement = None, None

    results = {
        "model":    "whisper-small + LoRA decoder",
        "WER":      round(wer, 4),
        "baseline_WER": baseline_wer,
        "relative_WER_improvement_pct": round(improvement, 2) if improvement else None,
        "avg_latency_s": round(sum(latencies)/len(latencies), 3),
        "num_clips": len(test),
    }

    Path("results").mkdir(exist_ok=True)
    json.dump(results, open(RESULTS_FILE, "w"), indent=2)

    print(f"\n--- LoRA Results ---")
    print(f"WER      : {wer*100:.1f}%")
    print(f"Baseline : {baseline_wer*100:.1f}%" if baseline_wer else "")
    print(f"Improvement: {improvement:.1f}%" if improvement else "")
    print(f"Saved to : {RESULTS_FILE}")

if __name__ == "__main__":
    main()