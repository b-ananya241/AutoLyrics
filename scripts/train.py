import sys
sys.path.append("src")
from autolyrics.audio_preprocess import preprocess_audio



import json
import torch
import whisper
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader
from peft import get_peft_model, LoraConfig, TaskType
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch.optim as optim

# ── config ──────────────────────────────────────────────
MODEL_NAME      = "openai/whisper-small"
DATASET_PATH    = "data/processed/nus48e/hf_dataset"
OUTPUT_DIR      = "models/lora_decoder"
RESULTS_FILE    = "results/lora_decoder.json"
EPOCHS          = 30
BATCH_SIZE      = 4
LEARNING_RATE   = 5e-5
MAX_DURATION_S  = 30
SAMPLE_RATE     = 16000
# ────────────────────────────────────────────────────────

class SingingDataset(Dataset):
    def __init__(self, hf_split, processor):
        self.clips     = hf_split
        self.processor = processor

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        import librosa
        audio, _ = librosa.load(clip["audio_path"], sr=SAMPLE_RATE, mono=True)
        audio = preprocess_audio(audio, SAMPLE_RATE)

        # truncate to 30 s
        max_samples = MAX_DURATION_S * SAMPLE_RATE
        audio = audio[:max_samples]

        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )
        input_features = inputs.input_features.squeeze(0)

        labels = self.processor.tokenizer(
            clip["text"],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=128
        ).input_ids.squeeze(0)

        return {"input_features": input_features, "labels": labels}


def collate_fn(batch):
    input_features = torch.stack([b["input_features"] for b in batch])
    label_list     = [b["labels"] for b in batch]

    # pad labels to same length
    max_len = max(l.size(0) for l in label_list)
    padded  = torch.full((len(label_list), max_len), -100, dtype=torch.long)
    for i, l in enumerate(label_list):
        padded[i, :l.size(0)] = l

    return {"input_features": input_features, "labels": padded}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── load processor + model ───────────────────────────
    print("Loading Whisper-small...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model     = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

   # ── attach LoRA to decoder only ──────────────────────
    lora_config = LoraConfig(
        r                = 16,
        lora_alpha       = 32,
        target_modules   = ["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout     = 0.1,
        bias             = "none",
        task_type        = TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    # ── dataset ──────────────────────────────────────────
    print("Loading dataset...")
    ds         = load_from_disk(DATASET_PATH)
    train_data = SingingDataset(ds["train"],      processor)
    val_data   = SingingDataset(ds["validation"], processor)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_fn)

    # ── optimizer ────────────────────────────────────────
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")
    history       = []

    # ── training loop ────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            input_features = batch["input_features"].to(device)
            labels         = batch["labels"].to(device)

           decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]] * input_features.size(0)).to(device)
outputs = model(input_features=input_features, labels=labels, decoder_input_ids=decoder_input_ids)
            loss    = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── validation ───────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_features = batch["input_features"].to(device)
                labels         = batch["labels"].to(device)
                outputs        = model(input_features=input_features,
                                       labels=labels)
                val_loss      += outputs.loss.item()

        val_loss /= len(val_loader)
        history.append({"epoch": epoch,
                         "train_loss": round(train_loss, 4),
                         "val_loss":   round(val_loss,   4)})

        print(f"Epoch {epoch}/{EPOCHS} — "
              f"train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}")

        # save best adapter
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            processor.save_pretrained(OUTPUT_DIR)
            print(f"  ✓ saved best model (val_loss={val_loss:.4f})")

    # ── save training history ─────────────────────────────
    Path("results").mkdir(exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump({"training_history": history,
                   "best_val_loss": round(best_val_loss, 4)}, f, indent=2)
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Adapter saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()