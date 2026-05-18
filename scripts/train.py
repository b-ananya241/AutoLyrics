import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, TaskType
import json
from pathlib import Path

# ── config ──────────────────────────────────────────────
MODEL_NAME   = "openai/whisper-small"
OUTPUT_DIR   = "models/lora_decoder"
LANGUAGE     = "english"
TASK         = "transcribe"
# ────────────────────────────────────────────────────────

def prepare_dataset(batch, feature_extractor, tokenizer):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

def compute_metrics(pred, tokenizer, metric=None):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    from jiwer import wer
    word_error_rate = wer(label_str, pred_str)
    return {"wer": round(word_error_rate * 100, 2)}

def main():
    print("Loading jam-alt dataset...")
    # load only english songs
    ds = load_dataset("audioshake/jam-alt", "en")
    print(ds)

    # jam-alt only has a test split — we manually split it
    split = ds["test"].train_test_split(test_size=0.2, seed=42)
    train_val = split["train"].train_test_split(test_size=0.15, seed=42)

    dataset = DatasetDict({
        "train":      train_val["train"],
        "validation": train_val["test"],
        "test":       split["test"],
    })
    print(f"Train: {len(dataset['train'])}  Val: {len(dataset['validation'])}  Test: {len(dataset['test'])}")

    # resample audio to 16kHz
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # load processor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained(
        MODEL_NAME, language=LANGUAGE, task=TASK
    )
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME, language=LANGUAGE, task=TASK
    )

    print("Preprocessing dataset...")
    dataset = dataset.map(
        lambda b: prepare_dataset(b, feature_extractor, tokenizer),
        remove_columns=dataset.column_names["train"],
        num_proc=1
    )

    # load model
    print("Loading model + attaching LoRA...")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.generation_config.language = LANGUAGE
    model.generation_config.task     = TASK
    model.generation_config.forced_decoder_ids = None

    lora_config = LoraConfig(
        r              = 32,
        lora_alpha     = 64,
        target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout   = 0.05,
        bias           = "none",
        task_type      = TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # metric
    wer_metric = None

    # training args
    training_args = Seq2SeqTrainingArguments(
        output_dir                  = OUTPUT_DIR,
        per_device_train_batch_size = 4,
        per_device_eval_batch_size  = 4,
        gradient_accumulation_steps = 2,
        learning_rate               = 1e-4,
        warmup_steps                = 50,
        num_train_epochs            = 10,
        eval_strategy         = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "wer",
        greater_is_better           = False,
        predict_with_generate       = True,
        generation_max_length       = 225,
        logging_steps               = 10,
        report_to                   = ["none"],
        push_to_hub                 = False,
        fp16                        = True,
    )

    trainer = Seq2SeqTrainer(
        model         = model,
        args          = training_args,
        train_dataset = dataset["train"],
        eval_dataset  = dataset["validation"],
        data_collator = data_collator,
        compute_metrics = lambda pred: compute_metrics(pred, processor.tokenizer, wer_metric),
    )

    print("Starting training...")
    trainer.train()

    # save adapter
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Saved adapter to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()