"""LoRA SFT trainer for the reviewed, source-grounded local Q&A dataset."""

import argparse
import json
from pathlib import Path

from app.feedback import FINETUNE_PATH, PROJECT_DATA

MIN_TRAINING_EXAMPLES = 10
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def load_training_rows(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed training JSONL at line {line_number}: {exc}") from exc
            if not isinstance(row.get("messages"), list) or len(row["messages"]) < 3:
                raise ValueError(f"Training row {line_number} must contain system, user, and assistant messages.")
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--allow-small-dataset", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DATA / "feedback" / "training_output")
    args = parser.parse_args()

    if not FINETUNE_PATH.exists():
        raise SystemExit(f"Training file not found: {FINETUNE_PATH}. Run python -m app.prepare_finetune first.")
    rows = load_training_rows(FINETUNE_PATH)
    if len(rows) < MIN_TRAINING_EXAMPLES and not args.allow_small_dataset:
        raise SystemExit(
            f"Only {len(rows)} reviewed example(s) are available. Add at least {MIN_TRAINING_EXAMPLES} "
            "source-checked examples first. Use --allow-small-dataset only for a deliberate pilot."
        )
    if not rows:
        raise SystemExit("No valid reviewed examples are available for training.")

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. On a CUDA-enabled Linux/WSL environment, "
            "install backend/requirements-training.txt first."
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was not found. LoRA training for this model is intended for a CUDA-enabled Linux/WSL "
            "environment; this project does not download or install a model automatically."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.config.use_cache = False

    text_rows = []
    for row in rows:
        text = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        text_rows.append({"text": text})
    dataset = Dataset.from_list(text_rows)

    training_args = SFTConfig(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        gradient_checkpointing=True,
        fp16=dtype == torch.float16,
        bf16=dtype == torch.bfloat16,
        max_length=2048,
        dataset_text_field="text",
        packing=False,
    )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()

    adapter_dir = args.output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"LoRA adapter saved to: {adapter_dir}")

    merged_dir = args.output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    try:
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="4GB")
        tokenizer.save_pretrained(merged_dir)
    except Exception as exc:
        print(f"The adapter was saved, but merging failed: {exc}")
        print("Use a compatible PEFT/Transformers environment to merge before importing into Ollama.")
        return

    modelfile = args.output_dir / "Modelfile"
    modelfile.write_text(
        f"FROM {merged_dir.resolve()}\nPARAMETER temperature 0\n",
        encoding="utf-8",
    )
    print(f"Merged model saved to: {merged_dir}")
    print(f"Create the Ollama model with: ollama create myanmar-law-finetuned -f \"{modelfile}\"")


if __name__ == "__main__":
    main()
