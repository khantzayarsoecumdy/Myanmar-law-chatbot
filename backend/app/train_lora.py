
"""QLoRA SFT trainer for reviewed, source-grounded legal Q&A."""

import argparse
import json
from pathlib import Path

from app.feedback import FINETUNE_PATH, PROJECT_DATA

MIN_TRAINING_EXAMPLES = 5
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
                raise ValueError(
                    f"Malformed JSONL at line {line_number}: {exc}"
                ) from exc

            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 3:
                raise ValueError(
                    f"Row {line_number} must contain system, user, "
                    "and assistant messages."
                )

            rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--allow-small-dataset", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DATA / "feedback" / "training_output",
    )
    args = parser.parse_args()

    if not FINETUNE_PATH.exists():
        raise SystemExit(
            f"Training file not found: {FINETUNE_PATH}. "
            "Run python -m app.prepare_finetune first."
        )

    rows = load_training_rows(FINETUNE_PATH)

    if len(rows) < MIN_TRAINING_EXAMPLES and not args.allow_small_dataset:
        raise SystemExit(
            f"Only {len(rows)} reviewed examples are available. "
            f"Add at least {MIN_TRAINING_EXAMPLES} examples, or use "
            "--allow-small-dataset for a deliberate pilot."
        )

    if not rows:
        raise SystemExit("No valid reviewed examples are available.")

    try:
        import torch
        from datasets import Dataset
        from peft import (
            LoraConfig,
            TaskType,
            prepare_model_for_kbit_training,
        )
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Check "
            "requirements-training.txt."
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this training configuration.")

    # This setup targets a single NVIDIA GPU.
    if torch.cuda.device_count() != 1:
        raise SystemExit(
            "This pilot script expects exactly one visible CUDA GPU."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    compute_dtype = torch.float16

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    print("Loading 4-bit model onto the single CUDA GPU...")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization_config,
        dtype=compute_dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False

    # Prepare the frozen quantized base model before adding LoRA.
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    text_rows = []

    for row in rows:
        text = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        text_rows.append({"text": text})

    dataset = Dataset.from_list(text_rows)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_args = SFTConfig(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=False,
        bf16=False,
        max_length=1024,
        dataset_text_field="text",
        packing=False,
        optim="adamw_torch",
        save_safetensors=True,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print(f"Training on {len(dataset)} reviewed example(s)...")
    trainer.train()

    adapter_dir = args.output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print(f"LoRA adapter saved to: {adapter_dir}")

    # Merging a 7B model may need substantial additional RAM.
    # Keep adapter saving independent from the optional merge step.
    try:
        print("Merging adapter into the base model...")
        merged_model = trainer.model.merge_and_unload()

        merged_dir = args.output_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)

        merged_model.save_pretrained(
            merged_dir,
            safe_serialization=True,
            max_shard_size="4GB",
        )
        tokenizer.save_pretrained(merged_dir)

        modelfile = args.output_dir / "Modelfile"
        modelfile.write_text(
            f"FROM {merged_dir.resolve()}\n"
            "PARAMETER temperature 0\n",
            encoding="utf-8",
        )

        print(f"Merged model saved to: {merged_dir}")
        print(
            "If supported by your Ollama installation, create it with:\n"
            f'ollama create myanmar-law-finetuned -f "{modelfile}"'
        )

    except Exception as exc:
        print(f"Adapter was saved, but merging failed: {exc}")
        print("The adapter is available; merging can be retried separately.")


if __name__ == "__main__":
    main()
