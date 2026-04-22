"""SFT Training Script for ARBITER.

Fine-tunes Qwen 2.5 1.5B on the generated trajectories using Unsloth + TRL.
Run this on Google Colab T4.

Usage:
    python train_sft.py \
        --dataset data/sft_trajectories.jsonl \
        --output lora_sft/ \
        --epochs 3 \
        --hub_repo your-username/arbiter-sft
"""
import argparse
import json
import os
import sys
from pathlib import Path

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",   default="data/sft_trajectories.jsonl")
parser.add_argument("--output",    default="lora_sft/")
parser.add_argument("--epochs",    type=int, default=3)
parser.add_argument("--batch_size",type=int, default=4)
parser.add_argument("--lr",        type=float, default=2e-4)
parser.add_argument("--max_len",   type=int, default=1024)
parser.add_argument("--hub_repo",  default=None, help="HuggingFace repo to push adapter")
args = parser.parse_args()

# ── Imports (Colab: pip install unsloth trl datasets) ─────────────────────────
try:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    import torch
    UNSLOTH = True
except ImportError:
    print("Unsloth not available. Falling back to standard transformers + TRL.")
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer
    from datasets import Dataset
    import torch
    UNSLOTH = False

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LEN = args.max_len

# ── Load model ─────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME}...")
if UNSLOTH:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,   # auto-detect
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    FastLanguageModel.for_training(model)
else:
    from transformers import BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                          target_modules=["q_proj","k_proj","v_proj","o_proj"])
    model = get_peft_model(model, lora_cfg)

tokenizer.pad_token = tokenizer.eos_token

# ── Load dataset ──────────────────────────────────────────────────────────────
SYSTEM = """You are an expert AI auditor investigating a synthetic AI Decision System for hidden anomalies.
Output exactly one JSON action per turn. Think step by step before acting."""

def load_trajectories(path: str):
    """Load JSONL trajectories and format as chat turns."""
    records = []
    with open(path) as f:
        for line in f:
            item = json.loads(line.strip())
            prompt   = item["prompt"]
            response = item["response"]
            # Format as instruction-following pair
            text = (
                f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{response}<|im_end|>"
            )
            records.append({"text": text, "level": item.get("level", 1)})
    return records

print(f"Loading dataset from {args.dataset}...")
if not Path(args.dataset).exists():
    print(f"ERROR: {args.dataset} not found. Run sft_generator.py first.")
    sys.exit(1)

records = load_trajectories(args.dataset)
print(f"  {len(records)} training pairs loaded.")

dataset = Dataset.from_list(records)
dataset = dataset.train_test_split(test_size=0.05, seed=42)

# ── Training config ────────────────────────────────────────────────────────────
training_args = SFTConfig(
    output_dir=args.output,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    learning_rate=args.lr,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",
    max_seq_length=MAX_SEQ_LEN,
    dataset_text_field="text",
    packing=True,   # pack short sequences for efficiency
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_args,
)

# ── Train ─────────────────────────────────────────────────────────────────────
print(f"Training for {args.epochs} epochs on {len(dataset['train'])} samples...")
trainer.train()
print("Training complete.")

# ── Save ─────────────────────────────────────────────────────────────────────
model.save_pretrained(args.output)
tokenizer.save_pretrained(args.output)
print(f"LoRA adapter saved to {args.output}/")

if args.hub_repo:
    print(f"Pushing to HuggingFace Hub: {args.hub_repo}...")
    model.push_to_hub(args.hub_repo)
    tokenizer.push_to_hub(args.hub_repo)
    print("Pushed.")

print("""
Done! Next step — run GRPO:
    python -m arbiter.training.grpo_trainer --checkpoint lora_sft/ --level 1 --episodes 100
    python -m arbiter.training.grpo_trainer --checkpoint lora_sft/ --level 3 --episodes 300
""")
