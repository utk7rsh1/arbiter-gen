"""SFT Training Script for ARBITER.

Fine-tunes Qwen 2.5 1.5B on the generated trajectories using Unsloth + TRL.
Run this on Google Colab T4.

Usage:
    python train_sft.py \
        --dataset sft_trajectories_clean.jsonl \
        --output lora_sft_v4/ \
        --epochs 3

Compatibility: Unsloth 2026.4.x, trl 0.19.x, transformers 5.x, torch 2.10+
"""
import argparse
import json
import sys
from pathlib import Path

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",    default="data/sft_trajectories.jsonl")
parser.add_argument("--output",     default="lora_sft/")
parser.add_argument("--epochs",     type=int,   default=3)
parser.add_argument("--batch_size", type=int,   default=4)
parser.add_argument("--lr",         type=float, default=2e-4)
parser.add_argument("--max_len",    type=int,   default=1024)
parser.add_argument("--hub_repo",   default=None)
args = parser.parse_args()

MODEL_NAME  = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LEN = args.max_len
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Imports ───────────────────────────────────────────────────────────────────
# Unsloth MUST be imported before trl/transformers to apply its patches.
try:
    import unsloth  # noqa: F401 — side-effect import, must be first
    from unsloth import FastLanguageModel
    UNSLOTH = True
except ImportError:
    print("Unsloth not available. Falling back to standard transformers + TRL.")
    UNSLOTH = False

# trl imports — handle multiple TRL versions
from trl import SFTTrainer, SFTConfig
try:
    from trl import DataCollatorForCompletionOnlyLM
    _HAS_COLLATOR = True
except ImportError:
    try:
        from trl.trainer.utils import DataCollatorForCompletionOnlyLM
        _HAS_COLLATOR = True
    except ImportError:
        _HAS_COLLATOR = False
        print("[INFO] DataCollatorForCompletionOnlyLM not available in this TRL version."
              " Training without response-only loss masking (still works fine).")
from datasets import Dataset
import torch
import transformers

if not UNSLOTH:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model

transformers.set_seed(42)

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME}...")
if UNSLOTH:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=32,
        lora_dropout=0,        # 0 enables Unsloth fast patching for all layers
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
else:
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_cfg)

tokenizer.pad_token = tokenizer.eos_token

# ── Dataset ───────────────────────────────────────────────────────────────────
SYSTEM = (
    "You are an expert AI auditor investigating a synthetic AI Decision System "
    "for hidden anomalies.\n"
    "Output exactly one JSON action per turn. Think step by step before acting."
)

RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


def load_trajectories(path: str) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = (
                f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
                f"<|im_start|>user\n{item['prompt']}<|im_end|>\n"
                f"<|im_start|>assistant\n{item['response']}<|im_end|>"
            )
            records.append({"text": text, "level": item.get("level", 1)})
    return records


print(f"Loading dataset from {args.dataset}...")
if not Path(args.dataset).exists():
    print(f"ERROR: {args.dataset} not found.")
    sys.exit(1)

records = load_trajectories(args.dataset)
print(f"  {len(records)} training pairs loaded.")

dataset = Dataset.from_list(records)
dataset = dataset.train_test_split(test_size=0.05, seed=42)
print(f"  Train: {len(dataset['train'])}  Eval: {len(dataset['test'])}")

# ── Loss masking — response tokens only ─────────────────────────────────────
if _HAS_COLLATOR:
    response_template_ids = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
    )
else:
    collator = None   # fall back: loss computed on full sequence

# ── Training config ───────────────────────────────────────────────────────────
# warmup_ratio removed in trl 0.19.x — use warmup_steps.
# 5% of ~427 total steps ≈ 21
training_args = SFTConfig(
    output_dir=args.output,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=4,
    warmup_steps=21,
    learning_rate=args.lr,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    save_steps=100,
    save_strategy="steps",
    eval_strategy="steps",
    eval_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",
    max_seq_length=MAX_SEQ_LEN,
    dataset_text_field="text",
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,   # 'tokenizer' kwarg removed in trl 0.19.x
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=collator,
    args=training_args,
)

# ── Train ─────────────────────────────────────────────────────────────────────
print(f"Training for {args.epochs} epochs on {len(dataset['train'])} samples...")
trainer.train()
print("Training complete.")

# ── Save ──────────────────────────────────────────────────────────────────────
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