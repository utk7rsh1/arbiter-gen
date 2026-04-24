from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# 🔹 Base + LoRA paths
base_model = "Qwen/Qwen2.5-1.5B-Instruct"
lora_path = "."   # current folder (important fix)

# 🔹 Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto" if torch.cuda.is_available() else "cpu"
)

# 🔹 Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)

# 🔹 Attach LoRA
model = PeftModel.from_pretrained(model, lora_path)

# 🔹 Strong prompt (VERY IMPORTANT)
prompt = """You are auditing an AI Decision System.

- Step: 1/20
- Budget remaining: 18
- Features available: [age, income, gender]

You must output EXACTLY ONE JSON object.

Rules:
- Do NOT repeat the prompt
- Do NOT include explanations
- Do NOT include markdown (no ```json)
- Output must be valid JSON

Schema:
{
  "type": "CLAIM",
  "strength": 0.5,
  "feature_id": "income",
  "predicted_outcome": "APPROVED",
  "actual_outcome": "DENIED"
}

Now output the JSON:
"""

# 🔹 Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 🔹 Generate (controlled)
outputs = model.generate(
    **inputs,
    max_new_tokens=80,
    do_sample=False,   # deterministic
    temperature=0.2,
    eos_token_id=tokenizer.eos_token_id
)

# 🔹 Decode
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 🔹 Extract only last JSON (optional cleanup)
import re
matches = re.findall(r"\{.*?\}", result)

if matches:
    print(matches[-1])
else:
    print(result)