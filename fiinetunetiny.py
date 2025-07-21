import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import SFTTrainer
from peft import LoraConfig

# ==== Step 1: Load & Clean Dataset ====
print("Loading dataset...")

with open("tinylama/formatted_dataset.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

cleaned_data = []
for item in raw_data:
    instruction = item.get("instruction", "").strip()
    output = item.get("output", "")
    if isinstance(output, list):
        output = "\n".join(
            o["point"] if isinstance(o, dict) and "point" in o else str(o)
            for o in output
        )
    output = output.strip()
    if instruction and output:
        cleaned_data.append({"instruction": instruction, "output": output})

print(f" Cleaned dataset size: {len(cleaned_data)} examples")

# Convert to HF dataset
dataset = Dataset.from_list(cleaned_data)

# ==== Step 2: Format Dataset (manual prompt construction) ====
def format_sample(example):
    return {
        "text": f"<s>[INST] {example['instruction']} [/INST] {example['output']}</s>"
    }

formatted_dataset = dataset.map(format_sample)

# ==== Step 3: Load Tokenizer & Model ====
print("Loading TinyLlama model...")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config
)

# ==== Step 4: LoRA Config ====
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# ==== Step 5: Training Arguments ====
training_args = TrainingArguments(
    output_dir="./tinyllama-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    save_steps=100,
    logging_steps=10,
    save_total_limit=2,
    report_to="none"
)

# ==== Step 6: Train ====
print("Starting fine-tuning...")

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    args=training_args,
    peft_config=peft_config
)

trainer.train()

# ==== Step 7: Save Model ====
print("Saving fine-tuned model...")
trainer.save_model("./tinyllama-trained")
tokenizer.save_pretrained("./tinyllama-trained")

print(" Done! Model trained and saved.")
