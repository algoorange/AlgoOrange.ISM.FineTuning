# 1. Install dependencies (run this in your terminal if not installed)
# pip install transformers datasets peft trl accelerate bitsandbytes

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import os

# 2. Load all your JSONL training files
jsonl_folder = "./datasets"  # Change this to your JSONL folder path
jsonl_files = [
    os.path.join(jsonl_folder, f)
    for f in os.listdir(jsonl_folder)
    if f.endswith(".jsonl")
]

# Load and merge all JSONL files into one training dataset
dataset = load_dataset("json", data_files={"train": jsonl_files}, split="train")
dataset = dataset.filter(lambda x: "messages" in x and isinstance(x["messages"], list))


# Format messages into ChatML-style training strings
def format_messages(example):
    parts = []
    for msg in example["messages"]:
        if msg["role"] == "user":
            parts.append(f"<|user|>\n{msg['content']}")
        elif msg["role"] == "assistant":
            parts.append(f"<|assistant|>\n{msg['content']}")
    return {"text": "\n".join(parts)}


# Apply formatter
formatted_dataset = dataset.map(format_messages)


# Optional: print first few examples to verify
for i in range(min(3, len(formatted_dataset))):
    print(f"\n--- Example {i} ---\n{formatted_dataset[i]['text']}")


# 3. Load tokenizer and base model (from Hugging Face or Ollama registry mirror)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # or path to local LLaMA 3.1 8B
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=False,  # or float16 if your GPU supports it
    device_map="auto",  # will fall back to CPU
    revision="main",
)

# 4. Prepare model for QLoRA
model = prepare_model_for_kbit_training(model)

# 5. LoRA (PEFT) configuration
peft_config = LoraConfig(
    r=8,  # Low-rank dimension
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.05,  # Dropout for regularization
    bias="none",  # No bias adaptation
    task_type="CAUSAL_LM",  # Causal Language Modeling
)

# 6. Apply LoRA
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # Check trainable parameters

# 7. Training arguments for QLoRA fine-tuning
# training_args = TrainingArguments(
#     output_dir="./e8-llama3-finetuned",  # Where to save the model
#     per_device_train_batch_size=2,  # Depends on VRAM (2 fits ~24GB)
#     gradient_accumulation_steps=4,  # For effective larger batch size
#     num_train_epochs=3,
#     learning_rate=2e-4,
#     logging_dir="./logs",
#     logging_steps=10,
#     save_strategy="epoch",
#     report_to="none",  # Disable WandB unless needed
#     fp16=True,  # Mixed precision for faster training
#     optim="paged_adamw_8bit",  # Optimizer for 8bit training
# )
sft_config = SFTConfig(
    output_dir="./e8-llama3-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    fp16=False,
    optim="adamw_torch",
    max_seq_length=2048,  # ✅ moved here
    label_names=["labels"],
)

# 8. Train using SFTTrainer (supports chat-style messages)
# trainer = SFTTrainer(
#     model=model,
#     train_dataset=dataset,
#     tokenizer=tokenizer,
#     args=training_args,
#     dataset_text_field="messages",  # JSONL field used for instruction tuning
#     max_seq_length=2048,  # Based on model context window
# )
print(formatted_dataset)


trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    args=sft_config,
)

# 9. Start training
trainer.train()

# 10. Save only the LoRA adapter (not full model to save space)
trainer.model.save_pretrained("./e8-llama3-finetuned/lora-adapter")
tokenizer.save_pretrained("./e8-llama3-finetuned/tokenizer")
