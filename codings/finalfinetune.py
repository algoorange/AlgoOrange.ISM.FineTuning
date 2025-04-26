import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# 1. Load Hugging Face token
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

# 2. Load model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# 3. Apply LoRA (standard PEFT)
lora_config = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. Load and format dataset
jsonl_folder = "./datasets"
jsonl_files = [
    os.path.join(jsonl_folder, f)
    for f in os.listdir(jsonl_folder)
    if f.endswith(".jsonl")
]

dataset = load_dataset("json", data_files={"train": jsonl_files}, split="train")
dataset = dataset.filter(lambda x: "messages" in x and isinstance(x["messages"], list))


def format_messages(example):
    parts = []
    for msg in example["messages"]:
        if msg["role"] == "user":
            parts.append(f"<|user|>\n{msg['content']}")
        elif msg["role"] == "assistant":
            parts.append(f"<|assistant|>\n{msg['content']}")
    return {"text": "\n".join(parts)}


formatted_dataset = dataset.map(format_messages)

# 5. Split dataset
# split_data = formatted_dataset.train_test_split(test_size=0.1)
# train_dataset = split_data["train"]
# eval_dataset = split_data["test"]

# 6. SFT config (no QLoRA settings)
sft_config = SFTConfig(
    output_dir="./FinalFinetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=2e-4,
    logging_dir="./FinalFinetuned/logs",
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    fp16=False,
    optim="adamw_torch",
    max_seq_length=1024,
    dataset_text_field="text",
)

# 7. Train using SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    # eval_dataset=eval_dataset,
    args=sft_config,
)

trainer.train()

# 8. Save LoRA adapter and tokenizer
trainer.model.save_pretrained("./FinalFinetuned/lora-adapter")
tokenizer.save_pretrained("./FinalFinetuned/tokenizer")
