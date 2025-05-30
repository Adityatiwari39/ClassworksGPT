import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# 1. Load or create your dataset (replace with your own data for real use)
def get_student_dataset():
    # Example: Replace this with your own dataset loading logic
    data = {
        "train": [
            {"text": "Q: What is photosynthesis?\nA: Photosynthesis is the process by which green plants make their own food using sunlight."},
            {"text": "Q: Solve 2x + 3 = 7\nA: x = 2"},
            {"text": "Q: Who wrote Romeo and Juliet?\nA: William Shakespeare wrote Romeo and Juliet."},
        ],
        "validation": [
            {"text": "Q: What is the capital of France?\nA: The capital of France is Paris."},
        ]
    }
    # Save to disk for Hugging Face datasets
    os.makedirs("data", exist_ok=True)
    for split in data:
        with open(f"data/{split}.txt", "w", encoding="utf-8") as f:
            for item in data[split]:
                f.write(item["text"] + "\n")
    dataset = load_dataset("text", data_files={"train": "data/train.txt", "validation": "data/validation.txt"})
    return dataset

# 2. Tokenizer and Model
model_name = "distilgpt2"  # Small and fast for demo; use 'gpt2' or larger for real use
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set pad token to eos token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.pad_token_id = tokenizer.pad_token_id

# 3. Prepare dataset
dataset = get_student_dataset()
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,  # Use True for dynamic padding
        max_length=128
    )
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 4. Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./trained/model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)

# 7. Train
if __name__ == "__main__":
    trainer.train()
    trainer.save_model("./trained/model")
    tokenizer.save_pretrained("./trained/model")
    print("Training complete! Model saved to ./trained/model")

# Directory structure
"""
ClassworksGPT-1/
├── ClassworksGPT/
│   ├── train.py
│   └── ClassworksGPT Version 1.0
├── trained/
│   ├── model/
│   └── datasets/
├── README.md
└── ...
"""