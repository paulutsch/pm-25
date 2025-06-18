from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
import os

# Model and dataset
model_name = "meta-llama/Llama-3.1-8B-Instruct"
dataset_name = "allenai/tulu-3-sft-personas-instruction-following"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # LLaMA does not have a pad token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # use float16 if GPU supports it
    device_map="auto",
    trust_remote_code=True
)

# Load and preprocess dataset
dataset = load_dataset(dataset_name, split="train")

def format_example(example):
    prompt = example["prompt"]
    completion = example["messages"][1]["content"]
    full_text = f"<|user|>\n{prompt}\n<|assistant|>\n{completion}"
    return tokenizer(
        full_text,
        max_length=4096,
        padding="max_length",
        truncation=True
    )

tokenized_dataset = dataset.map(format_example, remove_columns=dataset.column_names)

# Training configuration
training_args = TrainingArguments(
    output_dir="./llama3.1-8b-instruct-finetuned",
    per_device_train_batch_size=4,  # gradient_accumulation will scale this
    gradient_accumulation_steps=32,  # 4 x 32 = 128 effective batch
    learning_rate=5e-6,
    lr_scheduler_type="linear",
    warmup_ratio=0.03,
    num_train_epochs=2,
    logging_steps=10,
    save_steps=1000,
    save_total_limit=2,
    #evaluation_strategy="no",  # for now, just train
    fp16=True,  # or bf16=True if supported
    #bf16=torch.cuda.is_bf16_supported(),
    #report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Start training
trainer.train()

# Save model
trainer.save_model("./llama3.1-8b-instruct-finetuned")
tokenizer.save_pretrained("./llama3.1-8b-instruct-finetuned")