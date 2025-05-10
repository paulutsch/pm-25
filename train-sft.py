import yaml
import argparse
from unsloth import FastLanguageModel
from transformers import BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
import torch
from datasets import load_dataset
import json
from utils import formatting_func, run_wandb
import wandb
import shutil

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    type=str,
    required=True,
    help="Please set your training parameters in config-sft.yaml")
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

model_name = config['model']['name']
dataset_name = config['dataset']['name']
max_seq_length = int(config['max_seq_length']['length'])
#lora specific parameters
matrice_size = int(config['low_rank_matrice']['size'])
influence = int(config['influence']['rate'])
dropout = int(config['dropout']['rate'])
#hyperparameters
lr = float(config['training']['learning_rate'])
batch_size = int(config['training']['batch_size'])
grad_acc_steps = int(config['training']['grad_acc_steps'])
max_training_steps = int(config['training']['max_training_steps'])
warmup_steps = int(config["training"]["warmup_steps"])

print(config)

base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name, max_seq_length=max_seq_length, dtype=None, load_in_4bit=True
    )

model = FastLanguageModel.get_peft_model(
    base_model,
    r=matrice_size, #the size of the low-rank matrices
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=influence, # adjusts how strongly the LoRA modifications affect the model
    lora_dropout=dropout,  # Supports any, but = 0 is optimized. 
    bias="none",  # Supports any, but = "none" is optimized
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    max_seq_length=max_seq_length,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
    )

model.config.quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    )

train_dataset = load_dataset(dataset_name, split="train")

train_dataset = train_dataset.map(formatting_func, batched=True)

run_wandb(model_name, dataset_name, max_training_steps, lr, batch_size, grad_acc_steps)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc_steps,
        warmup_steps=warmup_steps,
        max_steps=max_training_steps,
        report_to="wandb",
        run_name=f"{model_name}_{dataset_name}",
        output_dir="outputs",
        optim="adamw_8bit",
    ),
    )

trainer.train()

wandb.finish()


save_dir = f"{model_name.split('/')[-1]}_finetuned_{dataset_name.split('/')[-1]}"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
shutil.copy(args.config, save_dir)
