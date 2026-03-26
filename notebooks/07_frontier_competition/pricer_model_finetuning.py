# Install required libraries (quiet mode to reduce output noise)
!pip install -q --upgrade bitsandbytes==0.48.2 trl==0.25.1

# Download helper utilities (evaluation, formatting, etc.)
!wget -q https://raw.githubusercontent.com/ed-donner/llm_engineering/main/week7/util.py -O util.py


# =========================
# IMPORTS
# =========================
# Standard libraries
import os
import re
import math
from datetime import datetime

# Progress bar
from tqdm import tqdm

# Colab secrets handling
from google.colab import userdata

# Hugging Face ecosystem
from huggingface_hub import login
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Dataset handling
from datasets import load_dataset

# Experiment tracking
import wandb

# Parameter-efficient fine-tuning (LoRA)
from peft import LoraConfig

# Trainer for supervised fine-tuning
from trl import SFTTrainer, SFTConfig

# Visualization (not actually used later)
import matplotlib.pyplot as plt


# =========================
# GENERAL CONFIGURATION
# =========================

# Base LLM to fine-tune
BASE_MODEL = "meta-llama/Llama-3.2-3B"

# Project naming (used in logging + model hub)
PROJECT_NAME = "price"
HF_USER = "Davidmeenderink"

# Lite mode reduces compute for testing/debugging
LITE_MODE = True

# Dataset selection (small vs full dataset)
DATA_USER = "ed-donner"
DATASET_NAME = f"{DATA_USER}/items_prompts_lite" if LITE_MODE else f"{DATA_USER}/items_prompts_full"

# Unique run name based on timestamp
RUN_NAME = f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
if LITE_MODE:
    RUN_NAME += "-lite"

PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_RUN_NAME}"


# =========================
# TRAINING HYPERPARAMETERS
# =========================

# Epochs and batch size scaled down in lite mode
EPOCHS = 1 if LITE_MODE else 3
BATCH_SIZE = 32 if LITE_MODE else 256

# Maximum input length (tokens)
MAX_SEQUENCE_LENGTH = 128

# Gradient accumulation simulates larger batch size
GRADIENT_ACCUMULATION_STEPS = 1


# =========================
# QLoRA (Efficient Fine-Tuning) CONFIG
# =========================

# Enable 4-bit quantization (huge memory savings)
QUANT_4_BIT = True

# LoRA rank (smaller in lite mode)
LORA_R = 32 if LITE_MODE else 256
LORA_ALPHA = LORA_R * 2

# Layers to apply LoRA to
ATTENTION_LAYERS = ["q_proj", "v_proj", "k_proj", "o_proj"]
MLP_LAYERS = ["gate_proj", "up_proj", "down_proj"]

# In lite mode only attention layers are tuned
TARGET_MODULES = ATTENTION_LAYERS if LITE_MODE else ATTENTION_LAYERS + MLP_LAYERS

# Regularization
LORA_DROPOUT = 0.1


# =========================
# OPTIMIZATION SETTINGS
# =========================

LEARNING_RATE = 1e-4
WARMUP_RATIO = 0.01
LR_SCHEDULER_TYPE = 'cosine'
WEIGHT_DECAY = 0.001
OPTIMIZER = "paged_adamw_32bit"

# Check GPU capability for bf16 support
capability = torch.cuda.get_device_capability()
use_bf16 = capability[0] >= 8


# =========================
# LOGGING / TRACKING
# =========================

VAL_SIZE = 500 if LITE_MODE else 1000
LOG_STEPS = 5 if LITE_MODE else 10
SAVE_STEPS = 100 if LITE_MODE else 200
LOG_TO_WANDB = True


# =========================
# AUTHENTICATION
# =========================

# Hugging Face login (for model push)
hf_token = userdata.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)

# Weights & Biases login (for experiment tracking)
wandb_api_key = userdata.get('WANDB_API_KEY')
os.environ["WANDB_API_KEY"] = wandb_api_key
wandb.login()

# Configure W&B project
os.environ["WANDB_PROJECT"] = PROJECT_NAME
os.environ["WANDB_LOG_MODEL"] = "false"
os.environ["WANDB_WATCH"] = "false"


# =========================
# DATA LOADING
# =========================

dataset = load_dataset(DATASET_NAME)

# Split dataset
train = dataset['train']
val = dataset['val'].select(range(VAL_SIZE))
test = dataset['test']

# Initialize W&B run
if LOG_TO_WANDB:
    wandb.init(project=PROJECT_NAME, name=RUN_NAME)


# =========================
# MODEL LOADING (QUANTIZED)
# =========================

# Configure quantization
if QUANT_4_BIT:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_quant_type="nf4"
    )
else:
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load base model with quantization
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
)

# Ensure padding works correctly
base_model.generation_config.pad_token_id = tokenizer.pad_token_id

print(f"Memory footprint: {base_model.get_memory_footprint() / 1e6:.1f} MB")


# =========================
# LORA CONFIGURATION
# =========================

lora_parameters = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=TARGET_MODULES,
)


# =========================
# TRAINING CONFIG
# =========================

train_parameters = SFTConfig(
    output_dir=PROJECT_RUN_NAME,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    optim=OPTIMIZER,
    save_steps=SAVE_STEPS,
    logging_steps=LOG_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    fp16=not use_bf16,
    bf16=use_bf16,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    report_to="wandb" if LOG_TO_WANDB else None,
    run_name=RUN_NAME,
    max_length=MAX_SEQUENCE_LENGTH,

    # Hugging Face Hub integration
    push_to_hub=True,
    hub_model_id=HUB_MODEL_NAME,
    hub_private_repo=True,

    # Evaluation during training
    eval_strategy="steps",
    eval_steps=SAVE_STEPS
)


# =========================
# TRAINER SETUP
# =========================

fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=train,
    eval_dataset=val,
    peft_config=lora_parameters,
    args=train_parameters
)


# =========================
# TRAINING
# =========================

fine_tuning.train()


# =========================
# SAVE MODEL
# =========================

fine_tuning.model.push_to_hub(PROJECT_RUN_NAME, private=True)
print(f"Saved to the hub: {PROJECT_RUN_NAME}")

if LOG_TO_WANDB:
    wandb.finish()