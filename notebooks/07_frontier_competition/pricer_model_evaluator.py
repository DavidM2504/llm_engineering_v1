# Install required libraries for quantization + training utilities
!pip install -q --upgrade bitsandbytes trl

# Download helper evaluation utilities
!wget -q https://raw.githubusercontent.com/ed-donner/llm_engineering/main/week7/util.py -O util.py


# =========================
# IMPORTS
# =========================

# Standard libraries
import os
import re
import math
from datetime import datetime

# Progress bar (not used but useful for loops)
from tqdm import tqdm

# Access secrets in Google Colab
from google.colab import userdata

# Hugging Face authentication
from huggingface_hub import login

# PyTorch + Transformers
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

# Dataset handling
from datasets import load_dataset

# Load fine-tuned LoRA adapters
from peft import PeftModel

# Custom evaluation function
from util import evaluate


# =========================
# OVERALL PURPOSE
# =========================
# This script:
# 1. Loads a pre-trained base model (LLaMA)
# 2. Loads a fine-tuned LoRA adapter from Hugging Face
# 3. Loads a test dataset
# 4. Runs inference (price prediction)
# 5. Evaluates model performance using a custom evaluation function


# =========================
# CONFIGURATION
# =========================

BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_NAME = "price"
HF_USER = "Davidmeenderink"  # Your Hugging Face username

# Lite mode = faster testing, smaller dataset
LITE_MODE = True

# Dataset configuration
DATA_USER = "ed-donner"
DATASET_NAME = (
    f"{DATA_USER}/items_prompts_lite"
    if LITE_MODE
    else f"{DATA_USER}/items_prompts_full"
)

# Select which trained model run to evaluate
if LITE_MODE:
    RUN_NAME = "2026-03-26_08.09.45-lite"
    REVISION = None  # latest version
else:
    RUN_NAME = "2025-11-28_18.47.07"
    REVISION = "b19c8bfea3b6ff62237fbb0a8da9779fc12cefbd"

PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_RUN_NAME}"


# =========================
# HARDWARE / PRECISION SETUP
# =========================

# Use 4-bit quantization for efficiency
QUANT_4_BIT = True

# Check GPU capability (for bf16 support)
capability = torch.cuda.get_device_capability()
use_bf16 = capability[0] >= 8


# =========================
# AUTHENTICATION
# =========================

# Login to Hugging Face (needed to load private models)
hf_token = userdata.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)


# =========================
# DATASET LOADING
# =========================

# Load dataset from Hugging Face
dataset = load_dataset(DATASET_NAME)

# Extract test split (used for evaluation)
test = dataset['test']

# Optional: inspect dataset structure
# print(test[0])


# =========================
# QUANTIZATION CONFIGURATION
# =========================

# Configure model loading in 4-bit or 8-bit mode
if QUANT_4_BIT:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,  # improves accuracy
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_quant_type="nf4"  # best quantization type for LLMs
    )
else:
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )


# =========================
# LOAD TOKENIZER + BASE MODEL
# =========================

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

# Ensure proper padding behavior
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load base LLM with quantization
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",  # automatically place on GPU/CPU
)

# Fix padding token for generation
base_model.generation_config.pad_token_id = tokenizer.pad_token_id


# =========================
# LOAD FINE-TUNED MODEL (LoRA)
# =========================

# Attach LoRA adapter weights to base model
if REVISION:
    fine_tuned_model = PeftModel.from_pretrained(
        base_model,
        HUB_MODEL_NAME,
        revision=REVISION
    )
else:
    fine_tuned_model = PeftModel.from_pretrained(
        base_model,
        HUB_MODEL_NAME
    )

# Print memory usage
print(f"Memory footprint: {fine_tuned_model.get_memory_footprint() / 1e6:.1f} MB")


# =========================
# PREDICTION FUNCTION
# =========================

def model_predict(item):
    """
    Generates a price prediction for a single dataset item.

    Steps:
    1. Tokenize input prompt
    2. Run model generation
    3. Remove prompt tokens from output
    4. Decode generated tokens to text
    """

    # Move inputs to GPU
    inputs = tokenizer(item["prompt"], return_tensors="pt").to("cuda")

    # Disable gradients for inference (faster + less memory)
    with torch.no_grad():
        output_ids = fine_tuned_model.generate(
            **inputs,
            max_new_tokens=8  # limit output length
        )

    # Remove original prompt tokens from output
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, prompt_len:]

    # Convert tokens to text
    return tokenizer.decode(generated_ids)


# =========================
# EVALUATION
# =========================

# Ensure reproducibility
set_seed(42)

# Run evaluation on test set
evaluate(model_predict, test)