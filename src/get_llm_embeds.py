from utils import set_seed, get_dataset, evaluate_predictions, serialize, get_embeddings
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from GatorTronForSequenceClassification import GatorTronForSequenceClassification
import gc
import os

_SEED = 0
_LABEL = 'label'
_BATCH_SIZE = 16
_MAX_LENGTH = 256
_DIMENSIONALITY = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("WARNING: No GPU detected, running on CPU which may be slow.", flush=True)

set_seed(_SEED)

models = [
    ('facebook/bart-base', 'bart'),
    ('microsoft/biogpt', 'biogpt'),
    ('UFNLP/gatortron-base', 'gatortron'),
]

datasets = [
    'adult',
    'bank',
    'blood',
    'car',
    'credit-g',
    'diabetes',
    'heart'
]


for dataset_name in datasets:

    # Load dataset, print shape and num labels
    df = get_dataset(dataset_name=dataset_name)
    num_labels = df[_LABEL].nunique()
    print(f"\nDataset: {dataset_name}, Shape: {df.shape}, Num Labels: {num_labels}")

    # Serialize dataset
    texts = serialize(df.drop(columns=[_LABEL]))

    for model_name, short_model_name in models:

        print(f"\Extracting {model_name} embeddings from {dataset_name}", flush=True)

        # Get model and tokenizer
        if model_name == 'UFNLP/gatortron-base':
            model = GatorTronForSequenceClassification(model_name, num_labels=num_labels)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif model_name == 'facebook/bart-base' or model_name == 'microsoft/biogpt':
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Get embeddings
        embeds = get_embeddings(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            text_list=texts,
            device=device,
            batch_size=_BATCH_SIZE,
            max_length=_MAX_LENGTH
        )
        labels = df[_LABEL]

        # Ensure the main folder exists
        os.makedirs("embeds", exist_ok=True)

        # Save embeddings + labels
        save_path = f'embeds/{short_model_name}-{dataset_name}-zeroshot'
        np.savez_compressed(save_path, embeddings=embeds, labels=labels.values)
        print(f"Saved embeddings and labels to {save_path}")

        # Clean up to free memory
        del model, tokenizer, embeds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

            

