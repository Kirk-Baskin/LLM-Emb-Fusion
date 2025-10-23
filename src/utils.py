from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    average_precision_score,
)
from sklearn.utils import resample
import pandas as pd
import kagglehub
import openml
import os
import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm
import copy
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def get_dataset(dataset_name: str) -> pd.DataFrame:

    if dataset_name == "adult":
        dataset = openml.datasets.get_dataset(1590)
        data_df, label, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
        data_df['label'] = label

    elif dataset_name == "bank":
        dataset = openml.datasets.get_dataset(1461)
        data_df, label, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
        data_df['label'] = label

    elif dataset_name == "blood":
        dataset = openml.datasets.get_dataset(1464)
        data_df, label, _, _ = dataset.get_data(
            target=dataset.default_target_attribute,
            dataset_format="dataframe"
        )
        data_df['label'] = label
        # data_df['label'] = data_df['label'].map({'2':0, '1':1})

    elif dataset_name == "car":
        dataset = openml.datasets.get_dataset(40975)
        data_df, label, _, _  = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
        data_df['label'] = label
    
    elif dataset_name == "credit-g":
        dataset = openml.datasets.get_dataset(31)
        data_df, label, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
        data_df['label'] = label

    elif dataset_name == "diabetes":
        path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
        train_path = os.path.join(path, "diabetes.csv")
        data_df = pd.read_csv(train_path, header=0)
        data_df.rename(columns={'Outcome': 'label'}, inplace=True)

    elif dataset_name == "heart":
        path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")
        train_path = os.path.join(path, "heart.csv")
        data_df = pd.read_csv(train_path, header=0)
        data_df.rename(columns={'HeartDisease': 'label'}, inplace=True)

    return data_df

def get_tabular(dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a tabular dataset and return (X, y) as numpy arrays ready for ML tasks.

    Numeric missing values -> 0
    Categorical missing values -> 'unknown'
    """
    # ===============================
    # Load dataset
    # ===============================
    if dataset_name == "antibiotic":
        data_df = pd.read_csv("data/antibiotic.csv", header=0)

    elif dataset_name == "diabetes":
        path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
        csv_path = os.path.join(path, "diabetes.csv")
        data_df = pd.read_csv(csv_path, header=0)
        data_df.rename(columns={'Outcome': 'label'}, inplace=True)

    elif dataset_name == "heart":
        path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")
        csv_path = os.path.join(path, "heart.csv")
        data_df = pd.read_csv(csv_path, header=0)
        data_df.rename(columns={'HeartDisease': 'label'}, inplace=True)

    elif dataset_name == "blood":
        dataset = openml.datasets.get_dataset(1464)
        data_df, label, _, _ = dataset.get_data(
            target=dataset.default_target_attribute,
            dataset_format="dataframe"
        )
        data_df['label'] = label
        data_df['label'] = data_df['label'].map({'2': 0, '1': 1})

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    if 'label' not in data_df.columns:
        raise ValueError(f"No 'label' column found in {dataset_name} dataset")

    # ===============================
    # Handle missing data
    # ===============================
    # Split numeric and categorical columns
    numeric_cols = data_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data_df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Fill missing numeric values with 0
    data_df[numeric_cols] = data_df[numeric_cols].fillna(0)

    # Fill missing categorical values safely
    for col in categorical_cols:
        # If column is categorical, add "unknown" as a category first
        if pd.api.types.is_categorical_dtype(data_df[col]):
            if "unknown" not in data_df[col].cat.categories:
                data_df[col] = data_df[col].cat.add_categories(["unknown"])
        # Fill missing values with "unknown"
        data_df[col] = data_df[col].fillna("unknown")
        # Ensure string type for dummy encoding
        data_df[col] = data_df[col].astype(str)

    # ===============================
    # Prepare features and labels
    # ===============================
    X = data_df.drop(columns=['label'])
    y = data_df['label']

    # Convert categorical features to numeric (one-hot encoding)
    X = pd.get_dummies(X, drop_first=True)

    # Convert to NumPy arrays
    X = X.to_numpy(dtype=np.float32)
    y = y.to_numpy(dtype=np.int64)

    return X, y

def evaluate_predictions(preds, probs, y_true):

    # Get classification report as dict
    cls_report = classification_report(y_true, preds, output_dict=True, digits=4)
    
    # Flatten the classification report
    results = {}
    for key, metrics in cls_report.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                results[f"{key}_{metric_name}"] = value
        else:  # for metrics like 'accuracy' which are single values
            results[key] = metrics
    
    # Confusion matrix
    cm = confusion_matrix(y_true, preds)
    
    # Additional metrics
    auc_roc = roc_auc_score(y_true, probs)
    auc_pr = average_precision_score(y_true, probs)
    
    # Combine everything
    results.update({
        "confusion_matrix": cm.tolist(),
        "auc_roc": auc_roc,
        "auc_pr": auc_pr
    })
    
    return results

def oversample_df(df, label_col='label', random_state=0):
    # Separate majority and minority classes
    class_counts = df[label_col].value_counts()
    max_count = class_counts.max()

    balanced_parts = []
    for label in class_counts.index:
        subset = df[df[label_col] == label]
        # Resample up to the majority count
        subset_upsampled = resample(
            subset,
            replace=True,            # sample with replacement
            n_samples=max_count,     # match majority class
            random_state=random_state
        )
        balanced_parts.append(subset_upsampled)

    # Concatenate and shuffle
    balanced_df = pd.concat(balanced_parts).sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_df

def serialize(df):
    texts = []
    for _, row in df.iterrows():
        parts = []
        for col, val in row.items():
            val_str = str(val)
            # Handle NaNs gracefully
            if pd.isna(val):
                val_str = "unknown"
            parts.append(f"The {col} is {val_str}.")
        texts.append(" ".join(parts))
    return texts

def train_model(
    model, tokenizer, train_texts, train_labels, 
    val_texts=None, val_labels=None, device="cpu",
    data_collator=None, epochs=10, batch_size=16, lr=5e-5, weight_decay=0.01, max_length=512
):

    model = model.to(device)

    # Tokenize training data
    train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    train_labels_tensor = torch.tensor(train_labels)
    train_dataset = TensorDataset(train_enc["input_ids"], train_enc["attention_mask"], train_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

    # Tokenize validation data if provided
    if val_texts is not None and val_labels is not None:
        val_enc = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        val_labels_tensor = torch.tensor(val_labels)
        val_dataset = TensorDataset(val_enc["input_ids"], val_enc["attention_mask"], val_labels_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    else:
        val_loader = None

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_model = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        total_train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()} if isinstance(batch, dict) else [x.to(device) for x in batch]

            if isinstance(batch, dict):
                outputs = model(**batch)
            else:
                input_ids, attention_mask, labels = batch
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation (optional) ---
        if val_loader is not None:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()} if isinstance(batch, dict) else [x.to(device) for x in batch]
                    if isinstance(batch, dict):
                        outputs = model(**batch)
                    else:
                        input_ids, attention_mask, labels = batch
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    total_val_loss += outputs.loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # Save best model if val set exists
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = copy.deepcopy(model.state_dict())
        else:
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} (no validation)")

    # Load best model if validation exists, else return last epoch
    if val_loader is not None:
        model.load_state_dict(best_model)

    print("Training complete.", "Best validation loss:" if val_loader is not None else "")
    if val_loader is not None:
        print(best_val_loss)
    return model

def predict(model, tokenizer, texts, device, batch_size=4, max_length=512):
    enc = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    dataset = TensorDataset(enc["input_ids"], enc["attention_mask"])
    loader = DataLoader(dataset, batch_size=batch_size)

    all_logits = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu())
    all_logits = torch.cat(all_logits, dim=0)
    return all_logits

def get_embeddings(model_name, model, tokenizer, text_list, device, batch_size=16, max_length=256):
    if model_name == 'facebook/bart-base':
        return get_bart_embeddings(text_list, model, tokenizer, device, batch_size, max_length)
    elif model_name == 'microsoft/biogpt':
        return get_biogpt_embeddings(text_list, model, tokenizer, device, batch_size, max_length)
    elif model_name == 'UFNLP/gatortron-base':
        return get_gatortron_embeddings(text_list, model, tokenizer, device, batch_size, max_length)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
# Extract mean-pooled BART embeddings from text_list
def get_bart_embeddings(text_list, model, tokenizer, device, batch_size=16, max_length=256):
    model.eval()
    model = model.to(device)
    embeddings = []

    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True,
                            padding=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            encoder_outputs = model.model.encoder(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                output_hidden_states=False,
                return_dict=True
            )
            hidden_states = encoder_outputs.last_hidden_state

        # Attention-masked mean pooling
        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_states.size()).float()
        masked_hidden = hidden_states * attention_mask
        summed = masked_hidden.sum(dim=1)
        counts = attention_mask.sum(dim=1)
        batch_embeddings = summed / counts
        embeddings.append(batch_embeddings.cpu().numpy())

    return np.vstack(embeddings)

def get_biogpt_embeddings(text_list, model, tokenizer, device, batch_size=16, max_length=256):
    model.eval()
    model = model.to(device)
    embeddings = []

    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            # Access the encoder/base model
            outputs = model.base_model(**inputs)
            hidden_states = outputs.last_hidden_state

        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_states.size()).float()
        masked_hidden = hidden_states * attention_mask
        summed = masked_hidden.sum(dim=1)
        counts = attention_mask.sum(dim=1)
        batch_embeddings = summed / counts

        embeddings.append(batch_embeddings.cpu().numpy())

    return np.vstack(embeddings)

def get_gatortron_embeddings(text_list, model, tokenizer, device, batch_size=16, max_length=256):
    model.eval()
    model = model.to(device)
    embeddings = []

    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            # Use the base GatorTron model, not the classifier wrapper
            outputs = model.gatortron(**inputs)
            hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

        # Attention-masked mean pooling
        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_states.size()).float()
        masked_hidden = hidden_states * attention_mask
        summed = masked_hidden.sum(dim=1)
        counts = attention_mask.sum(dim=1)
        batch_embeddings = summed / counts

        embeddings.append(batch_embeddings.cpu().numpy())

    return np.vstack(embeddings)


def get_classifier(clf_name, max_iter=10000, random_state=0):

    if clf_name == "logistic-regression":
        clf = LogisticRegression(max_iter=max_iter, random_state=random_state)

    elif clf_name == "rbf-svm":
        clf = SVC(kernel='rbf', probability=True, random_state=random_state)

    elif clf_name == "xgboost":
        clf = XGBClassifier(random_state=random_state)

    else:
        raise ValueError(f"Unknown classifier name: {clf_name}")

    return clf

def get_fewshot(train_texts, y_train, shots, random_state=None):
    """
    Selects a few-shot subset of texts and labels, stratified by class.
    
    train_texts: list of strings
    y_train: array-like of labels (pandas Series or np.array)
    shots: total number of few-shot examples (must be divisible by number of classes)
    random_state: int for reproducibility
    """
    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()
    
    y_np = np.array(y_train)
    classes = np.unique(y_np)
    shots_per_class = shots // len(classes)
    
    fewshot_texts = []
    fewshot_labels = []
    
    for cls in classes:
        idx_cls = np.where(y_np == cls)[0]
        chosen_idx = rng.choice(idx_cls, shots_per_class, replace=False)
        fewshot_texts.extend([train_texts[i] for i in chosen_idx])
        fewshot_labels.extend(y_np[chosen_idx])
    
    # Shuffle the few-shot set
    perm = rng.permutation(len(fewshot_labels))
    fewshot_texts = [fewshot_texts[i] for i in perm]
    fewshot_labels = np.array(fewshot_labels)[perm]
    
    return fewshot_texts, fewshot_labels