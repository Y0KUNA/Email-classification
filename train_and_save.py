"""
train_and_save.py
=================
Chay train ca 2 mo hinh va luu artifacts:
  - Naive Bayes  -> naive_bayes/naive_bayes_model.pkl
                    naive_bayes/count_vectorizer.pkl
  - DistilBERT   -> distilbert/email-spam-detection-distilbert/

Cach chay:
    # Train ca 2
    python train_and_save.py

    # Chi train Naive Bayes (nhanh, khong can GPU)
    python train_and_save.py --skip-bert

    # Tuy chinh duong dan
    python train_and_save.py --train-csv dataset/combined_data.csv --test-csv dataset/spam.csv
"""

import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent


# ------------------------------------
# DATA LOADING
# ------------------------------------

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="latin-1")
    for col in ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    if "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "text"})
    if "label" not in df.columns or "text" not in df.columns:
        cols = list(df.columns[:2])
        df = df.rename(columns={cols[0]: "label", cols[1]: "text"})

    def normalize(x):
        s = str(x).lower().strip()
        if s in ("ham", "0", "no spam", "no", "legitimate"):
            return 0
        if s in ("spam", "1", "yes"):
            return 1
        try:
            return int(float(x))
        except Exception:
            return 1 if "spam" in s else 0

    df["label"] = df["label"].apply(normalize)
    df = df.dropna(subset=["label", "text"])
    return df[["label", "text"]]


# ------------------------------------
# NAIVE BAYES TRAIN & SAVE
# ------------------------------------

def train_naive_bayes(train_csv: str, test_csv: str):
    print("\n" + "="*50)
    print("  [1/2] TRAIN NAIVE BAYES")
    print("="*50)

    import joblib
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import accuracy_score, classification_report

    print(f"  Loading training data: {train_csv}")
    df_train = load_data(train_csv)
    print(f"  Train shape: {df_train.shape} | Ham: {(df_train['label']==0).sum()} | Spam: {(df_train['label']==1).sum()}")

    print(f"  Loading test data: {test_csv}")
    df_test = load_data(test_csv)
    print(f"  Test shape:  {df_test.shape}  | Ham: {(df_test['label']==0).sum()} | Spam: {(df_test['label']==1).sum()}")

    X_train = df_train["text"]
    y_train = df_train["label"]
    X_test  = df_test["text"]
    y_test  = df_test["label"]

    print("\n  Vectorizing...")
    vectorizer = CountVectorizer()
    X_train_v  = vectorizer.fit_transform(X_train)
    X_test_v   = vectorizer.transform(X_test)

    print("  Training MultinomialNB...")
    model = MultinomialNB()
    model.fit(X_train_v, y_train)

    y_pred = model.predict(X_test_v)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

    # Save
    save_dir = ROOT / "naive_bayes"
    save_dir.mkdir(exist_ok=True)
    model_path = save_dir / "naive_bayes_model.pkl"
    vec_path   = save_dir / "count_vectorizer.pkl"
    joblib.dump(model,      str(model_path))
    joblib.dump(vectorizer, str(vec_path))
    print(f"  Model saved    -> {model_path}")
    print(f"  Vectorizer saved -> {vec_path}")
    return model, vectorizer


# ------------------------------------
# DISTILBERT TRAIN & SAVE
# ------------------------------------

def train_distilbert(train_csv: str, test_csv: str,
                     bert_model: str = "distilbert-base-cased",
                     num_epochs: int = 3,
                     train_batch: int = 8,
                     lr: float = 3e-5):
    print("\n" + "="*50)
    print("  [2/2] TRAIN DISTILBERT")
    print("="*50)

    import gc
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        Trainer, TrainingArguments, DataCollatorWithPadding,
    )
    from datasets import Dataset, ClassLabel
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import accuracy_score, classification_report

    output_dir = ROOT / "distilbert" / "email-spam-detection-distilbert"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Loading training data: {train_csv}")
    df_train = load_data(train_csv)
    print(f"  Loading test data: {test_csv}")
    df_test  = load_data(test_csv)

    # Rename column for HF Dataset
    df_train = df_train.rename(columns={"text": "title"})
    df_test  = df_test.rename(columns={"text": "title"})

    df_train["label"] = df_train["label"].map({0: "ham", 1: "spam"})
    df_test["label"]  = df_test["label"].map({0: "ham", 1: "spam"})

    labels_list = ["ham", "spam"]
    ClassLabels = ClassLabel(num_classes=2, names=labels_list)

    # Class weights
    all_labels = list(df_train["label"]) + list(df_test["label"])
    classes = np.unique(all_labels)
    weights = compute_class_weight("balanced", classes=classes, y=all_labels)
    class_weights = dict(zip(classes, weights))
    ordered_weights = [class_weights["ham"], class_weights["spam"]]

    train_ds = Dataset.from_pandas(df_train.reset_index(drop=True))
    test_ds  = Dataset.from_pandas(df_test.reset_index(drop=True))

    def map_label(ex):
        ex["label"] = ClassLabels.str2int(ex["label"])
        return ex

    train_ds = train_ds.map(map_label, batched=True).cast_column("label", ClassLabels)
    test_ds  = test_ds.map(map_label, batched=True).cast_column("label", ClassLabels)

    tokenizer = AutoTokenizer.from_pretrained(bert_model, use_fast=True)

    def tokenize(ex):
        return tokenizer(ex["title"], truncation=True)

    train_ds = train_ds.map(tokenize, batched=True).remove_columns(["title"])
    test_ds  = test_ds.map(tokenize, batched=True).remove_columns(["title"])
    gc.collect()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        bert_model, num_labels=2,
        output_attentions=False, output_hidden_states=False,
    )
    model.config.id2label = {0: "ham", 1: "spam"}
    model.config.label2id = {"ham": 0, "spam": 1}

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits  = outputs.get("logits")
            w = torch.tensor(ordered_weights, device=model.device).float()
            loss = torch.nn.CrossEntropyLoss(weight=w)(
                logits.view(-1, 2), labels.view(-1)
            )
            return (loss, outputs) if return_outputs else loss

    def compute_metrics_fn(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds)}

    training_args = TrainingArguments(
        output_dir=str(output_dir),

        num_train_epochs=num_epochs,

        per_device_train_batch_size=train_batch,
        per_device_eval_batch_size=64,

        learning_rate=lr,
        warmup_steps=50,
        weight_decay=0.02,

        save_strategy="epoch",

        save_total_limit=1,
        logging_steps=50,

        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )

    print(f"\n  Base model : {bert_model}")
    print(f"  Epochs     : {num_epochs}")
    print(f"  Batch size : {train_batch}")
    print(f"  Output dir : {output_dir}\n")

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Final eval
    outputs_pred = trainer.predict(test_ds)
    y_true = outputs_pred.label_ids
    y_pred = outputs_pred.predictions.argmax(1)
    print("\n  Final Test Metrics:")
    print(classification_report(y_true, y_pred, target_names=["Ham", "Spam"]))
    print(f"\n  Model saved -> {output_dir}")


# ------------------------------------
# MAIN
# ------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train va save 2 mo hinh spam detection")
    parser.add_argument("--train-csv",   type=str, default="dataset/combined_data.csv")
    parser.add_argument("--test-csv",    type=str, default="dataset/spam.csv")
    parser.add_argument("--skip-bert",   action="store_true", help="Bo qua DistilBERT")
    parser.add_argument("--skip-nb",     action="store_true", help="Bo qua Naive Bayes")
    parser.add_argument("--bert-model",  type=str, default="distilbert-base-cased")
    parser.add_argument("--epochs",      type=int, default=3)
    parser.add_argument("--batch-size",  type=int, default=16)
    parser.add_argument("--lr",          type=float, default=3e-5)
    args = parser.parse_args()

    train_csv = str(ROOT / args.train_csv)
    test_csv  = str(ROOT / args.test_csv)

    # If user left test_csv as the default spam.csv and train_csv is combined_data.csv,
    # perform an 80/20 stratified split on combined_data.csv and write temporary files
    # to use for training and testing.
    temp_train = None
    temp_test = None
    if (Path(train_csv).name == "combined_data.csv") and (Path(test_csv).name == "spam.csv"):
        print("\n[INFO] No explicit test file supplied - splitting combined_data.csv into 80/20 train/test...")
        import pandas as pd
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(train_csv, encoding="latin-1")
        # Normalize columns similarly to load_data
        for col in ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"]:
            if col in df.columns:
                df = df.drop(columns=[col])
        if "v1" in df.columns and "v2" in df.columns:
            df = df.rename(columns={"v1": "label", "v2": "text"})
        if "label" not in df.columns or "text" not in df.columns:
            cols = list(df.columns[:2])
            df = df.rename(columns={cols[0]: "label", cols[1]: "text"})

        def norm(x):
            s = str(x).lower().strip()
            if s in ("ham", "0", "no spam", "no", "legitimate"):
                return 0
            if s in ("spam", "1", "yes"):
                return 1
            try:
                return int(float(x))
            except Exception:
                return 1 if "spam" in s else 0

        df["label"] = df["label"].apply(norm)
        df = df.dropna(subset=["label", "text"]).reset_index(drop=True)

        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
        temp_train = ROOT / "dataset" / "combined_train_split.csv"
        temp_test = ROOT / "dataset" / "combined_test_split.csv"
        train_df.to_csv(temp_train, index=False, encoding="latin-1")
        test_df.to_csv(temp_test, index=False, encoding="latin-1")
        print(f"  Wrote split files: {temp_train} ({len(train_df)} rows), {temp_test} ({len(test_df)} rows)")

        train_csv = str(temp_train)
        test_csv = str(temp_test)

    print("="*50)
    print("  SPAM DETECTION - TRAIN & SAVE")
    print("="*50)
    print(f"  Train CSV : {train_csv}")
    print(f"  Test CSV  : {test_csv}")

    if not Path(train_csv).exists():
        print(f"\n[ERROR] Khong tim thay: {train_csv}")
        sys.exit(1)
    if not Path(test_csv).exists():
        print(f"\n[ERROR] Khong tim thay: {test_csv}")
        sys.exit(1)

    if not args.skip_nb:
        try:
            train_naive_bayes(train_csv, test_csv)
        except Exception as e:
            print(f"\n[ERROR] Naive Bayes: {e}")
    else:
        print("\n  [1/2] Bo qua Naive Bayes (--skip-nb)")

    if not args.skip_bert:
        try:
            train_distilbert(train_csv, test_csv,
                             bert_model=args.bert_model,
                             num_epochs=args.epochs,
                             train_batch=args.batch_size,
                             lr=args.lr)
        except Exception as e:
            print(f"\n[ERROR] DistilBERT: {e}")
    else:
        print("\n  [2/2] Bo qua DistilBERT (--skip-bert)")

    print("\n" + "="*50)
    print("  HOAN THANH TRAIN!")
    print("  Chay tiep: python visualizer.py")
    print("="*50)


if __name__ == "__main__":
    main()