#!/usr/bin/env python3
"""Evaluate DistilBERT checkpoints across epochs.

This script looks for checkpoint directories under the given model base dir (e.g.
distilbert/email-spam-detection-distilbert/checkpoint-1, checkpoint-2, ...), or
falls back to scanning for directories containing 'checkpoint' and sorts them.

For each checkpoint requested (by epoch numbers), it loads a pipeline from the
checkpoint, runs batched inference on the provided dataset, and records:
  - accuracy, precision, recall, f1
  - load time, total prediction time, per-sample latency, throughput
  - optional memory usage (requires psutil)

Output: JSON file with per-epoch metrics.
"""

from pathlib import Path
import time
import json
import os
from typing import List, Dict, Any


def preprocess_text(text: str) -> str:
    import re
    text = str(text).lower()
    text = re.sub(r"\d+", "escapenumber", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(path: Path):
    import pandas as pd
    df = pd.read_csv(path, encoding="latin-1")
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
    texts = df["text"].astype(str).apply(preprocess_text).tolist()
    labels = df["label"].astype(int).tolist()
    return texts, labels


def try_import_psutil():
    try:
        import psutil
        return psutil
    except Exception:
        return None


def mem_mb(psutil_mod=None):
    try:
        if psutil_mod is None:
            return None
        p = psutil_mod.Process(os.getpid())
        return p.memory_info().rss / (1024 ** 2)
    except Exception:
        return None


def find_checkpoints(model_base: Path, epochs: List[int]) -> Dict[int, Path]:
    # First try explicit checkpoint-{epoch} names
    found = {}
    for e in epochs:
        p = model_base / f"checkpoint-{e}"
        if p.exists():
            found[e] = p

    if len(found) == len(epochs):
        return found

    # Fallback: scan for any 'checkpoint' subdirs and sort by name/mtime
    candidates = [d for d in model_base.iterdir() if d.is_dir() and "checkpoint" in d.name]
    if not candidates:
        return found
    # sort by numeric suffix if present else by mtime
    def key_fn(d: Path):
        name = d.name
        import re
        m = re.search(r"checkpoint-?(\d+)", name)
        if m:
            return int(m.group(1))
        return int(d.stat().st_mtime)

    candidates = sorted(candidates, key=key_fn)
    # assign in order to epochs not found
    missing_epochs = [e for e in epochs if e not in found]
    for e, c in zip(missing_epochs, candidates):
        found[e] = c
    return found


def eval_checkpoint(checkpoint_dir: Path, texts: List[str], labels: List[int], batch_size: int, device: int):
    # device: 0 for cuda, -1 for cpu
    from transformers import pipeline
    import torch
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    psutil_mod = try_import_psutil()
    mem_before = mem_mb(psutil_mod)

    t0 = time.time()
    from transformers.utils import logging as hf_logging
    try:
        classifier = pipeline("text-classification", model=str(checkpoint_dir), tokenizer=str(checkpoint_dir), device=device)
    except OSError as e:
        # Some checkpoints contain only model weights but not tokenizer files.
        # Retry using tokenizer from parent directory (likely the model base dir).
        parent = checkpoint_dir.parent
        hf_logging.get_logger("transformers").warning(
            f"Tokenizer missing in checkpoint {checkpoint_dir}, falling back to tokenizer from {parent}: {e}"
        )
        classifier = pipeline("text-classification", model=str(checkpoint_dir), tokenizer=str(parent), device=device)
    load_time = time.time() - t0
    mem_after_load = mem_mb(psutil_mod)

    all_preds = []
    total_pred_time = 0.0
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        t0 = time.time()
        outs = classifier(batch, truncation=True)
        dt = time.time() - t0
        total_pred_time += dt
        for out in outs:
            if isinstance(out, list):
                best = max(out, key=lambda x: x.get("score", 0))
                lbl = str(best.get("label", "")).lower()
            else:
                lbl = str(out.get("label", "")).lower()
            is_spam = 1 if ("spam" in lbl or lbl in ("label_1", "1")) else 0
            all_preds.append(is_spam)

    per_sample = total_pred_time / max(1, len(texts))

    accuracy = float(accuracy_score(labels, all_preds))
    precision = float(precision_score(labels, all_preds, zero_division=0))
    recall = float(recall_score(labels, all_preds, zero_division=0))
    f1 = float(f1_score(labels, all_preds, zero_division=0))

    return {
        "checkpoint": str(checkpoint_dir),
        "device": "cuda" if device == 0 else "cpu",
        "load_time_s": load_time,
        "mem_before_mb": mem_before,
        "mem_after_load_mb": mem_after_load,
        "total_prediction_time_s": total_pred_time,
        "per_sample_latency_s": per_sample,
        "throughput_samples_per_s": len(texts) / max(1e-9, total_pred_time),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_samples": len(texts),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", default=str(Path("distilbert") / "email-spam-detection-distilbert"))
    parser.add_argument("--dataset", default=str(Path("dataset") / "combined_data.csv"))
    parser.add_argument("--epochs", default="1,2,3,4", help="Comma separated epoch numbers to evaluate")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default=str(Path("outputs") / "epoch_eval_distilbert.json"))
    args = parser.parse_args()

    model_base = Path(args.model_base)
    dataset = Path(args.dataset)
    epochs = [int(x) for x in args.epochs.split(",") if x.strip()]
    batch_size = args.batch_size
    out_file = Path(args.output)

    if not dataset.exists():
        print(f"Dataset not found: {dataset}")
        return
    if not model_base.exists():
        print(f"Model base not found: {model_base}")
        return

    texts, labels = load_dataset(dataset)

    # Determine device
    device = 0 if __import__("torch").cuda.is_available() else -1

    checkpoints = find_checkpoints(model_base, epochs)
    if not checkpoints:
        print("No checkpoint directories found under model base.")
        return

    results = {"model_base": str(model_base), "dataset": str(dataset), "epochs": {}}
    for e in epochs:
        ck = checkpoints.get(e)
        if not ck:
            results["epochs"][str(e)] = {"error": "checkpoint not found"}
            continue
        print(f"Evaluating epoch {e} -> {ck}")
        res = eval_checkpoint(ck, texts, labels, batch_size, device)
        results["epochs"][str(e)] = res

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print(f"Saved epoch evaluation -> {out_file}")


if __name__ == "__main__":
    main()
