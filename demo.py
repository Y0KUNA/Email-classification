#!/usr/bin/env python3
"""demo.py
Run both Naive Bayes and DistilBERT models (if present) on a dataset and collect
metrics: Precision, Recall, F1, latency/throughput/memory usage, and try to extract
training loss / validation accuracy from saved trainer artifacts if available.

Usage:
    python demo.py
    python demo.py --dataset dataset/combined_data.csv --nb-dir naive_bayes --bert-dir distilbert/email-spam-detection-distilbert
"""

from pathlib import Path
import time
import json
import os
import statistics
from typing import List, Tuple, Optional, Dict, Any

DEFAULT_DATASET = Path("dataset/combined_data.csv")
ROOT = Path(__file__).resolve().parent


def preprocess_text(text: str) -> str:
    import re
    text = str(text).lower()
    text = re.sub(r"\d+", "escapenumber", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(path: Path) -> Tuple[List[str], List[int]]:
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


def mem_mb(psutil_mod=None) -> Optional[float]:
    try:
        if psutil_mod is not None:
            p = psutil_mod.Process(os.getpid())
            return p.memory_info().rss / (1024 ** 2)
        else:
            return None
    except Exception:
        return None


def run_naive_bayes(texts: List[str], labels: List[int], nb_dir: Path) -> Dict[str, Any]:
    results = {"model": "naive_bayes"}
    try:
        import joblib
        from sklearn.metrics import precision_score, recall_score, f1_score
    except Exception as e:
        results["error"] = f"Missing dependency: {e}"
        return results

    model_path = nb_dir / "naive_bayes_model.pkl"
    vec_path = nb_dir / "count_vectorizer.pkl"
    if not model_path.exists() or not vec_path.exists():
        results["error"] = f"Missing NB artifacts in {nb_dir}"
        return results

    psutil_mod = try_import_psutil()
    mem_before = mem_mb(psutil_mod)

    t0 = time.time()
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    load_time = time.time() - t0
    mem_after_load = mem_mb(psutil_mod)

    # Predict in one batch (for sklearn it's fine) but measure time
    X = vectorizer.transform(texts)
    t0 = time.time()
    preds = model.predict(X)
    total_pred_time = time.time() - t0
    per_sample = total_pred_time / max(1, len(texts))

    # probabilities if available
    try:
        proba = model.predict_proba(X)[:, 1]
    except Exception:
        proba = None

    precision = float(precision_score(labels, preds, zero_division=0))
    recall = float(recall_score(labels, preds, zero_division=0))
    f1 = float(f1_score(labels, preds, zero_division=0))

    results.update({
        "load_time_s": load_time,
        "mem_before_mb": mem_before,
        "mem_after_load_mb": mem_after_load,
        "total_prediction_time_s": total_pred_time,
        "per_sample_latency_s": per_sample,
        "throughput_samples_per_s": len(texts) / max(1e-9, total_pred_time),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_samples": len(texts),
    })

    # Try to read training metrics if present in outputs/metrics.json
    try:
        metrics_file = ROOT / "outputs" / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r", encoding="utf-8") as fh:
                m = json.load(fh)
            if "naive_bayes" in m:
                results["saved_metrics"] = m["naive_bayes"]
    except Exception:
        pass

    return results


def run_distilbert(texts: List[str], labels: List[int], bert_dir: Path, batch_size: int = 32) -> Dict[str, Any]:
    results = {"model": "distilbert"}
    try:
        from transformers import pipeline
        import torch
        from sklearn.metrics import precision_score, recall_score, f1_score
    except Exception as e:
        results["error"] = f"Missing dependency: {e}"
        return results

    if not bert_dir.exists() or not (bert_dir / "config.json").exists():
        results["error"] = f"DistilBERT model not found in {bert_dir}"
        return results

    psutil_mod = try_import_psutil()
    mem_before = mem_mb(psutil_mod)

    # choose device automatically
    device = 0 if torch.cuda.is_available() else -1
    t0 = time.time()
    classifier = pipeline("text-classification", model=str(bert_dir), tokenizer=str(bert_dir), device=device)
    load_time = time.time() - t0
    mem_after_load = mem_mb(psutil_mod)

    all_preds = []
    all_scores = []
    total_pred_time = 0.0
    # batch predict
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        t0 = time.time()
        try:
            outs = classifier(batch, truncation=True)
        except TypeError:
            # older transformers may require return_all_scores
            outs = classifier(batch, truncation=True, return_all_scores=True)
        dt = time.time() - t0
        total_pred_time += dt
        # outs could be list of lists (scores) or list of dicts
        for out in outs:
            # normalize to best label + spam score
            if isinstance(out, list):
                # list of scores
                best = max(out, key=lambda x: x.get("score", 0))
                lbl = str(best.get("label", "")).lower()
                score = float(best.get("score", 0.0))
                spam_score = 0.0
                for item in out:
                    if "spam" in str(item.get("label", "")).lower() or str(item.get("label", "")).lower() in ("label_1", "1"):
                        spam_score = float(item.get("score", 0.0))
                is_spam = 1 if ("spam" in lbl or lbl in ("label_1", "1")) else 0
            else:
                lbl = str(out.get("label", "")).lower()
                score = float(out.get("score", 0.0))
                spam_score = score if ("spam" in lbl or lbl in ("label_1", "1")) else 0.0
                is_spam = 1 if ("spam" in lbl or lbl in ("label_1", "1")) else 0
            all_preds.append(is_spam)
            all_scores.append(spam_score)

    per_sample = total_pred_time / max(1, len(texts))

    precision = float(precision_score(labels, all_preds, zero_division=0))
    recall = float(recall_score(labels, all_preds, zero_division=0))
    f1 = float(f1_score(labels, all_preds, zero_division=0))

    results.update({
        "device": "cuda" if device == 0 else "cpu",
        "load_time_s": load_time,
        "mem_before_mb": mem_before,
        "mem_after_load_mb": mem_after_load,
        "total_prediction_time_s": total_pred_time,
        "per_sample_latency_s": per_sample,
        "throughput_samples_per_s": len(texts) / max(1e-9, total_pred_time),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_samples": len(texts),
    })

    # Try to extract training/validation stats from trainer_state.json or outputs/metrics.json
    try:
        # Check trainer_state.json in model dir
        ts = bert_dir / "trainer_state.json"
        if ts.exists():
            with open(ts, "r", encoding="utf-8") as fh:
                st = json.load(fh)
            # st may contain log_history
            logs = st.get("log_history", [])
            # find last training loss and eval accuracy entries
            train_losses = [x.get("loss") for x in logs if x.get("loss") is not None]
            eval_accs = [x.get("eval_accuracy") or x.get("eval_acc") for x in logs if (x.get("eval_accuracy") or x.get("eval_acc")) is not None]
            if train_losses:
                results["training_loss_from_trainer_state_last"] = train_losses[-1]
            if eval_accs:
                results["validation_accuracy_from_trainer_state_last"] = eval_accs[-1]
        # fallback to outputs/metrics.json
        metrics_file = ROOT / "outputs" / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r", encoding="utf-8") as fh:
                m = json.load(fh)
            if "distilbert" in m and isinstance(m["distilbert"], dict) and m["distilbert"]:
                results.setdefault("saved_metrics", {}).update(m["distilbert"])
    except Exception:
        pass

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--nb-dir", default=str(ROOT / "naive_bayes"))
    parser.add_argument("--bert-dir", default=str(ROOT / "distilbert" / "email-spam-detection-distilbert"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default=str(ROOT / "outputs" / "demo_metrics.json"))
    args = parser.parse_args()

    dataset = Path(args.dataset)
    nb_dir = Path(args.nb_dir)
    bert_dir = Path(args.bert_dir)
    out_file = Path(args.output)

    if not dataset.exists():
        print(f"Dataset not found: {dataset}")
        return

    texts, labels = load_dataset(dataset)
    print(f"Loaded {len(texts)} samples from {dataset}")

    report = {"dataset": str(dataset), "n_samples": len(texts), "results": {}}

    print("\nRunning Naive Bayes evaluation...")
    nb_res = run_naive_bayes(texts, labels, nb_dir)
    report["results"]["naive_bayes"] = nb_res

    print("\nRunning DistilBERT evaluation...")
    bert_res = run_distilbert(texts, labels, bert_dir, batch_size=args.batch_size)
    report["results"]["distilbert"] = bert_res

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"\nSaved report -> {out_file}")


if __name__ == "__main__":
    main()
