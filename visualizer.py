"""
visualizer.py
=============
Chay 2 mo hinh spam detection, so sanh va xuat bieu do vao outputs/.

Yeu cau truoc khi chay:
    python train_and_save.py          # train ca 2
    python train_and_save.py --skip-bert   # chi Naive Bayes

Sau do:
    python visualizer.py
    python visualizer.py --skip-bert  # neu chua co DistilBERT
"""

import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
)

warnings.filterwarnings("ignore")

ROOT         = Path(__file__).resolve().parent
METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
PALETTE      = {
    "nb":   "#4C72B0",
    "bert": "#DD8452",
    "ham":  "#55A868",
    "spam": "#C44E52",
    "bg":   "#F8F9FA",
    "grid": "#E0E0E0",
}


# ============================================================
# DATA HELPERS
# ============================================================

def load_test_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="latin-1")
    for col in ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    if "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "text"})
    if "label" not in df.columns or "text" not in df.columns:
        c = list(df.columns[:2])
        df = df.rename(columns={c[0]: "label", c[1]: "text"})

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
    df = df.dropna(subset=["label", "text"])
    return df[["label", "text"]]


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    m = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1-Score":  f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            m["ROC-AUC"] = roc_auc_score(y_true, y_prob)
        except Exception:
            m["ROC-AUC"] = float("nan")
    else:
        m["ROC-AUC"] = float("nan")
    return m


# ============================================================
# MODEL RUNNERS
# ============================================================

def run_naive_bayes(texts: pd.Series, nb_dir: Path):
    """Load Naive Bayes tu file .pkl. Tim trong nb_dir va ROOT."""
    print("\n[Naive Bayes] Dang tai mo hinh...")
    import joblib

    model_path, vec_path = None, None
    for d in [nb_dir, ROOT]:
        if model_path is None and (d / "naive_bayes_model.pkl").exists():
            model_path = d / "naive_bayes_model.pkl"
        if vec_path is None and (d / "count_vectorizer.pkl").exists():
            vec_path = d / "count_vectorizer.pkl"

    if model_path is None or vec_path is None:
        missing = []
        if not model_path: missing.append("naive_bayes_model.pkl")
        if not vec_path:   missing.append("count_vectorizer.pkl")
        raise FileNotFoundError(
            "Khong tim thay: " + str(missing) + "\n"
            "  -> Hay chay: python train_and_save.py --skip-bert"
        )

    print(f"  Model      : {model_path}")
    print(f"  Vectorizer : {vec_path}")
    model      = joblib.load(str(model_path))
    vectorizer = joblib.load(str(vec_path))
    X     = vectorizer.transform(texts.tolist())
    preds = np.array(model.predict(X))
    try:
        proba = model.predict_proba(X)[:, 1]
    except Exception:
        proba = None
    print(f"  Done. Ham: {(preds==0).sum():,} | Spam: {(preds==1).sum():,}")
    return preds, proba


def find_bert_model_dir(bert_dir: Path) -> Path:
    """Tim thu muc DistilBERT da train (co config.json ben trong)."""
    MODEL_NAME = "email-spam-detection-distilbert"
    candidates = [
        bert_dir / MODEL_NAME,
        ROOT / MODEL_NAME,
        bert_dir,
    ]
    for search_root in [bert_dir, ROOT]:
        try:
            for child in sorted(search_root.iterdir()):
                if child.is_dir() and (child / "config.json").exists():
                    if child not in candidates:
                        candidates.append(child)
        except Exception:
            pass
    for c in candidates:
        if c.exists() and (c / "config.json").exists():
            return c
    raise FileNotFoundError(
        "Khong tim thay thu muc model DistilBERT (can co config.json).\n"
        "  Da tim: " + str([str(c) for c in candidates]) + "\n"
        "  -> Hay chay: python train_and_save.py"
    )


def run_distilbert(texts: pd.Series, bert_dir: Path):
    """Load DistilBERT da train va chay inference theo batch."""
    print("\n[DistilBERT] Dang tai mo hinh...")
    model_dir = find_bert_model_dir(bert_dir)
    print(f"  Model dir : {model_dir}")
    from transformers import pipeline as hf_pipeline
    classifier = hf_pipeline(
        "text-classification",
        model=str(model_dir),
        tokenizer=str(model_dir),
        device = 0,
    )
    text_list = texts.tolist()
    BATCH = 64
    all_preds, all_proba = [], []
    for i in range(0, len(text_list), BATCH):
        batch = text_list[i: i + BATCH]
        try:
            outs = classifier(batch, truncation=True, top_k=None)
        except TypeError:
            outs = classifier(batch, truncation=True, return_all_scores=True)
        for out in outs:
            best_label, best_score, spam_score = "", 0.0, 0.0
            for item in out:
                lbl = str(item["label"]).lower()
                scr = float(item["score"])
                if scr > best_score:
                    best_score = scr
                    best_label = lbl
                if "spam" in lbl or lbl in ("label_1", "1"):
                    spam_score = scr
            is_spam = int("spam" in best_label or best_label in ("label_1", "1"))
            all_preds.append(is_spam)
            all_proba.append(spam_score)
        done = min(i + BATCH, len(text_list))
        print(f"  Progress: {done}/{len(text_list)}", end="\r")
    preds = np.array(all_preds)
    proba = np.array(all_proba) if any(s > 0 for s in all_proba) else None
    print(f"  Done. Ham: {(preds==0).sum():,} | Spam: {(preds==1).sum():,}    ")
    return preds, proba


# ============================================================
# CHART HELPERS
# ============================================================

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PALETTE["bg"])
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:  ax.set_title(title,   fontsize=13, fontweight="bold", pad=10)
    if xlabel: ax.set_xlabel(xlabel, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10)


def save_fig(fig, output_dir: Path, filename: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    p = output_dir / filename
    fig.savefig(str(p), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {filename}")


# ============================================================
# CHARTS
# ============================================================

def chart_email_distribution(y_true, nb_p, bert_p, output_dir: Path):
    def cnt(a):
        a = np.array(a)
        return int((a == 0).sum()), int((a == 1).sum())
    gh, gs = cnt(y_true)
    nh, ns = cnt(nb_p)
    bh, bs = cnt(bert_p)
    groups = ["Ground Truth", "Naive Bayes", "DistilBERT"]
    hams   = [gh, nh, bh]
    spams  = [gs, ns, bs]
    x, w   = np.arange(3), 0.35
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")
    b1 = ax.bar(x - w/2, hams,  w, label="Ham (Non-Spam)", color=PALETTE["ham"],  zorder=3)
    b2 = ax.bar(x + w/2, spams, w, label="Spam",           color=PALETTE["spam"], zorder=3)
    offset = max(hams + spams) * 0.012
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + offset,
                f"{h:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    style_ax(ax,
             title="Phan phoi Ham/Spam: Dataset vs. Du doan 2 mo hinh",
             xlabel="Nguon", ylabel="So email")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylim(0, max(max(hams), max(spams)) * 1.2)
    ax.legend(fontsize=10)
    save_fig(fig, output_dir, "01_email_distribution.png")


def chart_metrics_bar(nb_m: dict, bert_m: dict, output_dir: Path):
    nb_v   = [nb_m.get(m, float("nan"))   for m in METRIC_NAMES]
    bert_v = [bert_m.get(m, float("nan")) for m in METRIC_NAMES]
    x, w   = np.arange(len(METRIC_NAMES)), 0.32
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")
    b1 = ax.bar(x - w/2, nb_v,   w, label="Naive Bayes", color=PALETTE["nb"],   zorder=3)
    b2 = ax.bar(x + w/2, bert_v, w, label="DistilBERT",  color=PALETTE["bert"], zorder=3)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.006,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    style_ax(ax,
             title="So sanh chi so danh gia: Naive Bayes vs. DistilBERT",
             xlabel="Chi so", ylabel="Gia tri (0-1)")
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_NAMES, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    save_fig(fig, output_dir, "02_metrics_comparison.png")


def chart_radar(nb_m: dict, bert_m: dict, output_dir: Path):
    valid = [m for m in METRIC_NAMES
             if not np.isnan(nb_m.get(m, float("nan")))
             and not np.isnan(bert_m.get(m, float("nan")))]
    if len(valid) < 3:
        print("  Skip radar chart (qua it chi so hop le)")
        return
    nb_v   = [nb_m[m]   for m in valid] + [nb_m[valid[0]]]
    bert_v = [bert_m[m] for m in valid] + [bert_m[valid[0]]]
    N      = len(valid)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")
    ax.plot(angles, nb_v,   "o-", lw=2, color=PALETTE["nb"],   label="Naive Bayes")
    ax.fill(angles, nb_v,   alpha=0.18, color=PALETTE["nb"])
    ax.plot(angles, bert_v, "s-", lw=2, color=PALETTE["bert"], label="DistilBERT")
    ax.fill(angles, bert_v, alpha=0.18, color=PALETTE["bert"])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(valid, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=7, color="gray")
    ax.set_title("Radar Chart - So sanh tong the 2 mo hinh",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)
    ax.grid(color=PALETTE["grid"])
    save_fig(fig, output_dir, "03_radar_comparison.png")


def chart_confusion_matrices(y_true, nb_p, bert_p, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Confusion Matrix (chuan hoa theo hang)",
                 fontsize=14, fontweight="bold", y=1.02)
    clnames = ["Ham", "Spam"]
    pairs = [
        (nb_p,  "Naive Bayes", axes[0], plt.cm.Blues),
        (bert_p, "DistilBERT", axes[1], plt.cm.Oranges),
    ]
    for preds, title, ax, cmap in pairs:
        cm = confusion_matrix(y_true, preds, normalize="true")
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(clnames, fontsize=11)
        ax.set_yticklabels(clnames, fontsize=11)
        ax.set_xlabel("Du doan", fontsize=10)
        ax.set_ylabel("Thuc te",  fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]:.3f}", ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="white" if cm[i,j] > 0.5 else "black")
    plt.tight_layout()
    save_fig(fig, output_dir, "04_confusion_matrices.png")


def chart_roc(y_true, nb_prob, bert_prob, output_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("white")
    plotted = False
    for prob, label, color in [
        (nb_prob,   "Naive Bayes", PALETTE["nb"]),
        (bert_prob, "DistilBERT",  PALETTE["bert"]),
    ]:
        if prob is not None:
            try:
                fpr, tpr, _ = roc_curve(y_true, prob)
                auc = roc_auc_score(y_true, prob)
                ax.plot(fpr, tpr, lw=2, color=color,
                        label=f"{label}  (AUC={auc:.4f})")
                plotted = True
            except Exception as e:
                print(f"  Skip ROC {label}: {e}")
    if not plotted:
        plt.close(fig)
        return
    ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.5,label="Random (AUC=0.5)")
    style_ax(ax, title="Duong cong ROC - So sanh 2 mo hinh",
             xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
    ax.grid(color=PALETTE["grid"])
    ax.legend(fontsize=10)
    save_fig(fig, output_dir, "05_roc_curves.png")


def chart_summary_table(nb_m: dict, bert_m: dict, output_dir: Path):
    rows = []
    for m in METRIC_NAMES:
        nv = nb_m.get(m, float("nan"))
        bv = bert_m.get(m, float("nan"))
        if np.isnan(nv) and np.isnan(bv): better = "-"
        elif np.isnan(nv):  better = "DistilBERT"
        elif np.isnan(bv):  better = "Naive Bayes"
        elif abs(nv-bv) < 1e-6: better = "Tie"
        else: better = "Naive Bayes" if nv > bv else "DistilBERT"
        rows.append([
            m,
            f"{nv:.4f}" if not np.isnan(nv) else "N/A",
            f"{bv:.4f}" if not np.isnan(bv) else "N/A",
            better,
        ])
    fig, ax = plt.subplots(figsize=(9, 3.5))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    headers = ["Chi so", "Naive Bayes", "DistilBERT", "Mo hinh tot hon"]
    tbl = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.9)
    for j in range(4):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i, (_, _, _, better) in enumerate(rows, start=1):
        for j in range(4):
            tbl[i, j].set_facecolor("#EAF2F8" if i % 2 == 0 else "white")
        if better == "Naive Bayes":
            tbl[i, 1].set_facecolor("#D5F5E3")
            tbl[i, 3].set_text_props(color=PALETTE["nb"], fontweight="bold")
        elif better == "DistilBERT":
            tbl[i, 2].set_facecolor("#D5F5E3")
            tbl[i, 3].set_text_props(color=PALETTE["bert"], fontweight="bold")
    ax.set_title("Bang tong hop ket qua - Naive Bayes vs. DistilBERT",
                 fontsize=13, fontweight="bold", pad=20)
    save_fig(fig, output_dir, "06_summary_table.png")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv",   default=None,
                        help="Path to test CSV. If omitted, uses dataset/test_2000.csv when present, otherwise dataset/spam.csv")
    parser.add_argument("--nb-dir",     default="naive_bayes")
    parser.add_argument("--bert-dir",   default="distilbert")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--skip-bert",  action="store_true",
                        help="Bo qua DistilBERT")
    args = parser.parse_args()

    # Determine test CSV: prefer explicit --test-csv; if not provided use dataset/test_2000.csv when available,
    # otherwise fall back to dataset/spam.csv (legacy behavior)
    if args.test_csv:
        test_csv = ROOT / args.test_csv
    else:
        candidate = ROOT / "dataset" / "test_2000.csv"
        if candidate.exists():
            test_csv = candidate
        else:
            test_csv = ROOT / "dataset" / "spam.csv"
    nb_dir     = ROOT / args.nb_dir
    bert_dir   = ROOT / args.bert_dir
    output_dir = ROOT / args.output_dir

    print("=" * 55)
    print("  EMAIL SPAM - VISUALIZER & MODEL COMPARISON")
    print("=" * 55)
    print(f"  Test CSV   : {test_csv}")
    print(f"  Output dir : {output_dir}")
    print("=" * 55)

    # 1. Load data
    print("\n[1/4] Load test dataset...")
    if not test_csv.exists():
        print(f"  ERROR: Khong tim thay {test_csv}")
        sys.exit(1)
    # If the user did NOT provide --test-csv and combined_data.csv exists, perform an 80/20 split
    combined_path = ROOT / "dataset" / "combined_data.csv"
    # Note: if the user explicitly provided --test-csv we will always use it and not auto-split
    if getattr(args, 'test_csv', None) is None and Path(test_csv).name == "spam.csv" and combined_path.exists():
        print("  Found combined_data.csv - performing 80/20 stratified split and using the 20% test split for evaluation")
        import pandas as pd
        from sklearn.model_selection import train_test_split

        df_comb = pd.read_csv(combined_path, encoding="latin-1")
        for col in ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"]:
            if col in df_comb.columns:
                df_comb = df_comb.drop(columns=[col])
        if "v1" in df_comb.columns and "v2" in df_comb.columns:
            df_comb = df_comb.rename(columns={"v1": "label", "v2": "text"})
        if "label" not in df_comb.columns or "text" not in df_comb.columns:
            cols = list(df_comb.columns[:2])
            df_comb = df_comb.rename(columns={cols[0]: "label", cols[1]: "text"})

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

        df_comb["label"] = df_comb["label"].apply(norm)
        df_comb = df_comb.dropna(subset=["label", "text"]).reset_index(drop=True)
        _, df_test = train_test_split(df_comb, test_size=0.2, stratify=df_comb["label"], random_state=42)
    else:
        df_test = load_test_data(str(test_csv))
    y_true  = df_test["label"].values
    texts   = df_test["text"]
    print(f"  Loaded {len(df_test):,} emails | Ham: {(y_true==0).sum():,} | Spam: {(y_true==1).sum():,}")

    # 2. Inference
    print("\n[2/4] Chay suy luan...")
    nb_preds, nb_prob     = None, None
    bert_preds, bert_prob = None, None

    try:
        nb_preds, nb_prob = run_naive_bayes(texts, nb_dir)
    except Exception as e:
        print(f"  ERROR Naive Bayes: {e}")

    if not args.skip_bert:
        try:
            bert_preds, bert_prob = run_distilbert(texts, bert_dir)
        except Exception as e:
            print(f"  ERROR DistilBERT: {e}")
    else:
        print("\n  [DistilBERT] Bo qua (--skip-bert)")

    if nb_preds is None and bert_preds is None:
        print("\nERROR: Khong chay duoc mo hinh nao.")
        print("  -> Hay chay: python train_and_save.py  truoc")
        sys.exit(1)

    # If we didn't get DistilBERT predictions but the model directory exists, try to run inference now
    if bert_preds is None and not args.skip_bert:
        try:
            # This will raise FileNotFoundError if no model dir exists
            _ = find_bert_model_dir(bert_dir)
            print('\n[DistilBERT] Tim thay thu muc model nhung chua co du doan; dang chay inference de tinh chi so...')
            try:
                bert_preds, bert_prob = run_distilbert(texts, bert_dir)
            except Exception as e:
                print(f"  ERROR khi chay DistilBERT inference: {e}")
        except FileNotFoundError:
            # no model present; nothing to do
            pass

    # 3. Metrics
    print("\n[3/4] Tinh chi so...")
    nb_m, bert_m = {}, {}
    if nb_preds is not None:
        nb_m = compute_metrics(y_true, nb_preds, nb_prob)
        print("\n  [Naive Bayes]")
        for k, v in nb_m.items():
            print(f"    {k:12s}: {v:.4f}" if not np.isnan(v) else f"    {k:12s}: N/A")
        print(classification_report(y_true, nb_preds, target_names=["Ham", "Spam"]))
    if bert_preds is not None:
        # Try to load saved DistilBERT metrics from the model folder or outputs/metrics.json
        bert_m = None
        try:
            model_dir = find_bert_model_dir(bert_dir)
            cand_files = [
                model_dir / "metrics.json",
                ROOT / "outputs" / "metrics.json",
                ROOT / "metrics.json",
            ]
            import json
            loaded = None
            for cf in cand_files:
                try:
                    if cf.exists():
                        with open(cf, "r", encoding="utf-8") as fh:
                            loaded = json.load(fh)
                        break
                except Exception:
                    loaded = None
            # If loaded, try to extract distilbert/bert metrics, normalize keys
            if isinstance(loaded, dict):
                # possible keys: 'distilbert', 'DistilBERT', 'bert', or top-level metrics
                candidate_keys = [k for k in loaded.keys()]
                found = None
                for key in ("distilbert", "DistilBERT", "bert", "Distil-BERT", "email-spam-detection-distilbert"):
                    if key in loaded:
                        found = loaded[key]
                        break
                if found is None and all(k.lower() in ("accuracy","precision","recall","f1","f1-score","roc-auc","roc_auc","roc-auc") for k in loaded.keys()):
                    found = loaded

                def map_metrics(d):
                    # normalize various key formats to METRIC_NAMES
                    mapping = {}
                    for k, v in d.items():
                        kk = str(k).lower()
                        if kk in ("accuracy",): mapping["Accuracy"] = float(v)
                        if kk in ("precision",): mapping["Precision"] = float(v)
                        if kk in ("recall",): mapping["Recall"] = float(v)
                        if kk in ("f1", "f1_score", "f1-score", "f1_score_macro", "f1-score-macro"): mapping["F1-Score"] = float(v)
                        if kk in ("roc_auc", "roc-auc", "roc_auc_score", "roc-auc"): mapping["ROC-AUC"] = float(v)
                    return mapping

                if isinstance(found, dict):
                    bert_m = map_metrics(found)
        except Exception:
            bert_m = None

        # Fallback to computing metrics from predictions if no saved metrics found
        if not bert_m:
            bert_m = compute_metrics(y_true, bert_preds, bert_prob)
        print("\n  [DistilBERT]")
        for k, v in bert_m.items():
            print(f"    {k:12s}: {v:.4f}" if not np.isnan(v) else f"    {k:12s}: N/A")
        print(classification_report(y_true, bert_preds, target_names=["Ham", "Spam"]))

    # 4. Charts
    print("\n[4/4] Xuat bieu do...")
    _nb   = nb_preds   if nb_preds   is not None else np.zeros_like(y_true)
    _bert = bert_preds if bert_preds is not None else np.zeros_like(y_true)
    _nb_m   = nb_m   if nb_m   else {m: float("nan") for m in METRIC_NAMES}
    _bert_m = bert_m if bert_m else {m: float("nan") for m in METRIC_NAMES}

    chart_email_distribution(y_true, _nb, _bert, output_dir)
    chart_metrics_bar(_nb_m, _bert_m, output_dir)
    chart_radar(_nb_m, _bert_m, output_dir)
    chart_confusion_matrices(y_true, _nb, _bert, output_dir)
    chart_roc(y_true, nb_prob, bert_prob, output_dir)
    chart_summary_table(_nb_m, _bert_m, output_dir)

    # Save combined metrics JSON for record / downstream use
    try:
        import json
        output_dir.mkdir(parents=True, exist_ok=True)
        combined = {"naive_bayes": nb_m, "distilbert": bert_m}
        with open(output_dir / "metrics.json", "w", encoding="utf-8") as fh:
            json.dump(combined, fh, indent=2, ensure_ascii=False)
        print("\n" + "=" * 55)
        print(f"  XONG! Bieu do luu tai: {output_dir}/")
        print("=" * 55)
        for f in sorted(output_dir.glob("*.png")):
            print(f"    {f.name}")
        print(f"  Metrics JSON -> {output_dir / 'metrics.json'}")
    except Exception as e:
        print(f"  Warning: khong luu duoc metrics.json: {e}")


if __name__ == "__main__":
    main()