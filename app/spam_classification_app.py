#!/usr/bin/env python3
"""
Simple desktop app to classify an email as spam/ham using either Naive Bayes or DistilBERT.

Run:
    python app/spam_classification_app.py

The app will look for Naive Bayes artifacts in ../naive_bayes/ and a DistilBERT model
under ../distilbert/email-spam-detection-distilbert/ or ../distilbert/.
"""
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext


ROOT = Path(__file__).resolve().parent.parent


class ModelLoader:
    def __init__(self):
        self.nb_model = None
        self.nb_vec = None
        self.bert_pipe = None

    def load_naive_bayes(self):
        if self.nb_model is not None and self.nb_vec is not None:
            return
        try:
            import joblib
            nb_dir = ROOT / 'naive_bayes'
            model_path = nb_dir / 'naive_bayes_model.pkl'
            vec_path = nb_dir / 'count_vectorizer.pkl'
            if not model_path.exists() or not vec_path.exists():
                raise FileNotFoundError(f"Naive Bayes artifacts not found in {nb_dir}")
            self.nb_model = joblib.load(str(model_path))
            self.nb_vec = joblib.load(str(vec_path))
        except Exception:
            self.nb_model = None
            self.nb_vec = None
            raise

    def find_bert_model_dir(self):
        candidates = [ROOT / 'distilbert' / 'email-spam-detection-distilbert', ROOT / 'distilbert']
        for c in candidates:
            if c.exists() and (c / 'config.json').exists():
                return c
        # fallback: any child dir under distilbert that contains config.json
        droot = ROOT / 'distilbert'
        if droot.exists():
            for child in sorted(droot.iterdir()):
                if child.is_dir() and (child / 'config.json').exists():
                    return child
        raise FileNotFoundError('No DistilBERT model folder found under distilbert/')

    def load_distilbert(self):
        if self.bert_pipe is not None:
            return
        try:
            from transformers import pipeline
            model_dir = self.find_bert_model_dir()
            # device selection: use GPU if available through torch
            try:
                import torch
                device = 0 if torch.cuda.is_available() else -1
            except Exception:
                device = -1
            self.bert_pipe = pipeline('text-classification', model=str(model_dir), tokenizer=str(model_dir), device=device)
        except Exception:
            self.bert_pipe = None
            raise


class SpamApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Email Spam Classifier')
        self.geometry('760x560')
        self.loader = ModelLoader()
        self._build()
        # Start background auto-load of models when the app starts
        threading.Thread(target=self._auto_load_models, daemon=True).start()

    def _build(self):
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill='both', expand=True)

        lbl = ttk.Label(frm, text='Enter email text below:', font=('Segoe UI', 11))
        lbl.pack(anchor='w')

        self.txt = scrolledtext.ScrolledText(frm, wrap='word', height=18)
        self.txt.pack(fill='both', expand=True, pady=6)

        bottom = ttk.Frame(frm)
        bottom.pack(fill='x')

        self.model_var = tk.StringVar(value='nb')
        rb1 = ttk.Radiobutton(bottom, text='Naive Bayes (fast)', value='nb', variable=self.model_var)
        rb2 = ttk.Radiobutton(bottom, text='DistilBERT (requires transformers)', value='bert', variable=self.model_var)
        rb1.pack(side='left', padx=6)
        rb2.pack(side='left')

        btn_predict = ttk.Button(bottom, text='Predict', command=self._on_predict)
        btn_predict.pack(side='right', padx=6)

        btn_load = ttk.Button(bottom, text='Load Models', command=self._on_load_models)
        btn_load.pack(side='right')

        self.result_var = tk.StringVar(value='')
        res_lbl = ttk.Label(frm, textvariable=self.result_var, font=('Segoe UI', 12, 'bold'), foreground='blue')
        res_lbl.pack(anchor='w', pady=8)

    def _on_load_models(self):
        def task():
            try:
                self.loader.load_naive_bayes()
            except Exception as e:
                messagebox.showwarning('Naive Bayes', f'Failed to load Naive Bayes: {e}')
            try:
                self.loader.load_distilbert()
            except Exception as e:
                messagebox.showwarning('DistilBERT', f'Failed to load DistilBERT: {e}')
            messagebox.showinfo('Done', 'Model load attempts finished')
        threading.Thread(target=task, daemon=True).start()

    def _auto_load_models(self):
        """Background loader run at startup; updates status without modal dialogs."""
        msgs = []
        try:
            self.loader.load_naive_bayes()
            msgs.append('Naive Bayes loaded')
        except Exception as e:
            msgs.append(f'Naive Bayes failed: {e}')
        try:
            self.loader.load_distilbert()
            msgs.append('DistilBERT loaded')
        except Exception as e:
            msgs.append(f'DistilBERT failed: {e}')

        status = ' | '.join(msgs)
        # Update UI from main thread
        try:
            self.after(0, lambda: self.result_var.set(status))
        except Exception:
            pass

    def _on_predict(self):
        text = self.txt.get('1.0', 'end').strip()
        if not text:
            messagebox.showwarning('Input', 'Please enter email text to classify.')
            return
        model_choice = self.model_var.get()

        def task():
            try:
                if model_choice == 'nb':
                    try:
                        self.loader.load_naive_bayes()
                    except Exception as e:
                        messagebox.showerror('Error', f'Could not load Naive Bayes: {e}')
                        return
                    X = self.loader.nb_vec.transform([text])
                    pred = int(self.loader.nb_model.predict(X)[0])
                    proba = None
                    try:
                        proba = float(self.loader.nb_model.predict_proba(X)[0, 1])
                    except Exception:
                        proba = None
                    label = 'Spam' if pred == 1 else 'Ham'
                    out = f'Model: Naive Bayes\nPrediction: {label}'
                    if proba is not None:
                        out += f' (spam probability: {proba:.3f})'
                    self.result_var.set(out)
                else:
                    try:
                        self.loader.load_distilbert()
                    except Exception as e:
                        messagebox.showerror('Error', f'Could not load DistilBERT: {e}')
                        return
                    # pipeline returns list of dicts or dict depending
                    outs = self.loader.bert_pipe(text, truncation=True)
                    # normalize output
                    if isinstance(outs, list):
                        top = outs[0]
                    elif isinstance(outs, dict):
                        top = outs
                    else:
                        top = None
                    if top is None:
                        messagebox.showerror('Error', f'Unexpected pipeline output: {outs}')
                        return
                    lbl = str(top.get('label', '')).lower()
                    score = float(top.get('score', 0.0))
                    is_spam = 'spam' in lbl or lbl in ('label_1', '1')
                    label = 'Spam' if is_spam else 'Ham'
                    out = f'Model: DistilBERT\nPrediction: {label} (score: {score:.3f}, label: {lbl})'
                    self.result_var.set(out)
            except Exception as e:
                tb = traceback.format_exc()
                messagebox.showerror('Error', f'Exception during prediction:\n{e}\n\n{tb}')

        threading.Thread(target=task, daemon=True).start()


def main():
    app = SpamApp()
    app.mainloop()


if __name__ == '__main__':
    main()
