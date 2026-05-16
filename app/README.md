# Spam Classification Desktop App

This small Tkinter app lets you load two models (Naive Bayes and DistilBERT) and classify a single email text as spam or ham.

How to run

1. Install dependencies (recommended in a virtualenv):

    pip install -r app/requirements.txt

2. Train or place models:

    - Naive Bayes artifacts should be in `naive_bayes/naive_bayes_model.pkl` and `naive_bayes/count_vectorizer.pkl`.
    - DistilBERT should be under `distilbert/email-spam-detection-distilbert/` (or any subfolder under `distilbert/` that contains `config.json`).

3. Run the app:

    python app/spam_classification_app.py

Notes

- DistilBERT loading requires the `transformers` library and, ideally, PyTorch with CUDA if you want GPU acceleration.
- The UI is intentionally minimal and synchronous model loads/predictions are performed in background threads so the UI stays responsive.
