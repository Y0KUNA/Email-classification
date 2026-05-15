import gc
import re
import numpy as np
import pandas as pd
import warnings
import argparse
from tqdm import tqdm
from pathlib import Path
warnings.filterwarnings("ignore")
tqdm.pandas()


def preprocess_text(text: str) -> str:
    """Normalize text to match the preprocessing applied to combined_data.csv:
    lowercase, replace digits with 'escapenumber', strip punctuation."""
    text = str(text).lower()
    text = re.sub(r'\d+', 'escapenumber', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_preprocess(csv_path: str):
    """Load CSV, clean, and prepare a pandas DataFrame."""
    df = pd.read_csv(csv_path, encoding='latin-1')

    initial_count = df.shape[0]
    df = df.drop_duplicates()
    dedup_count = df.shape[0]
    print(f"There are {initial_count-dedup_count} duplicates found in the dataset")

    # Standardize label and text columns
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df.rename(columns={'v1': 'label', 'v2': 'text'})

    # Drop unnecessary columns if they exist
    for col in ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Rename text to title for consistency
    if 'text' in df.columns and 'title' not in df.columns:
        df = df.rename(columns={'text': 'title'})

    if 'label' in df.columns and 'title' in df.columns:
        df = df[['label', 'title']]
    else:
        raise ValueError("Expected 'label' and 'title' (or 'text') columns in CSV")

    df = df[~df['title'].isnull()]
    df = df[~df['label'].isnull()]

    # Normalize labels: convert to 'ham' and 'spam' for consistency
    def normalize_label(x):
        x_lower = str(x).lower().strip()
        if x_lower in ['ham', '0', 'no spam', 'no', 'legitimate']:
            return 'ham'
        elif x_lower in ['spam', '1', 'yes']:
            return 'spam'
        else:
            try:
                val = int(float(x))
                return 'spam' if val > 0 else 'ham'
            except:
                return 'spam' if 'spam' in x_lower else 'ham'

    df['label'] = df['label'].apply(normalize_label)

    # Apply text preprocessing to align vocabulary with combined_data.csv format.
    # combined_data.csv stores lowercased text with digits replaced by 'escapenumber'
    # and punctuation removed. Applying the same pipeline here ensures the DistilBERT
    # tokenizer sees the same token distribution at train and test time.
    print("Preprocessing text...")
    df['title'] = df['title'].progress_apply(preprocess_text)

    print('Data shape after preprocessing:', df.shape)
    return df


def main(argv=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, pipeline
    from datasets import Dataset, ClassLabel
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, f1_score
    import matplotlib.pyplot as plt
    import itertools

    parser = argparse.ArgumentParser(description='DistilBERT email spam training script')
    parser.add_argument('--train-csv', type=str, default='dataset/combined_data.csv')
    parser.add_argument('--test-csv', type=str, default='dataset/spam.csv')
    parser.add_argument('--num-train-epochs', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=3e-6)
    parser.add_argument('--train-batch-size', type=int, default=8)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--warmup-steps', type=int, default=50)
    parser.add_argument('--weight-decay', type=float, default=0.02)
    parser.add_argument('--bert-model', type=str, default='distilbert-base-cased')
    parser.add_argument('--output-dir', type=str, default='email-spam-detection-distilbert')
    args = parser.parse_args(argv)

    print("Loading training data from:", args.train_csv)
    # If defaults are used (combined_data.csv + spam.csv), perform an 80/20 split on combined_data.csv
    df_train_raw = None
    df_test_raw = None
    if Path(args.train_csv).name == 'combined_data.csv' and Path(args.test_csv).name == 'spam.csv':
        print('  Using combined_data.csv - performing 80/20 stratified split for train/test...')
        import pandas as pd
        from sklearn.model_selection import train_test_split

        df_all = load_and_preprocess(args.train_csv)
        # Map labels to 0/1 for stratify, then restore text labels
        df_all_temp = df_all.copy()
        df_all_temp['label_num'] = df_all_temp['label'].apply(lambda x: 1 if str(x).lower().strip() == 'spam' else 0)
        train_df, test_df = train_test_split(df_all_temp, test_size=0.2, stratify=df_all_temp['label_num'], random_state=42)
        # drop helper column
        train_df = train_df.drop(columns=['label_num']).reset_index(drop=True)
        test_df = test_df.drop(columns=['label_num']).reset_index(drop=True)
        df_train_raw = train_df
        df_test_raw = test_df
        print(f"  Split sizes: train={len(df_train_raw)}, test={len(df_test_raw)}")
    else:
        df_train_raw = load_and_preprocess(args.train_csv)
    print(f'Training data label distribution:\n{df_train_raw["label"].value_counts()}\n')

    if df_test_raw is None:
        print("Loading test data from:", args.test_csv)
        df_test_raw = load_and_preprocess(args.test_csv)
    print(f'Test data label distribution:\n{df_test_raw["label"].value_counts()}\n')

    df_combined = pd.concat([df_train_raw, df_test_raw], ignore_index=True)

    classes = np.unique(df_combined['label'])
    print('Classes found:', classes)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=df_combined['label'])
    class_weights = dict(zip(classes, weights))
    print('Class weights:', class_weights)

    labels_list = sorted(list(df_combined['label'].unique()))
    label2id = {label: i for i, label in enumerate(labels_list)}
    id2label = {i: label for label, i in label2id.items()}
    ordered_weigths = [class_weights[x] for x in id2label.values()]

    train_dataset = Dataset.from_pandas(df_train_raw)
    test_dataset = Dataset.from_pandas(df_test_raw)
    ClassLabels = ClassLabel(num_classes=len(labels_list), names=labels_list)

    def map_label2id(example):
        example['label'] = ClassLabels.str2int(example['label'])
        return example

    train_dataset = train_dataset.map(map_label2id, batched=True)
    train_dataset = train_dataset.cast_column('label', ClassLabels)

    test_dataset = test_dataset.map(map_label2id, batched=True)
    test_dataset = test_dataset.cast_column('label', ClassLabels)

    del df_train_raw, df_test_raw, df_combined
    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, use_fast=True)

    def preprocess_function(examples):
        return tokenizer(examples['title'], truncation=True)

    df_train = train_dataset.map(preprocess_function, batched=True)
    df_test = test_dataset.map(preprocess_function, batched=True)

    if 'title' in df_train.column_names:
        df_train = df_train.remove_columns(['title'])
    if 'title' in df_test.column_names:
        df_test = df_test.remove_columns(['title'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.bert_model,
        num_labels=len(labels_list),
        output_attentions=False,
        output_hidden_states=False,
    )
    model.config.id2label = {i: label for i, label in enumerate(labels_list)}
    model.config.label2id = {label: i for i, label in enumerate(labels_list)}

    print('Trainable params (M):', model.num_parameters(only_trainable=True) / 1e6)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        if len(labels_list) == 2:
            pos_label = label2id.get('spam', 1)
            precision = precision_score(labels, predictions, pos_label=pos_label, zero_division=0)
            recall = recall_score(labels, predictions, pos_label=pos_label, zero_division=0)
            f1 = f1_score(labels, predictions, pos_label=pos_label, zero_division=0)
        else:
            precision = precision_score(labels, predictions, average='macro', zero_division=0)
            recall = recall_score(labels, predictions, average='macro', zero_division=0)
            f1 = f1_score(labels, predictions, average='macro', zero_division=0)
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    class WeightedTrainer(Trainer):
        def __init__(self, *args, ordered_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._ordered_weights = ordered_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Accept extra kwargs from Trainer (e.g. num_items_in_batch) for compatibility
            labels = inputs.get('labels')
            new_inputs = {k: v for k, v in inputs.items() if k != 'labels'}
            outputs = model(**new_inputs)
            logits = outputs.get('logits')
            weight_tensor = None
            if self._ordered_weights is not None:
                weight_tensor = torch.tensor(self._ordered_weights, device=model.device).float()
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight_tensor)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, figsize=(10, 8), is_norm=True):
        import itertools
        plt.figure(figsize=figsize)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)
        fmt = '.3f' if is_norm else '.0f'
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir='./logs',
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        logging_steps=1,
        logging_first_step=True,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        save_total_limit=4,
        report_to='none',
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=df_train,
        eval_dataset=df_test,
        data_collator=data_collator,
        ordered_weights=ordered_weigths,
    )

    print('Initial evaluation:')
    trainer.evaluate()

    print('Starting training...')
    trainer.train()

    print('Final evaluation:')
    trainer.evaluate()

    outputs = trainer.predict(df_test)
    print('Prediction metrics:', outputs.metrics)

    y_true = outputs.label_ids
    y_pred = outputs.predictions.argmax(1)

    accuracy = accuracy_score(y_true, y_pred)
    if len(labels_list) == 2:
        pos_label = label2id.get('spam', 1)
        precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    else:
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    if len(labels_list) <= 120:
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        plot_confusion_matrix(cm, labels_list, figsize=(8, 6))

    print('Classification report:')
    print(classification_report(y_true, y_pred, target_names=labels_list, digits=4))

    trainer.save_model()
    tokenizer.save_vocabulary(save_directory=f"./{args.output_dir}")

    try:
        pipe = pipeline('text-classification', model=args.output_dir, tokenizer=tokenizer)
        sample_title = "Elon Musk buys Twitter, and so can you"
        print(pipe(sample_title, top_k=10))
    except Exception:
        pass


if __name__ == '__main__':
    main()


from transformers import pipeline
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "email-spam-detection-distilbert"

_classifier = None


def load_model():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "text-classification",
            model=str(MODEL_DIR),
            tokenizer=str(MODEL_DIR)
        )
    return _classifier


def predict(texts):
    # Apply the same preprocessing before inference
    preprocessed = [preprocess_text(t) for t in texts]

    classifier = load_model()
    outputs = classifier(preprocessed, truncation=True, batch_size=32)

    predictions = []
    for item in outputs:
        label = item["label"]
        if isinstance(label, str):
            label = label.lower()
            if "label_1" in label or "spam" in label:
                predictions.append(1)
            else:
                predictions.append(0)
        else:
            predictions.append(0)

    return predictions