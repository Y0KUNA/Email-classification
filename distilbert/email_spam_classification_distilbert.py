
import gc
import re
import numpy as np
import pandas as pd
import warnings
import argparse
from tqdm import tqdm

warnings.filterwarnings("ignore")
tqdm.pandas()


def load_and_preprocess(csv_path: str):
    """Load CSV, clean, and prepare a pandas DataFrame."""
    df = pd.read_csv(csv_path, encoding='latin-1')

    initial_count = df.shape[0]
    df = df.drop_duplicates()
    dedup_count = df.shape[0]
    print(f"There are {initial_count-dedup_count} duplicates found in the dataset")

    # Standardize label and text columns
    if 'v1' in df.columns and 'v2' in df.columns:
        # This is the spam.csv format
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
            # Try numeric conversion
            try:
                val = int(float(x))
                return 'spam' if val > 0 else 'ham'
            except:
                return 'spam' if 'spam' in x_lower else 'ham'

    df['label'] = df['label'].apply(normalize_label)

    print('Data shape after preprocessing:', df.shape)
    return df


def main(argv=None):
    # Defer heavy imports to runtime so module import doesn't fail when packages are missing.
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, pipeline
    from datasets import Dataset, ClassLabel
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
    import matplotlib.pyplot as plt
    import itertools

    parser = argparse.ArgumentParser(description='DistilBERT email spam training script')
    parser.add_argument('--train-csv', type=str, default='dataset/combined_data.csv', help='Path to training CSV file')
    parser.add_argument('--test-csv', type=str, default='dataset/spam.csv', help='Path to test CSV file')
    parser.add_argument('--num-train-epochs', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=3e-6)
    parser.add_argument('--train-batch-size', type=int, default=8)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--warmup-steps', type=int, default=50)
    parser.add_argument('--weight-decay', type=float, default=0.02)
    parser.add_argument('--bert-model', type=str, default='distilbert-base-cased')
    parser.add_argument('--output-dir', type=str, default='email-spam-detection-distilbert')
    args = parser.parse_args(argv)

    # Load training data
    print("Loading training data from:", args.train_csv)
    df_train_raw = load_and_preprocess(args.train_csv)
    print(f'Training data label distribution:\n{df_train_raw["label"].value_counts()}\n')

    # Load test data
    print("Loading test data from:", args.test_csv)
    df_test_raw = load_and_preprocess(args.test_csv)
    print(f'Test data label distribution:\n{df_test_raw["label"].value_counts()}\n')

    # Combine for label mapping (to ensure consistent labels)
    df_combined = pd.concat([df_train_raw, df_test_raw], ignore_index=True)

    # Compute class weights and mappings
    classes = np.unique(df_combined['label'])
    print('Classes found:', classes)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=df_combined['label'])
    class_weights = dict(zip(classes, weights))
    print('Class weights:', class_weights)

    labels_list = sorted(list(df_combined['label'].unique()))
    label2id = {label: i for i, label in enumerate(labels_list)}
    id2label = {i: label for label, i in label2id.items()}
    ordered_weigths = [class_weights[x] for x in id2label.values()]

    # Create HF datasets from raw data
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
        return {'accuracy': accuracy}

    class WeightedTrainer(Trainer):
        def __init__(self, *args, ordered_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._ordered_weights = ordered_weights

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop('labels')
            outputs = model(**inputs)
            logits = outputs.get('logits')
            weight_tensor = None
            if self._ordered_weights is not None:
                weight_tensor = torch.tensor(self._ordered_weights, device=model.device).float()
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight_tensor)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, figsize=(10, 8), is_norm=True):
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
        logging_strategy='steps',
        logging_first_step=True,
        load_best_model_at_end=True,
        logging_steps=1,
        learning_rate=args.learning_rate,
        evaluation_strategy='epoch',
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        eval_steps=1,
        save_strategy='epoch',
        save_total_limit=1,
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
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Accuracy: {accuracy:.4f}")
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
