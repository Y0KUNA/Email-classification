

import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_data(path: str) -> pd.DataFrame:
    """Load CSV and perform minimal cleaning to match the notebook expectations."""
    data = pd.read_csv(path, encoding='latin-1')

    # Drop the extra unnamed columns if they exist (not to raise KeyError)
    for col in ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']:
        if col in data.columns:
            data = data.drop(columns=[col])

    # If the dataset has more than 1 column and first two columns are label/text,
    # rename them to ['label', 'text'] to match the notebook.
    if len(data.columns) >= 2:
        # if columns are already named 'v1'/'v2' or similar, map to label/text
        first_two = list(data.columns[:2])
        if first_two != ['label', 'text']:
            try:
                data = data.rename(columns={first_two[0]: 'label', first_two[1]: 'text'})
            except Exception:
                pass

    # Normalize labels: convert to 0 (ham/no spam) and 1 (spam)
    def normalize_label(x):
        x_lower = str(x).lower().strip()
        if x_lower in ['ham', '0', 'no spam', 'no', 'legitimate']:
            return 0
        elif x_lower in ['spam', '1', 'yes']:
            return 1
        else:
            # Try to convert numeric values
            try:
                return int(float(x))
            except:
                return 1 if 'spam' in x_lower else 0

    data['label'] = data['label'].apply(normalize_label)
    
    return data


def train_and_evaluate(train_data: pd.DataFrame, test_data: pd.DataFrame, random_state: int = 42):
    """Train MultinomialNB on training data and evaluate on test data."""
    # Ensure we have the expected columns
    if 'label' not in train_data.columns or 'text' not in train_data.columns:
        raise ValueError("Training data must contain 'label' and 'text' columns")
    if 'label' not in test_data.columns or 'text' not in test_data.columns:
        raise ValueError("Test data must contain 'label' and 'text' columns")

    # Separate features and labels
    X_train = train_data['text']
    y_train = train_data['label']
    X_test = test_data['text']
    y_test = test_data['label']

    # Vectorize
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Train
    classifier = MultinomialNB()
    classifier.fit(X_train_vectorized, y_train)

    # Predict
    y_pred = classifier.predict(X_test_vectorized)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(classification_rep)

    # Plot counts of labels in the test set (optional)
    try:
        import matplotlib.pyplot as plt

        spam_counts = y_test.value_counts()
        plt.figure(figsize=(8, 6))
        # If labels are 0/1 map ticks accordingly, otherwise show raw labels
        labels = list(spam_counts.index)
        plt.bar(range(len(spam_counts)), spam_counts.values, color=['green', 'red'][:len(spam_counts)])
        plt.xlabel('Email Type')
        plt.ylabel('Number of Emails')
        plt.title('Number of Spam and Non-Spam Emails')
        if set(labels) <= {0, 1}:
            plt.xticks([0, 1], ['ham (Non-Spam)', 'spam'])
        else:
            plt.xticks(range(len(labels)), labels)
        plt.tight_layout()
        plt.show()
    except Exception:
        # If matplotlib is not available or plotting fails, ignore plotting.
        pass


def main(argv=None):
    parser = argparse.ArgumentParser(description='Email spam detection (Naive Bayes)')
    parser.add_argument('--train-csv', type=str, default='dataset/combined_data.csv',
                        help='Path to training CSV file (default: dataset/combined_data.csv)')
    parser.add_argument('--test-csv', type=str, default='dataset/spam.csv',
                        help='Path to test CSV file (default: dataset/spam.csv)')
    args = parser.parse_args(argv)

    # Load training data
    print("Loading training data...")
    train_data = load_data(args.train_csv)
    print(f'Loaded training data with shape: {train_data.shape}')
    print(f'Training label distribution:\n{train_data["label"].value_counts()}\n')

    # Load test data
    print("Loading test data...")
    test_data = load_data(args.test_csv)
    print(f'Loaded test data with shape: {test_data.shape}')
    print(f'Test label distribution:\n{test_data["label"].value_counts()}\n')

    # Train and evaluate
    print("Training Naive Bayes classifier...")
    train_and_evaluate(train_data, test_data)


if __name__ == '__main__':
    main()
