

import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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

    return data


def train_and_evaluate(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Train MultinomialNB on the text data and print evaluation metrics."""
    # Ensure we have the expected columns
    if 'label' not in data.columns or 'text' not in data.columns:
        raise ValueError("Data must contain 'label' and 'text' columns")

    # Separate features and labels
    X = data['text']
    y = data['label']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state)

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
    parser.add_argument('--csv', type=str, default='dataset/combined_data.csv',
                        help='Path to spam CSV file (default: dataset/spam.csv)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    args = parser.parse_args(argv)

    data = load_data(args.csv)
    print('Loaded data with shape:', data.shape)

    # Quick peek
    print(data.head())

    train_and_evaluate(data, test_size=args.test_size)


if __name__ == '__main__':
    main()
