"""
TASK 2: Intelligent Multi-Class Natural Language Text Sentiment Classifier
Custom implementation using NumPy only — TF-IDF + Softmax Classifier
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
from pathlib import Path
from collections import Counter
import time
import math

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(42)

# ─── 1. SYNTHETIC DATASET ────────────────────────────────────────────────────

STOP_WORDS = {
    'a', 'an', 'the', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
    'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their', 'mine',
    'yours', 'hers', 'ours', 'theirs', 'this', 'that', 'these', 'those',
    'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'for', 'with',
    'about', 'against', 'between', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
    'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'only', 'own', 'same', 'very', 'just', 'too', 'also',
    'if', 'then', 'else', 'than', 'as', 'well', 'like', 'because',
}

CATEGORIES = ['positive', 'negative', 'neutral', 'angry', 'sarcastic']

SEED_DATA = {
    'positive': [
        'absolutely loved this product it exceeded all expectations',
        'amazing experience highly recommend to everyone',
        'wonderful service and fantastic quality',
        'best purchase ever made truly delighted',
        'outstanding results could not be happier',
        'brilliant innovation changing the game completely',
        'fantastic support team resolved everything quickly',
        'incredible value for money very satisfied',
        'perfect solution works flawlessly every time',
        'superb craftsmanship attention to detail remarkable',
    ],
    'negative': [
        'terrible experience worst purchase ever regretted',
        'completely disappointed product broke immediately',
        'horrible quality wasted money completely useless',
        'awful service never again will recommend',
        'dreadful performance constantly crashing freezing',
        'pathetic excuse for product not worth it',
        'frustrating experience support completely ignored',
        'miserable quality falling apart within days',
        'horrendous customer service never responded',
        'atrocious product design fundamentally flawed',
    ],
    'neutral': [
        'the product arrived on time as expected',
        'standard quality nothing special but works',
        'average experience meets basic requirements',
        'usual service received what was ordered',
        'ordinary product does what it supposed to',
        'decent enough for the price nothing exceptional',
        'regular performance functions without issues',
        'typical quality acceptable for daily use',
        'moderate satisfaction neither good nor bad',
        'fair product adequate for basic needs',
    ],
    'angry': [
        'absolutely furious this is completely unacceptable',
        'outraged by terrible service you should be ashamed',
        'how dare you sell such defective garbage to customers',
        'this is infuriating completely wasted my entire day',
        'unbelievable incompetence you ruined everything',
        'seething with rage right now this is ridiculous',
        'completely fed up with pathetic excuses',
        'utterly disgusted by this horrible service',
        'this makes my blood boil total scam',
        'fuming mad this is not okay at all',
    ],
    'sarcastic': [
        'oh great another flawless product that broke instantly',
        'sure because waiting three hours is my idea of fun',
        'love how nothing ever works around here amazing',
        'yeah right totally believed it would work perfectly',
        'wonderful just what needed another problem to deal',
        'brilliant move charging premium for broken trash',
        'oh fantastic nothing beats paying extra for less',
        'sure this is exactly what i call quality service',
        'love waiting weeks for delivery that never arrives',
        'perfect because who needs customer support anyway',
    ],
}


def lemmatize_simple(word):
    """Basic lemmatization rules."""
    word = word.lower()
    if word.endswith('ing'):
        word = word[:-3]
    elif word.endswith('ly'):
        word = word[:-2]
    elif word.endswith('ed'):
        word = word[:-2]
    elif word.endswith('es') and len(word) > 4:
        word = word[:-2]
    elif word.endswith('s') and not word.endswith('ss') and len(word) > 3:
        word = word[:-1]
    return word


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    words = [lemmatize_simple(w) for w in words]
    return words


def build_dataset(augment=True):
    texts, labels = [], []
    for label, samples in SEED_DATA.items():
        for s in samples:
            texts.append(s)
            labels.append(label)
            if augment:
                words = s.split()
                if len(words) > 3:
                    words2 = list(words)
                    np.random.shuffle(words2)
                    texts.append(' '.join(words2))
                    labels.append(label)
    return texts, labels


# ─── 2. CUSTOM TF-IDF IMPLEMENTATION ─────────────────────────────────────────

class CustomTfidfVectorizer:
    def __init__(self, max_features=1000):
        self.max_features = max_features
        self.vocab = {}
        self.idf = {}
        self.doc_count = 0

    def fit(self, tokenized_docs):
        self.doc_count = len(tokenized_docs)
        df = Counter()
        for doc in tokenized_docs:
            unique_terms = set(doc)
            df.update(unique_terms)

        sorted_terms = sorted(df.items(), key=lambda x: -x[1])
        top_terms = sorted_terms[:self.max_features]
        self.vocab = {term: idx for idx, (term, _) in enumerate(top_terms)}

        for term, doc_freq in top_terms:
            self.idf[term] = math.log((self.doc_count + 1) / (doc_freq + 1)) + 1
        return self

    def transform(self, tokenized_docs):
        X = np.zeros((len(tokenized_docs), len(self.vocab)))
        for i, doc in enumerate(tokenized_docs):
            tf = Counter(doc)
            total = len(doc) if doc else 1
            for term, count in tf.items():
                if term in self.vocab:
                    tfidf = (count / total) * self.idf[term]
                    X[i, self.vocab[term]] = tfidf
        return X


# ─── 3. SOFTMAX CLASSIFIER (from scratch) ────────────────────────────────────

class SoftmaxClassifier:
    def __init__(self, n_features, n_classes, lr=0.01, reg=0.001):
        self.W = np.random.randn(n_features, n_classes) * 0.01
        self.b = np.zeros((1, n_classes))
        self.lr = lr
        self.reg = reg

    def softmax(self, z):
        z_max = np.max(z, axis=1, keepdims=True)
        z_stable = z - z_max
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        return self.softmax(X @ self.W + self.b)

    def loss(self, y_pred, y_true_onehot):
        n = y_pred.shape[0]
        log_probs = -np.log(np.clip(y_pred, 1e-10, 1.0))
        data_loss = np.sum(log_probs * y_true_onehot) / n
        reg_loss = 0.5 * self.reg * np.sum(self.W ** 2)
        return data_loss + reg_loss

    def train_step(self, X, y_true_onehot):
        n = X.shape[0]
        y_pred = self.forward(X)
        grad = (y_pred - y_true_onehot) / n
        dW = X.T @ grad + self.reg * self.W
        db = np.sum(grad, axis=0, keepdims=True)
        self.W -= self.lr * dW
        self.b -= self.lr * db
        return self.loss(y_pred, y_true_onehot)

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

    def predict_proba(self, X):
        return self.forward(X)


# ─── 4. EVALUATION METRICS ──────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, class_names):
    n_classes = len(class_names)
    metrics = {}
    for i, name in enumerate(class_names):
        tp = np.sum((y_true == i) & (y_pred == i))
        fp = np.sum((y_true != i) & (y_pred == i))
        fn = np.sum((y_true == i) & (y_pred != i))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        metrics[name] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': int(tp + fn)}
    total_support = sum(m['support'] for m in metrics.values())
    avg_f1 = np.mean([m['f1'] for m in metrics.values()])
    return metrics, avg_f1


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted', fontsize=13)
    ax.set_ylabel('Actual', fontsize=13)
    ax.set_title('Confusion Matrix - Sentiment Classification', fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return cm


def plot_metrics_bar(per_class_metrics, class_names, save_path):
    labels = class_names
    precision = [per_class_metrics[l]['precision'] for l in labels]
    recall = [per_class_metrics[l]['recall'] for l in labels]
    f1 = [per_class_metrics[l]['f1'] for l in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', color='#2E86AB')
    ax.bar(x, recall, width, label='Recall', color='#A23B72')
    ax.bar(x + width, f1, width, label='F1-Score', color='#F18F01')
    ax.set_xlabel('Sentiment Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.legend()
    ax.set_ylim(0, 1.1)
    for i in range(len(labels)):
        for j, vals in enumerate([precision, recall, f1]):
            ax.text(i + (j - 1) * width, vals[i] + 0.02, f'{vals[i]:.2f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_training_history(losses, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(losses, color='#2E86AB', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Convergence', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_f1_comparison(avg_f1, save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#2E86AB' if avg_f1 >= 0.7 else '#F18F01']
    ax.barh(['Classifier'], [avg_f1], color=colors, height=0.5)
    ax.set_xlim(0, 1.0)
    ax.axvline(0.7, color='green', linestyle='--', alpha=0.7, label='Target (0.70)')
    ax.set_xlabel('F1-Score', fontsize=12)
    ax.set_title('Overall F1-Score Performance', fontsize=14, fontweight='bold')
    ax.legend()
    ax.text(avg_f1 + 0.02, 0, f'{avg_f1:.4f}', va='center', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─── 5. MAIN PIPELINE ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("TASK 2: Multi-Class Natural Language Sentiment Classifier")
    print("=" * 60)

    print("\n[1/6] Building synthetic dataset...")
    texts, labels = build_dataset(augment=True)
    label_to_idx = {l: i for i, l in enumerate(CATEGORIES)}
    idx_to_label = {i: l for l, i in label_to_idx.items()}
    y = np.array([label_to_idx[l] for l in labels])
    print(f"   Dataset size: {len(texts)} samples")
    print(f"   Classes: {CATEGORIES}")

    print("\n[2/6] Preprocessing text (stop-word removal + lemmatization)...")
    tokenized = [preprocess(t) for t in texts]

    vocab_stats = [len(t) for t in tokenized]
    print(f"   Avg tokens/doc: {np.mean(vocab_stats):.1f}")
    print(f"   Min tokens/doc: {np.min(vocab_stats)}")
    print(f"   Max tokens/doc: {np.max(vocab_stats)}")

    print("\n[3/6] Converting text to TF-IDF vectors...")
    vectorizer = CustomTfidfVectorizer(max_features=500)
    vectorizer.fit(tokenized)
    X = vectorizer.transform(tokenized)
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Vocabulary size: {len(vectorizer.vocab)}")

    print("\n[4/6] Training Softmax Classifier...")

    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    n_classes = len(CATEGORIES)
    classifier = SoftmaxClassifier(n_features=X.shape[1], n_classes=n_classes, lr=0.05, reg=0.001)

    y_train_onehot = np.zeros((len(y_train), n_classes))
    y_train_onehot[np.arange(len(y_train)), y_train] = 1

    losses = []
    n_epochs = 500
    start_time = time.time()
    for epoch in range(n_epochs):
        batch_size = 16
        perm = np.random.permutation(len(X_train))
        epoch_losses = []
        for i in range(0, len(X_train), batch_size):
            batch_idx = perm[i:i + batch_size]
            loss = classifier.train_step(X_train[batch_idx], y_train_onehot[batch_idx])
            epoch_losses.append(loss)
        avg_epoch_loss = np.mean(epoch_losses)
        losses.append(avg_epoch_loss)
        if (epoch + 1) % 100 == 0:
            print(f"   Epoch {epoch + 1}/{n_epochs} — Loss: {avg_epoch_loss:.4f}")
    train_time = time.time() - start_time
    print(f"   Training completed in {train_time:.2f}s")

    print("\n[5/6] Evaluating model...")
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)

    accuracy = np.mean(y_pred == y_test)
    per_class_metrics, avg_f1 = compute_metrics(y_test, y_pred, CATEGORIES)

    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Avg F1:    {avg_f1:.4f}")
    for cls, m in per_class_metrics.items():
        print(f"   {cls:>10s}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")

    print("\n[6/6] Generating visualizations...")
    cm_path = OUTPUT_DIR / 'confusion_matrix.png'
    plot_confusion_matrix(y_test, y_pred, CATEGORIES, cm_path)

    metrics_path = OUTPUT_DIR / 'per_class_metrics.png'
    plot_metrics_bar(per_class_metrics, CATEGORIES, metrics_path)

    loss_path = OUTPUT_DIR / 'training_loss.png'
    plot_training_history(losses, loss_path)

    f1_path = OUTPUT_DIR / 'f1_score.png'
    plot_f1_comparison(avg_f1, f1_path)

    results = {
        'dataset_size': len(texts),
        'vocab_size': len(vectorizer.vocab),
        'feature_shape': list(X.shape),
        'test_size': len(X_test),
        'train_size': len(X_train),
        'accuracy': float(accuracy),
        'avg_f1': float(avg_f1),
        'training_time_s': float(train_time),
        'per_class_metrics': per_class_metrics,
        'categories': CATEGORIES,
    }
    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n   Outputs saved to: {OUTPUT_DIR}")
    print("[DONE] Task 2 completed successfully!\n")
    return results


if __name__ == '__main__':
    main()
