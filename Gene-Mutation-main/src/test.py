import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import Counter

# ------------------ Ayarlar ------------------
MAX_LEN = 20
EMBEDDING_DIM = 256
TEXT_LSTM_HIDDEN = 128
FC_HIDDEN = 64
FC_HIDDEN2 = 64

# ------------------ Temizleme ------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text

def text_to_sequence(text, vocab):
    tokens = text.split()
    seq = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    return seq + [vocab["<PAD>"]] * (MAX_LEN - len(seq)) if len(seq) < MAX_LEN else seq[:MAX_LEN]

# ------------------ Dataset ------------------
class TextOnlyDataset(Dataset):
    def __init__(self, df, vocab, labels=None):
        self.df = df
        self.vocab = vocab
        self.labels = labels.values if labels is not None else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['TEXT']
        text_seq = torch.tensor(text_to_sequence(clean_text(text), self.vocab), dtype=torch.long)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return text_seq, label
        return text_seq

# ------------------ Model ------------------
class TextOnlyModel(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super(TextOnlyModel, self).__init__()
        self.text_embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=0)
        self.text_lstm = nn.LSTM(EMBEDDING_DIM, TEXT_LSTM_HIDDEN, batch_first=True)
        self.fc1 = nn.Linear(TEXT_LSTM_HIDDEN, FC_HIDDEN)
        self.fc2 = nn.Linear(FC_HIDDEN, FC_HIDDEN2)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(FC_HIDDEN2, num_classes)

    def forward(self, text):
        x = self.text_embedding(text)
        _, (h_n, _) = self.text_lstm(x)
        x = h_n[-1]
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.out(x)

# ------------------ Görselleştirme ------------------
def plot_confusion_matrix(y_true, y_pred, class_labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gerçek")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plot_prediction_distribution(y_pred, class_labels):
    counter = Counter(y_pred)
    total = sum(counter.values())
    labels = [class_labels[i] for i in range(len(class_labels))]
    counts = [counter.get(i, 0) for i in range(len(class_labels))]
    percentages = [c / total * 100 for c in counts]

    plt.figure(figsize=(18, 8))
    bars = plt.bar(labels, counts, color='skyblue')

    for bar, pct in zip(bars, percentages):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f"%{pct:.1f}", ha='center', fontsize=9)

    plt.title("Test Seti Tahmin Dağılımı (Türkçe)")
    plt.ylabel("Tahmin Edilen Sayı")
    plt.xlabel("Kanser Türü (Türkçe)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# ------------------ Ana Akış ------------------
if __name__ == "__main__":
    # Veriyi oku
    df_test = pd.read_csv("final_text_based_dataset.csv")
    checkpoint = torch.load("pytorch_trained_model.pth", map_location=torch.device('cpu'))

    # Model bileşenleri
    vocab = checkpoint['text_vocab']
    class_labels = checkpoint['class_labels']
    num_classes = len(class_labels)

    # Temizleme
    df_test.columns = df_test.columns.str.strip()
    df_test = df_test.dropna(subset=['TEXT', 'Cancer Type'])
    df_test['Cancer Type'] = df_test['Cancer Type'].astype('category')
    class_map = {label: idx for idx, label in enumerate(class_labels)}
    df_test['Class'] = df_test['Cancer Type'].map(class_map)

    # Dataset ve DataLoader
    test_dataset = TextOnlyDataset(df_test, vocab, df_test['Class'])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model
    model = TextOnlyModel(vocab_size=len(vocab), num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Tahmin
    all_preds, all_labels = [], []
    with torch.no_grad():
        for text_seq, labels in test_loader:
            outputs = model(text_seq)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    # Performans
    print("\nSınıf Bazlı Performans:")
    print(classification_report(all_labels, all_preds, target_names=class_labels, zero_division=0))

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nToplam Doğruluk Oranı: {acc:.4f}")

    # Görselleştirme
    plot_confusion_matrix(all_labels, all_preds, class_labels)
    plot_prediction_distribution(all_preds, class_labels)
