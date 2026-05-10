import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ Hiperparametreler ------------------
MAX_LEN = 20
BATCH_SIZE = 16
EPOCHS = 20
EMBEDDING_DIM = 256
TEXT_LSTM_HIDDEN = 128
FC_HIDDEN = 64
FC_HIDDEN2 = 64
EARLY_STOPPING_PATIENCE = 5
LEARNING_RATE = 0.0007
BOOST_FACTOR = 2.0

# ------------------ Yardımcı Fonksiyonlar ------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text

def build_vocab(texts, min_freq=1):
    counter = Counter()
    for text in texts:
        tokens = text.split()
        counter.update(tokens)
    vocab = {word: i+2 for i, (word, freq) in enumerate(counter.items()) if freq >= min_freq}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab

def text_to_sequence(text, vocab):
    tokens = text.split()
    seq = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    return seq + [vocab["<PAD>"]] * (MAX_LEN - len(seq)) if len(seq) < MAX_LEN else seq[:MAX_LEN]

def plot_confusion_matrix_pyplot(y_true, y_pred, class_labels):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_labels)))
    plt.figure(figsize=(16, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Tüm Kanser Türleri İçin Confusion Matrix")
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gerçek")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(2)
    plt.close()

# ------------------ Dataset Sınıfı ------------------
class GeneticDataset(Dataset):
    def __init__(self, df, labels, text_vocab):
        self.df = df
        self.labels = labels.values if labels is not None else None
        self.text_vocab = text_vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text_seq = torch.tensor(text_to_sequence(clean_text(row['TEXT']), self.text_vocab), dtype=torch.long)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return text_seq, label
        return text_seq

# ------------------ Model Sınıfı ------------------
class GeneticModel(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super(GeneticModel, self).__init__()
        self.text_embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=0)
        self.text_lstm = nn.LSTM(EMBEDDING_DIM, TEXT_LSTM_HIDDEN, batch_first=True)
        self.fc1 = nn.Linear(TEXT_LSTM_HIDDEN, FC_HIDDEN)
        self.fc2 = nn.Linear(FC_HIDDEN, FC_HIDDEN2)
        self.dropout = nn.Dropout(0.6)
        self.out = nn.Linear(FC_HIDDEN2, num_classes)

    def forward(self, text):
        x_text = self.text_embedding(text)
        _, (h_n, _) = self.text_lstm(x_text)
        text_feat = h_n[-1]
        x = torch.relu(self.fc1(text_feat))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.out(x)

# ------------------ Eğitim ------------------
if __name__ == "__main__":
    df = pd.read_csv("final_text_based_dataset.csv")
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['TEXT', 'Cancer Type'])

    df['Cancer Type'] = df['Cancer Type'].astype('category')
    class_labels = df['Cancer Type'].cat.categories.tolist()
    df['Class'] = df['Cancer Type'].cat.codes

    labels = df['Class']
    df_train, df_val, y_train, y_val = train_test_split(df, labels, test_size=0.3, stratify=labels, random_state=42)

    text_vocab = build_vocab(df_train['TEXT'].apply(clean_text).tolist())

    train_dataset = GeneticDataset(df_train, y_train, text_vocab)
    val_dataset = GeneticDataset(df_val, y_val, text_vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeneticModel(len(text_vocab), len(class_labels)).to(device)

    final_weights = np.zeros(len(class_labels), dtype=np.float32)
    present_classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=present_classes, y=y_train)

    for cls_index, weight in zip(present_classes, weights):
        if weight > 2.0:
            final_weights[cls_index] = weight * BOOST_FACTOR
        else:
            final_weights[cls_index] = weight

    class_weights = torch.tensor(final_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

    best_f1 = 0
    patience_counter = 0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    train_losses = []
    val_accuracies = []
    val_f1_scores = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for text_seq, label in train_loader:
            text_seq, label = text_seq.to(device), label.to(device)
            optimizer.zero_grad()
            loss = criterion(model(text_seq), label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {avg_loss:.4f}")

        model.eval()
        val_preds, val_labels, val_loss = [], [], 0
        with torch.no_grad():
            for text_seq, label in val_loader:
                text_seq, label = text_seq.to(device), label.to(device)
                output = model(text_seq)
                val_loss += criterion(output, label).item()
                val_preds.extend(torch.argmax(output, dim=1).cpu().numpy())
                val_labels.extend(label.cpu().numpy())

        acc = accuracy_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds, average='macro')
        val_accuracies.append(acc)
        val_f1_scores.append(f1)

        print(f"Validation Accuracy: {acc:.4f}, F1 Score (macro): {f1:.4f}")
        scheduler.step(val_loss)

        plot_confusion_matrix_pyplot(val_labels, val_preds, class_labels)

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'text_vocab': text_vocab,
                'class_labels': class_labels
            }, "pytorch_trained_model.pth")
            print("Yeni en iyi model kaydedildi.")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping tetiklendi!")
                break

    print("Eğitim tamamlandı.")

    # ------------------ Görselleştirme ------------------
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, marker='o', label='Training Loss')
    plt.title("Eğitim Kaybı")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, val_accuracies, marker='o', color='green', label='Validation Accuracy')
    plt.title("Doğruluk Oranı")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, val_f1_scores, marker='o', color='purple', label='F1 Score (macro)')
    plt.title("F1 Skoru")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.grid(True)

    plt.tight_layout()
    plt.suptitle("Eğitim Süreci Gelişimi", fontsize=16, y=1.05)
    plt.show()