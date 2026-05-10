import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ Hiperparametreler ------------------
MAX_LEN = 20
BATCH_SIZE = 16
EPOCHS = 20
EMBEDDING_DIM = 128
TEXT_LSTM_HIDDEN = 64
GENE_EMBED_DIM = 16
VAR_EMBED_DIM = 16
FC_HIDDEN = 64
FC_HIDDEN2 = 64
EARLY_STOPPING_PATIENCE = 5
LEARNING_RATE = 0.0007

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

def build_category_vocab(series):
    unique_items = series.unique()
    return {item: i for i, item in enumerate(unique_items)}

def prepare_top_n_classes(y_true, y_pred, class_labels, n=9):
    top_n_indices = [idx for idx, _ in Counter(y_true).most_common(n)]
    other_index = len(top_n_indices)
    top_labels = [class_labels[i] for i in top_n_indices] + ['Diğer']

    def map_index(i):
        return top_n_indices.index(i) if i in top_n_indices else other_index

    y_true_mapped = [map_index(i) for i in y_true]
    y_pred_mapped = [map_index(i) for i in y_pred]
    return y_true_mapped, y_pred_mapped, top_labels

def plot_confusion_matrix_pyplot(y_true, y_pred, class_labels):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_labels)))
    plt.figure(figsize=(12, 10))
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
    def __init__(self, df, labels, text_vocab, gene_vocab, var_vocab):
        self.df = df
        self.labels = labels.values if labels is not None else None
        self.text_vocab = text_vocab
        self.gene_vocab = gene_vocab
        self.var_vocab = var_vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text_seq = torch.tensor(text_to_sequence(clean_text(row['TEXT']), self.text_vocab), dtype=torch.long)
        gene = torch.tensor(self.gene_vocab.get(row['Gene'], 0), dtype=torch.long)
        variation = torch.tensor(self.var_vocab.get(row['Variation'], 0), dtype=torch.long)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return text_seq, gene, variation, label
        return text_seq, gene, variation

# ------------------ Model Sınıfı ------------------
class GeneticModel(nn.Module):
    def __init__(self, vocab_size, num_genes, num_variations, num_classes):
        super(GeneticModel, self).__init__()
        self.text_embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=0)
        self.text_lstm = nn.LSTM(EMBEDDING_DIM, TEXT_LSTM_HIDDEN, batch_first=True)
        self.gene_embedding = nn.Embedding(num_genes, GENE_EMBED_DIM)
        self.var_embedding = nn.Embedding(num_variations, VAR_EMBED_DIM)
        self.fc1 = nn.Linear(TEXT_LSTM_HIDDEN + GENE_EMBED_DIM + VAR_EMBED_DIM, FC_HIDDEN)
        self.fc2 = nn.Linear(FC_HIDDEN, FC_HIDDEN2)
        self.dropout = nn.Dropout(0.6)
        self.out = nn.Linear(FC_HIDDEN2, num_classes)

    def forward(self, text, gene, variation):
        x_text = self.text_embedding(text)
        _, (h_n, _) = self.text_lstm(x_text)
        text_feat = h_n[-1]
        gene_feat = self.gene_embedding(gene)
        var_feat = self.var_embedding(variation)
        combined = torch.cat([text_feat, gene_feat, var_feat], dim=1)
        x = torch.relu(self.fc1(combined))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.out(x)

# ------------------ Eğitim ------------------
if __name__ == "__main__":
    df = pd.read_csv("realistic_model_dataset.csv")

    def first_n_words(text, n=5):
        return " ".join(str(text).split()[:n])

    df['TEXT'] = df['Biopsy Site'].apply(first_n_words)
    df['Gene'] = df['Sex']
    df['Variation'] = df['Ethnicity Category']

    df = df.dropna(subset=['TEXT', 'Gene', 'Variation', 'Cancer Type'])

    df['Cancer Type'] = df['Cancer Type'].astype('category')
    class_labels = df['Cancer Type'].cat.categories.tolist()
    df['Class'] = df['Cancer Type'].cat.codes

    labels = df['Class']
    df_train, df_val, y_train, y_val = train_test_split(df, labels, test_size=0.3, stratify=labels, random_state=42)

    text_vocab = build_vocab(df_train['TEXT'].apply(clean_text).tolist())
    gene_vocab = build_category_vocab(df_train['Gene'])
    var_vocab = build_category_vocab(df_train['Variation'])

    train_dataset = GeneticDataset(df_train, y_train, text_vocab, gene_vocab, var_vocab)
    val_dataset = GeneticDataset(df_val, y_val, text_vocab, gene_vocab, var_vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeneticModel(len(text_vocab), len(gene_vocab), len(var_vocab), len(class_labels)).to(device)

    final_weights = np.zeros(len(class_labels), dtype=np.float32)
    present_classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=present_classes, y=y_train)
    for cls_index, weight in zip(present_classes, weights):
        final_weights[cls_index] = weight

    class_weights = torch.tensor(final_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

    best_f1 = 0
    patience_counter = 0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for text_seq, gene, variation, label in train_loader:
            text_seq, gene, variation, label = text_seq.to(device), gene.to(device), variation.to(device), label.to(device)
            optimizer.zero_grad()
            loss = criterion(model(text_seq, gene, variation), label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {avg_loss:.4f}")

        model.eval()
        val_preds, val_labels, val_loss = [], [], 0
        with torch.no_grad():
            for text_seq, gene, variation, label in val_loader:
                text_seq, gene, variation, label = text_seq.to(device), gene.to(device), variation.to(device), label.to(device)
                output = model(text_seq, gene, variation)
                val_loss += criterion(output, label).item()
                val_preds.extend(torch.argmax(output, dim=1).cpu().numpy())
                val_labels.extend(label.cpu().numpy())

        acc = accuracy_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds, average='macro')
        print(f"Validation Accuracy: {acc:.4f}, F1 Score (macro): {f1:.4f}")
        print("\n📊 Sınıf Bazlı Rapor:")
        print(classification_report(val_labels, val_preds, target_names=class_labels, zero_division=0))

        scheduler.step(val_loss)

        val_labels_top9, val_preds_top9, new_labels = prepare_top_n_classes(val_labels, val_preds, class_labels)
        plot_confusion_matrix_pyplot(val_labels_top9, val_preds_top9, new_labels)

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'text_vocab': text_vocab,
                'gene_vocab': gene_vocab,
                'var_vocab': var_vocab,
                'class_labels': class_labels
            }, "pytorch_trained_model.pth")
            print("Yeni en iyi model kaydedildi.")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping tetiklendi!")
                break

    print("Eğitim tamamlandı.")
