import torch
import torch.nn as nn
import re

MAX_LEN = 20
EMBEDDING_DIM = 128
TEXT_LSTM_HIDDEN = 64
FC_HIDDEN = 64
FC_HIDDEN2 = 64

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text

def text_to_sequence(text, vocab):
    tokens = text.split()
    seq = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    return seq + [vocab["<PAD>"]] * (MAX_LEN - len(seq)) if len(seq) < MAX_LEN else seq[:MAX_LEN]

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

# ----------- Tahmin Fonksiyonu -----------
def load_model_and_predict(text_input):
    checkpoint = torch.load("pytorch_trained_model.pth", map_location=torch.device('cpu'))

    text_vocab = checkpoint['text_vocab']
    class_labels = checkpoint['class_labels']
    num_classes = len(class_labels)

    model = TextOnlyModel(len(text_vocab), num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Metni işle
    cleaned_text = clean_text(text_input)
    seq = torch.tensor([text_to_sequence(cleaned_text, text_vocab)], dtype=torch.long)

    # Tahmin
    with torch.no_grad():
        output = model(seq)
        probs = torch.softmax(output, dim=1)
        top_pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, top_pred].item()

    return class_labels[top_pred], confidence
