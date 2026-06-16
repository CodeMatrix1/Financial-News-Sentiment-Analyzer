import torch
import torch.nn as nn
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):

    def __init__(
        self,
        embedding_matrix
    ):

        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=False
        )
        self.lstm = nn.LSTM(input_size=100,hidden_size=128,num_layers=2,batch_first=True,bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256,3)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2], hidden[-1]),dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)

#loading the models

embedding_matrix = torch.load(
    "models/embedding_matrix.pt",
    map_location="cpu",
    weights_only=False
)

model = LSTMClassifier(
    embedding_matrix
)

model.load_state_dict(
    torch.load(
        "models/lstm_model.pth",
        map_location="cpu"
    )
)

model.eval()

#prediction
label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


def predict_lstm(text):
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(
        seq,
        maxlen=128,
        padding="post"
    )

    tensor = torch.LongTensor(seq)

    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).item()
    return label_map[pred]
