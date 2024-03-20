import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import Counter
import torchtext

def artist_to_label(artist_name):
    artist_label_map = {
        "ABBA": 0,
        "Bee Gees": 1,
        "Bob Dylan": 2,
    }
    return artist_label_map.get(artist_name, -1)

class LyricsDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        lyrics = self.data.iloc[idx]['text']
        artist = self.data.iloc[idx]['artist']
        return lyrics, artist

train_dataset = LyricsDataset("songdata_train.csv")
test_dataset = LyricsDataset("songdata_test.csv")

tokenizer = get_tokenizer("basic_english")

counter = Counter()
for lyrics, _ in train_dataset:
    counter.update(tokenizer(lyrics))

vocab = torchtext.vocab.Vocab(counter)

def text_to_tensor(text):
    return torch.tensor([vocab[token] for token in tokenizer(text)], dtype=torch.long)

def collate_batch(batch):
    lyrics, artists = zip(*batch)
    lyrics_tensor = [text_to_tensor(lyric) for lyric in lyrics]
    lyrics_tensor_padded = pad_sequence(lyrics_tensor, padding_value=0, batch_first=True)
    artists_tensor = torch.tensor([artist_to_label(artist) for artist in artists], dtype=torch.long)
    return lyrics_tensor_padded, artists_tensor

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_batch)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_layers = nn.ModuleList([nn.LSTM(embedding_dim if i == 0 else hidden_dim,
                                                  hidden_dim,
                                                  batch_first=True)
                                          for i in range(num_layers)])
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_output = self.embedding(x)
        for lstm_layer in self.lstm_layers:
            lstm_output, _ = lstm_layer(lstm_output)
        output = self.fc(lstm_output[:, -1, :])
        return output

# Define the evaluate_model function
def evaluate_model(model, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for lyrics, artists in test_loader:
            output = model(lyrics)
            loss = criterion(output, artists)
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == artists).sum().item()
            total += artists.size(0)

    test_accuracy = correct / total
    test_loss = total_loss / len(test_loader)

    print(f"Test Accuracy: {test_accuracy}")
    print(f"Test Loss: {test_loss}")

    return test_loss, test_accuracy

# Load the saved model
saved_model_path = 'saved_models/best_model.pth'
num_artists = len(set(artist_to_label(artist) for _, artist in train_dataset))
loaded_model = LSTMModel(len(vocab), embedding_dim=500, hidden_dim=128, output_dim=num_artists, num_layers=2)
loaded_model.load_state_dict(torch.load(saved_model_path))

# Define the test DataLoader
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_batch)

# Evaluate the loaded model
criterion = nn.CrossEntropyLoss()
evaluate_model(loaded_model, test_loader, criterion)
