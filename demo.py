import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from collections import Counter
import torchtext
from tqdm import tqdm
import matplotlib.pyplot as plt

def artist_to_label(artist_name):
    artist_label_map = {
        "ABBA": 0,
        "Bee Gees": 1,
        "Bob Dylan": 2,
        # Add more artists as needed
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

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    loss_values = []  # To store loss values for plotting
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as t_loader:
            for lyrics, artists in t_loader:
                optimizer.zero_grad()
                output = model(lyrics)
                loss = criterion(output, artists)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                _, predicted = torch.max(output, 1)
                correct += (predicted == artists).sum().item()
                total_samples += artists.size(0)
                t_loader.set_postfix(loss=total_loss/len(train_loader), accuracy=correct/total_samples)
            loss_values.append(total_loss/len(train_loader))  # Store loss for this epoch
    plt.plot(loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('training_loss_plot.png')
    plt.show()

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for lyrics, artists in test_loader:
            output = model(lyrics)
            _, predicted = torch.max(output, 1)
            total += artists.size(0)
            correct += (predicted == artists).sum().item()
    print(f"Accuracy: {correct/total}")

num_artists = len(set(artist_to_label(artist) for _, artist in train_dataset))
model = LSTMModel(len(vocab), embedding_dim=500, hidden_dim=128, output_dim=num_artists, num_layers=2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_model(model, train_loader, criterion, optimizer, num_epochs=500)

evaluate_model(model, test_loader)
