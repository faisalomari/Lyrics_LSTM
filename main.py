import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from collections import Counter
import torchtext
from tqdm import tqdm

def artist_to_label(artist_name):
    # Define a dictionary mapping artist names to numerical labels
    artist_label_map = {
        "ABBA": 0,
        "Bee Gees": 1,
        "Bob Dylan": 2,
        # Add more artists as needed
    }
    
    # Return the numerical label corresponding to the artist name
    return artist_label_map.get(artist_name, -1)  # Return -1 if artist name not found


# Define the dataset class
class LyricsDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        lyrics = self.data.iloc[idx]['text']
        artist = self.data.iloc[idx]['artist']
        return lyrics, artist

# Load the datasets
train_dataset = LyricsDataset("songdata_train.csv")
test_dataset = LyricsDataset("songdata_test.csv")

# Tokenization and vocabulary creation
tokenizer = get_tokenizer("basic_english")

counter = Counter()
for lyrics, _ in train_dataset:
    counter.update(tokenizer(lyrics))

vocab = torchtext.vocab.Vocab(counter)

# Function to encode text to tensor
def text_to_tensor(text):
    return torch.tensor([vocab[token] for token in tokenizer(text)], dtype=torch.long)

# Function to collate data samples into batches
def collate_batch(batch):
    lyrics, artists = zip(*batch)
    lyrics_tensor = [text_to_tensor(lyric) for lyric in lyrics]
    lyrics_tensor_padded = pad_sequence(lyrics_tensor, padding_value=0, batch_first=True)
    artists_tensor = torch.tensor([artist_to_label(artist) for artist in artists], dtype=torch.long)
    return lyrics_tensor_padded, artists_tensor

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        output = self.fc(output[:, -1, :])
        return output

# Define a function to train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
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


# Define the model
num_artists = len(set(artist_to_label(artist) for _, artist in train_dataset))
model = LSTMModel(len(vocab), embedding_dim=100, hidden_dim=128, output_dim=num_artists)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# Define a function to evaluate the model
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

# Evaluate the model
evaluate_model(model, test_loader)
