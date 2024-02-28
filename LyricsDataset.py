import torch
from torch.utils.data import Dataset
import pandas as pd
from collections import Counter
from torch.utils.data.dataloader import DataLoader


class LyricsDataset(Dataset):
    def __init__(self, filename, embedding=None, name=None, transform=None, target_transform=None):
        self.data = pd.read_csv(filename)
        self.name = name
        self.transform = transform
        self.target_transform = target_transform
        self.num_of_labels = len(self.unique_labels())
        self.labels = self.enumerate_labels()
        self.artists_list = list(self.labels.keys())
        self.embedding = embedding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        artist_name = self.data.iloc[idx]['artist']
        title = self.data.iloc[idx]['song']
        lyrics = self.data.iloc[idx]['text']
        if self.embedding is not None:
            lyrics = self.embedding(lyrics)
        return self.labels[artist_name], lyrics, title, artist_name

    def enumerate_labels(self):
        c = 0
        d = {}
        labels = self.unique_labels()
        for artist_name in labels:
            d[artist_name] = c
            c += 1
        return d

    def unique_labels(self):
        labels = self.data['artist'].tolist()
        return set(labels)

    def print_stats(self):
        print("#######################")
        print("dataset name: ", self.name)
        print("dataset length: ", len(self.data))
        print("number of labels: ", len(self.unique_labels()))
        print("number of words in all lyrics: ", sum(len(x) for x in self.data['text'].to_list()))
        print("#######################")

    def get_vocab(self):
        all_words = self.data['text'].to_list() + self.data['artist'].to_list()
        Vocab = Counter()
        for lyric in all_words:
            words_as_list = lyric.split(' ')
            if '' in words_as_list:
                words_as_list.remove("")
            Vocab += Counter(words_as_list)
        if '0' not in Vocab:
            Vocab += Counter(['0'])
        return Vocab





## THIS IS AN EXAMPLE SHOWING HOW TO ITERATE THROUGH THE DATA IN BATCHES##
d = LyricsDataset(filename = FILENAME)
loader = DataLoader(d, batch_size=2, shuffle=True, pin_memory=True)

for label, lyric, title, artist in loader:
  print("label: ",label)
  print("lyric: ",lyric)
  print("title: ",title)
  print("artist: ",artist)
  break