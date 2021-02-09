import torch
from torch.utils.data import Dataset
import csv


class Dataset(Dataset):
    def __init__(self, file_path, max_len, preprocessor):
        self.preprocessor = preprocessor
        self.max_len = max_len
        self.sentences = []
        self.labels = []
        self._load_data(file_path)

    def _load_data(self, file_path):
        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                self.sentences.append(row["document"])
                self.labels.append(int(row["label"]))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index: int):
        sentence = self.sentences[index]
        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.long)
        return self.preprocessor.get_input_features(sentence, self.max_len), label
