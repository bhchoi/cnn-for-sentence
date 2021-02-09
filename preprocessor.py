import torch
from konlpy.tag import Okt


class Preprocessor:
    def __init__(self, word2vec_model):
        self.model = word2vec_model
        self.vocab = self.model.wv.vocab
        self.okt = Okt()

    def tokenize(self, sentence):
        tokens = self.okt.morphs(sentence, stem=True)
        return tokens

    def get_input_features(self, sentence, max_len):
        tokenized_sentence = self.tokenize(sentence)

        word_indices = []

        for token in tokenized_sentence:
            if token in self.vocab.keys():
                word_indices.append(self.vocab[token].index)
            else:
                word_indices.append(self.vocab["<unk>"].index)

        if len(word_indices) > max_len:
            word_indices = word_indices[:max_len]

        word_indices = word_indices + [self.vocab["<pad>"].index] * (
            max_len - len(word_indices)
        )

        word_indices = torch.tensor(word_indices, dtype=torch.long)

        return word_indices
