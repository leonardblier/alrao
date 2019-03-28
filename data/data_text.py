import os
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.nb_occ = {}

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.nb_occ[word] = 1
        else:
            self.nb_occ[word] += 1
        return self.word2idx[word]

    def get_unique(self):
        dct_unique = {}
        for word, nb_occ in self.nb_occ.items():
            if nb_occ == 1:
                dct_unique[word] = True

        print(dct_unique)
        return dct_unique

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, char_prediction = False):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'), char_prediction)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'), char_prediction)
        self.test = self.tokenize(os.path.join(path, 'test.txt'), char_prediction)

    def tokenize(self, path, char_prediction = False):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                if char_prediction:
                    words = list(line)
                else:
                    words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                if char_prediction:
                    words = list(line)
                else:
                    words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
