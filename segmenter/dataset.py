import torch
import torch.utils.data as data
import os
import torch.nn as nn


class segmentationBinarizedDataset(data.Dataset):

    def __init__(self,file, vocab_dictionary):
        # Samples is a list of tuples. Element 0 is the Tensor of indices. Element 1 is the target
        self.samples = []
        self.vocab_dictionary = vocab_dictionary

        with open(file, encoding="utf-8") as f:
            for line in f:
                # Each line of the file contains a target, followed by a sentence. The target and the tokens of the
                # sentence are separeted by whitespace
                line = line.strip().split()

                target = int(line[0])
                tokens = line[1:]

                tokens_i = []

                for token in tokens:
                    tokens_i.append(vocab_dictionary.get_index(token))

                self.samples.append((torch.LongTensor(tokens_i),target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Returns a tuple. Element 0 is the Tensor of indices. Element 1 is the target
        return self.samples[idx]


def collateBinarizedBatch(batch):
    """
    Returns a padded sequence
    """
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    src_lengths = [len(x) for x in xs]

    X = nn.utils.rnn.pad_sequence(xs,batch_first=True)
    Y = torch.IntTensor(ys)

    return X, src_lengths, Y
