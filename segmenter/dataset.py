import torch
import torch.utils.data as data
import os
import torch.nn as nn


class segmentationBinarizedDataset(data.Dataset):

    def __init__(self,file, vocab_dictionary, upsample_split=1):
        # Samples is a list of tuples. Element 0 is the Tensor of indices. Element 1 is the target
        self.samples = []
        self.vocab_dictionary = vocab_dictionary

        # Keep count of how many of each class
        c0 = 0
        c1 = 0

        with open(file, encoding="utf-8") as f:
            targets = []
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

                if target == 0:
                    c0 += 1
                else:
                    c1 += 1
                targets.append(target)


        # Compute weights for each sample
        denom = c0 + c1 * upsample_split

        w0 = (c0 / denom) / c0
        w1 = (c1 * upsample_split / denom) / c1

        self.weights = [w0 if target == 0 else w1 for target in targets]

        if upsample_split > 1:
            print("Upsampling dataset. Original nr c0,",c0, "nr c1", c1, ". Upsample by a factor of ",
                  upsample_split,", final weights are:",w0,w1, "(originally ",(c0 / (c0 + c1)) / c0, c1 / (c0+c1) / c1," )")

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
    Y = torch.LongTensor(ys)

    return X, src_lengths, Y
