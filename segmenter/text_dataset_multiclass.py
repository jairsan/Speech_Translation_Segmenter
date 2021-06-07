import torch
import torch.utils.data as data
import random
import torch.nn as nn

class SegmentationTextDatasetMulticlass(data.Dataset):
    def __init__(self,file, vocab_dictionary, classes_dictionary, sampling_temperature=1.0):
        # Samples is a list of tuples. Element 0 is the Tensor of indices. Element 1 is the target
        self.samples = []
        self.vocab_dictionary = vocab_dictionary
        self.classes_dictionary = classes_dictionary
        counts = {} 
        self.sampling_temperature = sampling_temperature
        self.per_sample_probs = {}

        total_counts = 0
        with open(file, encoding="utf-8") as f:
            targets = []
            for line in f:
                # Each line of the file contains a target, followed by a sentence. The target and the tokens of the
                # sentence are separeted by whitespace
                line = line.strip().split()

                target = self.classes_dictionary.get_index(line[0])
                tokens = line[1:]

                tokens_i = []

                for token in tokens:
                    tokens_i.append(vocab_dictionary.get_index(token))

                self.samples.append((torch.IntTensor(tokens_i),target))

                c = counts.get(target, 0)
                counts[target] = c + 1
                total_counts += 1
                targets.append(target)


        if sampling_temperature > 1.0 :
            pseudo_probs = {}
            per_sample_probs = {}
            tot_pseudo = 0
            for cls in list(self.classes_dictionary.dictionary.keys()):
                if cls != "<pad>" and cls != "<unk>":
                    cls = self.classes_dictionary.get_index(cls)
                    cls_pseudo = (counts[cls] / total_counts) ** (1 / sampling_temperature)
                    pseudo_probs[cls] = cls_pseudo
                    tot_pseudo += cls_pseudo
            for cls in list(self.classes_dictionary.dictionary.keys()):
                if cls != "<pad>" and cls != "<unk>":
                    cls = self.classes_dictionary.get_index(cls)
                    cls_prob = pseudo_probs[cls] / tot_pseudo
                    self.per_sample_probs[cls] = cls_prob / counts[cls]

            self.weights = [ self.per_sample_probs[c] for c in targets]

            assert len(self.weights) == len(self.samples)
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Returns a tuple. Element 0 is the Tensor of indices. Element 1 is the target
        samp = self.samples[idx]
        tensor = samp[0]
        target = samp[1]

        return samp

    def collater(self,batch):
        """
        Returns a padded sequence
        """
        xs = [item[0] for item in batch]
        ys = [item[1] for item in batch]
        src_lengths = [len(x) for x in xs]

        X = nn.utils.rnn.pad_sequence(xs,batch_first=True).long()
        Y = torch.LongTensor(ys)

        return X, src_lengths, Y
