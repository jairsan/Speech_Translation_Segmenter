import torch
import torch.utils.data as data
import random
import torch.nn as nn

SEED=5


class SegmentationTextAudioDataset(data.Dataset):

    def __init__(self,text_file, audio_file, vocab_dictionary, min_split_samples_batch_ratio=0.0, unk_noise_prob=0.0):
        # Samples is a list of tuples. Element 0 is the Tensor of indices. Element 1 is the target
        self.samples = []
        self.vocab_dictionary = vocab_dictionary
        self.min_split_samples_batch_ratio = min_split_samples_batch_ratio
        self.unk_noise_prob = unk_noise_prob
        self.unk_index = vocab_dictionary.get_index("<unk>")
        # Keep count of how many of each class
        c0 = 0
        c1 = 0

        random.seed(SEED)

        with open(text_file, encoding="utf-8") as f, open(audio_file) as audio_f:
            targets = []
            for line_t,line_a in zip(f,audio_f):
                # Each line of the file contains a target, followed by a sentence. The target and the tokens of the
                # sentence are separeted by whitespace
                line = line_t.strip().split()

                target = int(line[0])
                tokens = line[1:]

                tokens_i = []

                for token in tokens:
                    tokens_i.append(vocab_dictionary.get_index(token))

                feas = line_a.split(";")

                # We convert the strings to floats
                fea_t = [list(map(float,fea.split())) for fea in feas]

                self.samples.append((torch.LongTensor(tokens_i),torch.FloatTensor(fea_t), target))

                if target == 0:
                    c0 += 1
                else:
                    c1 += 1
                targets.append(target)

        r1 = c1 / (c1 + c0)

        if self.min_split_samples_batch_ratio > 0.0 and r1 < self.min_split_samples_batch_ratio:
            w0 = (1 - self.min_split_samples_batch_ratio) / c0
            w1 = self.min_split_samples_batch_ratio / c1
            print("Upsampling dataset. Original nr c0,", c0, "nr c1", c1, ". Upsample to keep ratio of ",
                  self.min_split_samples_batch_ratio, ", final weights are:", w0, w1, "(originally ", (c0 / (c0 + c1)) / c0,
                  "for both classes )")

        else:
            w0 = (c0 / (c0 + c1)) / c0
            w1 = (c1 / (c0 + c1)) / c1

        self.weights = [w0 if target == 0 else w1 for target in targets]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Returns a tuple. Element 0 is the Tensor of indices. Element 1 is the Tensor of audio feas.
        # Element 2 is the target
        samp = self.samples[idx]
        tensor = samp[0]
        audio_feas = samp[1]
        target = samp[2]

        if self.unk_noise_prob > 0.0:
            return (self.noisify(tensor),audio_feas,target)
        else:
            return samp

    def noisify(self,item):

        for i in range(len(item)):
            if random.random() <= self.unk_noise_prob:
                item[i] = self.unk_index

        return item


def collater(batch):
    """
    Returns a padded sequence
    """
    xs = [item[0] for item in batch]
    x_feas_s = [item[1] for item in batch]
    ys = [item[2] for item in batch]
    src_lengths = [len(x) for x in xs]

    X = nn.utils.rnn.pad_sequence(xs,batch_first=True)
    X_feas = nn.utils.rnn.pad_sequence(x_feas_s,batch_first=True)
    Y = torch.LongTensor(ys)

    return X, X_feas, src_lengths, Y
