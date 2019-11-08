import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleRNN(nn.Module):

    def __init__(self,args,dictionary):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.GRU(batch_first=True, input_size=args.embedding_size,
                          hidden_size=args.rnn_layer_size)

        self.linear1 = nn.Linear(args.rnn_layer_size, args.n_classes,bias=False)

        self.embedding = nn.Embedding(
            num_embeddings=len(dictionary),
            embedding_dim=args.embedding_size,
            padding_idx=dictionary.pad_index,
        )

    def forward(self, x, src_lengths, h0=None):
        x = self.embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x, src_lengths, batch_first=True, enforce_sorted=False)
        x, hn = self.rnn(x,h0)
        x, lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, padding_value=0.0)
        x = self.linear1(x)
        return x, lengths, hn
