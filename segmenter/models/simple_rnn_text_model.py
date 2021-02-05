import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleRNNTextModel(nn.Module):

    def __init__(self,args,dictionary):
        super(SimpleRNNTextModel, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=len(dictionary),
            embedding_dim=args.embedding_size,
            padding_idx=dictionary.pad_index,
        )

        self.embedding_dropout = torch.nn.Dropout(p=args.dropout)

        # We put dropout in case we add multiple layers in the future
        self.rnn = nn.GRU(batch_first=True, input_size=args.embedding_size,
                          hidden_size=args.rnn_layer_size,dropout=args.dropout)

        self.post_rnn_dropout = torch.nn.Dropout(p=args.dropout)

        self.linear1 = nn.Linear(args.rnn_layer_size, args.n_classes, bias=True)

    def forward(self, x, src_lengths, device=None, h0=None):

        x, lengths, hn = self.extract_features(x, src_lengths, h0)
        x = self.linear1(x)
        return x, lengths, hn

    def extract_features(self, x, src_lengths, h0=None):
        """
        Extract the features computed by the model (ignoring the output layer)

        """
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        x = nn.utils.rnn.pack_padded_sequence(x, src_lengths, batch_first=True, enforce_sorted=False)
        x, hn = self.rnn(x,h0)
        x, lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, padding_value=0.0)
        x = self.post_rnn_dropout(x)

        return x, lengths, hn

    def get_sentence_prediction(self,model_output,lengths, device):
        """
        Returns the model output (which depends on the length),
        for each sample in the batch.
        """
        select = lengths - torch.ones(lengths.shape, dtype=torch.long)

        select = select.to(device)

        indices = torch.unsqueeze(select, 1)
        indices = torch.unsqueeze(indices, 2).repeat(1, 1, 2)
        results = torch.gather(model_output, 1, indices).squeeze(1)

        return results
