import torch
import torch.nn as nn
from segmenter.model_arguments import add_rnn_arguments


class SimpleRNNTextModel(nn.Module):
    name: str = "simple-rnn"

    @staticmethod
    def add_model_args(parser):
        parser = add_rnn_arguments(parser)
        return parser

    def __init__(self,args,dictionary):
        super(SimpleRNNTextModel, self).__init__()

        self.args = args

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

    def forward(self, batch, device):
        x, lengths, hn = self.extract_features(batch, device)
        x = self.linear1(x)
        return x

    def extract_features(self, batch, device):
        """
        Extract the features computed by the model (ignoring the output layer)

        """
        x = batch["idx"]
        x = x.to(device)
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        x = nn.utils.rnn.pack_sequence(x, enforce_sorted=False)
        x, _ = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, padding_value=0.0)
        x = self.post_rnn_dropout(x)

        return x

    def get_sentence_prediction(self, model_output):
        """
        Returns the model output (which depends on the length),
        for each sample in the batch.
        """

        results = model_output[:, -1, :]

        print(f"input: {model_output}")
        print(f"output: {results}")

        exit(0)
        return results
