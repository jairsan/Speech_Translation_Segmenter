from typing import Dict
import torch
import torch.nn as nn
from segmenter.model_arguments import add_ff_arguments, add_rnn_arguments
from segmenter.models.segmenter_model import SegmenterTextModel


class RNNFFTextModel(SegmenterTextModel):
    name: str = "rnn-ff-text"

    @staticmethod
    def add_model_args(parser):
        add_rnn_arguments(parser)
        add_ff_arguments(parser)

    def __init__(self, args, dictionary):
        super(RNNFFTextModel, self).__init__()
        self.args = args
        self.embedding = nn.Embedding(
            num_embeddings=len(dictionary),
            embedding_dim=args.embedding_size,
            padding_idx=dictionary.pad_index,
        )

        self.window_size = args.sample_window_size

        self.embedding_dropout = torch.nn.Dropout(p=args.dropout)

        # We put dropout in case we add multiple layers in the future
        self.rnn = nn.GRU(batch_first=True, input_size=args.embedding_size,
                          hidden_size=args.rnn_layer_size,dropout=args.dropout)

        self.post_rnn_dropout = torch.nn.Dropout(p=args.dropout)

        l = [nn.Linear(args.rnn_layer_size * (args.sample_window_size + 1), args.feedforward_size, bias=True), nn.ReLU(), nn.Dropout(p=args.dropout)]

        for i in range(1, self.args.feedforward_layers):
            l.append(nn.Linear(args.feedforward_size, args.feedforward_size, bias=True))
            l.append(nn.ReLU())
            l.append(nn.Dropout(p=args.dropout))

        self.feedforward = nn.ModuleList(l)

        self.output = nn.Linear(args.feedforward_size, args.n_classes, bias=True)

    def forward(self, batch: Dict, device: torch.device):
        x = self.extract_features(batch, device)

        x_sel = x[:, -(self.window_size+1):, :]

        x_ff = torch.flatten(x_sel, start_dim=1)

        for layer in self.feedforward:
            x_ff = layer(x_ff)

        x = self.output(x_ff)

        return x

    def extract_features(self, batch: Dict, device: torch.device):
        x = batch["idx"]
        x = x.to(device)
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        x = nn.utils.rnn.pack_sequence(x, enforce_sorted=False)
        x, hn = self.rnn(x)
        x, lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, padding_value=0.0)
        x = self.post_rnn_dropout(x)

        return x

    def get_sentence_prediction(self, model_output: torch.Tensor) -> torch.Tensor:

        return model_output
