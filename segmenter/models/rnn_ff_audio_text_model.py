import torch
import torch.nn as nn
from segmenter.model_arguments import add_audio_train_arguments, add_common_arguments


class RNNFFAudioTextModel(nn.Module):
    name: str = "rnn-ff-audio"

    @staticmethod
    def add_model_args(parser):
        add_audio_train_arguments(parser)
        add_common_arguments(parser)
        parser.add_argument("--audio_rnn_layer_size", type=int, default=8,
                            help="Hidden size of the RNN layers for the audio encoder")

    def __init__(self, args, text_model, text_features_size):
        super(RNNFFAudioTextModel, self).__init__()

        self.args = args

        self.window_size = args.sample_window_size

        # We put dropout in case we add multiple layers in the future
        self.rnn = nn.GRU(batch_first=True, input_size=args.audio_features_size,
                          hidden_size=args.audio_rnn_layer_size,dropout=args.dropout)

        self.post_rnn_dropout = torch.nn.Dropout(p=args.dropout)

        l = [nn.Linear( (args.audio_rnn_layer_size + text_features_size) * (args.sample_window_size + 1), args.feedforward_size, bias=True),
             nn.ReLU(), nn.Dropout(p=args.dropout)]

        for i in range(1,self.args.feedforward_layers):
            l.append(nn.Linear(args.feedforward_size, args.feedforward_size, bias=True))
            l.append(nn.ReLU())
            l.append(nn.Dropout(p=args.dropout))

        self.feedforward = nn.ModuleList(l)

        self.output = nn.Linear(args.feedforward_size, args.n_classes, bias=True)

        self.text_model = text_model

    def forward(self, batch, device):
        text_feas = self.text_model(batch, device)

        x_feas = batch["audio_features"].to(device)

        x_feas = nn.utils.rnn.pack_sequence(x_feas, enforce_sorted=False)
        x_feas, _ = self.rnn(x_feas)
        x_feas, _ = nn.utils.rnn.pad_packed_sequence(x_feas, batch_first=True, padding_value=0.0)

        # Concatenate on the last dimension
        x_comb = torch.cat((x_feas,text_feas),dim=2)

        # Keep only the ones we want
        x_comb_s = x_comb[:, -(self.window_size+1):, :]

        # Now flatten them
        x_ff = torch.flatten(x_comb_s, start_dim=1)

        for layer in self.feedforward:
            x_ff = layer(x_ff)

        x_ff = self.output(x_ff)

        return x_ff

    def get_sentence_prediction(self, model_output):

        return model_output
