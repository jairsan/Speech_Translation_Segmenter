import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNFFAudioTextFeasCopyModel(nn.Module):

    def __init__(self,args,text_features_size):
        super(RNNFFAudioTextFeasCopyModel, self).__init__()

        self.args = args

        self.window_size = args.sample_window_size

        l = [nn.Linear( (args.embedding_size + text_features_size) * (args.sample_window_size + 1), args.feedforward_size, bias=True),
             nn.ReLU(), nn.Dropout(p=args.dropout)]

        for i in range(1,self.args.feedforward_layers):
            l.append(nn.Linear(args.feedforward_size, args.feedforward_size, bias=True))
            l.append(nn.ReLU())
            l.append(nn.Dropout(p=args.dropout))

        self.feedforward = nn.ModuleList(l)

        self.output = nn.Linear(args.feedforward_size, args.n_classes, bias=True)

    def forward(self, x_feas, text_feas, src_lengths, h0=None):

        x_feas = nn.utils.rnn.pack_padded_sequence(x_feas, src_lengths, batch_first=True, enforce_sorted=False)
        x_feas, lengths = nn.utils.rnn.pad_packed_sequence(x_feas, batch_first=True, padding_value=0.0)

        # Concatenate on the last dimension
        x_comb = torch.cat((x_feas,text_feas),dim=2)

        # Keep only the ones we want
        x_comb_s = x_comb[:, -(self.window_size+1):, :]

        # Now flatten them
        x_ff = torch.flatten(x_comb_s, start_dim=1)

        for layer in self.feedforward:
            x_ff = layer(x_ff)

        x_ff = self.output(x_ff)

        return x_ff, lengths, None

    def get_sentence_prediction(self,model_output,lengths, device):
        """
        Returns the model output (which depends on the length),
        for each sample in the batch.

        select = lengths - torch.ones(lengths.shape, dtype=torch.long)

        select = select.to(device)

        indices = torch.unsqueeze(select, 1)
        indices = torch.unsqueeze(indices, 2).repeat(1, 1, 2)
        results = torch.gather(model_output, 1, indices).squeeze(1)
        """
        return model_output
