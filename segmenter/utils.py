import torch

from segmenter.models.rnn_ff_text_model import SimpleRNNFFTextModel
from segmenter.models.simple_rnn_text_model import SimpleRNNTextModel


def load_text_model(args):

    checkpoint = torch.load(args.text_model_path)

    saved_model_args = checkpoint['args']

    vocabulary = checkpoint['vocabulary']

    if saved_model_args.model_architecture == "ff_text":
        model = SimpleRNNFFTextModel(saved_model_args, vocabulary)
    else:
        model = SimpleRNNTextModel(saved_model_args, vocabulary)

    model.load_state_dict(checkpoint['model_state_dict'])

    return model, vocabulary, saved_model_args
