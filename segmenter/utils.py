import torch

from segmenter.models.rnn_ff_text_model import SimpleRNNFFTextModel
from segmenter.models.simple_rnn_text_model import SimpleRNNTextModel
from segmenter.models.rnn_ff_audio_text_model import RNNFFAudioTextModel


def load_text_model(args):

    checkpoint = torch.load(args.model_path)

    saved_model_args = checkpoint['args']

    vocabulary = checkpoint['vocabulary']

    if saved_model_args.model_architecture == "ff_text":
        model = SimpleRNNFFTextModel(saved_model_args, vocabulary)
    else:
        model = SimpleRNNTextModel(saved_model_args, vocabulary)

    model.load_state_dict(checkpoint['model_state_dict'])

    return model, vocabulary, saved_model_args

def load_text_and_audio_model(args):

    checkpoint = torch.load(args.model_path)

    text_model_args = checkpoint['text_model_args']

    audio_model_args = checkpoint['args']

    vocabulary = checkpoint['vocabulary']

    audio_model = RNNFFAudioTextModel(audio_model_args, text_model_args.rnn_layer_size)

    audio_model.load_state_dict(checkpoint['model_state_dict'])

    if text_model_args.model_architecture == "ff_text":
        text_model = SimpleRNNFFTextModel(text_model_args, vocabulary)
    else:
        text_model = SimpleRNNTextModel(text_model_args, vocabulary)

    text_model.load_state_dict(checkpoint['text_model_state_dict'])

    return text_model, vocabulary, text_model_args, audio_model, audio_model_args

