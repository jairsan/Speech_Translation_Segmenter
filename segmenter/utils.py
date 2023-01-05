from typing import Tuple, Any
from argparse import Namespace

import torch

from segmenter.models.rnn_ff_text_model import RNNFFTextModel
from segmenter.models.simple_rnn_text_model import SimpleRNNTextModel
from segmenter.models.rnn_ff_audio_text_model import RNNFFAudioTextModel
from segmenter.models.rnn_ff_audio_text_feas_copy_model import RNNFFAudioTextFeasCopyModel
from segmenter.models.bert_text_model import BERTTextModel
from segmenter.models.xlm_roberta_text_model import XLMRobertaTextModel


def model_picker(args) -> Tuple[Any, bool]:
    """Given a model name, returns the corresponding class and wheter the model requires a vocab to be passed"""
    if args.model_architecture == RNNFFTextModel.name:
        return RNNFFTextModel, True
    elif args.model_architecture == SimpleRNNTextModel.name:
        return SimpleRNNTextModel, True
    elif args.model_architecture == RNNFFAudioTextModel.name:
        return RNNFFAudioTextModel, False
    elif args.model_architecture == RNNFFAudioTextFeasCopyModel.name:
        return RNNFFAudioTextFeasCopyModel, False
    elif args.model_architecture == BERTTextModel.name:
        return BERTTextModel, False
    elif args.model_architecture == XLMRobertaTextModel.name:
        return XLMRobertaTextModel, False


def load_text_model(text_model_path: str):

    checkpoint = torch.load(text_model_path)

    saved_model_args = checkpoint['args']

    vocabulary = checkpoint['vocabulary']
    
    model_class, needs_vocab = model_picker(saved_model_args)
    if needs_vocab:
        model = model_class(saved_model_args, vocabulary)
    else:
        model = model_class(saved_model_args)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, vocabulary, saved_model_args
