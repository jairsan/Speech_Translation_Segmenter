from segmenter import text_audio_dataset,arguments,vocab
from segmenter.models.rnn_ff_audio_text_model import RNNFFAudioTextModel
from segmenter import utils
import argparse
import torch
import torch.utils.data as data
import torch.optim as optim
import time, os, math
import numpy as np

from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support,classification_report


if __name__ == "__main__":
    start_prep = time.time()

    parser = argparse.ArgumentParser()
    arguments.add_train_arguments(parser)
    arguments.add_model_arguments(parser)
    arguments.add_audio_train_arguments(parser)

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    vocabulary = vocab.VocabDictionary()
    vocabulary.create_from_count_file(args.vocabulary)

    train_dataset = text_audio_dataset.SegmentationTextAudioDataset(args.train_corpus, args.train_audio_features_corpus,
                                                                    vocabulary, args.min_split_samples_batch_ratio, args.unk_noise_prob)

    if args.min_split_samples_batch_ratio > 0.0:
        sampler = data.WeightedRandomSampler(train_dataset.weights, len(train_dataset.weights))

        train_dataloader = data.DataLoader(train_dataset, num_workers=3, batch_size=args.batch_size,
                                           drop_last=True,
                                           collate_fn=text_audio_dataset.collater, sampler=sampler)
    else:
        train_dataloader = data.DataLoader(train_dataset, num_workers=3, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                           collate_fn=text_audio_dataset.collater)

    dev_dataset = text_audio_dataset.SegmentationTextAudioDataset(args.dev_corpus, args.dev_audio_features_corpus, vocabulary)
    dev_dataloader = data.DataLoader(dev_dataset, num_workers=3, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                     collate_fn=text_audio_dataset.collater)

    text_model,vocabulary, saved_model_args = utils.load_text_model(args)
    text_model.to(device)
    text_model.eval()

    text_features_size = saved_model_args.rnn_layer_size

    model = RNNFFAudioTextModel(args,text_features_size)

    model.to(device)

    for epoch in range(1, args.epochs):

        epoch_cost = 0

        update = 0

        for x_text, x_feas_a,src_lengths, y in train_dataloader:

            x_text, x_feas_a, src_lengths, y = x_text.to(device), x_feas_a.to(device), src_lengths, y.to(device)

            with torch.no_grad():
                text_feas, _, _ = text_model.extract_features(x_text,src_lengths)

            prediction, _, _ = model.forward(x_feas_a, text_feas, src_lengths)

            print(prediction)

