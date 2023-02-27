import datasets.utils.logging

from segmenter import arguments, vocab, utils
from segmenter.huggingface_dataset import get_datasets
from segmenter.utils import load_text_model
from segmenter.models.segmenter_model import FeatureExtractorSegmenterModel, HuggingFaceSegmenterModel

import argparse
import torch
import torch.utils.data as data
import torch.optim as optim
import time
import os
import math
import numpy as np
import random
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import default_data_collator, DataCollatorWithPadding

logger = logging.getLogger(__name__)

def save_model(model, args, vocabulary, model_str):
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocabulary': vocabulary,
        'args': args
    }, args.output_folder + "/model." + model_str + ".pt", pickle_protocol=4)


def eval_model(dev_dataloader, model, epoch, update):
    with torch.no_grad():
        predicted_l = []
        true_l = []
        epoch_cost = 0
        model.eval()
        for batch in dev_dataloader:
            y = batch["labels"].to(device)
            model_output = model.forward(batch, device)

            results = model.get_sentence_prediction(model_output)
            yhat = torch.argmax(results, dim=1)

            predicted_l.extend(yhat.detach().cpu().numpy().tolist())
            true_l.extend(y.detach().cpu().numpy().tolist())

            cost = loss(results, y)
            epoch_cost += cost.detach().cpu().numpy()

        logger.info(f"Epoch  {epoch}, update  {update},  dev cost/batch:  {epoch_cost / len(dev_dataloader)}")
        logger.info(f"Dev Accuracy:{accuracy_score(true_l, predicted_l)}")
        precision, recall, f1, _ = precision_recall_fscore_support(true_l, predicted_l, average='macro')
        logger.info(f"Dev precision, Recall, F1 (Macro): {precision} {recall} {f1}")
        logger.info(classification_report(true_l, predicted_l))

        return f1


if __name__ == "__main__":
    start_prep = time.time()

    logging.basicConfig(level=logging.INFO)
    datasets.utils.disable_progress_bar()

    parser = argparse.ArgumentParser()
    arguments.add_train_arguments(parser)
    arguments.add_general_arguments(parser)

    known_args, unknown_args = parser.parse_known_args()

    model_arguments_parser = argparse.ArgumentParser()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.environ['PYTHONHASHSEED'] = str(known_args.seed)
    random.seed(known_args.seed)
    np.random.seed(known_args.seed)
    torch.manual_seed(known_args.seed)

    dataset_workers = 2

    vocabulary = vocab.VocabDictionary()
    vocabulary.create_from_count_file(path=known_args.vocabulary,
                                      vocab_max_size=known_args.vocabulary_max_size,
                                      word_min_frequency=known_args.vocabulary_min_frequency)

    model_class, requires_vocab = utils.model_picker(known_args)
    model_class.add_model_args(model_arguments_parser)
    model_specific_args = model_arguments_parser.parse_args(unknown_args)
    args = argparse.Namespace(**vars(known_args), **vars(model_specific_args))

    if hasattr(args, "frozen_text_model_path"):
        text_model, text_vocab, saved_model_args = load_text_model(args.frozen_text_model_path)
        text_model = text_model.to(device)
        assert isinstance(text_model, FeatureExtractorSegmenterModel)

        model = model_class(args, text_model, saved_model_args.rnn_layer_size).to(device)

        # overwrite the vocabulary so that it uses the one from the stored model
        vocabulary = text_vocab
    else:
        if requires_vocab:
            model = model_class(args, vocabulary).to(device)
        else:
            model = model_class(args).to(device)

    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.adam_b1, args.adam_b2), eps=args.adam_eps)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

    if args.lr_schedule == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_reduce_factor,
                                                               patience=args.lr_reduce_patience, verbose=True)
    loss = torch.nn.CrossEntropyLoss()

    best_result = -math.inf
    best_epoch = -math.inf

    if hasattr(args, "train_audio_features_corpus") and hasattr(args, "dev_audio_features_corpus"):
        hf_datasets = get_datasets(train_text_file=args.train_corpus, dev_text_file=args.dev_corpus,
                                   temperature=args.sampling_temperature,
                                   train_audio_features_file=args.train_audio_features_corpus,
                                   dev_audio_features_file=args.dev_audio_features_corpus, num_classes=args.num_classes)
    else:
        hf_datasets = get_datasets(train_text_file=args.train_corpus, dev_text_file=args.dev_corpus,
                                   temperature=args.sampling_temperature, num_classes=args.n_classes)

    if isinstance(model, HuggingFaceSegmenterModel):
        hf_datasets = hf_datasets.map(model.apply_tokenizer).remove_columns(column_names="words")
        collater = DataCollatorWithPadding(tokenizer=model.tokenizer)
    else:
        def text_to_idx(sample):
            return {"idx": [vocabulary.get_index(token) for token in sample["words"].split()]}


        hf_datasets = hf_datasets.map(text_to_idx)
        collater = default_data_collator

    train_dataset = hf_datasets["train"]
    dev_dataset = hf_datasets["dev"]

    train_dataloader = data.DataLoader(train_dataset, num_workers=dataset_workers, batch_size=args.batch_size,
                                       shuffle=True, collate_fn=collater)

    dev_dataloader = data.DataLoader(dev_dataset, num_workers=dataset_workers, batch_size=args.batch_size,
                                     shuffle=False, drop_last=False, collate_fn=collater)

    end_prep = time.time()
    logger.info(f"Training preparation took {str(end_prep - start_prep)} seconds.")

    if args.amp:
        if not torch.cuda.is_available():
            raise Exception
        scaler = torch.cuda.amp.GradScaler()

    update = 0
    for epoch in range(1, args.epochs):

        last_time = time.time()

        epoch_cost = 0

        optimizer.zero_grad()
        model.train()

        for batch in train_dataloader:

            with torch.cuda.amp.autocast(enabled=args.amp):
                model_output = model.forward(batch, device)
                results = model.get_sentence_prediction(model_output)
                # Scale loss by gradient accumulation steps, so that the mean is properly computed over the virtual
                # batch.
                cost = loss(results, batch["labels"].to(device)) / args.gradient_accumulation

            if args.amp:
                scaler.scale(cost).backward()
            else:
                cost.backward()

            epoch_cost += cost.detach().cpu().numpy()
            update += 1

            if update % args.gradient_accumulation == 0:
                if args.amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()

            if update % args.log_every == 0:
                curr_time = time.time()
                diff_t = curr_time - last_time
                last_time = curr_time
                logger.debug(f"Epoch {epoch} update {update} cost: {cost.detach().cpu().numpy()}, batches per second: "
                             f"{str(float(args.log_every) / float(diff_t))}")

            if args.checkpoint_every_n_updates is not None and (update + 1) % args.checkpoint_every_n_updates == 0:
                _ = eval_model(dev_dataloader, model, epoch, update)
                save_model(model, args, vocabulary, str(epoch) + "." + str(update))

        logger.info(f"Epoch {epoch}, train cost/batch: {epoch_cost / len(train_dataloader)}")

        f1 = eval_model(dev_dataloader, model, epoch, update)

        if args.lr_schedule == "reduce_on_plateau":
            scheduler.step(f1)

        if f1 > best_result:
            best_result = f1
            best_epoch = epoch

            save_model(model, args, vocabulary, "best")

        if args.checkpoint_interval > 0 and epoch % args.checkpoint_interval == 0:
            save_model(model, args, vocabulary, str(epoch))

    logger.info("Training finished.")
    logger.info(f"Best checkpoint F1: {best_result}, achieved at epoch: {best_epoch}")
