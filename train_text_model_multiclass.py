from segmenter import arguments, vocab
from segmenter.huggingface_dataset import get_text_datasets
from segmenter.models.simple_rnn_text_model import SimpleRNNTextModel
from segmenter.models.rnn_ff_text_model import RNNFFTextModel
from segmenter.models.bert_text_model import BERTTextModel
from segmenter.models.xlm_roberta_text_model import XLMRobertaTextModel

import argparse
import torch
import torch.utils.data as data
import torch.optim as optim
import time, os, math
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report
from transformers import default_data_collator

def save_model(model, args, optimizer, vocabulary, model_str, epoch, train_chunk):
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    torch.save({
        'version': "0.4",
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocabulary': vocabulary,
        'epoch': epoch,
        'train_chunk': train_chunk,
        'args': args
    }, args.output_folder + "/model." + model_str + ".pt", pickle_protocol=4)


def eval_model(args, dev_dataloader, model):
    with torch.no_grad():
        predicted_l = []
        true_l = []
        epoch_cost = 0
        model.eval()
        for x, src_lengths, y in dev_dataloader:
            if args.transformer_model_name is not None:
                # Transformer model does this internally
                y = y.to(device)
            else:
                x, y = x.to(device), y.to(device)

            model_output, lengths, hn = model.forward(x, src_lengths, device)

            results = model.get_sentence_prediction(model_output, lengths, device)
            yhat = torch.argmax(results, dim=1)

            predicted_l.extend(yhat.detach().cpu().numpy().tolist())
            true_l.extend(y.detach().cpu().numpy().tolist())

            cost = loss(results, y)
            epoch_cost += cost.detach().cpu().numpy()

        print("Epoch ", epoch, "chunk ", str(i), ", dev cost/batch: ", epoch_cost / len(dev_dataloader), sep="")
        print("Dev Accuracy:", accuracy_score(true_l, predicted_l))
        precision, recall, f1, _ = precision_recall_fscore_support(true_l, predicted_l, average='macro')
        print("Dev precision, Recall, F1 (Macro): ", precision, recall, f1)
        print(classification_report(true_l, predicted_l))

        return f1


if __name__ == "__main__":
    start_prep = time.time()

    parser = argparse.ArgumentParser()
    arguments.add_train_arguments(parser)
    arguments.add_general_arguments(parser)

    known_args, unknown_args = parser.parse_known_args()

    model_arguments_parser = argparse.ArgumentParser()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    os.environ['PYTHONHASHSEED'] = str(known_args.seed)
    random.seed(known_args.seed)
    np.random.seed(known_args.seed)
    torch.manual_seed(known_args.seed)

    dataset_workers = 2

    if known_args.transformer_model_name is None:
        vocabulary = vocab.VocabDictionary()
        vocabulary.create_from_count_file(known_args.vocabulary, known_args.vocabulary_max_size)

    hf_datasets = get_text_datasets(train_text_file=known_args.train_corpus, dev_text_file=known_args.dev_corpus,
                                    temperature=known_args.sampling_temperature)

    if known_args.model_architecture != BERTTextModel and known_args.model_architecture != XLMRobertaTextModel and known_args.transformer_model_name is not None:
        # Cant use transformer model name if we are not using transformer
        raise Exception

    if known_args.model_architecture == RNNFFTextModel.name:
        RNNFFTextModel.add_model_args(model_arguments_parser)
        model_specific_args = model_arguments_parser.parse_args(unknown_args)
        args = argparse.Namespace(**vars(known_args), **vars(model_specific_args))
        print(f" args {args}")
        model = RNNFFTextModel(args, vocabulary).to(device)
    elif known_args.model_architecture == SimpleRNNTextModel.name:
        SimpleRNNTextModel.add_model_args(model_arguments_parser)
        model_specific_args = model_arguments_parser.parse_args(unknown_args)
        args = argparse.Namespace(**vars(known_args), **vars(model_specific_args))
        print(f" args2 {args}")
        model = SimpleRNNTextModel(args, vocabulary).to(device)
    else:
        raise Exception

    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.adam_b1, args.adam_b2), eps=args.adam_eps)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

    if args.lr_schedule == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_reduce_factor,
                                                               patience=args.lr_reduce_patience, verbose=True)
    loss = torch.nn.CrossEntropyLoss()

    end_prep = time.time()
    print("Model preparation took: ", str(end_prep - start_prep), " seconds.")

    best_result = -math.inf
    best_epoch = -math.inf

    if args.transformer_model_name is not None:
        pass
    else:
        def text_to_idx(sample):
            return {"idx": [vocabulary.get_index(token) for token in sample["words"].split()]}


        hf_datasets = hf_datasets.map(text_to_idx).remove_columns(column_names="words")

    train_dataset = hf_datasets["train"]
    dev_dataset = hf_datasets["dev"]

    train_dataloader = data.DataLoader(train_dataset, num_workers=dataset_workers, batch_size=args.batch_size,
                                       shuffle=True, collate_fn=default_data_collator)

    dev_dataloader = data.DataLoader(dev_dataset, num_workers=dataset_workers, batch_size=args.batch_size,
                                     shuffle=False, drop_last=False, collate_fn=default_data_collator)

    for epoch in range(1, args.epochs):

        last_time = time.time()

        optimizer.zero_grad()

        epoch_cost = 0

        update = 0

        model.train()

        optimizer.zero_grad()
        model.train()

        for i, batch in enumerate(train_dataloader):

            model_output = model.forward(batch, device)

            results = model.get_sentence_prediction(model_output)

            cost = loss(results, batch["labels"])

            cost.backward()

            epoch_cost += cost.detach().cpu().numpy()
            update += 1

            if update % args.log_every == 0:
                curr_time = time.time()
                diff_t = curr_time - last_time
                last_time = curr_time
                print("Epoch ", epoch, " chunk ", i + 1, ", update ", update, ", cost: ",
                      cost.detach().cpu().numpy(), ", batches per second: ",
                      str(float(args.log_every) / float(diff_t)), sep="")
            if update % args.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
        print("Epoch ", epoch, ", train cost/batch: ", epoch_cost / len(train_dataloader), sep="")

        if args.checkpoint_every_n_chunks is not None and (i + 1) % args.checkpoint_every_n_chunks == 0:
            _ = eval_model(args, dev_dataloader, model)
            save_model(model, args, optimizer, vocabulary, str(epoch) + "." + str(i), epoch, i)

        f1 = eval_model(args, dev_dataloader, model)

        if args.lr_schedule == "reduce_on_plateau":
            scheduler.step(f1)

        if f1 > best_result:
            best_result = f1
            best_epoch = epoch

            save_model(model, args, optimizer, vocabulary, "best", epoch, None)

        if args.checkpoint_interval > 0 and epoch % args.checkpoint_interval == 0:
            save_model(model, args, optimizer, vocabulary, str(epoch), epoch, None)

    print("Training finished.")
    print("Best checkpoint F1:", best_result, ", achieved at epoch:", best_epoch)
