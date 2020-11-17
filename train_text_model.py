from segmenter import text_dataset,arguments,vocab
from segmenter.models.simple_rnn_text_model import SimpleRNNTextModel
from segmenter.models.rnn_ff_text_model import SimpleRNNFFTextModel

import argparse
import torch
import torch.utils.data as data
import torch.optim as optim
import time, os, math
import numpy as np
import random

from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support,classification_report


if __name__ == "__main__":
    start_prep = time.time()

    parser = argparse.ArgumentParser()
    arguments.add_train_arguments(parser)
    arguments.add_model_arguments(parser)

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    os.environ['PYTHONHASHSEED']=str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    vocabulary = vocab.VocabDictionary()
    vocabulary.create_from_count_file(args.vocabulary)

    train_dataset = text_dataset.SegmentationTextDataset(args.train_corpus, vocabulary, args.min_split_samples_batch_ratio, args.unk_noise_prob)

    if args.min_split_samples_batch_ratio > 0.0:
        sampler = data.WeightedRandomSampler(train_dataset.weights, len(train_dataset.weights))

        train_dataloader = data.DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size,
                                           drop_last=True,
                                           collate_fn=text_dataset.collater, sampler=sampler)
    else:
        train_dataloader = data.DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                           collate_fn=text_dataset.collater)

    dev_dataset = text_dataset.SegmentationTextDataset(args.dev_corpus, vocabulary)
    dev_dataloader = data.DataLoader(dev_dataset, num_workers=0, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                     collate_fn=text_dataset.collater)

    if args.model_architecture == "ff_text":
        model = SimpleRNNFFTextModel(args, vocabulary).to(device)
    else:
        model = SimpleRNNTextModel(args, vocabulary).to(device)

    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

    if args.lr_schedule == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_reduce_factor,
                                                               patience=args.lr_reduce_patience, verbose=True)


    loss_weights = torch.Tensor([1.0,args.split_weight]).to(device)
    loss = torch.nn.CrossEntropyLoss(weight=loss_weights)


    end_prep = time.time()
    print("Model preparation took: ",str(end_prep - start_prep), " seconds.")

    best_result = -math.inf
    best_epoch = -math.inf
    for epoch in range(1, args.epochs):
        optimizer.zero_grad()

        epoch_cost = 0

        update = 0

        model.train()
        for x, src_lengths, y in train_dataloader:

            x, y = x.to(device), y.to(device)
            model_output,lengths, hn = model.forward(x, src_lengths)

            results = model.get_sentence_prediction(model_output,lengths, device)

            cost = loss(results, y)
            
            cost.backward()

            epoch_cost += cost.detach().cpu().numpy()
            update+=1
            if update % args.log_every == 0:
                print("Epoch ", epoch, ", update ", update, "/",len(train_dataloader),", cost: ", cost.detach().cpu().numpy(),sep="")
            optimizer.step()
            optimizer.zero_grad()
        print("Epoch ", epoch, ", train cost/batch: ", epoch_cost / len(train_dataloader), sep="")

        with torch.no_grad():
            predicted_l = []
            true_l = []
            epoch_cost = 0
            model.eval()
            for x, src_lengths, y in dev_dataloader:

                x, y = x.to(device), y.to(device)
                model_output, lengths, hn = model.forward(x, src_lengths)

                results = model.get_sentence_prediction(model_output, lengths, device)
                yhat = torch.argmax(results,dim=1)


                predicted_l.extend(yhat.detach().cpu().numpy().tolist())
                true_l.extend(y.detach().cpu().numpy().tolist())

                cost = loss(results, y)
                epoch_cost += cost.detach().cpu().numpy()

            print("Epoch ", epoch, ", dev cost/batch: ", epoch_cost / len(dev_dataloader), sep="")
            print("Dev Accuracy:", accuracy_score(true_l, predicted_l))
            precision, recall, f1, _ = precision_recall_fscore_support(true_l, predicted_l, average='macro')
            print("Dev precision, Recall, F1 (Macro): ", precision, recall, f1)
            print(classification_report(true_l, predicted_l))


            if args.lr_schedule == "reduce_on_plateau":
                scheduler.step(f1)

            if f1 > best_result:
                if not os.path.exists(args.output_folder):
                    os.makedirs(args.output_folder)
                best_result = f1
                best_epoch = epoch
                torch.save({
                    'version': "0.2",
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'vocabulary': vocabulary,
                    'args' : args
                }, args.output_folder + "/model.best.pt", pickle_protocol=4)



        if args.checkpoint_interval > 0 and epoch % args.checkpoint_interval == 0:
            if not os.path.exists(args.output_folder):
                os.makedirs(args.output_folder)
            torch.save({
                'version': "0.2",
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocabulary': vocabulary,
                'args' : args
            }, args.output_folder + "/model."+str(epoch)+".pt", pickle_protocol=4)

    print("Training finished.")
    print("Best checkpoint F1:",best_result, ", achieved at epoch:", best_epoch)
