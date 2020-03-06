from segmenter import text_audio_dataset,arguments,vocab
from segmenter.models.rnn_ff_audio_text_model import RNNFFAudioTextModel
from segmenter.models.rnn_ff_audio_text_feas_copy_model import RNNFFAudioTextFeasCopyModel

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

    if args.model_architecture == "ff-audio-text-copy-feas" :

        model = RNNFFAudioTextFeasCopyModel(args, text_features_size)

    else:
        model = RNNFFAudioTextModel(args,text_features_size)

    model.to(device)


    loss_weights = torch.Tensor([1.0,args.split_weight]).to(device)
    loss = torch.nn.CrossEntropyLoss(weight=loss_weights)

    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

    if args.lr_schedule == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_reduce_factor,
                                                               patience=args.lr_reduce_patience, verbose=True)
    best_result = -math.inf
    best_epoch = -math.inf
    for epoch in range(1, args.epochs):
        optimizer.zero_grad()
        epoch_cost = 0

        update = 0

        model.train()

        for x_text, x_feas_a,src_lengths, y in train_dataloader:

            x_text, x_feas_a, src_lengths, y = x_text.to(device), x_feas_a.to(device), src_lengths, y.to(device)

            with torch.no_grad():
                text_feas, _, _ = text_model.extract_features(x_text,src_lengths)

            prediction, _, _ = model.forward(x_feas_a, text_feas, src_lengths)

            cost = loss(prediction, y)
            cost.backward()

            epoch_cost += cost.detach().cpu().numpy()
            update+=1
            if update % args.log_every == 0:
                print("Epoch ", epoch, ", update ", update, "/",len(train_dataloader),", cost: ", cost.detach().cpu().numpy(),sep="")
            optimizer.step()
            optimizer.zero_grad()



        with torch.no_grad():
            predicted_l = []
            true_l = []
            epoch_cost = 0
            model.eval()
            for x_text, x_feas_a,src_lengths, y in dev_dataloader:


                x_text, x_feas_a, src_lengths, y = x_text.to(device), x_feas_a.to(device), src_lengths, y.to(device)


                text_feas, _, _ = text_model.extract_features(x_text, src_lengths)

                prediction, _, _ = model.forward(x_feas_a, text_feas, src_lengths)

                yhat = torch.argmax(prediction,dim=1)


                predicted_l.extend(yhat.detach().cpu().numpy().tolist())
                true_l.extend(y.detach().cpu().numpy().tolist())

                cost = loss(prediction, y)
                epoch_cost += cost.detach().cpu().numpy()

            print("Epoch ", epoch, ", dev cost/batch: ", epoch_cost / len(dev_dataloader), sep="")
            print("Dev Accuracy:", accuracy_score(true_l, predicted_l))
            precision, recall, f1, _ = precision_recall_fscore_support(true_l, predicted_l, average='macro')
            print("Dev precision, Recall, F1 (Macro): ", precision, recall, f1)
            print(classification_report(true_l, predicted_l))


            if args.lr_schedule == "reduce_on_plateau":
                scheduler.step(f1)

        if args.checkpoint_interval > 0 and epoch % args.checkpoint_interval == 0:
            if not os.path.exists(args.output_folder):
                os.makedirs(args.output_folder)
            torch.save({
                'version': "0.2",
                'model_state_dict': model.state_dict(),
                'text_model_state_dict' : text_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocabulary': vocabulary,
                'args' : args,
                'text_model_args' : saved_model_args
            }, args.output_folder + "/model."+str(epoch)+".pt", pickle_protocol=4)

            if f1 > best_result:
                best_result = f1
                best_epoch = epoch
                torch.save({
                    'version': "0.2",
                    'model_state_dict': model.state_dict(),
                    'text_model_state_dict' : text_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'vocabulary': vocabulary,
                    'args' : args,
                    'text_model_args' : saved_model_args
                }, args.output_folder + "/model.best.pt", pickle_protocol=4)



        print("Epoch ", epoch, ", train cost/batch: ", epoch_cost / len(train_dataloader), sep="")


    print("Training finished.")
    print("Best checkpoint F1:",best_result, ", achieved at epoch:", best_epoch)

