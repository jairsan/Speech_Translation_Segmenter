from segmenter import text_dataset,text_dataset_multiclass,raw_text_dataset,arguments,vocab
from segmenter.models.simple_rnn_text_model import SimpleRNNTextModel
from segmenter.models.rnn_ff_text_model import SimpleRNNFFTextModel
from segmenter.models.bert_text_model import BERTTextModel
from segmenter.models.xlm_roberta_text_model import XLMRobertaTextModel

 
import argparse
import torch
import torch.utils.data as data
import torch.optim as optim
import time, os, math
import numpy as np
import random
import gpustat
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
    
    if False:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        dataset_workers = 1
    else:
        dataset_workers = 0

    vocabulary = vocab.VocabDictionary()
    vocabulary.create_from_count_file(args.vocabulary)
    classes_vocabulary = vocab.VocabDictionary(include_special=False)
    classes_vocabulary.create_from_count_file(args.classes_vocabulary)

    if args.transformer_architecture != None:
        raise Exception
        train_dataset = raw_text_dataset.SegmentationRawTextDataset(args.train_corpus, args.min_split_samples_batch_ratio, args.unk_noise_prob)
    else:
        train_dataset = text_dataset_multiclass.SegmentationTextDatasetMulticlass(args.train_corpus, vocabulary, classes_vocabulary, args.sampling_temperature)

    if args.samples_per_random_epoch == -1:
        samples_to_draw = len(train_dataset.weights)
    else:
        samples_to_draw = args.samples_per_random_epoch

    sampler = data.WeightedRandomSampler(train_dataset.weights, samples_to_draw)

    train_dataloader = data.DataLoader(train_dataset, num_workers=dataset_workers, batch_size=args.batch_size,
                                       drop_last=True,
                                       collate_fn=train_dataset.collater, sampler=sampler)

    if args.transformer_architecture != None:
        raise Exception
        dev_dataset = raw_text_dataset.SegmentationRawTextDataset(args.dev_corpus)
    else:
        dev_dataset = text_dataset_multiclass.SegmentationTextDatasetMulticlass(args.train_corpus, vocabulary, classes_vocabulary)

    dev_dataloader = data.DataLoader(dev_dataset, num_workers=dataset_workers, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                     collate_fn=train_dataset.collater)

    if args.transformer_architecture !=None:
        archetype, _ = args.transformer_architecture.split(":")
        if archetype == "bert":
            model = BERTTextModel(args).to(device)
        elif archetype == "xlm-roberta":
            model = XLMRobertaTextModel(args).to(device)
        else:
            raise Exception
    elif args.model_architecture == "ff-text":
        model = SimpleRNNFFTextModel(args, vocabulary).to(device)
    else:
        model = SimpleRNNTextModel(args, vocabulary).to(device)

    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.adam_b1, args.adam_b2), eps=args.adam_eps)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

    if args.lr_schedule == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_reduce_factor,
                                                               patience=args.lr_reduce_patience, verbose=True)
    loss = torch.nn.CrossEntropyLoss()


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
            if args.transformer_architecture !=None:
                #Transformer model does this internally
                y = y.to(device)
            else:
                x, y = x.to(device), y.to(device)

            model_output,lengths, hn = model.forward(x, src_lengths, device)

            results = model.get_sentence_prediction(model_output,lengths, device)
 
            cost = loss(results, y)
            
            cost.backward()

            epoch_cost += cost.detach().cpu().numpy()
            update+=1
            if update % args.log_every == 0:
                print("Epoch ", epoch, ", update ", update, "/",len(train_dataloader),", cost: ", cost.detach().cpu().numpy(),sep="")
            if (update % args.gradient_accumulation == 0):
                optimizer.step()
                optimizer.zero_grad()
        print("Epoch ", epoch, ", train cost/batch: ", epoch_cost / len(train_dataloader), sep="")

        with torch.no_grad():
            predicted_l = []
            true_l = []
            epoch_cost = 0
            model.eval()
            for x, src_lengths, y in dev_dataloader:
                if args.transformer_architecture !=None:
                    #Transformer model does this internally
                    y = y.to(device)
                else:
                    x, y = x.to(device), y.to(device)

                model_output, lengths, hn = model.forward(x, src_lengths, device)

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
                    'version': "0.3",
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'vocabulary': vocabulary,
                    'classes_vocabulary': classes_vocabulary,
                    'args' : args
                }, args.output_folder + "/model.best.pt", pickle_protocol=4)



        if args.checkpoint_interval > 0 and epoch % args.checkpoint_interval == 0:
            if not os.path.exists(args.output_folder):
                os.makedirs(args.output_folder)
            torch.save({
                'version': "0.3",
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocabulary': vocabulary,
                'classes_vocabulary' : classes_vocabulary,
                'args' : args
            }, args.output_folder + "/model."+str(epoch)+".pt", pickle_protocol=4)

    print("Training finished.")
    print("Best checkpoint F1:",best_result, ", achieved at epoch:", best_epoch)