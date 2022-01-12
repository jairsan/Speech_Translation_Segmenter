from segmenter import text_dataset,text_dataset_multiclass,raw_text_dataset,raw_text_dataset_multiclass,arguments,vocab
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
    
    dataset_workers = 2

    vocabulary = vocab.VocabDictionary()
    if args.transformer_architecture == None:
        vocabulary.create_from_count_file(args.vocabulary, args.vocabulary_max_size)
    classes_vocabulary = vocab.VocabDictionary(include_special=False)
    classes_vocabulary.create_from_count_file(args.classes_vocabulary)

    if args.transformer_architecture != None:
        dev_dataset = raw_text_dataset_multiclass.SegmentationRawTextDatasetMulticlass(args.dev_corpus, classes_vocabulary)
    else:
        dev_dataset = text_dataset_multiclass.SegmentationTextDatasetMulticlass(args.dev_corpus, vocabulary, classes_vocabulary)

    dev_dataloader = data.DataLoader(dev_dataset, num_workers=dataset_workers, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                     collate_fn=dev_dataset.collater)

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

    train_chunks = []
    if args.use_train_chunks_from_list != None:
        with open(args.use_train_chunks_from_list) as filfil:
            for line in filfil:
                file_path = line.strip()
                train_chunks.append(file_path)
    else:
        train_chunks.append(args.train_corpus)

    for epoch in range(1, args.epochs):


        optimizer.zero_grad()

        epoch_cost = 0

        update = 0

        model.train()

        #Iterate over all chunks
        for i in range(len(train_chunks)):

            optimizer.zero_grad()
            model.train()

            if args.transformer_architecture != None:
                train_dataset = raw_text_dataset_multiclass.SegmentationRawTextDatasetMulticlass(train_chunks[i], classes_vocabulary, args.sampling_temperature)
            else:
                train_dataset = text_dataset_multiclass.SegmentationTextDatasetMulticlass(train_chunks[i], vocabulary, classes_vocabulary, args.sampling_temperature)

            sampler = data.WeightedRandomSampler(train_dataset.weights, len(train_dataset.weights))

            train_dataloader = data.DataLoader(train_dataset, num_workers=dataset_workers, batch_size=args.batch_size,
                                       drop_last=True,
                                       collate_fn=train_dataset.collater, sampler=sampler)

            for x, src_lengths, y in train_dataloader:
                if args.transformer_architecture !=None:
                    #Transformer model does this internally
                    y = y.to(device)
                else:
                    x, y = x.to(device), y.to(device)

                model_output,lengths, hn = model.forward(x, src_lengths, device)

                results = model.get_sentence_prediction(model_output,lengths, device)

                #print(results, y)
     
                cost = loss(results, y)
                
                cost.backward()

                epoch_cost += cost.detach().cpu().numpy()
                update+=1
                if update % args.log_every == 0:
                    print("Epoch ", epoch, " chunk ", i + 1, ", update ", update,", cost: ", cost.detach().cpu().numpy(),sep="")
                if (update % args.gradient_accumulation == 0):
                    #a = model.parameters()
                    optimizer.step()
                    optimizer.zero_grad()
                    #b = model.parameters()
                    #print(a[0])
            print("Epoch ", epoch, ", train cost/batch: ", epoch_cost / len(train_dataloader), sep="")

            #Should change to i + 1
            if args.checkpoint_every_n_chunks != None  and i % args.checkpoint_every_n_chunks == 0:
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


                    print("Epoch ", epoch, "chunk ", str(i), ", dev cost/batch: ", epoch_cost / len(dev_dataloader), sep="")
                    print("Dev Accuracy:", accuracy_score(true_l, predicted_l))
                    true_l = [ classes_vocabulary.tokens[idx] for idx in true_l] 
                    predicted_l = [ classes_vocabulary.tokens[idx] for idx in predicted_l] 
                    precision, recall, f1, _ = precision_recall_fscore_support(true_l, predicted_l, average='macro')
                    print("Dev precision, Recall, F1 (Macro): ", precision, recall, f1)
                    print(classification_report(true_l, predicted_l))

                    if not os.path.exists(args.output_folder):
                        os.makedirs(args.output_folder)
     
                    torch.save({
                        'version': "0.3",
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'vocabulary': vocabulary,
                        'classes_vocabulary': classes_vocabulary,
                        'args' : args
                    }, args.output_folder + "/model." + str(epoch) + "." + str(i) + ".pt", pickle_protocol=4)

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
            true_l = [ classes_vocabulary.tokens[idx] for idx in true_l] 
            predicted_l = [ classes_vocabulary.tokens[idx] for idx in predicted_l] 
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
