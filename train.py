from segmenter import dataset,arguments,vocab
from segmenter.models.simple_rnn import SimpleRNN

import argparse
import torch
import torch.utils.data as data
import torch.optim as optim
import time, os
import numpy as np

from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support,classification_report



def get_sentence_prediction(model_output,lengths, device):
    select = lengths - torch.ones(lengths.shape, dtype=torch.long)

    select = select.to(device)

    indices = torch.unsqueeze(select, 1)
    indices = torch.unsqueeze(indices, 2).repeat(1, 1, 2)
    results = torch.gather(model_output, 1, indices).squeeze(1)

    return results

if __name__ == "__main__":
    start_prep = time.time()

    parser = argparse.ArgumentParser()
    arguments.add_train_arguments(parser)
    arguments.add_model_arguments(parser)

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    vocabulary = vocab.VocabDictionary()
    vocabulary.create_from_count_file(args.vocabulary)

    train_dataset = dataset.segmentationBinarizedDataset(args.train_corpus,vocabulary)
    train_dataloader = data.DataLoader(train_dataset,num_workers=3,batch_size=args.batch_size,shuffle=True,drop_last=True,
                                       collate_fn=dataset.collateBinarizedBatch)

    dev_dataset = dataset.segmentationBinarizedDataset(args.dev_corpus, vocabulary)
    dev_dataloader = data.DataLoader(dev_dataset, num_workers=3, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                       collate_fn=dataset.collateBinarizedBatch)


    model = SimpleRNN(args,vocabulary).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_weights = torch.Tensor([1.0,args.split_weight]).to(device)
    loss = torch.nn.CrossEntropyLoss(weight=loss_weights)


    end_prep = time.time()
    print("Model preparation took: ",str(end_prep - start_prep), " seconds.")

    for epoch in range(1, args.epochs):
        optimizer.zero_grad()

        epoch_cost = 0

        update = 0
        for x, src_lengths, y in train_dataloader:

            x, y = x.to(device), y.to(device)
            model_output,lengths, hn = model.forward(x, src_lengths)

            results = get_sentence_prediction(model_output,lengths, device)

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
            for x, src_lengths, y in dev_dataloader:

                x, y = x.to(device), y.to(device)
                model_output, lengths, hn = model.forward(x, src_lengths)

                results = get_sentence_prediction(model_output, lengths, device)
                yhat = torch.argmax(results,dim=1)


                predicted_l.extend(yhat.detach().cpu().numpy().tolist())
                true_l.extend(y.detach().cpu().numpy().tolist())

                cost = loss(results, y)
                epoch_cost += cost.detach().cpu().numpy()

            print("Epoch ", epoch, ", dev cost/batch: ", epoch_cost / len(dev_dataloader), sep="")
            print("Dev Accuracy:", accuracy_score(true_l, predicted_l))
            print("Dev precision, Recall, F1 (Macro): ", precision_recall_fscore_support(true_l, predicted_l, average='macro'))
            print(classification_report(true_l, predicted_l))

        if args.checkpoint_interval > 0 and epoch % args.checkpoint_interval == 0:
            if not os.path.exists(args.output_folder):
                os.makedirs(args.output_folder)
            torch.save({
                'version': "0.1",
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, args.output_folder + "/model.last.pt")
