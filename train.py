from segmenter import dataset,arguments,vocab
from segmenter.models.simple_rnn import SimpleRNN

import argparse
import torch
import torch.utils.data as data
import torch.optim as optim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arguments.add_train_arguments(parser)
    arguments.add_model_arguments(parser)

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    vocabulary = vocab.VocabDictionary()
    vocabulary.create_from_count_file(args.vocabulary)

    train_dataset = dataset.segmentationBinarizedDataset(args.train_corpus,vocabulary)
    train_dataloader = data.DataLoader(train_dataset,num_workers=3,batch_size=2,shuffle=True,drop_last=True,
                                       collate_fn=dataset.collateBinarizedBatch)


    model = SimpleRNN(args,vocabulary).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0001, momentum=0.0)
    loss = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs):
        optimizer.zero_grad()

        epoch_cost = 0
        for x, src_lengths, y in train_dataloader:

            x, y = x.to(device), y.to(device)
            model_output,lengths, hn = model.forward(x, src_lengths)

            select = lengths - torch.ones(lengths.shape, dtype=torch.long)

            select= select.to(device)

            indices = torch.unsqueeze(select, 1)
            indices = torch.unsqueeze(indices, 2).repeat(1, 1, 2)
            results = torch.gather(model_output, 1, indices).squeeze(1)

            cost = loss(results, y)
            cost.backward()

            epoch_cost += cost.detach().cpu().numpy()

            optimizer.step()
            optimizer.zero_grad()

        print("Train cost/batch :", epoch_cost / len(train_dataloader))