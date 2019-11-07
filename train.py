from segmenter import dataset,arguments,vocab

import argparse
import torch
import torch.utils.data as data

# TODO Create custom DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arguments.add_train_arguments(parser)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    vocabulary = vocab.VocabDictionary()
    vocabulary.load_from_count_file(args.vocabulary)

    train_dataset = dataset.segmentationBinarizedDataset(args.train_corpus,vocabulary)
    train_dataloader = data.DataLoader(train_dataset,num_workers=3,batch_size=2,shuffle=True,drop_last=True)

    for epoch in range(1, 5):
        for source_batch, target_batch in train_dataloader:
            print(source_batch, target_batch)
