import torch
import torch.nn as nn
import argparse

from segmenter import dataset,arguments,vocab
from segmenter.models.simple_rnn import SimpleRNN



def load_model(path,model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def get_decision(model,sentence,vocab_dictionary):
    #Sentence is a list of words

    tokens_i = []
    for word in sentence:
        tokens_i.append(vocab_dictionary.get_index(word))

    x = torch.LongTensor([tokens_i])
    X = nn.utils.rnn.pad_sequence(x, batch_first=True)
    model_output, lengths, hn = model.forward(X, [len(x)])

    results = model.get_sentence_prediction(model_output, lengths, device)

    #In the future, we should compute probs if we want to carry out search
    decision = torch.argmax(results,dim=1)


    return decision


def decode_from_sample_file(args, model):

    with open(args.file) as f:
        for line in f:
            line = line.strip().split()

            target = int(line[0])
            tokens = line[1:]

            decision = get_decision(model, tokens, vocabulary)

            decision = decision.detach().cpu().numpy().tolist()


def decode_from_stream(args, model):

    if args.stream != "sys.stdin":
        text = []

        with open(args.stream) as f:
            for l in f:
                l = l.strip().split()
                text.extend(l)

        max_len = 15
        window_size = 4

        history = ["</s>"] * (max_len - window_size - 1)

        for i in range(len(text)-window_size):


    else:
        pass


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()

    device = "cpu"
    #device = torch.device("cuda:0" if use_cuda else "cpu")

    parser = argparse.ArgumentParser()

    arguments.add_infer_arguments(parser)
    arguments.add_model_arguments(parser)
    args = parser.parse_args()

    vocabulary = vocab.VocabDictionary()
    vocabulary.create_from_count_file(args.vocabulary)

    model = SimpleRNN(args,vocabulary).to(device)
    model = load_model(args.model_path, model)

    if args.file != None:
        decode_from_sample_file(args, model)
    else:
        decode_from_stream(args,model)
