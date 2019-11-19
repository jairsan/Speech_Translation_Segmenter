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
    decision = torch.argmax(results,dim=1).detach().cpu().numpy().tolist()


    return decision


def decode_from_sample_file(args, model, vocabulary):

    with open(args.file) as f:
        for line in f:
            line = line.strip().split()

            target = int(line[0])
            tokens = line[1:]

            decision = get_decision(model, tokens, vocabulary)[0]

            print(decision)


def decode_from_file(file_path, args, model, vocabulary):

    text = []

    with open(file_path) as f:
        for l in f:
            l = l.strip().split()
            text.extend(l)

    max_len = 15
    window_size = 4

    history = ["</s>"] * (max_len - window_size - 1)

    buffer = []

    #Cuando llegamos a la ultima palabra, vamos a cortar siempre, asi que no hace falta evaluar ese caso
    for i in range(len(text)-window_size):
        buffer.append(text[i])
        sample = history + [text[i]] + text[i+1:i+window_size+1]
        assert len(sample) == max_len

        decision = get_decision(model,sample,vocabulary)[0]

        if decision == 0:
            history.pop(0)
            history.append(text[i])
        else:
            history.pop(0)
            history.pop(0)
            history.append(text[i])
            history.append("</s>")
            print(" ".join(buffer))
            buffer = []

def decode_from_list_of_files(args, model, vocabulary):

    with open(args.input_file_list) as f_lst:
        for line in f_lst:
            decode_from_file(line.strip(), args, model, vocabulary)


#TODO: Complete. This should decode only from stdin or server port. This is the true online version
def decode_from_stream(args, model, vocabulary):
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

    if args.input_format == "sample_file":
        decode_from_sample_file(args, model, vocabulary)
    elif args.input_format == "list_of_text_files":
        decode_from_list_of_files(args, model, vocabulary)
    else:
        decode_from_stream(args, model, vocabulary)
