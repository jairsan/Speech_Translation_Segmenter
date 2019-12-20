import torch
import torch.nn as nn
import argparse

from segmenter import arguments, utils

from sklearn.metrics import precision_recall_fscore_support,classification_report

from segmenter.utils import load_text_model


def get_decision(model,sentence,vocab_dictionary, device):
    """
    Given a list of words (sentence) and a model,
    returns the decision (split or not split) given by the model
    """

    tokens_i = []
    for word in sentence:
        tokens_i.append(vocab_dictionary.get_index(word))

    x = torch.LongTensor([tokens_i]).to(device)
    X = nn.utils.rnn.pad_sequence(x, batch_first=True)
    model_output, lengths, hn = model.forward(X, [len(tokens_i)])

    results = model.get_sentence_prediction(model_output, lengths, device)

    #In the future, we should compute probs if we want to carry out search
    decision = torch.argmax(results,dim=1).detach().cpu().numpy().tolist()

    return decision


def decode_from_sample_file(args, model, vocabulary, device):
    targets = []
    decisions = []
    with open(args.file) as f:
        for line in f:
            line = line.strip().split()

            target = int(line[0])
            tokens = line[1:]

            decision = get_decision(model, tokens, vocabulary)[0]

            targets.append(target)
            decisions.append(decision)

            print(decision)

    print("Dev precision, Recall, F1 (Macro): ", precision_recall_fscore_support(targets, decisions, average='macro'))
    print(classification_report(decisions, targets))


def decode_from_file(file_path, args, model, vocabulary, device):

    text = []

    with open(file_path) as f:
        for l in f:
            l = l.strip().split()
            text.extend(l)

    max_len = args.sample_max_len
    window_size = args.sample_window_size

    history = ["</s>"] * (max_len - window_size - 1)

    buffer = []

    #Cuando llegamos a la ultima palabra, vamos a cortar siempre, asi que no hace falta evaluar ese caso
    for i in range(len(text)-window_size):
        buffer.append(text[i])
        sample = history + [text[i]] + text[i+1:i+window_size+1]
        assert len(sample) == max_len

        decision = get_decision(model,sample,vocabulary, device)[0]

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

    buffer.extend(text[len(text)-window_size:])
    print(" ".join(buffer))


def decode_from_list_of_files(args, model, vocabulary, device):
    with open(args.input_file_list) as f_lst:
        for line in f_lst:
            decode_from_file(line.strip(), args, model, vocabulary, device)


#TODO: Complete. This should decode only from stdin or server port. This is the true online version
def decode_from_stream(args, model, vocabulary):
    raise NotImplementedError


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")

    parser = argparse.ArgumentParser()

    arguments.add_infer_arguments(parser)
    arguments.add_model_arguments(parser)
    args = parser.parse_args()

    model, vocabulary, _ = utils.load_text_model(args,device)

    model = model.to(device)
    model.eval()
    if args.input_format == "sample_file":
        decode_from_sample_file(args, model, vocabulary, device)
    elif args.input_format == "list_of_text_files":
        decode_from_list_of_files(args, model, vocabulary, device)
    else:
        decode_from_stream(args, model, vocabulary, device)
