import torch
import torch.nn as nn
import argparse
import copy
from segmenter.models.bert_text_model import BERTTextModel
from segmenter.models.xlm_roberta_text_model import XLMRobertaTextModel

from segmenter import arguments, utils

from sklearn.metrics import precision_recall_fscore_support,classification_report

from segmenter.utils import load_text_model_multiclass


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

def step(history, current_word, future_window, max_len, window_size, model, vocabulary, classes_vocabulary, device):
    sample = history + [current_word] + future_window
    assert len(sample) == max_len

    decision_idx = get_decision(model,sample,vocabulary, device)[0]
    decision = classes_vocabulary.tokens[decision_idx]

    casing = decision[0]

    output_word = current_word

    if casing == "A":
        output_word = output_word.upper()
    elif casing == "U":
        output_word = output_word[0].upper() + output_word[1:]
    
    sign=None

    if len(decision) > 1:
        sign=decision[1]
        output_word = output_word + sign
    
    history.append(current_word)

    if decision != "L":
        history.append(decision)

    if len(history) > (max_len - window_size - 1):
        history=history[-(max_len - window_size - 1):]

    return output_word, sign, history, casing

def decode_from_file(file_path, args, model, vocabulary, classes_vocabulary, device):

    text = []

    with open(file_path) as f:
        for l in f:
            l = l.strip().split()
            text.extend(l)

    max_len = args.sample_max_len
    window_size = args.sample_window_size

    history = ["<pad>"] * (max_len - window_size - 1)
    
    is_first = True
    
    #When we reach the last word, we will always segment, so no need to eval
    for i in range(len(text)):
        current_word = text[i]
        future_window = text[i+1: min(i+ window_size + 1, len(text)) ]
        if len(future_window) < window_size:
            future_window = future_window + ["<pad>"] * (window_size - len(future_window))
        out_word, sign, history, casing = step(history, current_word, future_window, max_len, window_size, model, vocabulary, classes_vocabulary, device)
        
        if is_first:
            out_word= out_word[0].upper() + out_word[1:]
            is_first = False

        if sign == ".":
            if args.return_type == "words":
                print(out_word, end="\n")
            else:
                print(casing, end=" ")
            is_first = True
        else:
            if args.return_type == "words":
                print(out_word, end=" ")
            else:
                print(casing, end=" ")
def decode_from_list_of_files(args, model, vocabulary, classes_vocabulary, device):
    with open(args.input_file_list) as f_lst:
        for line in f_lst:
            decode_from_file(line.strip(), args, model, vocabulary, classes_vocabulary, device)

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")

    parser = argparse.ArgumentParser()

    arguments.add_infer_arguments(parser)
    arguments.add_multiclass_infer_arguments(parser)
    arguments.add_general_arguments(parser)
    args = parser.parse_args()

    model, vocabulary, classes_vocabulary, _ = utils.load_text_model_multiclass(args)

    model = model.to(device)
    model.eval()

    decode_from_list_of_files(args, model, vocabulary, classes_vocabulary, device)
        
