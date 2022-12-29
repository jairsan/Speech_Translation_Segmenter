import torch
import torch.nn as nn
import argparse
import copy
from segmenter.models.bert_text_model import BERTTextModel
from segmenter.models.xlm_roberta_text_model import XLMRobertaTextModel

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

def get_probs(model,sentence,vocab_dictionary, device):
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

    return torch.nn.functional.log_softmax(results, dim=1).detach().cpu().numpy()

def get_decision_raw(model, sentence, device):
    x = [sentence]
    model_output, lengths, hn = model.forward(x, [len(x)], device)

    results = model.get_sentence_prediction(model_output, lengths, device)

    decision = torch.argmax(results,dim=1).detach().cpu().numpy().tolist()

    return decision

def get_probs_raw(model, sentence, device):
    x = [sentence]
    model_output, lengths, hn = model.forward(x, [len(x)], device)

    results = model.get_sentence_prediction(model_output, lengths, device)

    decision = torch.nn.functional.log_softmax(results,dim=1).detach().cpu().numpy().tolist()

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

    #When we reach the last word, we will always segment, so no need to eval
    for i in range(len(text)-window_size):
        buffer.append(text[i])
        sample = history + [text[i]] + text[i+1:i+window_size+1]
        assert len(sample) == max_len
        
        if isinstance(model,BERTTextModel) or isinstance(model, XLMRobertaTextModel):
            decision = get_decision_raw(model,sample, device)[0]
        else:
            decision = get_decision(model,sample,vocabulary, device)[0]

        if decision == 0 and len(buffer) <= args.segment_max_size :
            history.pop(0)
            history.append(text[i])
        else:
            history.pop(0)
            history.pop(0)
            history.append(text[i])
            history.append("</s>")
            print(" ".join(buffer))
            buffer = []

    buffer.extend(text[max(0,len(text)-window_size):])
    print(" ".join(buffer))

def beam_decode_from_file(file_path, args, model, vocabulary, device):

    text = []

    with open(file_path) as f:
        for l in f:
            l = l.strip().split()
            text.extend(l)

    max_len = args.sample_max_len
    window_size = args.sample_window_size

    history = ["</s>"] * (max_len - window_size - 1)

    #During beam decoding, we are going to use an unordered list for storing the hypos
    cubeta = []

    #Each hypo is a tuple (model_history, segmented_sentences, score)
    cubeta.append((history, [[]], 0))

    #Segmented_sentences is a list of lists
    #Every time a split decision is taken, a new empty list is added
    #Every time a non split decision is taken, the word is added to segmented_sentences

    #When we reach the last word, we will always segment, so no need to eval
    for i in range(len(text)-window_size):

        cubeta2 = []
        for j in range(len(cubeta)):

            history, segmentation_history, score = cubeta[j]

            sample = history + [text[i]] + text[i+1:i+window_size+1]


            probs = get_probs(model,sample,vocabulary, device)

            # No split
            history_0 = copy.deepcopy(history)
            segmentation_history_0 = copy.deepcopy(segmentation_history)

            history_0.pop(0)
            history_0.append(text[i])

            segmentation_history_0[-1].append(text[i])
            
            if len(segmentation_history_0[-1]) <= args.segment_max_size:
                cubeta2.append((history_0, segmentation_history_0, score + probs[0][0]))

            # Split
            history_1 = copy.deepcopy(history)
            segmentation_history_1 = copy.deepcopy(segmentation_history)

            history_1.pop(0)
            history_1.pop(0)
            history_1.append(text[i])
            history_1.append("</s>")

            segmentation_history_1[-1].append(text[i])

            segmentation_history_1.append([])

            cubeta2.append((history_1, segmentation_history_1, score + probs[0][1]))

        #Sort by scores and create the cubeta for next iteration
        cubeta2.sort(key=lambda hypo: hypo[2], reverse=True)
        cubeta = cubeta2[:min(args.beam,len(cubeta2)+1)]


    best_hypo = cubeta[0]

    best_hypo[1][-1].extend(text[max(0,len(text)-window_size):])

    for line in best_hypo[1]:
        print(" ".join(line))



def decode_from_list_of_files(args, model, vocabulary, device):
    with open(args.input_file_list) as f_lst:
        for line in f_lst:
            if args.beam > 1:
                beam_decode_from_file(line.strip(), args, model, vocabulary, device)
            else:
                decode_from_file(line.strip(), args, model, vocabulary, device)



if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")

    parser = argparse.ArgumentParser()

    arguments.add_infer_arguments(parser)
    arguments.add_general_arguments(parser)
    args = parser.parse_args()

    model, vocabulary, _ = utils.load_text_model(args)

    model = model.to(device)
    model.eval()
    if args.input_format == "sample_file":
        decode_from_sample_file(args, model, vocabulary, device)
    else:
        decode_from_list_of_files(args, model, vocabulary, device)
        
