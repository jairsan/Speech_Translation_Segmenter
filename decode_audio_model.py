import torch
import torch.nn as nn
import argparse

from segmenter import arguments, utils

from sklearn.metrics import precision_recall_fscore_support,classification_report

from segmenter.utils import load_text_model


FILLER_FEA=[0.0, 0.0, 0.0]


def get_decision(text_model,audio_model,sentence,audio_features, vocab_dictionary, device):
    """
    Given a list of words (sentence) and a model,
    returns the decision (split or not split) given by the model
    """

    tokens_i = []
    for word in sentence:
        tokens_i.append(vocab_dictionary.get_index(word))

    x = torch.LongTensor([tokens_i]).to(device)
    x_audio = torch.FloatTensor([audio_features]).to(device)



    X = nn.utils.rnn.pad_sequence(x, batch_first=True)
    X_feas = nn.utils.rnn.pad_sequence(x_audio,batch_first=True)



    text_feas, _, _ = text_model.extract_features(X, [len(tokens_i)])

    prediction, _, _ = audio_model.forward(X_feas, text_feas, [len(tokens_i)])

    #In the future, we should compute probs if we want to carry out search
    decision = torch.argmax(prediction,dim=1).detach().cpu().numpy().tolist()

    return decision


def decode_from_file_pair(text_file_path,audio_file_path, args, text_model, audio_model, vocabulary, device):

    text = []
    audio_features = []

    with open(text_file_path) as f:
        for l in f:
            l = l.strip().split()
            text.extend(l)


    with open(audio_file_path) as f:
        for line in f:
            audio_features.append(list(map(float, line.strip().split())))


    max_len = args.sample_max_len
    window_size = args.sample_window_size

    history = ["</s>"] * (max_len - window_size - 1)

    history_a = []
    for _ in range(max_len - window_size - 1):
        history_a.append(FILLER_FEA)


    buffer = []
    buffer_a = []

    #Cuando llegamos a la ultima palabra, vamos a cortar siempre, asi que no hace falta evaluar ese caso
    for i in range(len(text)-window_size):
        buffer.append(text[i])
        buffer_a.append(audio_features[i])


        sample = history + [text[i]] + text[i+1:i+window_size+1]
        sample_a = history_a + [audio_features[i]] + audio_features[i+1:i+window_size+1]

        decision = get_decision(text_model,audio_model,sample,sample_a,vocabulary, device)[0]

        if decision == 0:
            history.pop(0)
            history.append(text[i])

            history_a.pop(0)
            history_a.append(audio_features[i])
        else:
            history.pop(0)
            history.pop(0)
            history.append(text[i])
            history.append("</s>")

            history_a.pop(0)
            history_a.pop(0)
            history_a.append(audio_features[i])
            history_a.append(FILLER_FEA)


            print(" ".join(buffer))
            buffer = []
            buffer_a = []

    buffer.extend(text[len(text)-window_size:])
    print(" ".join(buffer))


def decode_from_list_of_file_pairs(args, text_model, audio_model, vocabulary, device):
    with open(args.input_file_list) as f_lst, open(args.input_audio_file_list) as a_lst:
        for text_f, audio_f in zip(f_lst,a_lst):
            decode_from_file_pair(text_f.strip(), audio_f.strip(), args, text_model, audio_model, vocabulary, device)


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")

    parser = argparse.ArgumentParser()

    arguments.add_infer_arguments(parser)
    arguments.add_model_arguments(parser)
    args = parser.parse_args()

    text_model, vocabulary, text_model_args, audio_model, audio_model_args = utils.load_text_and_audio_model(args)

    text_model = text_model.to(device)
    text_model.eval()

    audio_model = audio_model.to(device)
    audio_model.eval()

    decode_from_list_of_file_pairs(args, text_model, audio_model, vocabulary, device)
