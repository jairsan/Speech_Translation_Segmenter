# Direct Segmentation Models for Streaming Speech Translation
This repository contains the code of the paper [Direct Segmentation Models for Streaming Speech Translation](https://www.aclweb.org/anthology/2020.emnlp-main.206/).
Please refer to the publication:
```
@inproceedings{iranzo-sanchez-etal-2020-direct,
    title = "Direct Segmentation Models for Streaming Speech Translation",
    author = "Iranzo-S{\'a}nchez, Javier  and
      Gim{\'e}nez Pastor, Adri{\`a}  and
      Silvestre-Cerd{\`a}, Joan Albert  and
      Baquero-Arnal, Pau  and
      Civera Saiz, Jorge  and
      Juan, Alfons",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.206",
    pages = "2599--2611",
    abstract = "The cascade approach to Speech Translation (ST) is based on a pipeline that concatenates an Automatic Speech Recognition (ASR) system followed by a Machine Translation (MT) system. These systems are usually connected by a segmenter that splits the ASR output into hopefully, semantically self-contained chunks to be fed into the MT system. This is specially challenging in the case of streaming ST, where latency requirements must also be taken into account. This work proposes novel segmentation models for streaming ST that incorporate not only textual, but also acoustic information to decide when the ASR output is split into a chunk. An extensive and throughly experimental setup is carried out on the Europarl-ST dataset to prove the contribution of acoustic information to the performance of the segmentation model in terms of BLEU score in a streaming ST scenario. Finally, comparative results with previous work also show the superiority of the segmentation models proposed in this work.",
}
```

## Motivation

In the cascade approach to Speech Translation, an ASR system first transcribes the audio, and then the transcriptions are translated by a downstream MT system. 

Standard MT system are usually trained with sequences of around 100-150 tokens. Thus, if the audio is short, we can directly translate the transcription. However, when we have a long audio stream, the resulting transcription is many times longer than the maximum length seen by the MT model during training. This is why it is necessary to have a segmenter model that takes as input the stream of transcriped words, and outputs a stream of (hopefully) semantically self-contained segments, which are then translated independently by the MT model. The model presented here has been prepared to carry out segmentation in a streaming fashion.


## Overview

Main requirements:
* Python 3.8
* PyTorch 1.7
* CUDA 11.0

## Quickstart
1. Prepare your own data files using scripts/get_samples.py. You will need also need to define a vocabulary file.
2. Train a text model using train_text_model.py.
3. (Optional) Train an audio model with train_audio_model_from_pretrained_text.py. You will need a trained text model, and the audio data files. They can be produced with scripts/get_samples_with_audio_feas.py.
4. Segment the sentences, using decode_text_model.py or decode_audio_model.py

An example of the entire process for the Europarl-ST en-fr corpus is shown in the examples/ folder.

## Training Data preparation
An specific file must be prepared for each combination of HISTORY_LENGTH and FUTURE_WINDOW_SIZE.
### Text data
A text datafile consists in N lines, one for sample. Each line will consist in [1+HISTORY_LENGTH+1+FUTURE_WINDOW_SIZE] tokens separated by white-space. The first token is the label, 0 for no-split decision, 1 for split. Lets say that we are going to train a system with HISTORY_LENGTH=10, and FUTURE_WINDOW_SIZE=4.
We are given the sentence "madam president are only greed euphoria and cheap money to be blamed for the whole" to segment. The corresponding sample for the split decision at position 11, "be", and target 0, is the following:

```
0 madam president are only greed euphoria and cheap money to be blamed for the whole
```
* 0 is the label
* [madam president are only greed euphoria and cheap money to] is the part associated with the previous history (Remember that HISTORY_LENGTH=10 in this example)
* [be] is the word we are going to split
* [blamed for the whole] is the part associated with the future window (FUTURE_WINDOW_SIZE=4)

### Audio features
An audio features datafile consists in N lines, one for sample. It contains [HISTORY_LENGTH+1+FUTURE_WINDOW_SIZE] entries separated by semi-colons(;). Record j contains the audio features associated with word j, separated by white-space. This are the audio features computed by our ASR system for the same sample:

```
15 0 1;27 1 0;3 0 0;17 0 0;53 0 1;56 1 0;42 0 6;25 6 0;37 0 1;10 1 0;14 0 0;24 0 0;12 0 0;10 0 0;19 0 0
```

The acoustic features were extracted using the TLK software. Feel free to use any other ASR hybrid model toolkit, or use those features that might make more sense for your specific application. Remember that they must be aligned to words.

## Model Training

Training should be self-explanatory. For training a text model using the default configuration, you can do something like:

```
len=$(($HISTORY_LENGTH+1+$FUTURE_WINDOW_SIZE))
python train_text_model.py \
--train_corpus train.ML$len.WS$FUTURE_WINDOW_SIZE.txt \
--dev_corpus dev.ML$len.WS$FUTURE_WINDOW_SIZE.txt \
--output_folder text_model \
--vocabulary train.vocab.txt \
--model_architecture ff-text \
--sample_max_len $len \
--sample_window_size $FUTURE_WINDOW_SIZE
```

Then, to train an audio model on top of a pre-trained text model,

```
python train_audio_model_from_pretrained_text.py \
--train_corpus  train.ML$len.WS$FUTURE_WINDOW_SIZE.txt \
--dev_corpus dev.ML$len.WS$FUTURE_WINDOW_SIZE.txt \
--train_audio_features_corpus train.ML$len.WS$FUTURE_WINDOW_SIZE.feas.txt \
--dev_audio_features_corpus dev.ML$len.WS$FUTURE_WINDOW_SIZE.feas.txt \
--model_path text_model/model.best.pt \
--embedding_size 3 \
--output_folder audio_model \
--vocabulary train.vocab.txt \
--sample_max_len $len \
--sample_window_size $window \
--model_architecture ff-audio-text-copy-feas
```

It is required to provide --embedding_size [ACOUSTIC_FEATURES_DIM], which is 3 in our case.

The "ff-audio-text-copy-feas" architecture corresponds to the "Audio w/o RNN" on the paper. You can also train "Audio w/ RNN" models using --model_architecture ff-audio-text and --audio_rnn_layer_size [int].

## Segmentation

Decoding with a text model is very simple. You will provide a list file, each line of that file will contain the file path to a file whose contests you want to split.

```
ls test/ > test_files.lst

python decode_text_model.py \
    --input_format list_of_text_files \
    --input_file_list test_files.lst  \
    --model_path text_model/model.best.pt \
    --sample_max_len $len \
    --sample_window_size $FUTURE_WINDOW_SIZE
```

The contents of each file are treated as a stream, so preexisting line breaks are ignored, and the output is segmented into lines according to the best hypothesis of the model.

Decoding with an audio model follows the same principles, but now you also have to provide the file containing the paths of the audio feature files. Each audio feature file will contain N lines, N being the number of words of the corresponding text file. Each line provides white-space separated features for the corresponding word.

```
15 0 1
27 1 0
3 0 0
17 0 0
53 0 1
56 1 0
42 0 6
25 6 0
37 0 1
10 1 0
14 0 0
24 0 0
12 0 0
10 0 0
19 0 0
```

```
 python decode_audio_model.py \
     --input_format list_of_text_files \
     --input_file_list test_files.lst  \
     --input_audio_file_list test_files.features.lst \
     --model_path audio_model/model.best.pt \
     --sample_max_len $len \
     --sample_window_size $FUTURE_WINDOW_SIZE
 ```
