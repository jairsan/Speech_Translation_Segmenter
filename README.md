# Direct Segmentation Models for Streaming Speech Translation
This repository contains the code of the paper "Direct Segmentation Models for Streaming Speech Translation".
Please refer to the publication:
```
@inproceedings{Sanchez2020,
title = {Direct Segmentation Models for Streaming Speech Translation},
author = {Javier Iranzo-S\'{a}nchez and Adri\`{a} Gim\'{e}nez and Joan Albert Silvestre-Cerd\`{a} and Pau Baquero-Arnal and Jorge Civera and Alfons Juan},
year = {2020},
booktitle = {2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020)},

```

## Overview

Main requirements:
* Python 3.6
* PyTorch 1.2
* CUDA 10.0

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
We are given the sentence "madam president are only greed euphoria and cheap money to be blamed for the whole mess" to segment. The corresponding sample for the split decision at position 1, "be", and target 0, is the following:

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

Training should be self-explanatory. For training a text model, you can do something like:

```
len=$(($HISTORY_LENGTH+1+$FUTURE_WINDOW_SIZE))
python train_text_model.py \
--train_corpus train.ML$len.WS$FUTURE_WINDOW_SIZE.txt \
--dev_corpus dev.ML$len.WS$FUTURE_WINDOW_SIZE.txt \
--output_folder text_model \
--checkpoint_interval 10 \
--vocabulary train.vocab.txt \
--epochs 40 \
--rnn_layer_size 256 \
--embedding_size 256 \
--n_classes 2 \
--batch_size 256 \
--min_split_samples_batch_ratio 0.3 \
--optimizer adam \
--lr 0.0001 \
--lr_schedule reduce_on_plateau \
--lr_reduce_patience 10 \
--dropout 0.3 \
--model_architecture ff-text \
--feedforward_layers 2 \
--feedforward_size 128 \
--sample_max_len $len \
--sample_window_size $FUTURE_WINDOW_SIZE
```

"--min_split_samples_batch_ratio" upsamples the data with a split/1 label. Otherwise, the model might degenerate to always output 0.

Then, to train an audio model on top of a pre-trained text model,

```
python train_audio_model_from_pretrained_text.py \
--train_corpus  train.ML$len.WS$FUTURE_WINDOW_SIZE.txt \
--dev_corpus dev.ML$len.WS$FUTURE_WINDOW_SIZE.txt \
--train_audio_features_corpus train.ML$len.WS$FUTURE_WINDOW_SIZE.feas.txt \
--dev_audio_features_corpus dev.ML$len.WS$FUTURE_WINDOW_SIZE.feas.txt \
--model_path text_model/model.best.pt \
--output_folder audio_model \
--checkpoint_interval 1 \
--vocabulary train.vocab.txt \
--epochs 20 \
--embedding_size 3 \
--n_classes 2 \
--batch_size 256 \
--min_split_samples_batch_ratio 0.3 \
--optimizer adam \
--lr 0.0001 \
--lr_schedule reduce_on_plateau \
--lr_reduce_patience 1 \
--dropout 0.3 \
--feedforward_layers 2 \
--feedforward_size 128 \
--sample_max_len $len \
--sample_window_size $window \
--model_architecture ff-audio-text-copy-feas
```

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
