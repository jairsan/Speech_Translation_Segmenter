len=15
window=4

rm -r pre-trained_text_model

set -x

python $PWD/../train_model.py \
--train_corpus $PWD/train-mini.ML$len.WS$window.txt \
--dev_corpus $PWD/dev.ML$len.WS$window.txt \
--output_folder pre-trained_text_model \
--vocabulary $PWD/train.vocab.txt \
--model_architecture bert \
--transformer_model_name bert-base-multilingual-uncased \
--batch_size 16 \
--sample_max_len $len \
--sample_window_size $window
