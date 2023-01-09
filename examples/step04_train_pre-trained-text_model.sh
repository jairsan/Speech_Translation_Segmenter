len=15
window=4

rm -r pre-trained_text_model

python $PWD/../train_model.py \
--train_corpus $PWD/train.ML$len.WS$window.txt \
--dev_corpus $PWD/dev.ML$len.WS$window.txt \
--output_folder pre-trained_text_model \
--vocabulary $PWD/train.vocab.txt \
--model_architecture xlm-roberta \
--transformer_model_name xlm-roberta-base \
--sampling_temperature 100 \
--batch_size 64 \
--adam_b1 0.9 \
--adam_b2 0.98 \
--adam_eps 1e-8 \
--lr 2e-5 \
--lr_schedule reduce_on_plateau \
--lr_reduce_patience 1 \
--sample_max_len $len \
--sample_window_size $window
