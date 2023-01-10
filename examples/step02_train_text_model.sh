len=15
window=4

rm -r text_model

python $PWD/../train_model.py \
--model_architecture rnn-ff-text \
--train_corpus $PWD/train.ML$len.WS$window.txt \
--dev_corpus $PWD/dev.ML$len.WS$window.txt \
--output_folder text_model \
--vocabulary $PWD/train.vocab.txt \
--sampling_temperature 5 \
--batch_size 256 \
--sample_max_len $len \
--sample_window_size $window
