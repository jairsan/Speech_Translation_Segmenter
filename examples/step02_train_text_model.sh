len=15
window=4

rm -r text_model

python $PWD/../train_text_model.py \
--train_corpus $PWD/train.ML$len.WS$window.txt \
--dev_corpus $PWD/dev.ML$len.WS$window.txt \
--output_folder text_model \
--vocabulary $PWD/train.vocab.txt \
--model_architecture ff-text \
--sample_max_len $len \
--sample_window_size $window
