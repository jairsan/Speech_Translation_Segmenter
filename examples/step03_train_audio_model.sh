len=15
window=4

rm -r audio_model


python $PWD/../train_model.py \
--train_corpus  $PWD/train.ML$len.WS$window.txt \
--dev_corpus $PWD/dev.ML$len.WS$window.txt \
--train_audio_features_corpus $PWD/train.ML$len.WS$window.feas.txt \
--dev_audio_features_corpus $PWD/dev.ML$len.WS$window.feas.txt \
--frozen_text_model_path text_model/model.best.pt \
--audio_features_size 3 \
--output_folder audio_model \
--vocabulary train.vocab.txt \
--sample_max_len $len \
--sample_window_size $window \
--model_architecture rnn-ff-audio-copy

