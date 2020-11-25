len=15
window=4

rm -r audio_model


python $PWD/../train_audio_model_from_pretrained_text.py \
--train_corpus  $PWD/train.ML$len.WS$window.txt \
--dev_corpus $PWD/dev.ML$len.WS$window.txt \
--train_audio_features_corpus $PWD/train.ML$len.WS$window.feas.txt \
--dev_audio_features_corpus $PWD/dev.ML$len.WS$window.feas.txt \
--model_path text_model/model.best.pt \
--embedding_size 3 \
--output_folder audio_model \
--vocabulary train.vocab.txt \
--sample_max_len $len \
--sample_window_size $window \
--model_architecture ff-audio-text-copy-feas

