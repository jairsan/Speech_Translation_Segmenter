len=15
window=4

rm -r audio_model


python $PWD/../train_audio_model_from_pretrained_text.py \
--train_corpus  $PWD/train.ML$len.WS$window.txt \
--dev_corpus $PWD/dev.ML$len.WS$window.txt \
--train_audio_features_corpus $PWD/train.ML$len.WS$window.feas.txt \
--dev_audio_features_corpus $PWD/dev.ML$len.WS$window.feas.txt \
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

