ALIGN_FILE=$1
TMP_DIR=$2
OUTPUT_FILE=$3
CONFIG_FILE=$4

source $CONFIG_FILE
source $PYTHON_ENV

python /home/jiranzo/trabajo/git/my_gits/ST-Segmenter/scripts/tlkalign_to_features.py $ALIGN_FILE > $TMP_DIR/features
python /home/jiranzo/trabajo/git/my_gits/ST-Segmenter/scripts/tlkalign_to_text.py $ALIGN_FILE > $TMP_DIR/text

cat <<EOF > $TMP_DIR/text.lst
$TMP_DIR/text
EOF

cat <<EOF > $TMP_DIR/features.lst
$TMP_DIR/features
EOF

MODEL=$SYSTEM_DIRECTORY/model/
moses_scripts=/scratch/ttpuser/git/mosesdecoder/scripts
SUBWORD_FOLDER=/scratch/ttpuser/git/subword-nmt
BPE_FOLDER=$SYSTEM_DIRECTORY/bpe


python ~/trabajo/git/my_gits/ST-Segmenter/decode_audio_model.py \
    --input_format list_of_text_files \
    --input_file_list $TMP_DIR/text.lst \
    --input_audio_file_list $TMP_DIR/features.lst \
    --model_path $SEGMENTER_CHECKPOINT \
    --sample_max_len $SEGMENTER_MAX_LEN \
    --sample_window_size  $SEGMENTER_WINDOW_SIZE | sed -e 's/<unk>//g' | $SUBWORD_FOLDER/apply_bpe.py -c $BPE_FOLDER/bpe.codes --vocabulary $BPE_FOLDER/bpe.vocab.$src --vocabulary-threshold 50 | fairseq-interactive --path $MODEL/checkpoint_best.pt \
    --beam 6 \
    $MODEL --source-lang $src --target-lang $tgt > $OUTPUT_FILE

