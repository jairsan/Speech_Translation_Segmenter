#Prepare data to segment
rm test.lst

for file in $(ls test/);
do
        echo test/$file >> test.lst
done

#Now we segment
python $PWD/../decode_text_model.py \
    --input_format list_of_text_files \
    --input_file_list $PWD/test.lst  \
    --model_path text_model/model.best.pt \
    --sample_max_len $len \
    --sample_window_size $FUTURE_WINDOW_SIZE

