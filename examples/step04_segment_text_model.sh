#Prepare data to split. We will simulate an input stream, so only one line per file
rm -r test-1line/
mkdir test-1line/
rm test.1line.lst

for file in $(ls test/);

do
    echo $(cat test/$file | tr '\n' ' ') > test-1line/$file
    echo test-1line/$file >> test.1line.lst
done

#Now we segment
python $PWD/../decode_text_model.py \
    --input_format list_of_text_files \
    --input_file_list $PWD/test.1line.lst  \
    --model_path text_model/model.best.pt \
    --sample_max_len $len \
    --sample_window_size $FUTURE_WINDOW_SIZE
