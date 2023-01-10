import argparse

import torch

from segmenter.ds_segmenter import DsSegmenter
from segmenter.utils import load_text_model


def segment(input_file_path: str, ds_segmenter: DsSegmenter):
    words_to_output = []
    with open(input_file_path, "r") as inf:
        for line in inf:
            words = line.strip().split()
            for word in words:
                output_word, is_eos = ds_segmenter.step(new_word=word)
                if output_word is not None:
                    words_to_output.append(output_word)
                    if is_eos:
                        print(" ".join(words_to_output))
                        words_to_output = []

    if len(ds_segmenter.unprocessed_words) > 0:
        print(" ".join(ds_segmenter.unprocessed_words))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--segmenter", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    segmenter_model, segmenter_vocab, loaded_model_args = load_text_model(args.segmenter)
    assert loaded_model_args.n_classes == 2
    segmenter_model = segmenter_model.to(device)

    segmenter = DsSegmenter(segmenter_model=segmenter_model, segmenter_vocab_dictionary=segmenter_vocab,
                            max_segment_length=85, sample_max_len=loaded_model_args.sample_max_len,
                            sample_window_size=loaded_model_args.sample_window_size)

    segment(input_file_path=args.input_file, ds_segmenter=segmenter)
