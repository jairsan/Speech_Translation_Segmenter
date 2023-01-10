import logging
from typing import List, Tuple, Optional

import torch
import torch.nn as nn

from segmenter.models.segmenter_model import SegmenterModel, HuggingFaceSegmenterModel
from segmenter.vocab import VocabDictionary

logger = logging.getLogger(__name__)


class DsSegmenter:

    def __init__(self, segmenter_model: SegmenterModel, segmenter_vocab_dictionary: VocabDictionary,
                 sample_window_size: int, sample_max_len: int, max_segment_length: int):
        self.past_history: List[str] = ["</s>"] * (sample_max_len - sample_window_size - 1)
        self.unprocessed_words: List[str] = []
        self.sample_window_size = sample_window_size
        self.max_segment_length = max_segment_length
        self.sample_max_len = sample_max_len
        self.segmenter_model = segmenter_model
        self.segmenter_vocab_dictionary = segmenter_vocab_dictionary
        self.consecutive_no_split: int = 0

    def step(self, new_word: str) -> Tuple[Optional[str], bool]:
        self.unprocessed_words.append(new_word)

        if len(self.unprocessed_words) >= self.sample_window_size + 1:
            # The word for which we are going to decide split/no-split
            current_word = self.unprocessed_words[0]
            sample = self.past_history + [current_word] + self.unprocessed_words[1:self.sample_window_size + 1]
            logger.debug(f"Will segment sample {sample}, consits of history {self.past_history}, current_word "
                         f"{[current_word]}, and future_window {self.unprocessed_words[1:self.sample_window_size + 1]}")

            assert len(sample) == self.sample_max_len

            if self.consecutive_no_split >= self.max_segment_length - 1:
                decision = 1

            else:
                batch = {"words": " ".join(sample)}

                if isinstance(self.segmenter_model, HuggingFaceSegmenterModel):
                    batch = self.segmenter_model.apply_tokenizer(batch)
                    batch = {key: torch.tensor(value).view(1, -1) for key, value in batch.items()}
                else:
                    batch = {"idx": [self.segmenter_vocab_dictionary.get_index(token) for token in batch["words"].split()]}

                model_output = self.segmenter_model.forward(batch, device=torch.device('cuda'))
                sentence_prediction = self.segmenter_model.get_sentence_prediction(model_output)
                probs = torch.nn.functional.log_softmax(sentence_prediction, dim=1).detach().cpu().numpy()[0]

                logger.debug(f"Probs {probs}")

                self.unprocessed_words.pop(0)
                decision = probs[1] > probs[0]

            self.past_history.pop(0)
            self.past_history.append(current_word)

            logger.debug(f"Segmenter decision: {decision}")

            if decision == 1:
                self.consecutive_no_split = 0
                self.past_history.pop(0)
                self.past_history.append("</s>")
            else:
                self.consecutive_no_split += 1

            return current_word, decision == 1
        else:
            return None, False

    def reset(self):
        self.past_history = ["</s>"] * (self.sample_max_len - 1 - self.sample_window_size)
        self.consecutive_no_split = 0
        self.unprocessed_words = []
