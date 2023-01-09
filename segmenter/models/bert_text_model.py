import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

from segmenter.models.segmenter_model import SegmenterModel


class BERTTextModel(SegmenterModel):
    name: str = "bert"

    def __init__(self,args):
        super(BERTTextModel, self).__init__()
        
        self.transformer_model = BertForSequenceClassification.from_pretrained(args.transformer_architecture.split(":")[1], return_dict=True)
        self.tokenizer = BertTokenizer.from_pretrained(args.transformer_architecture.split(":")[1])

        self.window_size = args.sample_window_size

    def forward(self, batch, device):
        history = []
        future = []
        for l in batch:
            history.append(l[:-self.window_size])
            future.append(l[-self.window_size:])
        encoded_inputs = self.tokenizer(history, future,return_tensors="pt")

        return self.transformer_model(encoded_inputs['input_ids'].to(device), attention_mask=encoded_inputs['attention_mask'].to(device), token_type_ids=encoded_inputs['token_type_ids'].to(device)).logits, src_lengths, None


    def get_sentence_prediction(self, model_output):
        return model_output
