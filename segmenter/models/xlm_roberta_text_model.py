import torch.nn as nn
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

from segmenter.models.segmenter_model import SegmenterModel


class XLMRobertaTextModel(SegmenterModel):
    name: str = "xlm-roberta"

    def __init__(self, args):
        super(XLMRobertaTextModel, self).__init__()
        self.transformer_model = XLMRobertaForSequenceClassification.from_pretrained(args.transformer_architecture.split(":")[1], return_dict=True, num_labels=args.n_classes)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(args.transformer_architecture.split(":")[1])
        self.window_size = args.sample_window_size

    def forward(self, batch, device):
        samples = []
        for l in batch:
            new_l = ['nocontext' if x == "</s>" else x for x in l]
            a = new_l[:-self.window_size]+ ["</s>"] + new_l[-self.window_size:]
            samples.append(" ".join(a))
        encoded_inputs = self.tokenizer(samples,padding=True,return_tensors="pt")
        return self.transformer_model(encoded_inputs['input_ids'].to(device), attention_mask=encoded_inputs['attention_mask'].to(device)).logits, src_lengths, None

    def get_sentence_prediction(self,model_output):
        return model_output
