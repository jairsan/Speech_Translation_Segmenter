from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

from segmenter.models.segmenter_model import HuggingFaceSegmenterModel
from segmenter.model_arguments import add_transformer_arch_name


class XLMRobertaTextModel(HuggingFaceSegmenterModel):
    name: str = "xlm-roberta"

    @staticmethod
    def add_model_args(parser):
        add_transformer_arch_name(parser)

    def __init__(self, args):
        super(XLMRobertaTextModel, self).__init__()
        self.transformer_model = XLMRobertaForSequenceClassification.from_pretrained(args.transformer_model_name,
                                                                                     return_dict=True,
                                                                                     num_labels=args.n_classes)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(args.transformer_model_name)
        self.window_size = args.sample_window_size

    def apply_tokenizer(self, sample):
        new_l = ['nocontext' if x == "</s>" else x for x in sample["words"].split()]

        text_history = " ".join(new_l[:-self.window_size])
        text_future = " ".join(new_l[-self.window_size:])
        encoded_inputs = self.tokenizer(text=text_history, text_pair=text_future, add_special_tokens=True)

        return encoded_inputs

    def forward(self, batch, device):

        return self.transformer_model(batch['input_ids'].to(device),
                                      attention_mask=batch['attention_mask'].to(device)).logits

    def get_sentence_prediction(self, model_output):
        return model_output
