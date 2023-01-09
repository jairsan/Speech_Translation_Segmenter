from transformers import BertTokenizer, BertForSequenceClassification

from segmenter.models.segmenter_model import HuggingFaceSegmenterModel
from segmenter.model_arguments import add_transformer_arch_name


class BERTTextModel(HuggingFaceSegmenterModel):
    name: str = "bert"

    @staticmethod
    def add_model_args(parser):
        add_transformer_arch_name(parser)

    def __init__(self, args):
        super(BERTTextModel, self).__init__()
        
        self.transformer_model = BertForSequenceClassification.from_pretrained(args.transformer_model_name,
                                                                               return_dict=True)
        self.tokenizer = BertTokenizer.from_pretrained(args.transformer_model_name)

        self.window_size = args.sample_window_size

    def apply_tokenizer(self, sample):
        new_l = ['nocontext' if x == "</s>" else x for x in sample["words"].split()]
        a = new_l[:-self.window_size] + ["</s>"] + new_l[-self.window_size:]
        text = " ".join(a)
        encoded_inputs = self.tokenizer(text)
        return encoded_inputs

    def forward(self, batch, device):
        history = []
        future = []
        for l in batch:
            history.append(l[:-self.window_size])
            future.append(l[-self.window_size:])
        encoded_inputs = self.tokenizer(history, future, return_tensors="pt")

        return self.transformer_model(encoded_inputs['input_ids'].to(device),
                                      attention_mask=encoded_inputs['attention_mask'].to(device),
                                      token_type_ids=encoded_inputs['token_type_ids'].to(device)).logits

    def get_sentence_prediction(self, model_output):
        return model_output
