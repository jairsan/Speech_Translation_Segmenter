import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

class XLMRobertaTextModel(nn.Module):

    def __init__(self,args):
        super(XLMRobertaTextModel, self).__init__()
        
        self.transformer_model = XLMRobertaForSequenceClassification.from_pretrained(args.transformer_architecture.split(":")[1], return_dict=True)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(args.transformer_architecture.split(":")[1])
        #self.tokenizer.add_special_tokens({'additional_special_tokens':['[NO_CONTEXT]']})

        self.window_size = args.sample_window_size

        self.embedding_dropout = torch.nn.Dropout(p=args.dropout)

    def forward(self, x, src_lengths, device):
        
        samples = []
        for l in x:
            new_l = ['nocontext' if x == "</s>" else x for x in l]
            a = new_l[:-self.window_size]+ ["</s>"] + new_l[-self.window_size:]
            samples.append(" ".join(a))
        encoded_inputs = self.tokenizer(samples,padding=True,return_tensors="pt")
        return self.transformer_model(encoded_inputs['input_ids'].to(device), attention_mask=encoded_inputs['attention_mask'].to(device)).logits, src_lengths, None

    def extract_features(self, x, src_lengths, h0=None):
        raise NotImplementedError

    def get_sentence_prediction(self,model_output,lengths, device):
        """
        Returns the model output (which depends on the length),
        for each sample in the batch.

        """
        return model_output
