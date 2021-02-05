import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

class BERTTextModel(nn.Module):

    def __init__(self,args):
        super(BERTTextModel, self).__init__()
        
        self.transformer_model = BertForSequenceClassification.from_pretrained(args.transformer_architecture.split(":")[1], return_dict=True)
        self.tokenizer = BertTokenizer.from_pretrained(args.transformer_architecture.split(":")[1])

        self.window_size = args.sample_window_size

        self.embedding_dropout = torch.nn.Dropout(p=args.dropout)

    def forward(self, x, src_lengths, device):
        
        history = []
        future = []
        for l in x:
            history.append(l[:-self.window_size])
            future.append(l[-self.window_size:])
        encoded_inputs = self.tokenizer(history, future,return_tensors="pt")

        return self.transformer_model(encoded_inputs['input_ids'].to(device), attention_mask=encoded_inputs['attention_mask'].to(device), token_type_ids=encoded_inputs['token_type_ids'].to(device)).logits, src_lengths, None

    def extract_features(self, x, src_lengths, h0=None):
        raise NotImplementedError

    def get_sentence_prediction(self,model_output,lengths, device):
        """
        Returns the model output (which depends on the length),
        for each sample in the batch.

        """
        return model_output
