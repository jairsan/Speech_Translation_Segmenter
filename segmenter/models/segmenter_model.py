from typing import Dict
import torch
import torch.nn as nn


class SegmenterModel(nn.Module):
    def forward(self, batch: Dict, device: torch.device) -> torch.Tensor:
        raise NotImplementedError

    def extract_features(self, batch: Dict, device: torch.device) -> torch.Tensor:
        """
        Extract the features computed by the model (ignoring the output layer)

        """
        raise NotImplementedError

    def get_sentence_prediction(self, model_output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
