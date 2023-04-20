import torch
import sys

from transformers import RobertaModel
from typing import Union, Any

try:
    if sys.version_info >= (3, 10):
        from types import NoneType
    else:
        from builtins import NoneType
except ImportError:
    NoneType = type(None)


class CustomRobertaModel(torch.nn.Module):
    def __init__(
            self,
            pretrained_weights: str = "roberta-large",
            linear_units: int = 768,
            num_labels: int = 3,
            dropout_rate: float = 1e-1
        ) -> NoneType:
        super(CustomRobertaModel, self).__init__()
        self.pretrained_weights = pretrained_weights
        self.roberta = RobertaModel.from_pretrained(self.pretrained_weights)
        self.linear_units = linear_units
        self.num_labels = num_labels
        self.dropout_rate = dropout_rate 
        self.linear1 = torch.nn.Linear(self.linear_units, self.linear_units)
        #self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.linear2 = torch.nn.Linear(self.linear_units, self.num_labels)

    def forward(
            self, 
            input_ids: torch.Tensor, 
            attention_mask: Union[torch.Tensor, NoneType] = None
        ) -> torch.Tensor:
        output = self.roberta(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        pooled_output = output.pooler_output
        pooled_output = self.linear1(pooled_output)
        #pooled_output = self.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.linear2(pooled_output)
        return logits
    
    def __call__(
            self, 
            input_ids: torch.Tensor,
            attention_mask: Union[torch.Tensor, NoneType] = None,
            labels: Any = None
        ) -> torch.Tensor:
        return self.forward(input_ids, attention_mask = attention_mask)