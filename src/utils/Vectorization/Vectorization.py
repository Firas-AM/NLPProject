from ..Preprocessing.Preprocessing import *
from transformers import BertModel
import torch

class WordVectorizer(object):
    def __init__(
            self,
            file_name: str,
            preprocesser: type = DataProcesser,
            encoder: type = BertModel,
            pretrained_encoder: str = "bert-base-uncased",
            sentence_field: str = "concatenated_sentence",
            polarity_field: str = "polarity"
        ) -> None:
        self.file_name = file_name
        self.encoder = encoder
        self.preprocesser = preprocesser(file_name)
        self.sentence_field = sentence_field
        self.polarity_field = polarity_field
        self.preprocesser.fit()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenized_sentences = self.preprocesser.data_frame[self.sentence_field]
        self.polarities = self.preprocesser.data_frame[self.polarity_field]

    def __to_torch_tensor(
            self, 
            values: list[float] | int,
            target_type: type = torch.int64
        ) -> torch.Tensor:
        return torch.Tensor([values]).to(target_type).to(self.device)

    def __vectorize(
            self, 
            sentence_tensor: torch.Tensor
        ) -> torch.Tensor:
        ...

    def __getitem__(
            self, 
            idx: int
        ) -> tuple[torch.Tensor]:
        sentence = self.tokenized_sentences.iloc[idx]
        polarity = self.polarities.iloc[idx]
        sentence_tensor = self.__to_torch_tensor(sentence)
        sentence_tensor = self.__vectorize(sentence_tensor)
        polarity_tensor = self.__to_torch_tensor(polarity)
        return sentence_tensor, polarity_tensor

    