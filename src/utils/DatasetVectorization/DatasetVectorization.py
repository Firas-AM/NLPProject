from ..Preprocessing.Preprocessing import *

from transformers import BertModel, AutoTokenizer, BertTokenizer
from torch.utils.data import Dataset

import torch


class VectorizedDataset(Dataset):
    def __init__(
            self,
            file_name: str,
            preprocesser: object = DataProcesser,
            bert_tokenization: bool = False,
            batch_encode: bool = False,
            bert_tokenizer: object = AutoTokenizer, 
            pretrained_encoder: str = "bert-base-uncased",
            sentence_field: str = "concatenated_sentence",
            polarity_field: str = "polarity",
            encoder: object = BertModel,
        ) -> NoneType:
        self.file_name = file_name
        self.preprocesser = preprocesser(file_name)
        self.bert_tokenization = bert_tokenization
        self.pretrained_encoder = pretrained_encoder
        self.sentence_field = sentence_field
        self.polarity_field = polarity_field
        self.encoder = encoder.from_pretrained(self.pretrained_encoder)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.encoder = self.encoder.to(self.device)
        self.batch_encode = batch_encode
        if not self.bert_tokenization:
            self.preprocesser.fit()
        else: 
            self.bert_tokenizer = bert_tokenizer.from_pretrained(self.pretrained_encoder)
        self.tokenized_sentences = self.preprocesser.data_frame[self.sentence_field]
        self.polarities = self.preprocesser.data_frame[self.polarity_field]

    def __to_torch_tensor(
            self, 
            values: Union[list[float], int],
            target_type: object = torch.int64
        ) -> torch.Tensor:
        return torch.Tensor([values]).to(target_type).to(self.device)

    def __create_encoder_input(
            self, 
            sentence_tensor: torch.Tensor,
            target_type: object = torch.int64
        ) -> torch.Tensor:
        token_type_ids = torch.zeros_like(sentence_tensor).to(target_type).to(self.device)
        attention_mask = torch.ones_like(sentence_tensor).to(target_type).to(self.device)
        encoder_input_dictionary = {
            "input_ids": sentence_tensor, 
            "token_type_ids": token_type_ids, 
            "attention_mask": attention_mask
        } 
        return encoder_input_dictionary

    def __encode(
            self, 
            sentence: Union[torch.Tensor, str], 
            target_type: object = torch.int64,
            max_length: int = 128,
            padding_type: str = "max_length",
            truncation: bool = True,
            return_tensors: str = "pt"
        ) -> torch.Tensor:
        if not self.bert_tokenization:
            encoder_input = self.__create_encoder_input(
                sentence, 
                target_type = target_type
            )
        elif self.bert_tokenization and not self.batch_encode:
            assert isinstance(self.bert_tokenizer, AutoTokenizer), "The given tokenizer is not of the right type, a AutoTokenizer is expected"
            encoder_input = self.bert_tokenizer(
                sentence, 
                return_tensors = return_tensors
            )
        elif self.bert_tokenization and self.batch_encode:
            assert isinstance(self.bert_tokenizer, BertTokenizer), "The given tokenizer is not of the right type, a BertTokenizer is expected"
            print("performing berto tokenization with batch encoding plus")
            encoded_input = self.bert_tokenizer.batch_encode_plus(
                sentence, 
                max_length = max_length,
                padding = padding_type,
                truncation = truncation, 
                return_tensors = return_tensors
            )
            return encoded_input['input_ids'], encoded_input['attention_mask']
        encoder_input = {key: value for key, value in encoder_input.items()}#encoder_input = {key: value.to(self.device) for key, value in encoder_input.items()}
        encoded_sentence = self.encoder(**encoder_input)
        return encoded_sentence.last_hidden_state

    def __len__(
            self
        ) -> int:
        return len(self.preprocesser.data_frame)

    def __getitem__(
            self, 
            idx: int
        ) -> tuple[torch.Tensor]:
        sentence = self.tokenized_sentences.iloc[idx]
        if not self.bert_tokenization:
            sentence = self.__to_torch_tensor(
                sentence
            )
        polarity = self.polarities.iloc[idx]
        sentence = self.__encode(
            sentence
        )
        polarity_tensor = self.__to_torch_tensor(
            polarity
        )
        return sentence, polarity_tensor