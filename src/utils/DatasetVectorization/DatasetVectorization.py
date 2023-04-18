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
            bert_tokenizer: object = AutoTokenizer, 
            pretrained_encoder: str = "bert-base-uncased",
            sentence_field: str = "concatenated_sentence",
            polarity_field: str = "polarity",
            max_length: int = 128,
            padding_type: str = "max_length",
            truncation: bool = True,
            return_tensors: str = "pt",
        ) -> NoneType:
        self.file_name = file_name
        self.preprocesser = preprocesser(file_name)
        self.bert_tokenization = bert_tokenization
        self.pretrained_encoder = pretrained_encoder
        self.sentence_field = sentence_field
        self.polarity_field = polarity_field
        self.max_length = max_length
        self.padding_type = padding_type
        self.truncation = truncation
        self.return_tensors = return_tensors
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_tokenizer = bert_tokenizer.from_pretrained(self.pretrained_encoder)
        # tokenize sentences all at once using custom tokenization
        if not self.bert_tokenization:
            self.preprocesser.fit()
        self.tokenized_sentences = self.preprocesser.data_frame[self.sentence_field]
        self.polarities = self.preprocesser.data_frame[self.polarity_field]
        # tokenize sentences all at once using hugging_face tokenizers
        if self.bert_tokenization:
            self.__pre_tokenize_sentences()

    def __pre_tokenize_sentences(
            self
        )-> NoneType:
        encoded_input = self.bert_tokenizer(
            self.tokenized_sentences.tolist(), 
            max_length = self.max_length,
            padding = self.padding_type,
            truncation = self.truncation, 
            return_tensors = self.return_tensors
        )
        self.tokenized_sentences = encoded_input['input_ids']
        self.attention_masks = encoded_input['attention_mask']

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
        ) -> torch.Tensor:
        encoder_input = self.__create_encoder_input(
            sentence, 
            target_type = target_type
        )
        encoded_sentence = self.bert_tokenizer(**encoder_input)
        return encoded_sentence.last_hidden_state
        
    def __len__(
            self
        ) -> int:
        return len(self.preprocesser.data_frame)

    def __get_already_tokenized_item(
            self, 
            idx: int
        ) -> tuple[torch.Tensor]:
        
        sentence = self.tokenized_sentences[idx]
        attention_mask = self.attention_masks[idx]
        return sentence, attention_mask

    def __get_and_tokenize_sentence(
            self, 
            idx: int
        ) -> torch.Tensor:
        sentence = self.tokenized_sentences.iloc[idx]
        sentence = self.__to_torch_tensor(
            sentence
        )   
        sentence = self.__encode(
            sentence
        ) 
        return sentence

    def __getitem__(
            self, 
            idx: int
        ) -> tuple[torch.Tensor]:
        polarity = self.polarities.iloc[idx]
        polarity_tensor = self.__to_torch_tensor(
            polarity
        )
        if not self.bert_tokenization:
            sentence = self.__get_and_tokenize_sentence(idx)
            return sentence, polarity_tensor    
        else:   
            sentence, attention_mask =  self.__get_already_tokenized_item(idx)
            return sentence, attention_mask, polarity_tensor