from __future__ import annotations

import pandas as pd
import sentencepiece as spm

import nltk
import re

from typing import Any
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.preprocessing import LabelEncoder

class DataProcesser(object):
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    def __init__(
            self,
            file_name: str,
            sep: str = "\t",
            header: Any = None,
            to_lowercase: bool = True, 
            to_remove_non_alphabetic_characters: bool = True,
            to_split_sentence: bool = True,
            to_filter_one_char_words: bool = True,
            to_lemmatize: bool = True,
            to_subword_tokenize: bool = True,
            to_remove_stopwords: bool = True,
            encode_as_ids: bool = True,
            polarity_encoder: Any = LabelEncoder,
            lemmatizer: Any = WordNetLemmatizer,
            subword_tokenizer_trainer: Any = spm.SentencePieceTrainer,
            subword_tokenizer: Any = spm.SentencePieceProcessor,
            target_vocabulary_size: int = 5000, 
            corpus_dump_file: str = "../data/corpus.txt",
            tokenizer_model_type: str ="bpe",
            tokenizer_model_prefix: str = "subword_tokenizer"
        ) -> None:
        self.file_name = file_name
        self.sep = sep
        self.header = header
        self.to_lowercase = to_lowercase
        self.to_remove_non_alphabetic_characters = to_remove_non_alphabetic_characters
        self.to_split_sentence = to_split_sentence
        self.to_filter_one_char_words = to_filter_one_char_words
        self.to_lemmatize = to_lemmatize
        self.to_subword_tokenize = to_subword_tokenize
        self.to_remove_stopwords = to_remove_stopwords
        self.encode_as_ids = encode_as_ids
        self.stopwords = stopwords.words('english')
        self.polarity_encoder = polarity_encoder()
        self.corpus_dump_file = corpus_dump_file
        self.lemmatizer = lemmatizer()
        self.subword_tokenizer_trainer = subword_tokenizer_trainer
        self.target_vocabulary_size = target_vocabulary_size
        self.tokenizer_model_type = tokenizer_model_type
        self.tokenizer_model_prefix = tokenizer_model_prefix
        self.rename_map = {
            0: "polarity",
            1: "aspect_category",
            2: "target_term",
            3: "character_offset",
            4: "sentence"
        }
        self.data_frame = pd.read_csv(
            self.file_name, 
            sep = self.sep,
            header = self.header
        )
        self.data_frame = self.data_frame.\
            rename(columns = self.rename_map)
        self.__write_corpus_file()
        self.__train_subword_tokenizer()
        self.__encode_polarity_categories()
        self.aspect_categories = self.data_frame["aspect_category"].\
            unique()   
        self.aspect_categories_map = {category: f"C{idx}" for idx, category in enumerate(self.aspect_categories)} 
        self.data_frame["aspect_category"] = self.data_frame["aspect_category"].\
            map(self.aspect_categories_map)
        self.data_frame["character_offset_start"] = self.data_frame["character_offset"].\
            map(self.__find_offset_start)
        self.data_frame["character_offset_end"] = self.data_frame["character_offset"].\
            map(self.__find_offset_end)
        self.data_frame["concatenated_sentence"] = self.data_frame.\
            apply(self.__insert_aspect_tokens, axis = 1)
        self.data_frame = self.data_frame.drop(
            [
                "character_offset",
                "character_offset_start", 
                "character_offset_end",
                "aspect_category",
                "sentence",
                "target_term"
            ], 
            axis = 1
        ) 
        self.subword_tokenizer = subword_tokenizer()
        self.subword_tokenizer.load("./subword_tokenizer.model")

    @staticmethod
    def __find_offset_start(
            offset_string: str,
        ) -> int:
        start_offset = re.findall("(\d*):\d*", offset_string)[0]
        return int(start_offset)

    @staticmethod
    def __find_offset_end(
            offset_string: str,
        ) -> int:
        end_offset = re.findall("\d*:(\d*)", offset_string)[0]
        return int(end_offset)

    @staticmethod
    def __insert_aspect_tokens(
            data_frame_row: pd.core.series.Series,
            aspect_chategory_field: str = "aspect_category", 
            sentence_field: str = "sentence",
            character_offset_start_field: str  = "character_offset_start",
            character_offset_end_field: str  = "character_offset_end"
        ) -> str:
        sentence = data_frame_row[sentence_field]
        offset_start = data_frame_row[character_offset_start_field]
        offset_end = data_frame_row[character_offset_end_field]
        category = data_frame_row[aspect_chategory_field]
        concatenated_string = f"{sentence[:offset_start]} {category} {sentence[offset_start: offset_end]} {category} {sentence[offset_end: ]}"
        return concatenated_string
    
    @staticmethod
    def __lowercase(
            sentence: str
        ) -> str:
        return sentence.lower()

    @staticmethod
    def __remove_non_alphabetic_characters(
            sentence: str
        ) -> str:
        return re.sub("[^a-z0-9\s]", " ", sentence)
    
    @staticmethod
    def __tokenize_sentence(
            sentence: str
        ) -> list[str]:
        return nltk.word_tokenize(sentence)

    @staticmethod
    def __detokenize_sentence(
            tokenized_sentence: list[str]
        ) -> str:
        detokenizer = TreebankWordDetokenizer()
        detokenized_sentence = detokenizer.detokenize(tokenized_sentence)
        return detokenized_sentence

    @staticmethod
    def __filter_one_char_words(
            tokenized_sentence: list[str]
        ) -> list[str]:
        return [word for word in tokenized_sentence if len(word) > 1]

    @staticmethod
    def __get_wordnet_pos(
            treebank_tag: str
        ) -> str | None:
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    @staticmethod
    def __tag_word_pos(
            tokenized_sentence: list[str]
        ) -> list[tuple[str]]:
        pos_tags = nltk.pos_tag(tokenized_sentence)
        return pos_tags

    def __encode_polarity_categories(
            self
        ) -> None:
        polarities = self.data_frame["polarity"].values
        encoded_polarities = self.polarity_encoder.fit_transform(polarities)
        self.data_frame["polarity"] = encoded_polarities

    def __filter_stopwords(
            self, 
            tokenized_sentence: list[str]
        ) -> list[str]:
        return [word for word in tokenized_sentence if word not in self.stopwords]
        
    def __lemmatize_word(
            self, 
            word: str, 
            pos: str | None
        ) -> str:
        pos = self.__get_wordnet_pos(pos)
        return self.lemmatizer.lemmatize(word, pos) if pos \
            else word

    def __lemmatize(
            self,
            tokenized_sentence: list[str]
        ) -> list[str]:
        tagged_sentence = self.__tag_word_pos(tokenized_sentence)
        lemmatized_sentence = [
            self.__lemmatize_word(word, pos) for word, pos in tagged_sentence
        ]
        return lemmatized_sentence

    def __write_corpus_file(
            self,
        ) -> None:
        with open(self.corpus_dump_file, "w+") as dump_file_handle:
            for sentence in self.data_frame["sentence"]:
                dump_file_handle.write(sentence)
                dump_file_handle.write("\n")

    def __train_subword_tokenizer(
            self, 
        ) -> None:
        train_arg_string = f"--input={self.corpus_dump_file} --vocab_size={self.target_vocabulary_size} --model_prefix={self.tokenizer_model_prefix} --model_type={self.tokenizer_model_type}"
        self.subword_tokenizer_trainer.train(train_arg_string)

    def __subword_tokenize_sentence(
            self, 
            sentence: str
        ) -> list[str]:
        tokenized_sentence = self.subword_tokenizer.encode_as_ids(sentence) if self.encode_as_ids \
            else self.subword_tokenizer.encode_as_pieces(sentence)
        return tokenized_sentence

    def __preprocess_text(
            self,
        ) -> None:
        if self.to_lowercase:
            self.data_frame["concatenated_sentence"] = self.data_frame["concatenated_sentence"].\
                map(self.__lowercase)
        if self.to_remove_non_alphabetic_characters:
            self.data_frame["concatenated_sentence"] = self.data_frame["concatenated_sentence"].\
                map(self.__remove_non_alphabetic_characters)
        if self.to_split_sentence:
            self.data_frame["concatenated_sentence"] = self.data_frame["concatenated_sentence"].\
                map(self.__tokenize_sentence)
        if self.to_filter_one_char_words:
            self.data_frame["concatenated_sentence"] = self.data_frame["concatenated_sentence"].\
                map(self.__filter_one_char_words)
        if self.to_remove_stopwords:
            self.data_frame["concatenated_sentence"] = self.data_frame["concatenated_sentence"].\
                map(self.__filter_stopwords)
        if self.to_lemmatize:
            self.data_frame["concatenated_sentence"] = self.data_frame["concatenated_sentence"].\
                map(self.__lemmatize)
        if self.to_subword_tokenize and self.tokenizer_model_type != "word":
            self.data_frame["concatenated_sentence"] = self.data_frame["concatenated_sentence"].\
                map(self.__detokenize_sentence).\
                map(self.__subword_tokenize_sentence)
        if self.to_subword_tokenize and self.tokenizer_model_type == "word":
            self.data_frame["concatenated_sentence"] = self.data_frame["concatenated_sentence"].\
                map(self.__subword_tokenize_sentence)
    
    def fit(
            self,
            preprocess_text: bool = True, 
            encode_aspect_categories: bool = True
        ) -> None:
        self.__preprocess_text()