from __future__ import annotations

import pandas as pd
import nltk
import re

from typing import Any
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


class DataLoader(object):
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
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
            lemmatizer: Any = WordNetLemmatizer
        ) -> None:

        self.file_name = file_name
        self.sep = sep
        self.header = header
        self.to_lowercase = to_lowercase
        self.to_remove_non_alphabetic_characters = to_remove_non_alphabetic_characters
        self.to_split_sentence = to_split_sentence
        self.to_filter_one_char_words = to_filter_one_char_words
        self.to_lemmatize = to_lemmatize
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
        self.data_frame = self.data_frame.rename(columns = self.rename_map)
        
        self.lemmatizer = lemmatizer()

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

    def preprocess_text(
            self,
        ) -> None:
        """
        
        1) Lowercase the sentence
        2) Replace non-alphabetic characters from sentence
        
        """
        remove_non_alphabetic_characters = lambda s: re.sub("[^a-z0-9\s]", " ", s)
        filter_one_char_words = lambda l: [w for w in l if len(w) > 1]
        
        if self.to_lowercase:
            self.data_frame["sentence"] = self.data_frame["sentence"].map(self.__lowercase)
        if self.to_remove_non_alphabetic_characters:
            self.data_frame["sentence"] = self.data_frame["sentence"].map(self.__remove_non_alphabetic_characters)
        if self.to_split_sentence:
            self.data_frame["sentence"] = self.data_frame["sentence"].map(self.__tokenize_sentence)
        if self.to_filter_one_char_words:
            self.data_frame["sentence"] = self.data_frame["sentence"].map(self.__filter_one_char_words)
        if self.to_lemmatize:
            self.data_frame["sentence"] = self.data_frame["sentence"].map(self.__lemmatize)
        