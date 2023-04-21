# **NLP Project**

This repository contains the work of our group for the NLP Course Assignment.

A detailed description of the working pipeline can be found below.

## Description of the Task (**Aspect Category Based Sentiment Analysis**)

As described in the ```nlp_assignment_doc.pdf``` file, the goal of this assignment is to build a Sentiment Analysis model in order to predict the valence of each sentence with respect to some referene Aspect Category.

In order to perform this fine-grained Sentuiment Analysis Task, we decided to incorporate the information about the aspect categories in the following way:

Given a character offset (i.e.: ```dataset["character_offset"]```), indicating the beginning and the end of the target word for a given aspect category (i.e.: ```dataset["aspect_category"]```), we inserted the string representing the aspect category around the target word.

For example, given:
* the sentence: "short and sweet – seating is great:it's romantic,cozy and private." 
* the aspect: category "AMBIENCE#GENERAL" 
* the character: offset "18:25", 

our new representation of the target sentence that would take into account both the aspect category and the target word would be:

* "short and sweet –  AMBIENCE#GENERAL seating AMBIENCE#GENERAL  is great:it's romantic,cozy and private."

## Data Preprocessing

Data preprocessing is a crucial step for any NLP pipeline. The quality of the preprocessing has a very important influence in the performance of the model in the downstream tasks.

As there is not a general and universal recipe for the best preprocessing strategy, it is very important to try several things, and check the performance of each preprocessing decision on a validation dataset.

In our implementation, the preprocessing tasks are handled by the ```DataProcesser``` object, located in the ```.\src\utils\Preprocessing\Preprocessing.py``` file.

A detailed description of the object and its arguments and methods can be found directly in the script, as multi-line comments.

The ```DataProcesser``` object is designed to perform several 
preprocessing steps, with maximum flexibility in the choice of how to do it. In particular, it offers the possibility to:


* lowercase input texts
* remove non alphabetic characters from the input text (TODO: Remove Stopwords)
* perform word-level tokenization of the input text
* filter out one character words from the input text
* perform lemmatization (by using the ```nltk`` implementation of ```WordNetLemmatizer```)
* perform sub-word tokenization (by using arbitrary type of sub-word tokenizers provided in the ```sentencepiece```)


## Feature Extraction

For feature extraction, we used the pre-trained RoBERTa tokenizer to generate input IDs and attention masks. The tokenizer performed better than other techniques, such as removing punctuation, stopwords, tokenizing, lemmatizing, and vectorizing. The model used in this project is RoBERTa-large.
To improve the tokenizer's ability to identify the target, we performed some preprocessing steps:
First, we added special tokens before and after the target term in the sentence to emphasize it. We then transformed the category column into questions manually and added the same special tokens before and after the category question. We then concatenated the category question column and the sentence with special token column together. Our final step was to pass that to the tokenizer, which generated input IDs and attention masks that were used in the RoBERTa transformer model.
For the tokenizer hyperparameters, we set the max length for padding to 128 to ensure every sentence is mapped to the same length, and add_special_tokens=True, which will add other special tokens to the beginning and end of the sentence.

## Models

For the model, we imported RoBERTaModel and added a classifier head ourselves, consisting of a linear layer, a dropout, and another linear layer. 

We used the AdamW optimizer with a linear scheduler so that the learning rate would decrease to 0 with the epochs and our loss function was CrossEntropyLoss.

Since the data is very unbalanced, we used a function to compute class weights and passed these weights to the loss function. This ensured that classes with fewer samples had more weight when computing losses. 

We set the training epoch by a function, and stopped the training when the loss didn’t decrease for 5 epochs. We also had a patience attribute set to 15 to ensure at least 15 epochs of training.

Finally, we wrote a function to save the model with the best accuracy, this function overwrites the current model if a better model is found in subsequent epochs.


#### Authors

Hassan AMRAOUI, Lorenzo CONSOLI, Firas ABO MRAD, Jiayi WU
