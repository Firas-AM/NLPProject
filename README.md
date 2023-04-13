# **NLP Project**

This repository contains the work of our group for the NLP Course Assignment.

A detailed description of the working pipeline can be found below.

#### Description of the Task (**Aspect Category Based Sentiment Analysis**)

As described in the ```nlp_assignment_doc.pdf``` file, the goal of this assignment is to build a Sentiment Analysis model in order to predict the valence of each sentence with respect to some referene Aspect Category.

#### Data Preprocessing

Data preprocessing is a crucial step for any NLP pipeline. The quality of the preprocessing has a very important influence in the performance of the model in the downstream tasks.

As there is not a general and universal recipe for the best preprocessing strategy, it is very important to try several things, and check the performance of each preprocessing decision on a validation dataset.

In our implementation, the preprocessing tasks are handled by the ```DataProcesser``` object, located in the ```.\src\utils\Preprocessing\Preprocessing.py``` file.

A detailed description of the object and its arguments and methods can be found directly in the script, as multi-line comments.

The ```DataProcesser``` object is designed to perform several 
preprocessing steps, with maximum flexibility in the choice of how to do it. In particular, it offers the possibility to:

* lowercase input texts
* remove non alphabetic characters from the input text
* perform word-level tokenization of the input text
* filter out one character words from the input text
* perform lemmatization (by using the ```nltk`` implementation of ```WordNetLemmatizer```)
* perform sub-word tokenization (by using arbitrary type of sub-word tokenizers provided in the ```sentencepiece```)



##### Authors

Hassan AMRAOUI, Lorenzo CONSOLI, Firas MRAD, Jiayi WU
