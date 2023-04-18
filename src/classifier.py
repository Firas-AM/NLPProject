from typing import List
from .DatasetVectorization.DatasetVectorization import *

import torch


class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please donot change
     """
    def __init__(
        self, 
        model: object
      ) -> NoneType:
      ...
      


    ############################################# comp
    def train(
          self, 
          train_filename: str, 
          dev_filename: str,
          device: torch.device
        ) -> NoneType:
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        train_dataset = VectorizedDataset(
          train_filename
        )
        val_dataset = VectorizedDataset(
          dev_filename
        )


    def predict(
          self, 
          data_filename: str, 
          device: torch.device
        ) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """      