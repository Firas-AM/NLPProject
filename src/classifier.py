from utils.ModelTrainer.ModelTrainer import *
from utils.CustomModels.CustomRoberta import *
from utils.DatasetVectorization.DatasetVectorization import *

from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader


import torch, os


class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please donot change
     """
    def __init__(
        self,
        trainer: object = ModelTrainer,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        model: object = CustomRobertaModel,
        tokenizer: object = RobertaTokenizer,
        optimizer: torch.nn.Module = torch.optim.AdamW,
        scheduler: torch.nn.Module = get_linear_schedule_with_warmup,
        loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss,
        pretrained_encoder: str = "roberta-large", 
        max_length: int = 128,
        training_batch_size: int = 16,
        val_batch_size: int = 16,
        initial_learning_rate: float = 1e-5,
        patience: int = 15, 
        class_weights: Union[np.ndarray, NoneType] = None,
        bert_tokenization: bool = True, 
        input_already_vectorized: bool = False,
        epochs: int = 100, 
        save_folder: str = "./TrainedModels",
        model_name: str = "roberta_large",
      ) -> NoneType:
      self.device = device
      self.model = model
      self.optimizer = optimizer
      self.scheduler = scheduler
      self.initial_learning_rate = initial_learning_rate
      self.loss_fn = loss_fn
      self.training_batch_size = training_batch_size
      self.val_batch_size = val_batch_size
      self.class_weigths = class_weights
      self.bert_tokenization = bert_tokenization
      self.input_already_vectorized = input_already_vectorized
      self.tokenizer = tokenizer
      self.pretrained_encoder = pretrained_encoder
      self.max_length = max_length
      self.model_name = model_name
      self.patience = patience
      self.save_folder = save_folder
      self.epochs = epochs
      self.load_path = os.path.join(
        self.save_folder, 
        f"{self.model_name}-best.pt"
      )
      self.trainer = trainer(
        self.model, 
        self.device, 
        self.optimizer, 
        self.scheduler, 
        self.initial_learning_rate, 
        loss = self.loss_fn,
        training_batch_size = self.training_batch_size,
        bert_tokenization = self.bert_tokenization,
        input_already_vectorized = self.input_already_vectorized,
        bert_tokenizer = self.tokenizer, 
        pretrained_encoder = self.pretrained_encoder,
        max_length = self.max_length,
        model_name = self.model_name,
        patience = self.patience,
        epochs = self.epochs
      )

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
        self.trainer.train(
          train_filename, 
          dev_filename,
        )


    def predict(
          self, 
          data_filename: str, 
          device: torch.device
        ) -> list[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
        predictions = []
        kwargs = self.trainer.kwargs
        test_dataset = VectorizedDataset(
          data_filename, 
          pretrained_encoder= self.pretrained_encoder,
          **kwargs
        )
        polarity_encoder = test_dataset.preprocesser.polarity_encoder
        test_dataloader = DataLoader(
          test_dataset,
        )
        self.model = torch.load(self.load_path)
        with torch.no_grad():
          for vectorized_sentence in test_dataloader:
            input_ids, attention_mask, labels = vectorized_sentence
            attention_mask = attention_mask.to(self.device)
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            prediction = self.model(
              input_ids, 
              attention_mask = attention_mask, 
              labels = labels
            )
            predictions += torch.argmax(
              prediction, 
              axis = 1
            ).tolist()
        decoded_predictions = polarity_encoder.inverse_transform(
          predictions
        )
        return list(decoded_predictions)
        

