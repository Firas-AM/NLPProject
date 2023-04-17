import torch
import gc

import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report

from ..DatasetVectorization.DatasetVectorization import *


class ModelTrainer(object):
    def __init__(
            self,
            model: object,
            device: torch.device,
            optimizer: torch.nn.Module, 
            scheduler: torch.nn.Module,
            class_weights: Union[np.ndarray, NoneType],
            initial_learning_rate: float,
            pretrained_encoder: str = "bert-base-uncased",
            loss: Union[torch.nn.Module, NoneType] = None,
            training_batch_size: int = 32, 
            val_batch_size: int = 32,
            num_labels: int = 3,
            input_already_vectorized: bool = True,
            epochs: int = 100,
            patience: int = 5,
            **kwargs # for vectorized dataset
        ) -> NoneType:
        self.model = model
        self.device = device
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.class_weights = class_weights
        self.initial_learning_rate = initial_learning_rate
        self.pretrained_encoder = pretrained_encoder
        self.training_batch_size = training_batch_size
        self.val_batch_size = val_batch_size
        self.num_labels = num_labels
        self.input_already_vectorized = input_already_vectorized
        self.epochs = epochs 
        self.patience = patience
        self.epochs_with_no_improvement = 0
        self.train_losses = [float("inf")] 
        self.train_accuracies = []
        self.train_precisions = []
        self.train_recalls = []
        self.train_f1s = []
        self.val_losses = [float("inf")]
        self.val_accuracies = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_f1s = []
        self.kwargs = kwargs # for vectorized dataset
    
    def __track_validation_progress(
            self,
        ) -> None:
        previous_loss = self.val_losses[-2]
        current_loss = self.val_losses[-1]
        if current_loss <= previous_loss:
            self.epochs_with_no_improvement = 0
        else: 
            self.epochs_with_no_improvement += 1

    def __init_training(
            self,
            train_dataset: Dataset, 
            val_dataset: Dataset,
            n_warmup_steps: int = 0 
        ) -> NoneType:
        num_training_steps = len(train_dataset) * self.epochs
        self.model = self.model.from_pretrained(self.pretrained_encoder, num_labels = self.num_labels)
        self.model.to(self.device)
        self.class_weights = torch.Tensor(self.class_weights).to(self.device) if self.class_weights is not None\
            else None
        self.loss = self.loss(weight = self.class_weights) if self.loss\
            else None
        self.optimizer = self.optimizer(self.model.parameters(), lr = self.initial_learning_rate)
        self.scheduler = self.scheduler(self.optimizer, num_warmup_steps = n_warmup_steps, num_training_steps = num_training_steps)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size = self.training_batch_size
        )
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size = self.val_batch_size
        )
        return train_dataloader, val_dataloader

    def __training_pass(
            self, 
            train_dataloader: DataLoader, 
        ) -> float:
        train_loss = 0
        self.model.train()
        train_preds = []; train_labels = []
        for batch in tqdm(train_dataloader):
            if self.input_already_vectorized:
                inputs_ids, labels = batch
            else:
                input_ids, attention_mask, labels = batch
                attention_mask = attention_mask.to(self.device)
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, labels = labels) if self.input_already_vectorized\
                else self.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = self.loss(outputs, labels) if self.loss\
                else outputs.loss
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            train_loss += loss.item()
            train_preds += torch.argmax(logits, axis=1).tolist()
            train_labels += labels.tolist()
            del input_ids
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        train_loss /= len(train_dataloader.dataset)
        train_accuracy = accuracy_score(train_labels, train_preds)
        train_precision = precision_score(train_labels, train_preds, average='weighted', zero_division=1)
        train_recall = recall_score(train_labels, train_preds, average='weighted')
        train_f1 = f1_score(train_labels, train_preds, average='weighted')
        return train_loss, train_accuracy, train_precision, train_recall, train_f1
    
    def __eval_pass(
            self, 
            eval_dataloader: DataLoader
        ) -> tuple[float]:
        self.model.eval()
        eval_loss = 0
        val_preds = []; val_labels = []
        with torch.no_grad():
            for batch in eval_dataloader:
                if self.input_already_vectorized:
                    inputs_ids, labels = batch
                else: 
                    input_ids, attention_mask, labels = batch
                    attention_mask = attention_mask.to(self.device)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(input_ids, labels = labels) if self.input_already_vectorized\
                    else self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = self.loss(outputs, labels) if self.loss\
                    else outputs.loss
                eval_loss += loss.item() * input_ids.size(0)
                val_preds += torch.argmax(logits, axis=1).tolist()
                val_labels += labels.tolist()
                del input_ids
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        eval_loss /= len(eval_dataloader.dataset)    
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, average='weighted', zero_division=1)
        val_recall = recall_score(val_labels, val_preds, average='weighted')
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        return eval_loss, val_accuracy, val_precision, val_recall, val_f1
        
    def __epoch(
            self,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader
        ) -> tuple[float]:
        train_logs = self.__training_pass(train_dataloader)
        val_logs = self.__eval_pass(eval_dataloader)
        return train_logs, val_logs

    def __train(
            self, 
            train_dataset: Dataset, 
            val_dataset: Dataset,
        ) -> NoneType:
        train_dataloader, val_dataloader = self.__init_training(
            train_dataset, 
            val_dataset,
        )
        for epoch in range(self.epochs):
            train_logs, val_logs = self.__epoch(train_dataloader, val_dataloader)
            train_loss, train_accuracy, train_precision, train_recall, train_f1 = train_logs
            eval_loss, val_accuracy, val_precision, val_recall, val_f1 = val_logs

            print(f'Epoch {epoch+1} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f} - Val Precision: {val_precision:.4f} - Val Recall: {val_recall:.4f} - Val F1: {val_f1:4f}')

            self.train_losses.append(train_loss); self.train_accuracies.append(train_accuracy)
            self.train_precisions.append(train_precision); self.train_recalls.append(train_recall)
            self.train_f1s.append(train_f1)
            self.val_losses.append(val_loss); self.val_accuracies.append(val_accuracy)
            self.val_precisions.append(val_precision); self.val_recalls.append(val_recall)
            self.val_f1s.append(val_f1)
            self.__track_validation_progress()
            if self.epochs_with_no_improvement == self.patience:
                break
            
    def train(
            self,
            train_path: str, 
            eval_path: str,
        ) -> NoneType:
        train_dataset = VectorizedDataset(train_path, **self.kwargs)
        eval_dataset = VectorizedDataset(eval_path, **self.kwargs)
        self.__train(train_dataset, eval_dataset)