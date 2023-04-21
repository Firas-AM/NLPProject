from utils.ModelTrainer.ModelTrainer import *
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from utils.CustomModels.CustomRoberta import *



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomRobertaModel
tokenizer = RobertaTokenizer
optimizer = torch.optim.AdamW
scheduler = get_linear_schedule_with_warmup
loss_fn = torch.nn.CrossEntropyLoss

train_path =  r"/content/NLPProject/data/traindata.csv"# change this back when running locally "./NLPProject/data/traindata.csv"
eval_path = r"/content/NLPProject/data/devdata.csv"# change this back when running locally "./NLPProject/data/devdata.csv"


initial_learning_rates = 1e-5

for initial_learning_rate in initial_learning_rates:
    trainer = ModelTrainer(
        model, 
        device, 
        optimizer, 
        scheduler, 
        initial_learning_rate,
        loss = loss_fn,
        training_batch_size = 16,
        val_batch_size = 16, 
        class_weights = None,
        bert_tokenization = True, 
        input_already_vectorized = False,
        bert_tokenizer = tokenizer,
        pretrained_encoder = "roberta-large",
        max_length = 128,
        model_name = f"Roberta-lr{initial_learning_rate}",
        patience = 5,
    )

    trainer.train(
        train_path, 
        eval_path
    )
