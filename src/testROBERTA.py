from utils.ModelTrainer.ModelTrainer import *
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW,  get_linear_schedule_with_warmup


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RobertaForSequenceClassification
tokenizer = RobertaTokenizer
optimizer = AdamW
scheduler = get_linear_schedule_with_warmup

class_weights = torch.FloatTensor(class_weights).to(device)

initial_learning_rate = 2e-5

trainer = ModelTrainer(
    model, 
    device, 
    optimizer, 
    scheduler, 
    initial_learning_rate,
    class_weights = None,
    bert_tokenization = True, 
    input_already_vectorized = False,
    bert_tokenizer = tokenizer,
    pretrained_encoder = "roberta-base",
    max_length = 256 
)
train_path = "./NLPProject/data/traindata.csv" ## change this back when running locally
eval_path = "./NLPProject/data/devdata.csv" ## change this back when running locally


trainer.train(
    train_path, 
    eval_path
)