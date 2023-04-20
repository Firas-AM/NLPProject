from utils.ModelTrainer.ModelTrainer import *
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from utils.CustomModels.CustomRoberta import *



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomRobertaModel
tokenizer = RobertaTokenizer
optimizer = torch.optim.AdamW
scheduler = get_linear_schedule_with_warmup
loss_fn = torch.nn.CrossEntropyLoss


initial_learning_rate = 2e-5

trainer = ModelTrainer(
    model, 
    device, 
    optimizer, 
    scheduler, 
    initial_learning_rate,
    loss = loss_fn,
    class_weights = None,
    bert_tokenization = True, 
    input_already_vectorized = False,
    bert_tokenizer = tokenizer,
    pretrained_encoder = "roberta-base",
    max_length = 256,
    model_name = "Roberta",
    patience = 15
)
train_path = r".\data\traindata.csv"#"./NLPProject/data/traindata.csv" ## change this back when running locally
eval_path = r".\data\traindata.csv"#"./NLPProject/data/devdata.csv" ## change this back when running locally


trainer.train(
    train_path, 
    eval_path
)