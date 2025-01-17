from utils.ModelTrainer.ModelTrainer import *
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification
optimizer = AdamW
scheduler = get_linear_schedule_with_warmup
class_weights = None#np.array([3.,40.,1.])
initial_learning_rate = 2e-5

trainer = ModelTrainer(
    model, 
    device, 
    optimizer, 
    scheduler, 
    class_weights, 
    initial_learning_rate,
    bert_tokenization = True, 
    input_already_vectorized = False,
    bert_tokenizer = BertTokenizer,
    pretrained_encoder = "bert-base-uncased",
    model_name = "Bert"
)
train_path = "./NLPProject/data/traindata.csv" ## change this back when running locally
eval_path = "./NLPProject/data/devdata.csv" ## change this back when running locally


trainer.train(
    train_path, 
    eval_path
)