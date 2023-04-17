from utils.ModelTrainer.ModelTrainer import *
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

from transformers import get_linear_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification
optimizer = AdamW
scheduler = get_linear_schedule_with_warmup
class_weights = np.array([3.,40.,1.])
initial_learning_rate = 2e-5

trainer = ModelTrainer(
    model, 
    device, 
    optimizer, 
    scheduler, 
    class_weights, 
    initial_learning_rate,
    input_already_vectorized = False,
    batch_encode = True
)

train_path = "..\data\traindata.csv"
eval_path = "..\data\devdata.csv"


trainer.train(
    train_path, 
    eval_path
)