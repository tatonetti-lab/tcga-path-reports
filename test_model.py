#Apply Best Trained Model on Held-Out Test Set 
import numpy as np
import pandas as pd
import torch
import random
import time, os, pickle, glob, sys
from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
from scipy.special import softmax
import matplotlib.pyplot as plt
from eval_metrics import get_metrics

#Model Checkpoint 
dir_list = pickle.load(open('testing_directories.p','rb')) 
model_output_dir = dir_list[int(sys.argv[1])]
tissue = model_output_dir.split('/')[-2].split('_')[0]
model_output_dir_glob = glob.glob(model_output_dir + 'checkpoint-*/')
if len(model_output_dir_glob) == 1: 
    checkpoint_dir = model_output_dir_glob[0]
else:
    print('Error: Exists >1 checkpoint for Tissue/Random Seed Combination')

#Held-Out Test Set Directory 
input_dir = '../Target_Selection/final_data_proto_full/'
test_pickle = input_dir + 'Target_Data'+'_'+tissue+'_test.p' 

#Output Directory 
test_output_dir = model_output_dir + 'test_best_model_evaluate_tmp/'
os.makedirs(test_output_dir, exist_ok=True)

#Define Dataset Class 
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

seed=0
np.random.seed(seed) 
torch.manual_seed(seed) 
random.seed(seed)

#Load Tokenizer and Model from Checkpoint 
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
test_model = BertForSequenceClassification.from_pretrained(checkpoint_dir, num_labels=2, local_files_only=True) 

#Load Test Data
test_data = pickle.load(open(test_pickle,'rb')).sample(frac=1)
X_test = list(test_data["text"])
X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)
test_dataset = Dataset(X_test_tokenized)
y_true = list(test_data['label'])

#Run Model 
test_trainer = Trainer(model=test_model, args=TrainingArguments(output_dir = test_output_dir))
y_pred, _, _  = test_trainer.predict(test_dataset)

#Evaluation
ac, au_roc, au_prc, n_pred_pos, actual_pos  = get_metrics(y_pred, y_true, output_dir=model_output_dir, suffix='test_best_model', plot=True)
print('\nAccuracy - ', round(ac,4), 'AU-ROC - ', round(au_roc,4), 'AU-PRC - ', round(au_prc,4)) 
