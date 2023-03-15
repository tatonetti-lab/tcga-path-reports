#Fine-Tune ClinicalBERT on TCGA Pathology Reports across All Tissues. Binary Cancer-Type Classification. 
#Expects GPU-training and external command line inputs (tissue, random seed)
#Example usasge: CUDA_VISIBLE_DEVICES=3 python3 train_model.py BRCA 0
import numpy as np
import pandas as pd
import torch
import random
from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
import time, os, pickle, glob, shutil, sys 
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
from scipy.special import softmax
from eval_metrics import get_metrics 

#Track Run-time
start = time.time()
curr_datetime = datetime.now().strftime('%m-%d-%Y_%Hh-%Mm')

#Set Random Seed
seed=int(sys.argv[2]) #Input - Random Seed
np.random.seed(seed) 
torch.manual_seed(seed) 
random.seed(seed)

#Establish Directories
tissue=sys.argv[1] #Input - Tissue Name 
input_dir = 'Target_Selection/final_data_proto_full/' #Input Directory 
pickle_path = input_dir + 'Target_Data'+'_'+tissue+'.p' 
input_data = pickle.load(open(pickle_path,'rb')) 
root_dir = "model_output/all_subtypes_10_randomseeds_10e/" #Output Directory 
model_output_dir = root_dir +tissue +'_rs'+str(seed)+'_'+curr_datetime+ '/'
val_best_model_evaluate_dir = model_output_dir + 'val_best_model_evaluate_tmp/'

for directory in [root_dir, model_output_dir, val_best_model_evaluate_dir]:
    os.makedirs(directory, exist_ok=True)

#Training Parameters
eval_metric = 'eval_roc_auc' 
training_args_dict = {'save_strategy':'steps',
		                  'save_steps':32, 
                      'save_total_limit':2,
                      'num_train_epochs':10, 
                      'logging_steps':128, 
                      'per_device_train_batch_size':16,
 		                   'per_device_eval_batch_size':16,
                      'evaluation_strategy':'steps',
                      'eval_steps':32,
                      'load_best_model_at_end':True, 
                      'metric_for_best_model':eval_metric}

#Define Dataset 
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

#Define Performance Metrics 
def compute_metrics(eval_pred):
    raw_pred, labels = eval_pred
    score_pred = softmax(raw_pred, axis=1)[:,1]
    binary_pred = np.argmax(raw_pred, axis=1)
    accuracy = accuracy_score(labels, binary_pred)
    roc = roc_auc_score(labels, score_pred)
    prc = average_precision_score(labels, score_pred)
    f1 = f1_score(labels, binary_pred)
    return {"accuracy": accuracy, "roc_auc": roc, "prc_auc": prc, "f1": f1} 

#Track Model Performance 
output_file = tissue+'_output_rs'+str(seed)+'.txt'
meta_df = pd.DataFrame({'tissue':[tissue]})

#Model and Tokenizer - ClinicalBERT with Binary Classification 
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = BertForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=2) 

#Load and Shuffle Data 
data=input_data.sample(frac=1, random_state=seed) 
X = list(data["text"])
y = list(data["label"])
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.1765, random_state=seed) 
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)
train_dataset = Dataset(X_train_tokenized, y_train)
val_dataset = Dataset(X_val_tokenized, y_val)

#Print Imbalance of Validation Set
print('Tissue:', tissue)
pos_prop_train = round(sum(y_train)/len(y_train),4)
pos_prop_val = round(sum(y_val)/len(y_val),4)
print('Proportion positive: Train:', pos_prop_train, 'Validation:', pos_prop_val)

#Define Training Parameters 
#Automatically keeps the best model checkpoint saved (even if other models come after it)
#Note: Save_steps must be a round multiple of eval_steps (as per docs)
training_args = TrainingArguments(model_output_dir, 
                                  report_to=None,
                                  seed=0, 
                                  save_strategy = training_args_dict['save_strategy'],
                                  save_steps = training_args_dict['save_steps'], 
                                  save_total_limit = training_args_dict['save_total_limit'], #Deletes all but last X checkpoints 
                                  num_train_epochs = training_args_dict['num_train_epochs'], 
                                  logging_steps = training_args_dict['logging_steps'], #Logs training_loss every X steps 
                                  per_device_train_batch_size = training_args_dict['per_device_train_batch_size'], 
				                          per_device_eval_batch_size = training_args_dict['per_device_eval_batch_size'], 
				                          evaluation_strategy = training_args_dict['evaluation_strategy'],
                                  eval_steps = training_args_dict['eval_steps'],
                                  load_best_model_at_end = training_args_dict['load_best_model_at_end'], 
                                  metric_for_best_model = training_args_dict['metric_for_best_model'])   

#Set Model Trainer 
trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=train_dataset,
                  eval_dataset=val_dataset, 
                  compute_metrics=compute_metrics)

#Fine-Tune ClinicalBERT 
print(training_args_dict) 
trainer.train()
end = time.time()
runtime = round((end - start)/60,3)
print('\nElapsed Time: ', runtime, 'Minutes')

#Save Performance Data
history_list = trainer.state.log_history
pickle.dump(history_list, open(model_output_dir+'state_log_history.p','wb'))

#Training - Performance 
train_info = [a for a in history_list if 'loss' in a.keys()]
train_info_dict = {'step':[a['step'] for a in train_info],
                  'training_loss':[a['loss'] for a in train_info],
                  'epoch':[a['epoch'] for a in train_info],
                  'learning_rate':[a['learning_rate'] for a in train_info]}
train_info_df=pd.DataFrame(train_info_dict)

#Evaluation - Performance
e_info = [a for a in history_list if 'eval_loss' in a.keys()]
e_info_dict = {key:[a[key] for a in e_info] for key in e_info[0].keys()}
e_info_df=pd.DataFrame(e_info_dict)
train_info_df.to_csv(model_output_dir+'train_history.csv',index=False)
e_info_df.to_csv(model_output_dir+'eval_history.csv',index=False)
history_df = e_info_df.merge(train_info_df, on=['step','epoch'],how='outer')
history_df.sort_values(by='step',inplace=True)
history_df.to_csv(model_output_dir+'full_history.csv',index=False)

#Performance Plots 

#Training Loss Plot 
train_steps = [a['step'] for a in trainer.state.log_history if 'loss' in a.keys()]
train_loss = [a['loss'] for a in trainer.state.log_history if 'loss' in a.keys()]
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.plot(train_steps, train_loss)
plt.savefig(model_output_dir+'training_loss.png', bbox_inches='tight', dpi=600, facecolor='w')
plt.close() 

#Evalation Loss Plot 
e_steps = [a['step'] for a in trainer.state.log_history if 'eval_loss' in a.keys()]
e_loss = [a['eval_loss'] for a in trainer.state.log_history if 'eval_loss' in a.keys()]
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss during Training')
plt.plot(train_steps, train_loss, label='Train')
plt.plot(e_steps, e_loss, label='Validation')
plt.legend()
plt.savefig(model_output_dir+'validation_loss.png', bbox_inches='tight', dpi=600, facecolor='w')
plt.close() 

#Evaluation Accuracy Plot 
e_steps = [a['step'] for a in trainer.state.log_history if 'eval_loss' in a.keys()]
e_acc = [a['eval_accuracy'] for a in trainer.state.log_history if 'eval_loss' in a.keys()]
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.plot(e_steps, e_acc)
plt.savefig(model_output_dir+'validation_acc.png', bbox_inches='tight', dpi=600, facecolor='w')
plt.close() 

#AU-ROC, AU-PRC Plots 
e_steps = [a['step'] for a in trainer.state.log_history if 'eval_loss' in a.keys()]
e_roc = [a['eval_roc_auc'] for a in trainer.state.log_history if 'eval_loss' in a.keys()]
e_prc = [a['eval_prc_auc'] for a in trainer.state.log_history if 'eval_loss' in a.keys()]
plt.xlabel('Step')
plt.ylabel('Area Under Curve')
plt.title('Validation Metrics')
plt.plot(e_steps, e_roc,label='AU-ROC')
plt.plot(e_steps, e_prc,label='AU-PRC')
plt.legend()
plt.savefig(model_output_dir+'validation_auroc_auprc.png', bbox_inches='tight', dpi=600, facecolor='w')
plt.close() 

#Identify Best_model 
model_checkpoints = glob.glob(model_output_dir+'checkpoint*')
checkpoint_steps = [int(a.split('-')[-1]) for a in model_checkpoints]
checkpoint_data = [a for a in history_list if ((eval_metric in a.keys()) and (a['step'] in checkpoint_steps))]
eval_full_list = [a[eval_metric] for a in checkpoint_data]
best_checkpoint_steps = [a['step'] for a in checkpoint_data if a[eval_metric] == max(eval_full_list)]
#If multiple checkpoints have the same eval_metric, choose the chronologically later one (trained on more data)
best_model_checkpoint = [a for a in model_checkpoints if str(max(best_checkpoint_steps)) == a.split('-')[-1]][0]
print(best_model_checkpoint)

#Save Best Model Metadata  
info_for_test_evaluation = {'best_model_checkpoint':best_model_checkpoint,
                            'model_output_dir':model_output_dir}
pickle.dump(info_for_test_evaluation, open(model_output_dir+'info_for_test_evaluation.p','wb')) 

#Print Best Model Metrics (Validation Set Performance)
print('Validation - Best Model Performance')
best_model = BertForSequenceClassification.from_pretrained(best_model_checkpoint, num_labels=2, local_files_only=True) 
best_trainer = Trainer(model=best_model, args=TrainingArguments(output_dir = val_best_model_evaluate_dir))
y_pred, _, _  = best_trainer.predict(Dataset(X_val_tokenized))
ac, au_roc, au_prc, n_pred_pos, n_actual_pos  = get_metrics(y_pred, y_val, output_dir=model_output_dir, suffix='val_best_model', plot=True)
print('\nAccuracy - ', round(ac,4), 'AU-ROC - ', round(au_roc,4), 'AU-PRC - ', round(au_prc,4))

#Update meta_df with Validation Set Performance and other Metadata 
meta_df['runtime_min'] = [runtime]
meta_df['best_model_steps'] = [best_model_checkpoint.split('-')[-1]]
meta_df['best_model_dir'] = [best_model_checkpoint]
meta_df['pos_prop_train'] = [pos_prop_train] 
meta_df['pos_prop_val'] = [pos_prop_val]
meta_df['save_steps'] = [training_args_dict['save_steps']]
meta_df['num_train_epochs'] = [training_args_dict['num_train_epochs']]
meta_df['logging_steps'] = [training_args_dict['logging_steps']]
meta_df['per_device_train_batch_size'] = [training_args_dict['per_device_train_batch_size']]
meta_df['per_device_eval_batch_size'] = [training_args_dict['per_device_eval_batch_size']]
meta_df['eval_steps'] = [training_args_dict['eval_steps']]
meta_df['val_accuracy_bestmodel'] = [ac]
meta_df['val_au_roc_bestmodel'] = [au_roc]
meta_df['val_au_prc_bestmodel'] = [au_prc]
meta_df['random_seed'] = [seed]
meta_df['n_pred_pos'] = [n_pred_pos]
meta_df['n_actual_pos'] = [n_actual_pos] 

#Save meta_df to csv
meta_df.to_csv(model_output_dir+tissue+'_rs'+str(seed)+'_meta_df.csv',index=False)

#Delete checkpoint that is not the best model 
non_best_model_checkpoint = [a for a in model_checkpoints if str(max(best_checkpoint_steps)) != a.split('-')[-1]][0]
shutil.rmtree(non_best_model_checkpoint)
print('Non-Best-Model checkpoint deleted')
