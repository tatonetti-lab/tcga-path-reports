#Add Binary Cancer Type Label to Report Set and Separate into Train/Held-out Test Set (for Cancer-type Classification Task)
import pandas as pd
from collections import Counter
import random, os, glob
import pickle 
from sklearn.model_selection import train_test_split
random.seed(0)=
output_directory = 'Target_Selection/final_data_proto_full/' 
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

#Match Cancer Type with Patient Metadata 
patient_type = pd.read_csv('data/tcga_metadata/tcga_patient_to_cancer_type.csv')[['patient_id','cancer_type']]
current_dir = 'post_keyword_and_empty_filtering/'
current_list = glob.glob(current_dir+'*.p')
patients_full = list(set([a.replace(current_dir,'').split('_Page')[0] for a in current_list]))
patients = [a.split('.')[0] for a in patients_full]

#Subset patient_type dataset according to which patients are in the final TCGA Pathology Report Dataset 
target_df = patient_type[patient_type['patient_id'].isin(patients)].copy()
target_df.rename(columns={'patient_id': 'patient'}, inplace=True)
patients_filename=[]
for patient in target_df['patient']:
    for filename in patients_full:
        if patient in filename:
            patients_filename.append(filename)
target_df['patient_filename'] = patients_filename
target_df.reset_index(inplace=True, drop=True)

#Convert Cancer Type Columns to Binary Values
tissue_list = list(target_df['cancer_type'])
for tissue_type in list(set(tissue_list)):
    tissue_binary = []
    for a in tissue_list:
        if a == tissue_type:
            tissue_binary.append(1)
        else:
            tissue_binary.append(0)
    target_df[tissue_type] = tissue_binary

#Separate Finalized Dataset into Full-Training (contains both training/validation) and Held-Out Test Set; Stratify by tissue type  
for tissue in tissue_list: 
    
    print(tissue,' patients:', sum(full_df[tissue]))
    tissue_df = target_df[['patient_filename',tissue]] 
    shuffled_df = tissue_df.sample(frac=1, random_state = 0)
    train, test = train_test_split(shuffled_df, test_size=0.15, 
                                   stratify=shuffled_df[tissue], random_state = 0)
    text_list_train= []

    #Training set
    for patient in list(train['patient_filename']):
        pages = glob.glob(current_dir+patient+ '*.p')
        pages=sorted(pages)
        patient_list = []
        for page in pages:
            lines = pickle.load(open(page, "rb"))
            lines = [a if a[-1] in ['.',':'] else a +'.' for a in lines]
            page_string = " ".join(lines)
            patient_list.append(page_string)
        patient_list = [a if a[-1] in ['.',':'] else a +'.' for a in patient_list]
        patient_text = " ".join(patient_list)
        text_list_train.append(patient_text)

    
    train['text'] = text_list_train

    #Rename column
    train.rename({tissue:'label'},axis=1, inplace=True)
    #Save as pickle
    pickle.dump(train,open(output_directory + 'Target_Data_'+tissue.lower()+'.p','wb')) 
    #Save as csv 
    train.to_csv(output_directory+ 'Target_Data_'+tissue.lower()+'.csv', index=False) 
    
    #Held-Out Test Set 
    text_list_test= []
    for patient in list(test['patient_filename']):
        pages = glob.glob(current_dir+patient+ '*.p')
        pages=sorted(pages)
        patient_list = []
        for page in pages:
            lines = pickle.load(open(page, "rb"))
            lines = [a if a[-1] in ['.',':'] else a +'.' for a in lines]
            page_string = " ".join(lines)
            patient_list.append(page_string)
        patient_list = [a if a[-1] in ['.',':'] else a +'.' for a in patient_list]
        patient_text = " ".join(patient_list)
        text_list_test.append(patient_text)
        
    test['text'] = text_list_test

    #Rename column
    test.rename({tissue:'label'},axis=1, inplace=True)
    #Save as pickle
    pickle.dump(test,open(output_directory + 'Target_Data_'+tissue.lower()+'_test.p','wb')) 
    #Save as csv
    test.to_csv(output_directory+ 'Target_Data_'+tissue.lower()+'_test.csv', index=False) 
    
    print('# Train/Val Patients:',train.shape[0],', # Test Patients',test.shape[0])
 