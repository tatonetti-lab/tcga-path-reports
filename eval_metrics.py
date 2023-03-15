#Evaluate Best-Performing Model on Validation or Test Set 
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from scipy.special import softmax
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

def get_metrics(raw_pred, y_true, output_dir, suffix='', plot=True):

    #Calculate Performance Metrics 
    y_pred = np.argmax(raw_pred, axis=1)
    n_pred_pos = sum(y_pred)     
    actual_pos = sum(y_true)  
    ac = accuracy_score(y_true, y_pred) #Accuracy 
    sm_scores = softmax(raw_pred, axis=1) #Compute softmax on a per-row basis to normalize raw predictions 
    y_score = sm_scores[:,1]
    au_roc = roc_auc_score(y_true, y_score) #AU-ROC
    au_prc = average_precision_score(y_true, y_score) #AU-PRC

    #Save raw and normalized scores 
    raw_df = pd.DataFrame({'raw_pred':raw_pred[:,1],
                            'transformed_pred':y_score,
                            'y_pred':y_pred,
                            'y_true':y_true})
    raw_df.to_csv(output_dir+'raw_predictions_'+suffix+'.csv', index=False)

    #Save final evaluation data 
    eval_df = pd.DataFrame({'Accuracy':[ac],
                            'AUROC':[au_roc],
                            'AUPRC':[au_prc],
                            'n_pred_pos':[n_pred_pos],
                            'actual_pos':[actual_pos]})
    eval_df.to_csv(output_dir+'eval_metrics_'+suffix+'.csv', index=False)

    #Generate Performance Plots 

    if plot: 

        #Plot ROC Curve 
        fpr, tpr, threshold = roc_curve(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)
        plt.title('ROC Curve')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(output_dir+'roc_curve_'+suffix+'.png', dpi=1200, facecolor='w')
        plt.close()
        
        #Plot PR Curve
        p, r, threshold = precision_recall_curve(y_true, y_score)
        auc = average_precision_score(y_true, y_score)
        plt.title('Precision-Recall Curve')
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.plot(r,p, 'b', label = 'AUC = %0.2f' % auc)
        plt.legend(loc = 'lower right')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.savefig(output_dir+'pr_curve_'+suffix+'.png', bbox_inches='tight', dpi=1200, facecolor='w')
        plt.close()
	    
    return ac, au_roc, au_prc, n_pred_pos, actual_pos

