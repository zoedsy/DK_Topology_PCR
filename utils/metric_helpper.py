from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
import numpy as np

def comp_auc(y_label,y_predict):
    fpr, tpr, threshods = metrics.roc_curve(np.array(y_label), np.array(y_predict), pos_label=1)

    return metrics.auc(fpr,tpr)

def comp_f1score(y_label,y_predict):

    return f1_score(y_label,y_predict)


def classi_report(y_label,y_predict):
    return classification_report(y_label,y_predict,labels=[0,1],output_dict=True)



def comp_specificity(y_label,y_predict):
#     fpr, tpr, threshods = metrics.roc_curve(np.array(y_label), np.array(y_predict), pos_label=0)
    
#     return recall_score(y_label,y_predict,pos_label=0,average='weighted')
    return recall_score(y_label,y_predict,pos_label=0,average='weighted')

