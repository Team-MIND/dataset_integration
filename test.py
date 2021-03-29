import sklearn 
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics import roc_auc_score 

import numpy as np 

y_preds = np.array([1,0])
y_true = np.array([0,0])

print(precision_recall_fscore_support(y_preds, y_true))
