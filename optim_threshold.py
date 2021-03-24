import numpy as np
import pandas as pd


def optim_threshold(y_pred, y_train):
    df_train = pd.DataFrame({"proba":y_pred,"label":y_train})
    thresholds = np.linspace(0,1,101)
    precisions = []
    recalls = []
    for threshold in thresholds:
        df_train['pred'] = np.where(df_train['proba']>threshold,1,0)
        precision = np.mean(df_train[df_train['pred']==1]['label']==1)
        recall = np.mean(df_train[df_train['label']==1]['pred']==1)
        if (precision > 0.95):
            return threshold, precision, recall
        precisions.append(precision)
        recalls.append(recall)
    return thresholds[np.argmax(precisions)], np.max(precisions), recalls[np.argmax(precisions)]
