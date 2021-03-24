
from sklearn.preprocessing import StandardScaler


max_len = 20

def vec_concatenate(tokens):
    words = [word for word in tokens if word in model.wv.vocab]
    if len(words) == 0:
        return None
    vec = model.wv[words]
    return vec
 

def optim_threshold(net):
    df_train = pd.DataFrame({"proba":net.predict(X_train_vector).reshape(-1),"label":y_train})
    thresholds = np.linspace(0.1,1,101)
    precisions = []
    for threshold in thresholds:
        df_train['pred'] = np.where(df_train['proba']>threshold,1,0)
        precision = np.mean(df_train[df_train['pred']==1]['label']==1)
        if (precision > 0.95):
            return threshold, precision
        precisions.append(precision)
    return threshold[np.argmax(precisions)], np.max(precisions)
  


  
  
  
