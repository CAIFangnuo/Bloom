from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def buildWordVector(tokens, size=200):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += model.wv[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError:
                         
            continue
    if count != 0:
        vec /= count
    return vec.reshape(-1)
  
  
def optim_threshold(Net):
    df_train = pd.DataFrame({"proba":Net.predict(X_train_vector).reshape(-1),"label":y_train})
    thresholds = np.linspace(0.1,1,101)
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
  
  
  
  
  
  
  
