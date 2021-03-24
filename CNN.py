import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import optim_threshold

max_len = 20

def vec_concatenate(tokens, embedding):
    words = [word for word in tokens if word in embedding.wv.vocab]
    if len(words) == 0:
        return None
    vec = embedding.wv[words]
    return vec
    

def pad(x):
    if x.shape[0] > max_len:
        return x[:max_len,]
    else:
        return np.concatenate((x, np.zeros((max_len - x.shape[0] , 200))))
    
           
def train(X_train, y_train):
    
    embedding = Word2Vec(X_train, min_count=1, window=3, size=200)

    X_train_vector = X_train.apply(lambda x: vec_concatenate(x, embedding))

    y_train = y_train[X_train_vector.notnull()]
    X_train_vector = X_train_vector[X_train_vector.notnull()]

    X_train_vector = X_train_vector.apply(lambda x: pad(x))

    X_train_vector = np.array(X_train_vector.tolist())

    scaler = StandardScaler()
    X_train_vector = scaler.fit_transform(X_train_vector.reshape(-1, X_train_vector.shape[-1])).reshape(X_train_vector.shape)

    Net = keras.Sequential()
    Net.add(layers.Conv1D(filters=128, kernel_size=2, activation='relu',input_shape=(max_len,200)))
    Net.add(layers.MaxPooling1D(pool_size=2))
    Net.add(layers.Flatten())
    Net.add(layers.Dense(1, activation='sigmoid'))
    Net.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=[keras.metrics.Precision(), keras.metrics.Recall()])

    Net.fit(X_train_vector, y_train, epochs=30, batch_size=32)
    
    y_pred = Net.predict(X_train_vector).reshape(-1)

    threshold, train_precision, train_recall = optim_threshold.optim_threshold(y_pred, y_train)
    
    print("train_precision: {}".format(train_precision))
    print("train_recall: {}".format(train_recall))

    return Net, embedding, threshold, train_precision, train_recall

    
def test(Net, embedding, X_test, y_test, threshold, visualization=False):

    X_test_vector = X_test.apply(lambda x: vec_concatenate(x, embedding))
    y_test = y_test[X_test_vector.notnull()]
    X_test = X_test[X_test_vector.notnull()]
    X_test_vector = X_test_vector[X_test_vector.notnull()]
    X_test_vector = X_test_vector.apply(lambda x: pad(x))
    X_test_vector = np.array(X_test_vector.tolist())
    scaler = StandardScaler()
    X_test_vector = scaler.fit_transform(X_test_vector.reshape(-1, X_test_vector.shape[-1])).reshape(X_test_vector.shape)

    df_test = pd.DataFrame({"proba":Net.predict(X_test_vector).reshape(-1),"label":y_test})
    df_test['pred'] = np.where(df_test['proba']>threshold,1,0)
    test_precision = np.mean(df_test[df_test['pred']==1]['label']==1)
    test_recall = np.mean(df_test[df_test['label']==1]['pred']==1)
    
    if visualization == True:
        for item in sorted(df_test['label'].unique()):
            sns.distplot((df_test[df_test['label']==item]['proba']),label=item)
        plt.axvline(x=threshold, c='g',label='cut')
        plt.legend()
        plt.title("test data distribution")
        plt.show()
    
    print("test_precision: {}".format(test_precision))
    print("test_recall: {}".format(test_recall))
    
    return test_precision, test_recall