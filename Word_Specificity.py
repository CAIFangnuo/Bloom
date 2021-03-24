import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import optim_threshold

from nltk.corpus import stopwords
stop = stopwords.words('english')

def sumWordSpecificity(tokens, F, e):
    score = 0.
    for word in tokens:
        try:
            s = F[e].loc[word,'S']
            if s > 1:
                score += s
        except KeyError:                   
            continue
    return score
    
def train(X_train, y_train, X, y, content, emotions, visualization=False):

    # frequency of words in each emotion / number of sentences in that emotion
    Fe = {}
    for i,e in enumerate(emotions):
        Fe[e] = pd.Series([content[j] for j in X_train[y_train[:,i]==1].index]).str.split().apply(lambda x: ' '.join(list(set([item for item in x if item not in stop])))).str.split(expand=True).stack().value_counts()
        Fe[e] = Fe[e][Fe[e] > 2]
        Fe[e] = Fe[e] / np.mean(y_train[:,i]==1)
    
    # frequency of words outside each emotion / number of sentences
    Fne = {}
    for i,e in enumerate(emotions):
        Fne[e] = pd.Series([content[j] for j in X[y[:,i]==0].index]).str.split().apply(lambda x: ' '.join([item for item in x if item not in stop])).str.split(expand=True).stack().value_counts()/(1-np.mean(y[:,i]==1))

    # word specific dataframe for each emotion
    F = {}
    for i,e in enumerate(emotions):
        F[e] = pd.DataFrame(Fe[e], columns=['Fe']).join(pd.DataFrame(Fne[e], columns=['Fne']))
        F[e]['Fne'] = F[e]['Fne'].fillna(1/(1-np.mean(y[:,i]==1)))
        F[e]['S'] = F[e]['Fe'] / F[e]['Fne']

    data_train = pd.DataFrame(X_train, columns=['sentence'])
    for e in emotions:
        data_train[e] = X_train.apply(lambda x: sumWordSpecificity(x, F, e))
    for i,e in enumerate(emotions):
        data_train['label_{}'.format(e)] = y[data_train.index,i]

    thresholds = {}
    train_precisions = {}
    train_recalls = {}
    
    for e in emotions:
        subdf = data_train[['sentence',e, 'label_{}'.format(e)]]
        subdf = subdf[subdf[e] != 0]
        scaler = MinMaxScaler()
        subdf[e] = scaler.fit_transform(subdf[[e]])
    
        if visualization == True:
            for l in sorted(subdf['label_{}'.format(e)].unique()):
                sns.distplot((subdf[subdf['label_{}'.format(e)]==l][e]),label=l)
            plt.legend()
            plt.title(e)
            plt.show()
        
        threshold, train_precision, train_recall = optim_threshold.optim_threshold(np.array(subdf[e]), np.array(subdf['label_{}'.format(e)]))
    
        thresholds[e] = threshold
        train_precisions[e] = train_precision
        train_recalls[e] = train_recall
    
        print("emotion: {}".format(e))
        print("precision: {}".format(train_precision))
        print("recall: {}".format(train_recall))
    
    return F, thresholds, train_precisions, train_recalls
    

def test(X_test, y_test, y, F, emotions, thresholds, visualization=False):
    data_test = pd.DataFrame(X_test, columns=['sentence'])
    for e in emotions:
        data_test[e] = X_test.apply(lambda x: sumWordSpecificity(x,F,e))
    for i,e in enumerate(emotions):
        data_test['label_{}'.format(e)] = y[data_test.index,i]

    test_precisions = {}
    test_recalls = {}
    for e in emotions:
        subdf = data_test[['sentence',e, 'label_{}'.format(e)]]
        scaler = MinMaxScaler()
        subdf[e] = scaler.fit_transform(subdf[[e]])
        
        if visualization == True:
            for l in sorted(subdf['label_{}'.format(e)].unique()):
                sns.distplot((subdf[subdf['label_{}'.format(e)]==l][e]),label=l)
            plt.legend()
            plt.title(e)
            plt.show()
    
        threshold = thresholds[e]
        subdf['pred'] = np.where(subdf[e]>threshold, 1, 0)
        data_test['pred_{}'.format(e)] = subdf['pred']
        precision = np.mean(subdf[subdf['pred']==1]['label_{}'.format(e)]==1)
        recall = np.mean(subdf[subdf['label_{}'.format(e)]==1]['pred']==1)
    
        print("emotion: {}".format(e))
        print("precsion: {}".format(precision))
        print("recall: {}".format(recall))
    
        test_precisions[e] = precision
        test_recalls[e] = recall
    
    return test_precisions, test_recalls
    
    
    

    
    
