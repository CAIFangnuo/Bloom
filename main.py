import pandas as pd
import numpy as np
from gensim.utils import simple_preprocess
from tqdm.notebook import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

import NN
import CNN
import Word_Specificity

mappings = {'🗓':'anticipation', '🤞':'anticipation', '😡':'colère', '🤬':'colère', '👿':'colère', '😾':'colère',
           '🖕':'colère', '😤':'colère', '😠':'colère', '💪':'confiance', '🤝':'confiance', '👍':'confiance',
           '👌':'confiance', '😒':'déception', '👎':'déception', '😞':'déception', '🙄':'déception', '😖':'déception',
           '😕':'déception', '🤮':'déception', '🤢':'déception', '😘':'joie', '❤️':'joie', '💋':'joie', '💓':'joie',
            '👍':'joie', '👌':'joie', '😻':'joie', '💏':'joie', '💖':'joie', '🤩':'joie', '💕':'joie', '🤗':'joie',
            '💑':'joie', '🎊':'joie', '🎉':'joie', '😍':'joie', '💗':'joie', '👏':'joie', '🌞':'joie', '😨':'peur',
           '😮':'surprise', '😲':'surprise', ':(':'tristesse', '🙁':'tristesse', '😢':'tristesse', ':-(':'tristesse',
           '😞':'tristesse', '💔':'tristesse', '😿':'tristesse'}


def extract_content_emotions_with_emojis(file_name):
    for row in open(file_name, "r",encoding='utf-16'):
        yield (row.split('\t')[0],
             [(s, mappings[s]) for s in row.split('\t')[0] if s in mappings])
        
        
def preprocessing_with_emojis(file_name):
    content_emotions = []
    for row in extract_content_emotions_with_emojis(file_name):
        content_emotions.append(row)
    content_emotions = [x for x in content_emotions if simple_preprocess(x[0], deacc=True)!=[] and x[1]!=[]]
    emotions = [list(set(x[1])) for x in content_emotions]
    content = [x[0] for x in content_emotions]
    return content, emotions



if __name__ == '__main__':
    content_with_emojis = []
    emotions_with_emojis = []
    for project in tqdm(['DANONE','FLYING_BLUE','NESPRESSO','SANOFI']):
        file_name = "{}.csv".format(project)
        data = preprocessing_with_emojis(file_name)
        content_with_emojis += data[0]
        emotions_with_emojis += data[1]
    emotions = [list(set([e[1] for e in x])) for x in emotions_with_emojis]
    X = pd.Series([simple_preprocess(x, deacc=True) for x in content_with_emojis])
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(emotions)
    
    #training for NN and CNN approach
    Nets = {}
    train_precisions = {}
    train_recalls = {}
    test_precisions = {}
    test_recalls = {}
    
    for label,e in enumerate(mlb.classes_):
        print("emtion:{}".format(e))
        if np.mean(y[:,label]==1) > 0.5:
            idx0 = np.array(X[y[:,label]==0].index)
            idx1 = np.random.choice(np.array(X[y[:,label]!=0].index), len(X[y[:,label]==0]), replace=False)
        else:
            idx0 = np.random.choice(np.array(X[y[:,label]!=1].index), len(X[y[:,label]==1]), replace=False)
            idx1 = np.array(X[y[:,label]==1].index)
            
        idx = np.concatenate((idx0,idx1))
        np.random.shuffle(idx)

        X_train, X_test, y_train, y_test = train_test_split(X[idx], y[idx,label], test_size=0.2, random_state=0)
        
        #using classical neural networks
        #Net, embedding, tfidf, threshold, train_precision, train_recall = NN.train(X_train, y_train)       
        #test_precision, test_recall = NN.test(Net, embedding, tfidf, X_test, y_test, threshold)
        
        #using convolutional neural networks
        Net, embedding, threshold, train_precision, train_recall = CNN.train(X_train, y_train)       
        test_precision, test_recall = CNN.test(Net, embedding, X_test, y_test, threshold)
        
        Nets[e] = Net
        train_precisions[e] = train_precision
        train_recalls[e] = train_recall
        test_precisions[e] = test_precision
        test_recalls[e] = test_recall
    
    #training for Word Specificity approach
    #content = [' '.join(simple_preprocess(x, deacc=True)) for x in content_with_emojis]
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    #F, thresholds, train_precisions, train_recalls = Word_Specificity.train(X_train, y_train, X, y, content, mlb.classes_)
    #test_precisions, test_recalls = Word_Specificity.test(X_test, y_test, y, F, mlb.classes_, thresholds)
	
		
        
        
        
        
