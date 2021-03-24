from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import scale
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
  
def train(X_train, y_train):
    
    word2vec = Word2Vec(X_train, min_count=1, window=3, size=200)
    
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([word for x in X_train for word in x])
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

    X_train_vector = X_train.apply(lambda x: buildWordVector(x))
    X_train_vector = np.array(X_train_vector.tolist())
    X_train_vector = scale(X_train_vector)

    Net = keras.Sequential()
    Net.add(layers.Dense(16, activation='relu', input_dim=200))
    Net.add(layers.Dense(1, activation='sigmoid'))
    Net.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[keras.metrics.Precision(), keras.metrics.Recall()])

    Net.fit(X_train_vector, y_train, epochs=30, batch_size=32)

    threshold, train_precision, train_recall = optim_threshold(Net)
    return threshold, train_precision, train_recall


def test(X_test, y_test, threshold):

    X_test_vector = X_test.apply(lambda x: buildWordVector(x))
    X_test_vector = np.array(X_test_vector.tolist())
    X_test_vector = scale(X_test_vector)

    df_test = pd.DataFrame({"proba":Net.predict(X_test_vector).reshape(-1),"label":y_test})
    df_test['pred'] = np.where(df_test['proba']>threshold,1,0)
    test_precision = np.mean(df_test[df_test['pred']==1]['label']==1)
    test_recall = np.mean(df_test[df_test['label']==1]['pred']==1)
    
    for item in sorted(df_test['label'].unique()):
        sns.distplot((df_test[df_test['label']==item]['proba']),label=item)
    plt.axvline(x=threshold, c='g',label='cut')
    plt.legend()
    plt.title("anticipation (Tfidf + Neural Networks)")
    plt.show()
    
    return test_precision, test_recall
  
  
  
  
  
