import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import itertools
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score

### Preprocessing function ###
def strip(text):
    t = text.lower()
    t = re.sub('\&amp;', ' ', t)
    t = re.sub('http\S+', ' ', t)
    t = re.sub('@\w+', ' ', t)
    t = re.sub("[^\w']", ' ', t)
    t = re.sub('\s+', ' ', t)
    return t

### Stats of dataset ###
def explore(df):
    print('Observations:', len(df.index))
    for label in set(df.label):
        print('{:.2f} % {}'.format(len(df[df.label == label])/df.shape[0]*100, label))

### Loading Dataset ###
def load_dataset(path):
    columns = ['id', 'topic', 'label', 'text']
    df = pd.read_csv(path, sep='\t', header=None, names=columns)
    df = df[df.text != 'Not Available']
    df.reset_index(drop=True, inplace=True)
    df.text = df.text.apply(lambda x: strip(x))
    return df

### Evaluating feature set on arbitrary classifier ###
def evaluate(pipeline, x_train, y_train, x_test, y_test):
    results = {}
    fit = pipeline.fit(x_train, y_train)
    y_pred = fit.predict(x_test)
    results['accuracy'] = accuracy_score(y_test, y_pred)
    results['f1'] = f1_score(y_test, y_pred, average='weighted')
    return results

### Function that finds best feature set of all combinations ###
def checker(train_x, train_y, dev_x, dev_y, params, metric="f1"):
    result = pd.DataFrame(columns=["vec","accuracy","f1"])
    total = 0
    for elem in itertools.product(*params):
        total+= 1
    for i, (vec,n,sw,ngram) in enumerate(itertools.product(*params)):
        temp = {}
        print("Trying {} out of {}.".format(i+1,total))
        temp["vec"] = vec
        vec.set_params(stop_words=sw, max_features=n, ngram_range=ngram)
        pipeline = Pipeline([('vectorizer', vec),('classifier', LogisticRegression())])
        metrics = evaluate(pipeline, train_x, train_y, dev_x, dev_y)
        temp['accuracy'] = metrics['accuracy']
        temp['f1'] = metrics['f1']
        result = result.append(temp, ignore_index=True)
    print(result.iloc[result[metric].idxmax()])
    return result.iloc[result[metric].idxmax()].vec
