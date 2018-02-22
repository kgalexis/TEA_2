import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from tools import labelEncoder, normalizer, select_features, shrink_features, sample, benchmark, plot
import pandas as pd

df = pd.read_csv("./train_2_labels.csv",  sep='\t', header=None, names=["id", "user", "label", "text"])
X = df.text
Y = df.label

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=42)
train_y, test_y = labelEncoder(train_y, test_y)

train_x, test_x = select_features(train_x, test_x, choice="tf")
train_x, test_x = normalizer(train_x, test_x)
train_x, test_x = shrink_features(train_x, test_x, k=10)
train_x, train_y = sample(train_x,train_y)

print('Training instances')
for i in range(0,int(np.max(train_y))):
    print('instances of class '+str(i)+' : '+str(len(train_y[train_y==i])))

print('\nTesting instances')
for i in range(0,int(np.max(test_y))):
    print('instances of class '+str(i)+' : '+str(len(test_y[test_y==i])))

from sklearn.utils import shuffle
train_x, train_y = shuffle(train_x, train_y, random_state=1989)

#metric = 'Accuracy'
metric = 'f1'

results = {'train_size': [], 'on_test': [], 'on_train': []}
for i in range(1,11):
    if(i==10):
        train_x_part = train_x
        train_y_part = train_y
    else:
        to = int(i*(train_x.shape[0]/10))
        #print(to)
        train_x_part = train_x[0:to,:]
        train_y_part = train_y[0:to]
    print(train_x_part.shape)
    results['train_size'].append(train_x_part.shape[0])
    clf = svm.LinearSVC(random_state = 1989, C=100., penalty = 'l2', max_iter =1000)
    result = benchmark(clf, train_x_part, train_y_part, test_x, test_y, metric)
    results['on_test'].append(result[metric])
    result = benchmark(clf, train_x_part, train_y_part, train_x_part, train_y_part, metric)
    results['on_train'].append(result[metric])

plot(results, metric)
