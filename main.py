import numpy as np
from sklearn.model_selection import train_test_split
from tools import *
from tools2 import *
from tools3 import *
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt



### Training Dataset #####
train_df = load_dataset("./train_2_labels.csv")
print(explore(train_df))
train_x = train_df.text
train_y = train_df.label
print(train_x.shape, train_y.shape)

### Dev Dataset #####
dev_df = load_dataset('./dev_2_labels.csv')
print(explore(dev_df))
dev_x = dev_df.text
dev_y = dev_df.label
print(dev_x.shape, dev_y.shape)

### Test Dataset #####
test_df = load_dataset('./test_2_labels.csv')
print(explore(test_df))
test_x = test_df.text
test_y = test_df.label
print(test_x.shape, test_y.shape)

train_y, dev_y, test_y = labelEncoder1(train_y, dev_y, test_y)


train_x, train_x_2, train_y, train_y_2 = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

### Params to play with ###
vectorizers = [CountVectorizer(), TfidfVectorizer()]
max_features = np.arange(1000, 9001, 500)
stopwords = [None, 'english']
ngrams = [(1,1), (1,2), (1,3)]

### Find bests feature set ###
best = checker(train_x, train_y, train_x_2, train_y_2,  [vectorizers,max_features,stopwords,ngrams])
print(best)

### Play dif tricks with best feature set ###
transformed_train_x, transformed_train_x_2, transformed_dev_x, transformed_test_x = select_features(train_x, train_x_2, dev_x, test_x, choice="given", v=best)
transformed_train_x, transformed_train_x_2, transformed_dev_x, transformed_test_x = normalizer(transformed_train_x, transformed_train_x_2, transformed_dev_x, transformed_test_x)
transformed_train_x, transformed_train_x_2, transformed_dev_x, transformed_test_x = shrink_features(transformed_train_x, transformed_train_x_2, transformed_dev_x, transformed_test_x, k=100, choice="SVD")
transformed_train_x, train_y = sample(transformed_train_x, train_y)


transformed_train_x, train_y = shuffle(transformed_train_x, train_y, random_state=1989)

### Test best final feature set ###
test_lines(transformed_train_x, train_y, transformed_train_x_2, train_y_2)

 
classifiers = candidate_families()
best_classifier = best_model(classifiers, transformed_train_x, train_y, transformed_dev_x, dev_y)

print('\n\n Test dataset classification report')
pred = best_classifier.predict(transformed_test_x)
print(classification_report(test_y, pred))

