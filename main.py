import numpy as np
from sklearn.model_selection import train_test_split
from tools import *
from tools2 import *
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

### Training Dataset #####
train_df = load_dataset("./train_2_labels.csv")
print(explore(train_df))
train_x = train_df.text
train_y = train_df.label
print(train_x.shape, train_y.shape)

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
train_x, train_x_2 = select_features(train_x, train_x_2, choice="given", v=best)
train_x, train_x_2 = normalizer(train_x, train_x_2)
#train_x, train_x_2 = shrink_features(train_x, train_x_2, k=10)
train_x, train_x_2 = shrink_features(train_x, train_x_2, k=100, choice="SVD")
train_x, train_y = sample(train_x, train_y)

from sklearn.utils import shuffle
train_x, train_y = shuffle(train_x, train_y, random_state=1989)

### Test best final feature set ###
test_lines(train_x, train_y, train_x_2, train_y_2)

### Dev Dataset #####
dev_df = load_dataset('./dev_2_labels.csv')
print(explore(dev_df))
dev_x = dev_df.text
dev_y = dev_df.label
train_y, dev_y = labelEncoder(train_y, dev_y)
print(dev_x.shape, dev_y.shape)
