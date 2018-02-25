import numpy as np
from sklearn.decomposition import TruncatedSVD, PCA
#from sklearn.decomposition import TruncatedSVD, ProjectedGradientNMF
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, Normalizer, KernelCenterer
from sklearn import metrics
from pprint import pprint
from sklearn import svm


### Function that returns labels encoded to numbers ###
def labelEncoder1(train_y, dev_y, test_y):
    normalizer = LabelEncoder()
    train_y = normalizer.fit_transform(train_y)
    dev_y = normalizer.transform(dev_y)
    test_y = normalizer.transform(test_y)
    return train_y, dev_y, test_y

### Normalizer of values based on choice ###
def normalizer(train_X, train_X_2, dev_X, test_X, feature_range=(-1.0, 1.0), choice=None):
    if choice==None:
        return train_X, train_X_2, dev_X, test_X
    elif choice=="MinMax":
        normalizer = MinMaxScaler(feature_range=feature_range)
    train_X = normalizer.fit_transform(train_X)
    train_X_2 = normalizer.transform(train_X_2)
    dev_X = normalizer.transform(dev_X)
    test_X = normalizer.transform(test_X)
    return train_X, train_X_2, dev_X, test_X

### Function that constructs features from text ###
def select_features(train_X, train_X_2, dev_X, test_X, k=10, choice="tf", v=None, stop_words=None, ngram=1, analyzer="word", max_df=1.0, min_df=1, max_features=None):
    if choice=="given":
        v = v
    elif choice=="tf":
        v = CountVectorizer(stop_words=stop_words,ngram_range=(1, ngram), analyzer="word", max_df=max_df, min_df=min_df, max_features=max_features)
    elif choice=="tf-idf":
        v = TfidfVectorizer(stop_words=stop_words,ngram_range=(1, ngram), analyzer="word", max_df=max_df, min_df=min_df, max_features=max_features)
    train_X = v.fit_transform(train_X)
    train_X_2 = v.transform(train_X_2)
    dev_X = v.transform(dev_X)
    test_X = v.transform(test_X)
    return train_X, train_X_2, dev_X, test_X
    
### Dimensionality reduction based on choice ###
def shrink_features(train_X, train_X_2, dev_X, test_X, k=10, choice=None):
    if choice==None:
        return train_X, train_X_2, dev_X, test_X
    elif choice == "PCA":
        selector = PCA(n_components=k)
    elif choice == "SVD":
        selector = TruncatedSVD(n_components=k, random_state=1989)       
    selector.fit(train_X)
    train_X = selector.transform(train_X)
    train_X_2 = selector.transform(train_X_2)
    dev_X = selector.transform(dev_X)
    test_X = selector.transform(test_X)
    return train_X, train_X_2, dev_X, test_X
	
### Sampling function based on choice ###
def sample(train_X,train_y,choice=None):
    unique, counts = np.unique(train_y, return_counts=True)
    labels = dict(zip(unique, counts))

    if choice==None:
        return train_X, train_y
    elif choice == "over":
        no = max(labels.values())
        total_X = train_X  
        total_y = train_y
        rep = True
    elif choice == "under":
        no = min(labels.values())
        total_X = np.empty([0, 2])
        total_y = np.empty([0, 1])
        rep = False
    
    for l,c in labels.items():
        temp= train_X[np.random.choice(np.argwhere(train_y==l).flatten(), no-c*rep, replace=rep)] 
        total_X = np.append(total_X,temp,axis=0)
        total_y = np.append(total_y, np.array([l for i in range(no-c*rep)]))

    return total_X, total_y
#sample(np.array([(0,0),(0,1),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7)]),np.array([0,0,0,0,1,1,2,2,2]),"under")    


# return the results of the classification in a dictionary
def benchmark(clf, train_X, train_y, test_X, test_y, metric):
    """
    evaluate classification
    """
    clf.fit(train_X, train_y)
    pred = clf.predict(test_X)
    #print(pred)
    f1 = metrics.f1_score(test_y, pred, average='weighted')
    accuracy = metrics.accuracy_score(test_y, pred)
    result = {'f1' : f1,'Accuracy' : accuracy,'train size' : len(train_y), 'test size' : len(test_y), 'predictions': pred }
    print(" {}: {} ".format(metric, result[metric]))
    return result

def plot(results, metric='Accuracy'):
    
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    
    fontP = FontProperties()
    fontP.set_size('small')
    fig = plt.figure()
    fig.suptitle('Learning Curves', fontsize=20)
    ax = fig.add_subplot(111)
    ax.axis([0, 3000, 0, 1.1])
    line_up, = ax.plot( results['train_size'], results['on_train'], 'o-',label=metric+' on Train')
    line_down, = ax.plot( results['train_size'] , results['on_test'], 'o-',label=metric+' on Test')
    
    plt.xlabel('N. of training instances', fontsize=18)
    plt.ylabel(metric, fontsize=16)
    plt.legend([line_up, line_down], [metric+' on Train', metric+' on Test'], prop = fontP)
    plt.grid(True)
    
    fig.savefig('best_100_svd.png')
    
def test_lines(train_x, train_y, test_x, test_y):
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