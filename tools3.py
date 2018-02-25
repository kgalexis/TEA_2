from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier

# Returns the best configuration for a model using crosvalidation
# and grid search
def best_config(model, name, parameters, train_x, train_y):
    print('\n\nGrid search for ' + name)
    clf = GridSearchCV(model, parameters, cv=10, scoring="f1")
    clf.fit(train_x, train_y)
    best_estimator = clf.best_estimator_
    print('Best hyperparameters: ' + str(clf.best_params_))
 
    return [name, str(clf.best_params_), best_estimator]
    
# Returns the best model from a set of model families given
# training data using cross-validation.
def best_model(classifier_families, train_x, train_y, dev_x, dev_y):
    best_f1_score = 0.0
    classifier_f1_score = 0.0
    best_classifier = None
    classifiers = []
    for name, model, parameters in classifier_families:
        classifiers.append(best_config(model, name, parameters, train_x, train_y))
 
    for name, parameters, classifier in classifiers:
        print('\n\nConsidering classifier ' + name)
        y_true, y_pred = dev_y, classifier.predict(dev_x)
        print(classification_report(y_true, y_pred))
        classifier_f1_score = f1_score(y_true, y_pred, average = 'weighted')
        print(name," f1 score: ", classifier_f1_score)
        if(classifier_f1_score > best_f1_score):
            best_f1_score = classifier_f1_score
            best_classifier = [name, classifier]
        
        print("Best f1 score:", best_f1_score)
 
    print('Best classifier: ' + best_classifier[0])
    return best_classifier[1]
 
# List of candidate family classifiers with parameters for grid
# search [name, classifier object, parameters].
def candidate_families():
    candidates = []
 
    dummy_tuned_parameters = [{'strategy': ['stratified', 'most_frequent']}]
    candidates.append(["Majority Classifier", DummyClassifier(), dummy_tuned_parameters])
    
    
    lr_tuned_parameters = [{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l1', 'l2']}]
    candidates.append(["Logistic Regression Classifier", LogisticRegression(class_weight = 'balanced'), lr_tuned_parameters])
    
  # nb_tuned_parameters = [{'alpha': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]}]
  # candidates.append(["Naive Bayes", MultinomialNB(), nb_tuned_parameters])
 
    knn_tuned_parameters = [{"n_neighbors": list(range(20, 30))}]
    candidates.append(["kNN", KNeighborsClassifier(), knn_tuned_parameters])
        
    svm_tuned_parameters = [{'kernel': ['poly','linear'],
                             'degree': [1, 2, 3, 4]}]
    candidates.append(["SVM", SVC(C=1), svm_tuned_parameters])
    
    return candidates