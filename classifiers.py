import time
from itertools import combinations

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics
from mlxtend.evaluate import paired_ttest_5x2cv

from Data import *


def calculate_metrics(label, pred, clf):
    precision = metrics.precision_score(label, pred, average=None)
    recall = metrics.recall_score(label, pred, average=None)
    f1 = metrics.f1_score(label, pred, average=None)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)

    print(metrics.classification_report(label, pred))
    cm = confusion_matrix(label, pred, labels=clf.classes_)
    print("Confusion matrix: ")
    print(cm)


def grid_search(model, param_grid, X_train, y_train, X_test, y_test):
    clf = GridSearchCV(estimator=model, param_grid=param_grid, cv=10)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    print('\033[94m' + "Accuracy with below parameters: ", accuracy)
    print(clf.best_params_)

    calculate_metrics(y_test, pred, clf)

    return clf


# measuring runtime
start = time.time()

data_unigrams= Data(1)
data_bigrams = Data(2)

# loading pre-processed data
X_train_uni, y_train_uni, X_test_uni, y_test_uni, feature_names_uni = data_unigrams.X_train, data_unigrams.Y_train, data_unigrams.X_test, data_unigrams.Y_test, data_unigrams.feature_names
X_train_bi, y_train_bi, X_test_bi, y_test_bi, feature_names_bi = data_bigrams.X_train, data_bigrams.Y_train, data_bigrams.X_test, data_bigrams.Y_test, data_bigrams.feature_names

all_classifiers = []

# 1. Classifier: Multinomial NB
model = MultinomialNB(force_alpha=True)
param_grid = {
    "alpha": list(np.linspace(0.1, 2, num=11)),
    "fit_prior": [True, False]
}
print("\n")
print('\033[92m' + "Multinomial Naive Bayes with unigram features:")
clf_multinb_uni = grid_search(model, param_grid, X_train_uni, y_train_uni, X_test_uni, y_test_uni)
all_classifiers.append(clf_multinb_uni)
print('\033[92m' + "Multinomial Naive Bayes with unigram and bigram features:")
clf_multinb_bi = grid_search(model, param_grid, X_train_bi, y_train_bi, X_test_bi, y_test_bi)
all_classifiers.append(clf_multinb_bi)


# 2. Classifier: Logistic regression
model = LogisticRegression()
param_grid = {
    "penalty": ["l1"],  # lasso penalty
    "solver": ["liblinear"],
    "C": list(np.ones(10) - (np.logspace(0, 2, num=10) / 100) + 0.01),
}
print("\n")
print('\033[92m' + "Logistic Regression with unigram features:")
clf_logreg_uni = grid_search(model, param_grid, X_train_uni, y_train_uni, X_test_uni, y_test_uni)
all_classifiers.append(clf_logreg_uni)
print('\033[92m' + "Logistic Regression with unigram and bigram features:")
clf_logreg_bi = grid_search(model, param_grid, X_train_bi, y_train_bi, X_test_bi, y_test_bi)
all_classifiers.append(clf_logreg_bi)


# 3. Classifier: Classification trees
model = DecisionTreeClassifier()
param_grid = {
    "criterion": ["gini", "entropy", "log_loss"],  # ToDo: maybe delete if takes too long
    "max_depth": [1],  # ToDo: choose parameters
    "min_samples_split": [2],
    "min_samples_leaf": [1],
    "ccp_alpha": [0.0],
}
print("\n")
print('\033[92m' + "Classification trees with unigram features:")
clf_ctrees_uni = grid_search(model, param_grid, X_train_uni, y_train_uni, X_test_uni, y_test_uni)
all_classifiers.append(clf_ctrees_uni)
print('\033[92m' + "Classification trees with unigram and bigram features:")
clf_ctrees_bi = grid_search(model, param_grid, X_train_bi, y_train_bi, X_test_bi, y_test_bi)
all_classifiers.append(clf_ctrees_bi)


# 4. Classifier: Random forests
model = RandomForestClassifier()
param_grid = {
    "n_estimators": [100],  # ToDo: choose parameters
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [1],
    "min_samples_split": [2],
    "min_samples_leaf": [1],
    "ccp_alpha": [0.0],
}
print("\n")
print('\033[92m' + "Random Forests with unigram features:")
clf_randforest_uni = grid_search(model, param_grid, X_train_uni, y_train_uni, X_test_uni, y_test_uni)
all_classifiers.append(clf_randforest_uni)
print('\033[92m' + "Random forest with unigram and bigram features:")
clf_ranforest_bi = grid_search(model, param_grid, X_train_bi, y_train_bi, X_test_bi, y_test_bi)
all_classifiers.append(clf_ranforest_bi)


def accuracy_t_test(pair):  # ToDo: implement t test for two classifiers + print relevant results
    pass


# ToDo: put classifiers in dict with name as key and predictions as value
# pairs_of_models = list(map(dict, combinations(all_classifiers.items(), 2)))
# for pair in pairs_of_models:
  #  accuracy_t_test(pair)


# measuring runtime
runtime = time.time() - start
print('\033[92m' + "Execution time in seconds: ", runtime)




