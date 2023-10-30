import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics
from scipy import stats
from itertools import combinations
from sklearn.feature_selection import mutual_info_classif
from Data import *


def print_metrics(label, pred, clf):
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

    print_metrics(y_test, pred, clf)

    print("Class name: ", model.__class__.__name__)
    return model.__class__.__name__, pred

def mutual_information(X, y):
    """
        Takes an review-gram array and a label array. 
        Transforms the review-gram array to binary term present/absent
        Calculates the mutual info and returns an single array
        with length "#columns in review-gram array"

        returns : 1D ndarray with len X.shape[1]
    """
    binary_X = np.where(X>0, 1, 0)
    print(binary_X.shape)
    return mutual_info_classif(binary_X, y) # this is ndarray

#######################################################################################################################


# measuring runtime
start = time.time()

data_unigrams= Data(1)
data_bigrams = Data(2)

# loading pre-processed data
X_train_uni, y_train_uni, X_test_uni, y_test_uni, feature_names_uni = data_unigrams.X_train, data_unigrams.Y_train, data_unigrams.X_test, data_unigrams.Y_test, data_unigrams.feature_names
X_train_bi, y_train_bi, X_test_bi, y_test_bi, feature_names_bi = data_bigrams.X_train, data_bigrams.Y_train, data_bigrams.X_test, data_bigrams.Y_test, data_bigrams.feature_names

mutual_information_X_train_uni = mutual_information(X_train_uni, y_train_uni) # those are ndarrays
mutual_information_X_train_bi = mutual_information(X_train_bi, y_train_bi) # those are ndarrays


p = 0.35
best_n_uni = int(3799 * p)
best_n_bi = int(38717 * p)

X_train_uni_top_percent = X_train_uni[:, np.argsort(mutual_information_X_train_uni)[-(best_n_uni):]]
X_train_bi_top_percent = X_train_bi[:, np.argsort(mutual_information_X_train_bi)[-best_n_bi:]]

X_test_uni_top_percent = X_test_uni[:, np.argsort(mutual_information_X_train_uni)[-(best_n_uni):]]
X_test_bi_top_percent = X_test_bi[:, np.argsort(mutual_information_X_train_bi)[-(best_n_bi):]]

#######################################################################################################################


all_classifiers = {}

# 1. Classifier: Multinomial NB
model = MultinomialNB(force_alpha=True)
param_grid_uni = {
    "alpha": [0.48],  # list(np.linspace(0.1, 2, num=11)),
    "fit_prior": [True]
}
param_grid_bi = {
    "alpha": [0.29],  # list(np.linspace(0.1, 2, num=11)),
    "fit_prior": [True]
}
print("\n")
print('\033[92m' + "Multinomial Naive Bayes with unigram features:")
clf_multinb_uni, pred_multinb_uni = grid_search(model, param_grid_uni, X_train_uni_top_percent, y_train_uni, X_test_uni_top_percent, y_test_uni)
all_classifiers[clf_multinb_uni + "_uni"] = pred_multinb_uni
print('\033[92m' + "Multinomial Naive Bayes with unigram and bigram features:")
clf_multinb_bi, pred_multinb_bi = grid_search(model, param_grid_bi, X_train_bi_top_percent, y_train_bi, X_test_bi_top_percent, y_test_bi)
all_classifiers[clf_multinb_bi + "_bi"] = pred_multinb_bi


# 2. Classifier: Logistic regression
model = LogisticRegression()
param_grid = {
    "penalty": ["l1"],  # lasso penalty
    "solver": ["liblinear"],
    "C": [1]  # list(np.ones(10) - (np.logspace(0, 2, num=10) / 100) + 0.01),
}
print("\n")
print('\033[92m' + "Logistic Regression with unigram features:")
clf_logreg_uni, pred_logreg_uni = grid_search(model, param_grid, X_train_uni, y_train_uni, X_test_uni, y_test_uni)
all_classifiers[clf_logreg_uni + "_uni"] = pred_logreg_uni
print('\033[92m' + "Logistic Regression with unigram and bigram features:")
clf_logreg_bi, pred_logreg_bi = grid_search(model, param_grid, X_train_bi, y_train_bi, X_test_bi, y_test_bi)
all_classifiers[clf_logreg_bi + "_bi"] = pred_logreg_bi


# 3. Classifier: Classification trees
model = DecisionTreeClassifier()
param_grid = {
    "criterion": ["gini"],  # ["gini", "entropy", "log_loss"],
    "max_depth": [2],  # list(np.arange(5) + 1),
    "min_samples_split": [1.0],  # list(np.arange(5) + 1),
    "min_samples_leaf": [2],  # list(np.arange(5) + 1),
    "ccp_alpha": [0.0],  # list(np.arange(11) / 5),
}
print("\n")
print('\033[92m' + "Classification trees with unigram features:")
clf_ctrees_uni, pred_ctrees_uni = grid_search(model, param_grid, X_train_uni, y_train_uni, X_test_uni, y_test_uni)
all_classifiers[clf_ctrees_uni + "_uni"] = pred_ctrees_uni
print('\033[92m' + "Classification trees with unigram and bigram features:")
clf_ctrees_bi, pred_ctrees_bi = grid_search(model, param_grid, X_train_bi, y_train_bi, X_test_bi, y_test_bi)
all_classifiers[clf_ctrees_bi + "_bi"] = pred_ctrees_bi


# 4. Classifier: Random forests
model = RandomForestClassifier()
param_grid = {
    "n_estimators": [500],  # [10, 20, 50, 100, 200, 500],
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [3],  # list(np.arange(5) + 1),
    "min_samples_split": [4],  # list(np.arange(5) + 1),
    "min_samples_leaf": [5],  # list(np.arange(5) + 1),
    "ccp_alpha": [0.2],  # list(np.arange(11) / 5),
}
print("\n")
print('\033[92m' + "Random Forests with unigram features:")
clf_randforest_uni, pred_randforest_uni = grid_search(model, param_grid, X_train_uni, y_train_uni, X_test_uni, y_test_uni)
all_classifiers[clf_randforest_uni + "_uni"] = pred_randforest_uni
print('\033[92m' + "Random forest with unigram and bigram features:")
clf_randforest_bi, pred_randforest_bi = grid_search(model, param_grid, X_train_bi, y_train_bi, X_test_bi, y_test_bi)
all_classifiers[clf_randforest_bi + "_bi"] = pred_randforest_bi


#######################################################################################################################


def accuracy_t_test(model_pair, model_dict):
    pred1 = model_dict[model_pair[0]]
    pred2 = model_dict[model_pair[1]]
    t_statistic, p_value = stats.ttest_rel(pred1, pred2)

    print("Paired t-test results for classifiers ", model_pair[0], " and ", model_pair[1], ": ")
    if p_value < 0.05:
        print("Difference is significant!")
    print("T-statistic:", t_statistic)
    print("P-value:", p_value)


# calculate pairwise t-tests
model_pairs = list(combinations(list(all_classifiers.keys()), 2))
for pair in model_pairs:
    accuracy_t_test(pair, all_classifiers)


# measuring runtime
runtime = time.time() - start
print('\033[92m' + "Execution time in seconds: ", runtime)




