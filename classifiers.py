from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics


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

    print("Accuracy with below parameters: ", accuracy)
    print(clf.best_params_)

    calculate_metrics(y_test, pred, clf)

    print("All results:")
    print(clf.cv_results_)


# 1. Classifier: Multinomial NB
model = MultinomialNB(force_alpha=True)
param_grid = {
    "alpha": [1.0],  # ToDo: choose parameters (maybe np.logspace)
}
print("\n")
print("Multinomial Naive Bayes:")
grid_search(model, param_grid)

# 2. Classifier: Logistic regression
model = LogisticRegression()
param_grid = {
    "penalty": ["l1"],  # lasso penalty
    "solver": ["liblinear"],
    "C": [0.0],  # ToDo: choose parameters (maybe np.logspace)
}
print("\n")
print("Logistic Regression:")
grid_search(model, param_grid)

# 3. Classifier: Classification trees
model = DecisionTreeClassifier()
param_grid = {
    "criterion": ["gini", "entropy", "log_loss"],  # ToDo: maybe delete if takes too long
    "max_depth": [0],  # ToDo: choose parameters
    "min_samples_split": [2],
    "min_samples_leaf": [1],
    "ccp_alpha": [0.0],
}
print("\n")
print("Classification trees:")
grid_search(model, param_grid)

# 4. Classifier: Random forests
model = RandomForestClassifier()
param_grid = {
    "n_estimators": [100],  # ToDo: choose parameters
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [0],
    "min_samples_split": [2],
    "min_samples_leaf": [1],
    "ccp_alpha": [0.0],
}
print("\n")
print("Random Forests:")
grid_search(model, param_grid)
































