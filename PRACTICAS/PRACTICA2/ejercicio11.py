# Ejercicio 11 de la practica 2 de IMD
# Autor: Jaime Lorenzo Sanchez

# Listado de bibliotecas
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# Cross validation
from sklearn.model_selection import cross_val_score
# rbf kernel
from sklearn.svm import SVC
# Clasificador SVM con kernel lineal


# Dataset a entrenar y testear
iris = datasets.load_iris()
wine = datasets.load_wine()
breast_cancer = datasets.load_breast_cancer()
digits = datasets.load_digits()
datasets = ['iris','wine','breast cancer','digits']

# Parametros para el grid search
parameters = [ 
    {"kernel": ["rbf"], "gamma": [0.01, 0.1, 1.0], "C": [1, 10, 100, 1000]}, 
    {"kernel": ["linear"], "C": [1, 10, 100, 1000]}, ]

# Entrenamiento y testeo de los datasets con los parametros del grid search

# Dataset iris 

# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)
# clf = GridSearchCV(SVC(), parameters, cv=5)
# clf.fit(X_train, y_train)
# scores = cross_val_score(clf, iris.data, iris.target, cv=5)
# print("Dataset\t\tBest Parameters\t\t\tAccuracy")
# print(datasets[0],"\t\t",clf.best_params_,"\t\t",scores.mean())

# Dataset wine

# X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.2, random_state=0)
# clf = GridSearchCV(SVC(), parameters, cv=5)
# clf.fit(X_train, y_train)
# scores = cross_val_score(clf, wine.data, wine.target, cv=5)
# print(datasets[1],"\t\t",clf.best_params_,"\t\t",scores.mean())

# Dataset breast cancer

X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.2, random_state=0)
clf = GridSearchCV(SVC(), parameters, cv=5)
clf.fit(X_train, y_train)
scores = cross_val_score(clf, breast_cancer.data, breast_cancer.target, cv=5)
print(datasets[2],"\t",clf.best_params_,"\t\t",scores.mean())

# Dataset digits

# X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=0)
# clf = GridSearchCV(SVC(), parameters, cv=5)
# clf.fit(X_train, y_train)
# scores = cross_val_score(clf, digits.data, digits.target, cv=5)
# print(datasets[3],"\t\t",clf.best_params_,"\t\t",scores.mean())

