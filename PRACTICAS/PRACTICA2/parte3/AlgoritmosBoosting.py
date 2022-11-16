# Seleccionamos 2 algoritmos Boosting y los aplicamos a cada uno de los conjuntos
# Autor: Jaime Lorenzo Sanchez

# Listado de bibliotecas
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Dataset a entrenar y testear
iris = datasets.load_iris()
wine = datasets.load_wine()
breast_cancer = datasets.load_breast_cancer()
digits = datasets.load_digits()
datasets = ['iris','wine','breast cancer','digits']

# Variables para almacenar los resultados
precision_adaboost = []; precision_gradientboost=[]

# Aplicamos la validacion cruzada con 10 folds y el clasificador AdaBoost
scores = cross_val_score(AdaBoostClassifier(), iris.data, iris.target, cv=10, scoring='accuracy')
precision_adaboost.append(scores.mean())
scores = cross_val_score(GradientBoostingClassifier(), iris.data, iris.target, cv=10, scoring='accuracy')
precision_gradientboost.append(scores.mean())

# Procesamiento del dataset wine
scores = cross_val_score(AdaBoostClassifier(), wine.data, wine.target, cv=10, scoring='accuracy')
precision_adaboost.append(scores.mean())
scores = cross_val_score(GradientBoostingClassifier(), wine.data, wine.target, cv=10, scoring='accuracy')
precision_gradientboost.append(scores.mean())

# Procesamiento del dataset breast cancer
scores = cross_val_score(AdaBoostClassifier(), breast_cancer.data, breast_cancer.target, cv=10, scoring='accuracy')
precision_adaboost.append(scores.mean())
scores = cross_val_score(GradientBoostingClassifier(), breast_cancer.data, breast_cancer.target, cv=10, scoring='accuracy')
precision_gradientboost.append(scores.mean())

# Procesamiento del dataset digits
scores = cross_val_score(AdaBoostClassifier(), digits.data, digits.target, cv=10, scoring='accuracy')
precision_adaboost.append(scores.mean())
scores = cross_val_score(GradientBoostingClassifier(), digits.data, digits.target, cv=10, scoring='accuracy')
precision_gradientboost.append(scores.mean())

# Mostramos los resultados

for i in datasets:
    print('Dataset: ',i)
    print('Precision adaboost: ',precision_adaboost[datasets.index(i)])
    print('Precision gradientboost: ',precision_gradientboost[datasets.index(i)])

