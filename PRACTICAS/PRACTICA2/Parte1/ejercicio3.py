# Ejercicio 3 practica 2 de IMD
# Realizado por Jaime Lorenzo Sanchez

# Listado de bibliotecas
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

# validacion cruzada
from sklearn.model_selection import cross_val_score
# k-vecinos
from sklearn.neighbors import KNeighborsClassifier
# Máquinas de vectores de soporte
from sklearn.svm import SVC
dataset_names = [];precision_arbol = []; error_arbol = []

precisionknn = []; error_knn = []

precision_svm = []; error_svm = []

# IRIS DATASET
dataset = datasets.load_iris(); dataset_names.append("Iris")
# Aplicamos la validacion cruzada con 10 folds y el modelo de arbol de decision 
scores = cross_val_score(DecisionTreeClassifier(), dataset.data, dataset.target, cv=10, scoring='accuracy')
precision_arbol.append(scores.mean())
error_arbol.append(1 - scores.mean())

# Aplicamos la validacion cruzada con 10 folds y el modelo de k-vecinos
scores = cross_val_score(KNeighborsClassifier(n_neighbors=5), dataset.data, dataset.target, cv=10, scoring='accuracy')
precisionknn.append(scores.mean())
error_knn.append(1 - scores.mean())

# Aplicamos la validacion cruzada con 10 folds y el modelo de máquinas de vectores de soporte
scores = cross_val_score(SVC(kernel='linear', C=1), dataset.data, dataset.target, cv=10, scoring='accuracy')
precision_svm.append(scores.mean())
error_svm.append(1 - scores.mean())

# WINE DATASET
dataset = datasets.load_wine(); dataset_names.append("Wine")
scores = cross_val_score(DecisionTreeClassifier(), dataset.data, dataset.target, cv=10, scoring='accuracy')
precision_arbol.append(scores.mean())
error_arbol.append(1 - scores.mean())

scores = cross_val_score(KNeighborsClassifier(n_neighbors=5), dataset.data, dataset.target, cv=10, scoring='accuracy')
precisionknn.append(scores.mean())
error_knn.append(1 - scores.mean())

scores = cross_val_score(SVC(kernel='linear', C=1), dataset.data, dataset.target, cv=10, scoring='accuracy')
precision_svm.append(scores.mean())
error_svm.append(1 - scores.mean())

# Breast Cancer DATASET
dataset = datasets.load_breast_cancer(); dataset_names.append("Breast Cancer")
scores = cross_val_score(DecisionTreeClassifier(), dataset.data, dataset.target, cv=10, scoring='accuracy')
precision_arbol.append(scores.mean())
error_arbol.append(1 - scores.mean())

scores = cross_val_score(KNeighborsClassifier(n_neighbors=5), dataset.data, dataset.target, cv=10, scoring='accuracy')
precisionknn.append(scores.mean())
error_knn.append(1 - scores.mean())

scores = cross_val_score(SVC(kernel='linear', C=1), dataset.data, dataset.target, cv=10, scoring='accuracy')
precision_svm.append(scores.mean())
error_svm.append(1 - scores.mean())

# Digits DATASET
dataset = datasets.load_digits(); dataset_names.append("Digits")
scores = cross_val_score(DecisionTreeClassifier(), dataset.data, dataset.target, cv=10, scoring='accuracy')
precision_arbol.append(scores.mean())
error_arbol.append(1 - scores.mean())

scores = cross_val_score(KNeighborsClassifier(n_neighbors=5), dataset.data, dataset.target, cv=10, scoring='accuracy')
precisionknn.append(scores.mean())
error_knn.append(1 - scores.mean())

scores = cross_val_score(SVC(kernel='linear', C=1), dataset.data, dataset.target, cv=10, scoring='accuracy')
precision_svm.append(scores.mean())
error_svm.append(1 - scores.mean())

# Mostramos la informacion en forma de tabla

df = pd.DataFrame({'Dataset': dataset_names, 'Precision arbol': precision_arbol, 'Error arbol': error_arbol, 'Precision knn': precisionknn, 'Error knn': error_knn, 'Precision svm': precision_svm, 'Error svm': error_svm})
print(df)