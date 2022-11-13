# Ejercicio 6 practica 2 de IMD
# Compare los métodos por parejas usando el procedimiento de Bonferroni-Dunn.
# Autor: Jaime Lorenzo Sanchez

# Listado de bibliotecas
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# Biblioteca de Ponferroni-Dunn
import scikit_posthocs as sp


# Dataset a entrenar y testear
iris = datasets.load_iris()
wine = datasets.load_wine()
breast_cancer = datasets.load_breast_cancer()
digits = datasets.load_digits()

datasets = ['iris','wine','breast cancer','digits']

# Modelos de entrenamiento
arbol = DecisionTreeClassifier()
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(kernel='linear', C=1)
pred_arbol = []; pred_knn = []; pred_svm = []

# Entrenamos el arbol de decision
arbol.fit(iris.data, iris.target)
pred_arbol.append(arbol.predict(iris.data))
arbol.fit(wine.data, wine.target)
pred_arbol.append(arbol.predict(wine.data))
arbol.fit(breast_cancer.data, breast_cancer.target)
pred_arbol.append(arbol.predict(breast_cancer.data))
arbol.fit(digits.data, digits.target)
pred_arbol.append(arbol.predict(digits.data))

# Entrenamos el k-vecinos
knn.fit(iris.data, iris.target)
pred_knn.append(knn.predict(iris.data))
knn.fit(wine.data, wine.target)
pred_knn.append(knn.predict(wine.data))
knn.fit(breast_cancer.data, breast_cancer.target)
pred_knn.append(knn.predict(breast_cancer.data))
knn.fit(digits.data, digits.target)
pred_knn.append(knn.predict(digits.data))

# Entrenamos el SVM
svm.fit(iris.data, iris.target)
pred_svm.append(svm.predict(iris.data))
svm.fit(wine.data, wine.target)
pred_svm.append(svm.predict(wine.data))
svm.fit(breast_cancer.data, breast_cancer.target)
pred_svm.append(svm.predict(breast_cancer.data))
svm.fit(digits.data, digits.target)
pred_svm.append(svm.predict(digits.data))

# Prueba de Dunn usando una correccion de Bonferroni para los valores p

p_arbol_knn = []; p_arbol_svm = []; p_knn_svm = []

# Calculamos los valores p
for i in range(4):
    p_arbol_knn.append(sp.posthoc_dunn([pred_arbol[i], pred_knn[i]], p_adjust='bonferroni'))
    p_arbol_svm.append(sp.posthoc_dunn([pred_arbol[i], pred_svm[i]], p_adjust='bonferroni'))
    p_knn_svm.append(sp.posthoc_dunn([pred_knn[i], pred_svm[i]], p_adjust='bonferroni'))

# Resultados del dataset Iris
print('Dataset Iris')
print('p_arbol_knn: \n', p_arbol_knn[0])
print('p_arbol_svm: \n', p_arbol_svm[0])
print('p_knn_svm: \n', p_knn_svm[0])

# Resultados del dataset Wine
print('Dataset Wine')
print('p_arbol_knn: \n', p_arbol_knn[1])
print('p_arbol_svm: \n', p_arbol_svm[1])
print('p_knn_svm: \n', p_knn_svm[1])

# Resultados del dataset Breast Cancer
print('Dataset Breast Cancer')
print('p_arbol_knn: \n', p_arbol_knn[2])
print('p_arbol_svm: \n', p_arbol_svm[2])
print('p_knn_svm: \n', p_knn_svm[2])

# Resultados del dataset Digits
print('Dataset Digits')
print('p_arbol_knn: \n', p_arbol_knn[3])
print('p_arbol_svm: \n', p_arbol_svm[3])
print('p_knn_svm: \n', p_knn_svm[3])

