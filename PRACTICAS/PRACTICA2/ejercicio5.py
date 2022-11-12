# Ejercicio 5 practica 2 de IMD
# Realizado por Jaime Lorenzo Sanchez

# Listado de bibliotecas
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# Test de Friedman para comparar tres metodos de clasificacion en N datasets
from scipy.stats import friedmanchisquare
# Procedimiento de Holm para comparar tres metodos de clasificacion en N datasets (p<0.05)
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison

# Dataset a entrenar y testear
iris = datasets.load_iris()
wine = datasets.load_wine()
breast_cancer = datasets.load_breast_cancer()
digits = datasets.load_digits()

# Modelos de entrenamiento
arbol = DecisionTreeClassifier()
knn = KNeighborsClassifier(n_neighbors=5)
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
svm = SVC(kernel='linear', C=1).fit(iris.data, iris.target)
pred_svm.append(svm.predict(iris.data))
svm = SVC(kernel='linear', C=1).fit(wine.data, wine.target)
pred_svm.append(svm.predict(wine.data))
svm = SVC(kernel='linear', C=1).fit(breast_cancer.data, breast_cancer.target)
pred_svm.append(svm.predict(breast_cancer.data))
svm = SVC(kernel='linear', C=1).fit(digits.data, digits.target)
pred_svm.append(svm.predict(digits.data))

# Metodo de comparacion de tres metodos de clasificacion en N datasets (p<0.05) usando Friedman
friedman_arbol_knn_svm_iris = friedmanchisquare(pred_arbol[0], pred_knn[0], pred_svm[0])
friedman_arbol_knn_svm_wine = friedmanchisquare(pred_arbol[1], pred_knn[1], pred_svm[1])
friedman_arbol_knn_svm_breast_cancer = friedmanchisquare(pred_arbol[2], pred_knn[2], pred_svm[2])
friedman_arbol_knn_svm_digits = friedmanchisquare(pred_arbol[3], pred_knn[3], pred_svm[3])

# Comparamos el mejor metodo segun Friedman con el resto de metodos usando Holm
mc = MultiComparison(pred_arbol[0], pred_knn[0])
result = mc.tukeyhsd()
# Almacenamos los resultados en un fichero de texto
f = open('ejercicio5.txt', 'w')
f.write('Resultados de comparacion de tres metodos de clasificacion en N datasets (p<0.05) usando Friedman y Holm')


print(result)
