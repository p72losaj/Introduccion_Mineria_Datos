# Ejercicio 4 practica 2 de IMD
# Realizado por Jaime Lorenzo Sanchez

# Listado de bibliotecas
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
# Arbol de decision
from sklearn.tree import DecisionTreeClassifier
# k-vecinos
from sklearn.neighbors import KNeighborsClassifier

# SVM
from sklearn.svm import SVC

# Wilcoxon test para comparar dos metodos de clasificacion en N datasets
from scipy.stats import wilcoxon

# Test de Friedman para comparar tres metodos de clasificacion en N datasets
from scipy.stats import friedmanchisquare

# Biblioteca test Iman-Davenport: mannwhitneyu
from scipy.stats import mannwhitneyu

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

# Test de Wilcoxon para comparar dos metodos de clasificacion en N datasets
# Comparacion arbol de decision y k-vecinos

# Almacenamos los resultados de la comparacion
wilcoxon_arbol_knn_iris = wilcoxon(pred_arbol[0], pred_knn[0])
wilcoxon_arbol_knn_wine = wilcoxon(pred_arbol[1], pred_knn[1])
wilcoxon_arbol_knn_breast_cancer = wilcoxon(pred_arbol[2], pred_knn[2])
wilcoxon_arbol_knn_digits = wilcoxon(pred_arbol[3], pred_knn[3])

# Grafico de comparacion del test de Wilcoxon
plt.figure()
# Titulo de la figura
plt.title("Wilcoxon: arbol de decision y k-vecinos")
# Etiquetas de los ejes
plt.xlabel("Dataset")
plt.ylabel("P-value")
# Datos de los ejes
plt.plot(["Iris", "Wine", "Breast cancer", "Digits"], [wilcoxon_arbol_knn_iris[1], wilcoxon_arbol_knn_wine[1], wilcoxon_arbol_knn_breast_cancer[1], wilcoxon_arbol_knn_digits[1]], 'ro')
plt.legend()
# Guardamos la figura en el directorio images
plt.savefig("images/wilcoxon_arbol_knn.png")
# Limpiamos la imagen 
plt.clf()

# Test de Friedman para comparar tres metodos de clasificacion en N datasets
# Comparacion arbol de decision, k-vecinos y SVM

# Almacenamos los resultados de la comparacion
friedman_arbol_knn_svm_iris = friedmanchisquare(pred_arbol[0], pred_knn[0], pred_svm[0])
friedman_arbol_knn_svm_wine = friedmanchisquare(pred_arbol[1], pred_knn[1], pred_svm[1])
friedman_arbol_knn_svm_breast_cancer = friedmanchisquare(pred_arbol[2], pred_knn[2], pred_svm[2])
friedman_arbol_knn_svm_digits = friedmanchisquare(pred_arbol[3], pred_knn[3], pred_svm[3])

# Generamos un grafico con los resultados del test de Friedman
plt.figure()
# Titulo de la figura
plt.title("Friedman: arbol de decision, k-vecinos y SVM")
# Etiquetas de los ejes
plt.xlabel("Dataset")
plt.ylabel("P-value")
# Datos de los ejes
plt.plot(["Iris", "Wine", "Breast cancer", "Digits"], [friedman_arbol_knn_svm_iris[1], friedman_arbol_knn_svm_wine[1], friedman_arbol_knn_svm_breast_cancer[1], friedman_arbol_knn_svm_digits[1]], 'ro')

plt.legend()
# Guardamos la figura en el directorio images
plt.savefig("images/friedman_arbol_knn_svm.png")
# Limpiamos la imagen
plt.clf()

# Test de Iman-Davenport para comparar tres metodos de clasificacion en N datasets
# Comparacion arbol de decision, k-vecinos y SVM

# Almacenamos los resultados de la comparacion
imandavenport_arbol_knn_svm_iris = mannwhitneyu(pred_arbol[0], pred_knn[0])
imandavenport_arbol_knn_svm_wine = mannwhitneyu(pred_arbol[1], pred_knn[1])
imandavenport_arbol_knn_svm_breast_cancer = mannwhitneyu(pred_arbol[2], pred_knn[2])
imandavenport_arbol_knn_svm_digits = mannwhitneyu(pred_arbol[3], pred_knn[3])

# Generamos un grafico con los resultados del test de Iman-Davenport
plt.figure()
# Titulo de la figura
plt.title("Iman-Davenport: arbol de decision, k-vecinos y SVM")
# Etiquetas de los ejes
plt.xlabel("Dataset")
plt.ylabel("P-value")
# Datos de los ejes
plt.plot(["Iris", "Wine", "Breast cancer", "Digits"], [imandavenport_arbol_knn_svm_iris[1], imandavenport_arbol_knn_svm_wine[1], imandavenport_arbol_knn_svm_breast_cancer[1], imandavenport_arbol_knn_svm_digits[1]], 'ro')
plt.legend()
# Guardamos la figura en el directorio images
plt.savefig("images/imandavenport_arbol_knn_svm.png")
# Limpiamos la imagen
plt.clf()