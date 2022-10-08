# Realizado por Jaime Lorenzo Sanchez
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import Binarizer

# Programa que estudia el efecto de la binarizacion de caracteristicas en el modelo de KNN y arbol de decision
# Cargamos el dataset breast cancer
breast_cancer = datasets.load_breast_cancer(); X_breast_cancer = breast_cancer.data; y_breast_cancer = breast_cancer.target
# Creamos los arrays para los datos de los graficos
knn_size = []; knn_accuracy = []; knn_error=[]
arbol_size = []; arbol_accuracy = []; arbol_error=[]
# Creamos los modelos
KNN = KNeighborsClassifier()
DecisionTree = DecisionTreeClassifier()

# Split data into training and test sets
X_train_breast_cancer, X_test_breast_cancer, y_train_breast_cancer, y_test_breast_cancer = train_test_split(X_breast_cancer, y_breast_cancer, test_size=0.1, random_state=0)

# Creamos el modelo de KNN
KNN.fit(X_train_breast_cancer, y_train_breast_cancer)
# Creamos el modelo de arbol
DecisionTree.fit(X_train_breast_cancer, y_train_breast_cancer)

# Estudiamos el efecto de la binarizacion en el modelo de KNN
for i in range(1, 11):
    # Creamos el modelo de binarizacion
    binarizer = Binarizer(threshold=i/10)
    # Binarizamos los datos de entrenamiento
    X_train_binarized = binarizer.fit_transform(X_train_breast_cancer)
    # Binarizamos los datos de test
    X_test_binarized = binarizer.fit_transform(X_test_breast_cancer)
    # Creamos el modelo de KNN
    KNN.fit(X_train_binarized, y_train_breast_cancer)
    # Creamos el modelo de arbol
    DecisionTree.fit(X_train_binarized, y_train_breast_cancer)
    # Calculamos el accuracy del modelo de KNN
    knn_accuracy.append(KNN.score(X_test_binarized, y_test_breast_cancer))
    # Calculamos el accuracy del modelo de arbol
    arbol_accuracy.append(DecisionTree.score(X_test_binarized, y_test_breast_cancer))
    # Calculamos el error del modelo de KNN
    knn_error.append(1-KNN.score(X_test_binarized, y_test_breast_cancer))
    # Calculamos el error del modelo de arbol
    arbol_error.append(1-DecisionTree.score(X_test_binarized, y_test_breast_cancer))
    # Guardamos el valor de la binarizacion
    knn_size.append(i/10)
    arbol_size.append(i/10)

# Creamos el grafico de accuracy y error del modelo de KNN y arbol de decision
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(knn_size, knn_accuracy, label='KNN_accuracy')
plt.plot(knn_size, arbol_accuracy, label='Arbol_accuracy')
plt.xlabel('Umbral') # Label x-axis with 'Threshold' 
plt.ylabel('Accuracy') # Label y-axis with 'Accuracy'
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(knn_size, knn_error, label='KNN_error')
plt.plot(knn_size, arbol_error, label='Arbol_error')
plt.xlabel('Umbral') # Label x-axis with 'Threshold'
plt.ylabel('Error')
plt.legend()
plt.savefig('images/ejercicio8.png')

