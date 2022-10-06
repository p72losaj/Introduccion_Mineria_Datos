# Realizado por Jaime Lorenzo Sanchez
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import preprocessing

# variables
knn_size = []; knn_accuracy = []; knn_error=[]
arbol_size = []; arbol_accuracy = []; arbol_error=[]

# Creamos el modelo
model = KNeighborsClassifier()
model2 = DecisionTreeClassifier()

# Cargamos el dataset Iris.data
dataset = datasets.load_iris()
# Normalizamos el modelo
dataset.data = preprocessing.normalize(dataset.data)
# Dividimos el dataset en train y test
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=1)
# Tamano del modelo
knn_size.append(len(X_train))
arbol_size.append(len(X_train))
# Entrenamos el modelo
model.fit(X_train, y_train)
model2.fit(X_train, y_train)
# Hacemos las predicciones
y_pred = model.predict(X_test)
y_pred2 = model2.predict(X_test)
# Calculamos la precision
precision = metrics.accuracy_score(y_test, y_pred)
precision2 = metrics.accuracy_score(y_test, y_pred2)
# Calculamos el error
error = 1 - precision
error2 = 1 - precision2
# Guardamos los resultados
knn_accuracy.append(precision)
knn_error.append(error)
arbol_accuracy.append(precision2)
arbol_error.append(error2)

# Cargamos el dataset Wine data
wine = datasets.load_wine()
# Normalizamos el modelo
wine.data = preprocessing.normalize(wine.data)
# Dividimos el dataset en train y test
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.2, random_state=1)
#Tamano del modelo
knn_size.append(len(X_train))
arbol_size.append(len(X_train))
# Entrenamos el modelo
model.fit(X_train, y_train)
model2.fit(X_train, y_train)
# Hacemos las predicciones
y_pred = model.predict(X_test)
y_pred2 = model2.predict(X_test)
# Calculamos la precision
precision = metrics.accuracy_score(y_test, y_pred)
precision2 = metrics.accuracy_score(y_test, y_pred2)
# Calculamos el error
error = 1 - precision
error2 = 1 - precision2
# Guardamos los resultados
knn_accuracy.append(precision)
knn_error.append(error)
arbol_accuracy.append(precision2)
arbol_error.append(error2)

# Cargamos el dataset Diabetes.data
diabetes = datasets.load_diabetes()
# Normalizamos el modelo
diabetes.data = preprocessing.normalize(diabetes.data)
# Dividimos el dataset en train y test
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=1)
# Tamano del modelo
knn_size.append(len(X_train))
arbol_size.append(len(X_train))
# Entrenamos el modelo
model.fit(X_train, y_train)
model2.fit(X_train, y_train)
# Hacemos las predicciones
y_pred = model.predict(X_test)
y_pred2 = model2.predict(X_test)
# Calculamos la precision
precision = metrics.accuracy_score(y_test, y_pred)
precision2 = metrics.accuracy_score(y_test, y_pred2)
# Calculamos el error
error = 1 - precision
error2 = 1 - precision2
# Guardamos los resultados
knn_accuracy.append(precision)
knn_error.append(error)
arbol_accuracy.append(precision2)
arbol_error.append(error2)

# Cargamos el dataset Breast Cancer.data
cancer = datasets.load_breast_cancer()
# Normalizamos el modelo
cancer.data = preprocessing.normalize(cancer.data)
# Dividimos el dataset en train y test
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=1)
# Tamano del modelo
knn_size.append(len(X_train))
arbol_size.append(len(X_train))
# Entrenamos el modelo
model.fit(X_train, y_train)
model2.fit(X_train, y_train)
# Hacemos las predicciones
y_pred = model.predict(X_test)
y_pred2 = model2.predict(X_test)
# Calculamos la precision
precision = metrics.accuracy_score(y_test, y_pred)
precision2 = metrics.accuracy_score(y_test, y_pred2)
# Calculamos el error
error = 1 - precision
error2 = 1 - precision2
# Guardamos los resultados
knn_accuracy.append(precision)
knn_error.append(error)
arbol_accuracy.append(precision2)
arbol_error.append(error2)

# Generamos el grafico de comparacion

plt.plot(knn_size, knn_accuracy, label='KNN_Accuracy')
plt.plot(knn_size, knn_error, label='KNN_Error')
plt.plot(arbol_size, arbol_accuracy, label='Arbol_Accuracy')
plt.plot(arbol_size, arbol_error, label='Arbol_Error')
plt.xlabel('Tamaño del modelo (train)')
plt.ylabel('Accuracy/Error')
plt.title('Comparacion de algoritmos de clasificacion (KNN vs Arbol de decision)') 
plt.legend()
plt.savefig("images/Comparacion.png")
