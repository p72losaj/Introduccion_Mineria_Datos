# Realizado por Jaime Lorenzo Sanchez
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
# variables
arbol_size = []; arbol_accuracy = []; arbol_error=[]
# Cargamos el dataset Iris.data
dataset = datasets.load_iris()
# Dividimos el dataset en train y test
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=1)
# Tamano del modelo
arbol_size.append(len(X_train))
# Creamos el modelo
model = DecisionTreeClassifier()
# Entrenamos el modelo
model.fit(X_train, y_train)
# Hacemos las predicciones
y_pred = model.predict(X_test)
# Calculamos la precision
precision = metrics.accuracy_score(y_test, y_pred)
# Calculamos el error
error = 1 - precision
# Guardamos los resultados
arbol_accuracy.append(precision)
arbol_error.append(error)

# Cargamos el dataset Wine data
wine = datasets.load_wine()
# Dividimos el dataset en train y test
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.2, random_state=1)
# Tamano del modelo
arbol_size.append(len(X_train))
# Creamos el modelo
model = DecisionTreeClassifier()
# Entrenamos el modelo
model.fit(X_train, y_train)
# Hacemos las predicciones
y_pred = model.predict(X_test)
# Calculamos la precision
precision = metrics.accuracy_score(y_test, y_pred)
# Calculamos el error
error = 1 - precision
# Guardamos los resultados
arbol_accuracy.append(precision)
arbol_error.append(error)

# Cargamos el dataset Diabetes.data
diabetes = datasets.load_diabetes()
# Dividimos el dataset en train y test
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=1)
# Tamano del modelo
arbol_size.append(len(X_train))
# Creamos el modelo
model = DecisionTreeClassifier()
# Entrenamos el modelo
model.fit(X_train, y_train)
# Hacemos las predicciones
y_pred = model.predict(X_test)
# Calculamos la precision
precision = metrics.accuracy_score(y_test, y_pred)
# Calculamos el error
error = 1 - precision
# Guardamos los resultados
arbol_accuracy.append(precision)
arbol_error.append(error)

# Cargamos el dataset Breast Cancer.data
cancer = datasets.load_breast_cancer()
# Dividimos el dataset en train y test
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=1)
# Tamano del modelo
arbol_size.append(len(X_train))
# Creamos el modelo
model = DecisionTreeClassifier()
# Entrenamos el modelo
model.fit(X_train, y_train)
# Hacemos las predicciones
y_pred = model.predict(X_test)
# Calculamos la precision
precision = metrics.accuracy_score(y_test, y_pred)
# Calculamos el error
error = 1 - precision
# Guardamos los resultados
arbol_accuracy.append(precision)
arbol_error.append(error)

# Cargamos el dataset Boston.data
boston = datasets.load_boston()
# Dividimos el dataset en train y test
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=1)
# Tamano del modelo
arbol_size.append(len(X_train))
# Creamos el modelo
model = DecisionTreeClassifier()
# Entrenamos el modelo
model.fit(X_train, y_train)
# Hacemos las predicciones
y_pred = model.predict(X_test)
# Calculamos la precision
precision = metrics.accuracy_score(y_test, y_pred)
# Calculamos el error
error = 1 - precision
# Guardamos los resultados
arbol_accuracy.append(precision)
arbol_error.append(error)


# Mostramos el arbol
plt.plot(arbol_size, arbol_accuracy, label='Accuracy')
plt.plot(arbol_size, arbol_error, label='Error')
plt.xlabel('Tamaño del modelo')
plt.ylabel('Accuracy/Error')
plt.title('Arbol de decision')
plt.legend()
plt.savefig("images/ArbolDecision.png")

