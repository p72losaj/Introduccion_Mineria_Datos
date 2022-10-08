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
from sklearn import preprocessing
from sklearn.utils import resample

# Creamos los modelos
model = KNeighborsClassifier()
model2 = DecisionTreeClassifier()

# Leemos el dataset iris
iris = datasets.load_iris(); X_iris = iris.data; y_iris = iris.target
# Leemos el dataset wine
wine = datasets.load_wine(); X_wine = wine.data; y_wine = wine.target
# Leemos el dataset diabetes
diabetes = datasets.load_diabetes(); X_diabetes = diabetes.data; y_diabetes = diabetes.target
# Leemos el dataset breast cancer
breast_cancer = datasets.load_breast_cancer(); X_breast_cancer = breast_cancer.data; y_breast_cancer = breast_cancer.target
# Creamos los arrays para los datos de los graficos
knn_size = []; knn_accuracy = []; knn_error=[]
arbol_size = []; arbol_accuracy = []; arbol_error=[]

# Split data into training and test sets 10% instancias sin reemplazo
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.1, random_state=0)

# Split data into training and test sets 10% instancias sin reemplazo
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, y_wine, test_size=0.1, random_state=0)

# Split data into training and test sets 10% instancias sin reemplazo
X_train_diabetes, X_test_diabetes, y_train_diabetes, y_test_diabetes = train_test_split(X_diabetes, y_diabetes, test_size=0.1, random_state=0)

# Split data into training and test sets 10% instancias sin reemplazo
X_train_breast_cancer, X_test_breast_cancer, y_train_breast_cancer, y_test_breast_cancer = train_test_split(X_breast_cancer, y_breast_cancer, test_size=0.1, random_state=0)

# Tamano del modelo
knn_size.append(len(X_train_iris))
arbol_size.append(len(X_train_iris))
knn_size.append(len(X_train_wine))
arbol_size.append(len(X_train_wine))
knn_size.append(len(X_train_diabetes))
arbol_size.append(len(X_train_diabetes))
knn_size.append(len(X_train_breast_cancer))
arbol_size.append(len(X_train_breast_cancer))

# Entrenamos el modelo iris
model.fit(X_train_iris, y_train_iris)
model2.fit(X_train_iris, y_train_iris)

# Hacemos las predicciones iris
y_pred_iris = model.predict(X_test_iris)
y_pred2_iris = model2.predict(X_test_iris)

# Calculamos la precision iris
knn_accuracy.append(metrics.accuracy_score(y_test_iris, y_pred_iris))
arbol_accuracy.append(metrics.accuracy_score(y_test_iris, y_pred2_iris))

# Calculamos el error iris
knn_error.append(1-metrics.accuracy_score(y_test_iris, y_pred_iris))
arbol_error.append(1-metrics.accuracy_score(y_test_iris, y_pred2_iris))

# Entrenamos el modelo wine
model.fit(X_train_wine, y_train_wine)
model2.fit(X_train_wine, y_train_wine)

# Hacemos las predicciones wine
y_pred_wine = model.predict(X_test_wine)
y_pred2_wine = model2.predict(X_test_wine)

# Calculamos la precision wine
knn_accuracy.append(metrics.accuracy_score(y_test_wine, y_pred_wine))
arbol_accuracy.append(metrics.accuracy_score(y_test_wine, y_pred2_wine))

# Calculamos el error wine
knn_error.append(1-metrics.accuracy_score(y_test_wine, y_pred_wine))
arbol_error.append(1-metrics.accuracy_score(y_test_wine, y_pred2_wine))

# Entrenamos el modelo diabetes
model.fit(X_train_diabetes, y_train_diabetes)
model2.fit(X_train_diabetes, y_train_diabetes)

# Hacemos las predicciones diabetes
y_pred_diabetes = model.predict(X_test_diabetes)
y_pred2_diabetes = model2.predict(X_test_diabetes)

# Calculamos la precision diabetes
knn_accuracy.append(metrics.accuracy_score(y_test_diabetes, y_pred_diabetes))
arbol_accuracy.append(metrics.accuracy_score(y_test_diabetes, y_pred2_diabetes))

# Calculamos el error diabetes
knn_error.append(1-metrics.accuracy_score(y_test_diabetes, y_pred_diabetes))
arbol_error.append(1-metrics.accuracy_score(y_test_diabetes, y_pred2_diabetes))

# Entrenamos el modelo breast cancer
model.fit(X_train_breast_cancer, y_train_breast_cancer)
model2.fit(X_train_breast_cancer, y_train_breast_cancer)

# Hacemos las predicciones breast cancer
y_pred_breast_cancer = model.predict(X_test_breast_cancer)
y_pred2_breast_cancer = model2.predict(X_test_breast_cancer)

# Calculamos la precision breast cancer
knn_accuracy.append(metrics.accuracy_score(y_test_breast_cancer, y_pred_breast_cancer))
arbol_accuracy.append(metrics.accuracy_score(y_test_breast_cancer, y_pred2_breast_cancer))

# Calculamos el error breast cancer
knn_error.append(1-metrics.accuracy_score(y_test_breast_cancer, y_pred_breast_cancer))
arbol_error.append(1-metrics.accuracy_score(y_test_breast_cancer, y_pred2_breast_cancer))


# Generamos el grafico de accuracy y error
plt.plot(knn_size, knn_accuracy, label='KNN_Accuracy')
plt.plot(knn_size, knn_error, label='KNN_Error')
plt.plot(arbol_size, arbol_accuracy, label='Arbol_Accuracy')
plt.plot(arbol_size, arbol_error, label='Arbol_Error')
plt.xlabel('Tamaño del modelo (train)')
plt.ylabel('Accuracy/Error')
plt.title('Muestreo aleatorio sin reemplazamiento del 10% ')
plt.legend()
plt.savefig("images/Comparacion.png")

# Limpiamos las listas
knn_size.clear(); knn_accuracy.clear(); knn_error.clear()
arbol_size.clear(); arbol_accuracy.clear(); arbol_error.clear()

# Limpiamos los datos de generacion del grafico
plt.clf() # Clear the figure

# Split data into training and test sets 10% instancias con muestreo estratificado
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.1, random_state=0, stratify=y_iris)
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, y_wine, test_size=0.1, random_state=0, stratify=y_wine)
# X_train_diabetes, X_test_diabetes, y_train_diabetes, y_test_diabetes = train_test_split(X_diabetes, y_diabetes, test_size=0.33, random_state=32, stratify=y_diabetes)
X_train_breast_cancer, X_test_breast_cancer, y_train_breast_cancer, y_test_breast_cancer = train_test_split(X_breast_cancer, y_breast_cancer, test_size=0.1, random_state=0, stratify=y_breast_cancer)

# Tamano de los datos de entrenamiento
knn_size.append(len(X_train_iris))
knn_size.append(len(X_train_wine))
# knn_size.append(len(X_train_diabetes))
knn_size.append(len(X_train_breast_cancer))
arbol_size.append(len(X_train_iris))
arbol_size.append(len(X_train_wine))
# arbol_size.append(len(X_train_diabetes))
arbol_size.append(len(X_train_breast_cancer))


# Entrenamos el modelo iris
model.fit(X_train_iris, y_train_iris)
model2.fit(X_train_iris, y_train_iris)

# Hacemos las predicciones iris
y_pred_iris = model.predict(X_test_iris)
y_pred2_iris = model2.predict(X_test_iris)

# Calculamos la precision iris
knn_accuracy.append(metrics.accuracy_score(y_test_iris, y_pred_iris))
arbol_accuracy.append(metrics.accuracy_score(y_test_iris, y_pred2_iris))

# Calculamos el error iris
knn_error.append(1-metrics.accuracy_score(y_test_iris, y_pred_iris))
arbol_error.append(1-metrics.accuracy_score(y_test_iris, y_pred2_iris))

# Entrenamos el modelo wine
model.fit(X_train_wine, y_train_wine)
model2.fit(X_train_wine, y_train_wine)

# Hacemos las predicciones wine
y_pred_wine = model.predict(X_test_wine)
y_pred2_wine = model2.predict(X_test_wine)

# Calculamos la precision wine
knn_accuracy.append(metrics.accuracy_score(y_test_wine, y_pred_wine))
arbol_accuracy.append(metrics.accuracy_score(y_test_wine, y_pred2_wine))

# Calculamos el error wine
knn_error.append(1-metrics.accuracy_score(y_test_wine, y_pred_wine))
arbol_error.append(1-metrics.accuracy_score(y_test_wine, y_pred2_wine))

# Entrenamos el modelo diabetes

# Hacemos las predicciones diabetes

# Calculamos la precision diabetes

# Calculamos el error diabetes

# Entrenamos el modelo breast cancer
model.fit(X_train_breast_cancer, y_train_breast_cancer)
model2.fit(X_train_breast_cancer, y_train_breast_cancer)

# Hacemos las predicciones breast cancer
y_pred_breast_cancer = model.predict(X_test_breast_cancer)
y_pred2_breast_cancer = model2.predict(X_test_breast_cancer)

# Calculamos la precision breast cancer
knn_accuracy.append(metrics.accuracy_score(y_test_breast_cancer, y_pred_breast_cancer))
arbol_accuracy.append(metrics.accuracy_score(y_test_breast_cancer, y_pred2_breast_cancer))

# Calculamos el error breast cancer
knn_error.append(1-metrics.accuracy_score(y_test_breast_cancer, y_pred_breast_cancer))
arbol_error.append(1-metrics.accuracy_score(y_test_breast_cancer, y_pred2_breast_cancer))

# Generamos el grafico de accuracy y error
plt.plot(knn_size, knn_accuracy, label='KNN_Accuracy')
plt.plot(knn_size, knn_error, label='KNN_Error')
plt.plot(arbol_size, arbol_accuracy, label='Arbol_Accuracy')
plt.plot(arbol_size, arbol_error, label='Arbol_Error')
plt.xlabel('Tamaño del modelo (train)')
plt.ylabel('Accuracy/Error')
plt.title('Muestreo estratificado del 10% ')
plt.legend()
plt.savefig("images/MuestreoEstratificado.png")
plt.show()

