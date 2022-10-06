# Realizado por Jaime Lorenzo Sanchez
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt # Grafica PCA 3D
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# variables para el grafico de PCA de 3 dimensiones
knn_size = []; knn_accuracy = []; knn_error=[]
arbol_size = []; arbol_accuracy = []; arbol_error=[]

# Cargamos el dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA 3D
pca = PCA(n_components=3)
pca.fit(X_train)

# Fit and transform data 3D
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Creamos el modelo
model = KNeighborsClassifier()
model2 = DecisionTreeClassifier()

# Tamano del modelo 3D
knn_size.append(len(X_train))
arbol_size.append(len(X_train))


# Entrenamos el modelo 3D
model.fit(X_train, y_train)
model2.fit(X_train, y_train)


# Hacemos las predicciones 3D
y_pred = model.predict(X_test)
y_pred2 = model2.predict(X_test)


# Calculamos la precision 3D
precision = metrics.accuracy_score(y_test, y_pred)
precision2 = metrics.accuracy_score(y_test, y_pred2)

# Calculamos el error 3D
error = 1 - precision
error2 = 1 - precision2

# Guardamos los resultados 3D
knn_accuracy.append(precision)
knn_error.append(error)
arbol_accuracy.append(precision2)
arbol_error.append(error2)

# Cargamos el dataset Wine data
wine = datasets.load_wine()
X = wine.data
y = wine.target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize the data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA 3D
pca.fit(X_train)

# Fit and transform data 3D
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Tamano del modelo 3D
knn_size.append(len(X_train))
arbol_size.append(len(X_train))

# Entrenamos el modelo 3D
model.fit(X_train, y_train)
model2.fit(X_train, y_train)

# Hacemos las predicciones 3D
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

# Cargamos el dataset Diabetes data
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize the data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA
pca.fit(X_train)

# Fit and transform data
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

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

# Cargamos el dataset Breast cancer data
cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize the data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA
pca.fit(X_train)

# Fit and transform data
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

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

# Generamos el grafico de los resultados de los modelos KNN y Arbol de decision
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(knn_size, knn_accuracy, label='KNN Accuracy')
plt.plot(arbol_size, arbol_accuracy, label='Arbol de decision Accuracy')
plt.plot(knn_size, knn_error, label='KNN Error')
plt.plot(arbol_size, arbol_error, label='Arbol de decision Error')
plt.xlabel('Tamano del modelo (train) ')
plt.ylabel('Precision/Error del modelo')
plt.title('Algoritmos KNN y Arbol de decision en 3D')
plt.legend()
plt.savefig('images/ejercicio4_3dimesiones.png')
