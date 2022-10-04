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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# variables
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

# Apply PCA
pca = PCA(n_components=2) # 2 componentes principales
pca.fit(X_train)

# Fit and transform data
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Apply KNN
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    knn_size.append(i)
    knn_accuracy.append(metrics.accuracy_score(y_test, y_pred))
    knn_error.append(1-metrics.accuracy_score(y_test, y_pred))

# Apply Decision Tree
for i in range(1, 20):
    arbol = DecisionTreeClassifier(max_depth=i)
    arbol.fit(X_train, y_train)
    y_pred = arbol.predict(X_test)
    arbol_size.append(i)
    arbol_accuracy.append(metrics.accuracy_score(y_test, y_pred))
    arbol_error.append(1-metrics.accuracy_score(y_test, y_pred))

# Plot the results of KNN and Decision Tree in the same graph (accuracy) 
plt.plot(knn_size, knn_accuracy, label='KNN')
plt.plot(arbol_size, arbol_accuracy, label='Arbol')
plt.xlabel('Tamaño')
plt.ylabel('Accuracy')
plt.title('Accuracy KNN vs Arbol de Decisión en 2D')
plt.legend()
plt.show()
plt.savefig('images/ejercicio4_accuracy_2dimesiones.png')

# Plot error rate vs k for KNN and Decision Tree 
plt.plot(knn_size, knn_error, label='KNN')
plt.plot(arbol_size, arbol_error, label='Arbol')
plt.xlabel('Tamaño')
plt.ylabel('Error')
plt.title('Error KNN vs Arbol de Decisión en 2D')
plt.legend()
plt.show()
plt.savefig('images/ejercicio4_error_2dimesiones.png')






