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
from sklearn.datasets import fetch_openml # fetch_openml is able to fetch data from openml.org, the open machine learning repository.
from sklearn.impute import SimpleImputer
# iterative imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Cargamos el dataset breast cancer
breast_cancer = datasets.load_breast_cancer(); X_breast_cancer = breast_cancer.data; y_breast_cancer = breast_cancer.target
breast_cancer2 = datasets.load_breast_cancer(); X_breast_cancer2 = breast_cancer2.data; y_breast_cancer2 = breast_cancer2.target
# SimpleImputer es una clase que permite rellenar valores faltantes en un array de numpy
breast_cancer_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# IterativeImputer es una clase que permite rellenar valores faltantes en un array de numpy
breast_cancer_imputer2 = IterativeImputer(max_iter=10, random_state=0)
# Ajustamos el imputer a los datos
breast_cancer_imputer.fit(X_breast_cancer)
breast_cancer_imputer2.fit(X_breast_cancer2)
# Transformamos los datos
X_breast_cancer = breast_cancer_imputer.transform(X_breast_cancer)
X_breast_cancer2 = breast_cancer_imputer2.transform(X_breast_cancer2)


# Creamos los arrays para los datos de los graficos SimpleImputer
knn_size = []; knn_accuracy = []; knn_error=[]
arbol_size = []; arbol_accuracy = []; arbol_error=[]

# Creamos los arrays para los datos de los graficos IterativeImputer
knn_size2 = []; knn_accuracy2 = []; knn_error2=[]
arbol_size2 = []; arbol_accuracy2 = []; arbol_error2=[]
# Creamos los modelos
KNN = KNeighborsClassifier()
DecisionTree = DecisionTreeClassifier()
KNN2 = KNeighborsClassifier()
DecisionTree2 = DecisionTreeClassifier()

# Split data into training and test sets
X_train_breast_cancer, X_test_breast_cancer, y_train_breast_cancer, y_test_breast_cancer = train_test_split(X_breast_cancer, y_breast_cancer, test_size=0.1, random_state=0)
X_train_breast_cancer2, X_test_breast_cancer2, y_train_breast_cancer2, y_test_breast_cancer2 = train_test_split(X_breast_cancer2, y_breast_cancer2, test_size=0.1, random_state=0)

# Tamano del modelo
knn_size.append(len(X_train_breast_cancer))
arbol_size.append(len(X_train_breast_cancer))
knn_size2.append(len(X_train_breast_cancer2))
arbol_size2.append(len(X_train_breast_cancer2))

# Entrenamos el modelo breast cancer
KNN.fit(X_train_breast_cancer, y_train_breast_cancer)
DecisionTree.fit(X_train_breast_cancer, y_train_breast_cancer)
KNN2.fit(X_train_breast_cancer2, y_train_breast_cancer2)
DecisionTree2.fit(X_train_breast_cancer2, y_train_breast_cancer2)

# Hacemos las predicciones
y_pred_knn = KNN.predict(X_test_breast_cancer)
y_pred_arbol = DecisionTree.predict(X_test_breast_cancer)
y_pred_knn2 = KNN2.predict(X_test_breast_cancer2)
y_pred_arbol2 = DecisionTree2.predict(X_test_breast_cancer2)

# Calculamos la precision
knn_accuracy.append(metrics.accuracy_score(y_test_breast_cancer, y_pred_knn))
arbol_accuracy.append(metrics.accuracy_score(y_test_breast_cancer, y_pred_arbol))
knn_accuracy2.append(metrics.accuracy_score(y_test_breast_cancer2, y_pred_knn2))
arbol_accuracy2.append(metrics.accuracy_score(y_test_breast_cancer2, y_pred_arbol2))

# Calculamos el error
knn_error.append(1 - metrics.accuracy_score(y_test_breast_cancer, y_pred_knn))
arbol_error.append(1 - metrics.accuracy_score(y_test_breast_cancer, y_pred_arbol))
knn_error2.append(1 - metrics.accuracy_score(y_test_breast_cancer2, y_pred_knn2))
arbol_error2.append(1 - metrics.accuracy_score(y_test_breast_cancer2, y_pred_arbol2))

# Creamos una tabla con los valores de error y precision para ambos modelos
tabla = pd.DataFrame({'KNN Accuracy': knn_accuracy, 'Arbol Accuracy': arbol_accuracy, 'KNN Error': knn_error, 'Arbol Error': arbol_error}, index=knn_size)
print(tabla)

# Creamos una tabla con los valores de error y precision para ambos modelos
tabla2 = pd.DataFrame({'KNN Accuracy': knn_accuracy2, 'Arbol Accuracy': arbol_accuracy2, 'KNN Error': knn_error2, 'Arbol Error': arbol_error2}, index=knn_size2)
print(tabla2)


