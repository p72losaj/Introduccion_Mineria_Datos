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
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SequentialFeatureSelector # Sequential feature selection

# Cargamos el dataset breast cancer
breast_cancer = datasets.load_breast_cancer(); X_breast_cancer = breast_cancer.data; y_breast_cancer = breast_cancer.target
# Creamos los arrays para los datos de los graficos
knn_size = []; knn_accuracy = []; knn_error=[]
arbol_size = []; arbol_accuracy = []; arbol_error=[]
# Creamos los modelos
KNN = KNeighborsClassifier()
DecisionTree = DecisionTreeClassifier()

# Using RFECV
rfecv = RFECV(estimator=DecisionTree, step=1, cv=5, scoring='accuracy')
rfecv.fit(X_breast_cancer, y_breast_cancer)

# Eliminamos las columnas que no son relevantes
X_breast_cancer = X_breast_cancer[:,rfecv.support_]

# Split data into training and test sets
X_train_breast_cancer, X_test_breast_cancer, y_train_breast_cancer, y_test_breast_cancer = train_test_split(X_breast_cancer, y_breast_cancer, test_size=0.1, random_state=0)

# Creamos el modelo de KNN
KNN.fit(X_train_breast_cancer, y_train_breast_cancer)
# Creamos el modelo de arbol
DecisionTree.fit(X_train_breast_cancer, y_train_breast_cancer)

# Estudiamos el efecto del RFECV en el modelo de KNN
for i in range(1, X_train_breast_cancer.shape[1]+1):
    # Creamos el modelo de KNN
    KNN.fit(X_train_breast_cancer[:,0:i], y_train_breast_cancer)
    # Creamos el modelo de arbol
    DecisionTree.fit(X_train_breast_cancer[:,0:i], y_train_breast_cancer)
    # Calculamos el accuracy del modelo
    knn_accuracy.append(KNN.score(X_test_breast_cancer[:,0:i], y_test_breast_cancer))
    arbol_accuracy.append(DecisionTree.score(X_test_breast_cancer[:,0:i], y_test_breast_cancer))
    # Calculamos el error del modelo
    knn_error.append(1-KNN.score(X_test_breast_cancer[:,0:i], y_test_breast_cancer))
    arbol_error.append(1-DecisionTree.score(X_test_breast_cancer[:,0:i], y_test_breast_cancer))
    # Añadimos el tamaño de los datos
    knn_size.append(i)
    arbol_size.append(i)

# Creamos el dataframe para el grafico de KNN
df_knn = pd.DataFrame({'Tamaño': knn_size, 'Accuracy': knn_accuracy, 'Error': knn_error})
# Creamos el dataframe para el grafico de arbol
df_arbol = pd.DataFrame({'Tamaño': arbol_size, 'Accuracy': arbol_accuracy, 'Error': arbol_error})

# Creamos el grafico de KNN
sns.lineplot(data=df_knn, x="Tamaño", y="Accuracy", label="KNN_Accuracy")
sns.lineplot(data=df_arbol, x="Tamaño", y="Error", label="Arbol_Error")
sns.lineplot(data=df_knn, x="Tamaño", y="Error", label="KNN_Error")
sns.lineplot(data=df_arbol, x="Tamaño", y="Accuracy", label="Arbol_Accuracy")
plt.title("Efecto del RFCV en el modelo de KNN y arbol Decision")

# Guardamos el grafico
plt.savefig('images/RFCV.png')

# Limpiamos los graficos
plt.clf()

# Using SequentialFeatureSelector
sfs = SequentialFeatureSelector(DecisionTree, n_features_to_select=1, direction='backward')
sfs.fit(X_breast_cancer, y_breast_cancer)

# Eliminamos las columnas que no son relevantes
X_breast_cancer = X_breast_cancer[:,sfs.get_support()]

# Split data into training and test sets
X_train_breast_cancer, X_test_breast_cancer, y_train_breast_cancer, y_test_breast_cancer = train_test_split(X_breast_cancer, y_breast_cancer, test_size=0.1, random_state=0)

# Creamos el modelo de KNN
KNN.fit(X_train_breast_cancer, y_train_breast_cancer)
# Creamos el modelo de arbol
DecisionTree.fit(X_train_breast_cancer, y_train_breast_cancer)

# Estudiamos el efecto del SequentialFeatureSelector en el modelo de KNN
for i in range(1, X_train_breast_cancer.shape[1]+1):
    # Creamos el modelo de KNN
    KNN.fit(X_train_breast_cancer[:,0:i], y_train_breast_cancer)
    # Creamos el modelo de arbol
    DecisionTree.fit(X_train_breast_cancer[:,0:i], y_train_breast_cancer)
    # Calculamos el accuracy del modelo
    knn_accuracy.append(KNN.score(X_test_breast_cancer[:,0:i], y_test_breast_cancer))
    arbol_accuracy.append(DecisionTree.score(X_test_breast_cancer[:,0:i], y_test_breast_cancer))
    # Calculamos el error del modelo
    knn_error.append(1-KNN.score(X_test_breast_cancer[:,0:i], y_test_breast_cancer))
    arbol_error.append(1-DecisionTree.score(X_test_breast_cancer[:,0:i], y_test_breast_cancer))
    # Añadimos el tamaño de los datos
    knn_size.append(i)
    arbol_size.append(i)

# Creamos el dataframe para el grafico de KNN
df_knn = pd.DataFrame({'Tamaño': knn_size, 'Accuracy': knn_accuracy, 'Error': knn_error})
# Creamos el dataframe para el grafico de arbol
df_arbol = pd.DataFrame({'Tamaño': arbol_size, 'Accuracy': arbol_accuracy, 'Error': arbol_error})

# Creamos el grafico de KNN
sns.lineplot(data=df_knn, x="Tamaño", y="Accuracy", label="KNN_Accuracy")
sns.lineplot(data=df_arbol, x="Tamaño", y="Error", label="Arbol_Error")
sns.lineplot(data=df_knn, x="Tamaño", y="Error", label="KNN_Error")
sns.lineplot(data=df_arbol, x="Tamaño", y="Accuracy", label="Arbol_Accuracy")
plt.title("Efecto del SequentialFeatureSelector en el modelo de KNN y arbol Decision")

# Guardamos el grafico
plt.savefig('images/SequentialFeatureSelector.png')
