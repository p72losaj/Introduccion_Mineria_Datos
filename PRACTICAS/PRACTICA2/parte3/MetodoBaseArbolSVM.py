# Enunciado del ejercicio: Aplicar el metodo base a cada uno de los conjuntos
# Autor: Jaime Lorenzo Sanchez

# Listado de bibliotecas
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


# Dataset a entrenar y testear
iris = datasets.load_iris()
wine = datasets.load_wine()
breast_cancer = datasets.load_breast_cancer()
digits = datasets.load_digits()
datasets = ['iris','wine','breast cancer','digits']

# Variables para almacenar los resultados
precision_arbol = []; precision_svc=[]

# Aplicamos la validacion cruzada con 10 folds y el modelo de arbol de decision 
scores = cross_val_score(DecisionTreeClassifier(), iris.data, iris.target, cv=10, scoring='accuracy')
precision_arbol.append(scores.mean())
# Aplicamos la validacion cruzada con 10 folds y el modelo de máquinas de vectores de soporte
scores = cross_val_score(SVC(kernel='linear', C=1), iris.data, iris.target, cv=10, scoring='accuracy')
precision_svc.append(scores.mean())

# Procesamiento del dataset wine
scores = cross_val_score(DecisionTreeClassifier(), wine.data, wine.target, cv=10, scoring='accuracy')
precision_arbol.append(scores.mean())
scores = cross_val_score(SVC(kernel='linear', C=1), wine.data, wine.target, cv=10, scoring='accuracy')
precision_svc.append(scores.mean())

# Procesamiento del dataset breast cancer
scores = cross_val_score(DecisionTreeClassifier(), breast_cancer.data, breast_cancer.target, cv=10, scoring='accuracy')
precision_arbol.append(scores.mean())
scores = cross_val_score(SVC(kernel='linear', C=1), breast_cancer.data, breast_cancer.target, cv=10, scoring='accuracy')
precision_svc.append(scores.mean())

# Procesamiento del dataset digits
scores = cross_val_score(DecisionTreeClassifier(), digits.data, digits.target, cv=10, scoring='accuracy')
precision_arbol.append(scores.mean())
scores = cross_val_score(SVC(kernel='linear', C=1), digits.data, digits.target, cv=10, scoring='accuracy')
precision_svc.append(scores.mean())

# Mostramos los resultados

for i in datasets:
    print('Dataset: ',i)
    print('Precision arbol: ',precision_arbol[datasets.index(i)])
    print('Precision svc: ',precision_svc[datasets.index(i)])

