# Aplicamos el método de combinación de clasificadores bagging a cada
# conjunto de datos.

# Listado de bibliotecas
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier

# Dataset a entrenar y testear
iris = datasets.load_iris()
wine = datasets.load_wine()
breast_cancer = datasets.load_breast_cancer()
digits = datasets.load_digits()
datasets = ['iris','wine','breast cancer','digits']
# Variables para almacenar los resultados
precision_arbol = []; precision_svc=[]

# Procesamiento del dataset iris
scores = cross_val_score(BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=0.5), iris.data, iris.target, cv=10, scoring='accuracy')
precision_arbol.append(scores.mean())
scores = cross_val_score(BaggingClassifier(SVC(kernel='linear', C=1), max_samples=0.5, max_features=0.5), iris.data, iris.target, cv=10, scoring='accuracy')
precision_svc.append(scores.mean())

# Procesamiento del dataset wine
scores = cross_val_score(BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=0.5), wine.data, wine.target, cv=10, scoring='accuracy')
precision_arbol.append(scores.mean())
scores = cross_val_score(BaggingClassifier(SVC(kernel='linear', C=1), max_samples=0.5, max_features=0.5), wine.data, wine.target, cv=10, scoring='accuracy')
precision_svc.append(scores.mean())

# Procesamiento del dataset breast cancer
scores = cross_val_score(BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=0.5), breast_cancer.data, breast_cancer.target, cv=10, scoring='accuracy')
precision_arbol.append(scores.mean())
scores = cross_val_score(BaggingClassifier(SVC(kernel='linear', C=1), max_samples=0.5, max_features=0.5), breast_cancer.data, breast_cancer.target, cv=10, scoring='accuracy')
precision_svc.append(scores.mean())

# Procesamiento del dataset digits
scores = cross_val_score(BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=0.5), digits.data, digits.target, cv=10, scoring='accuracy')
precision_arbol.append(scores.mean())
scores = cross_val_score(BaggingClassifier(SVC(kernel='linear', C=1), max_samples=0.5, max_features=0.5), digits.data, digits.target, cv=10, scoring='accuracy')
precision_svc.append(scores.mean())

# Mostramos los resultados
print('Dataset ', 'Arbol ', 'SVC ', sep='\t')
for i in range(0,4):
    print(datasets[i], '\t', precision_arbol[i], '\t', precision_svc[i], sep='\t')

