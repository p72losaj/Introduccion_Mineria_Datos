# Ejercicio 10 de la practica 2 de IMD
# Autor: Jaime Lorenzo Sanchez

# Listado de bibliotecas
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
# Clasificador SVM con kernel lineal
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Dataset a entrenar y testear
iris = datasets.load_iris()
wine = datasets.load_wine()
breast_cancer = datasets.load_breast_cancer()
digits = datasets.load_digits()
datasets = ['iris','wine','breast cancer','digits']

# Obtenemos la precision del modelo Arbol de decision 

arbol = DecisionTreeClassifier(); accuracy_arbol = []
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=1)
arbol.fit(X_train, y_train)
accuracy_arbol.append(metrics.accuracy_score(y_test, arbol.predict(X_test)))

X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=1)
arbol.fit(X_train, y_train)
accuracy_arbol.append(metrics.accuracy_score(y_test, arbol.predict(X_test)))

X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.3, random_state=1)
arbol.fit(X_train, y_train)
accuracy_arbol.append(metrics.accuracy_score(y_test, arbol.predict(X_test)))

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=1)
arbol.fit(X_train, y_train)
accuracy_arbol.append(metrics.accuracy_score(y_test, arbol.predict(X_test)))

# Obtenemos la precision del modelo SVM con kernel lineal y valor fijo C=1

svm = SVC(kernel='linear', C=1); accuracy_svm = []
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=1)
svm.fit(X_train, y_train)
accuracy_svm.append(metrics.accuracy_score(y_test, svm.predict(X_test)))

X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=1)
svm.fit(X_train, y_train)
accuracy_svm.append(metrics.accuracy_score(y_test, svm.predict(X_test)))

X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.3, random_state=1)
svm.fit(X_train, y_train)
accuracy_svm.append(metrics.accuracy_score(y_test, svm.predict(X_test)))

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=1)
svm.fit(X_train, y_train)
accuracy_svm.append(metrics.accuracy_score(y_test, svm.predict(X_test)))

# Resultados en grafico

plt.plot(datasets, accuracy_arbol, label='Arbol de decision')
plt.plot(datasets, accuracy_svm, label='SVM con kernel lineal')
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.title('SVM vs Arbol de decision')
plt.legend()
plt.savefig('images/SVMvsArbolDecision.png')