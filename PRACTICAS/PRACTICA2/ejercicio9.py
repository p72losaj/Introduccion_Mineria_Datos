# Ejercicio 9 de la practica 2 de IMD
# Autor: Jaime Lorenzo Sanchez
# Aplicar el metodo RIPPER del modulo wiitgenstein en al menos 3 conjuntos de datos
# de 2 clases diferentes. Evaluar los modelos obtenidos y compararlos con el arbol de decision

# Listado de bibliotecas
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import wittgenstein as lw

# Dataset a entrenar y testear
iris = datasets.load_iris()
wine = datasets.load_wine()
breast_cancer = datasets.load_breast_cancer()

# Modelos de entrenamiento
ripper = lw.RIPPER()
arbol = DecisionTreeClassifier()
pred_ripper = []; precision_ripper = []
pred_arbol = []; precision_arbol = []

# Entrenamos el arbol de decision
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=1)
arbol.fit(X_train, y_train)
pred_arbol = arbol.predict(X_test)
precision_arbol.append(metrics.accuracy_score(y_test, pred_arbol))

X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=1)
arbol.fit(X_train, y_train)
pred_arbol = arbol.predict(X_test)
precision_arbol.append(metrics.accuracy_score(y_test, pred_arbol))

X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.3, random_state=1)
arbol.fit(X_train, y_train)
pred_arbol = arbol.predict(X_test)
precision_arbol.append(metrics.accuracy_score(y_test, pred_arbol))

