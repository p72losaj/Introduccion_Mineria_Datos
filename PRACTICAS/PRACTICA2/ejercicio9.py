# Ejercicio 8 de la practica 2 de imd
# Autor: Jaime Lorenzo Sanchez

# Listado de bibliotecas
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import wittgenstein as lw
import pandas as pd

# Read dataset para el modelo ripper
iris = pd.read_csv("iris.csv")
titanic = pd.read_csv("titanic.csv")
# Create and train model
ripper_clf = lw.RIPPER()
ripper_precision = []
ripper_clf.fit(iris, class_feat="variety", pos_class='Versicolor')
# Score
train, test = train_test_split(iris, test_size=.33)
X_test = test.drop('variety', axis=1)
y_test = test['variety']
ripper_clf.score(X_test, y_test)
# Calculamos la precision del modelo
ripper_precision.append(ripper_clf.score(X_test, y_test))

# Read dataset para el modelo ripper 2 (con 2 reglas) 
train, test = train_test_split(titanic, test_size=.33)
# Create and train model
ripper_clf.fit(titanic, class_feat="Embarked", pos_class='C')
# Score
X_test = test.drop('Embarked', axis=1)
y_test = test['Embarked']
ripper_clf.score(X_test, y_test)
# Calculamos la precision del modelo
ripper_precision.append(ripper_clf.score(X_test, y_test))
# Read dataset para el modelo ripper 2 (con 2 reglas)


print("Precision de los modelos ripper: ", ripper_precision)