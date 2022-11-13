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
df = pd.read_csv("iris.csv")
train, test = train_test_split(df, test_size=.33)
# Create and train model
ripper_clf = lw.RIPPER()
ripper_precision = []
ripper_clf.fit(df, class_feat="variety", pos_class='Versicolor')
# Score
X_test = test.drop('variety', axis=1)
y_test = test['variety']
ripper_clf.score(X_test, y_test)
# Calculamos la precision del modelo
ripper_precision.append(ripper_clf.score(X_test, y_test))

# Read dataset para el modelo ripper
# Descargar el archivo wine.csv de https://www.kaggle.com/brynja/wineuci

df = pd.read_csv("wine.csv")