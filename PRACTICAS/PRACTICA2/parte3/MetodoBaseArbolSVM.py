# Enunciado del ejercicio: Aplicar el metodo base a cada uno de los conjuntos
# Autor: Jaime Lorenzo Sanchez

# Listado de bibliotecas
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
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

# Procesamiento del dataset iris
dataset = datasets.load_iris()

# Procesamiento del dataset wine

# Procesamiento del dataset breast cancer

# Procesamiento del dataset digits




