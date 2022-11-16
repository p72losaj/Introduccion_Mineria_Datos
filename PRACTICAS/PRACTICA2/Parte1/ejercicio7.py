# Ejercicio 7 practica 2 de IMD
# Para uno de los clasificadores elegidos utilice una validación de los hiper parámetros 
# con grid search y compare su rendimiento con el método con hiper parámetros fijados 
# a priori. 
# autor: Jaime Lorenzo Sanchez

# Listado de bibliotecas
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# Dataset a entrenar y testear
iris = datasets.load_iris()
wine = datasets.load_wine()
breast_cancer = datasets.load_breast_cancer()
digits = datasets.load_digits()
datasets = ['iris','wine','breast cancer','digits']
# Modelos de entrenamiento
# Hiperparametros a priori fijados para el arbol de decision
arbol = DecisionTreeClassifier()
pred_arbol = []; precision_apriori = []

# Entrenamos el arbol de decision
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=1)
arbol.fit(X_train, y_train)
pred_arbol.append(arbol.predict(X_test))
precision_apriori.append(metrics.accuracy_score(y_test, pred_arbol[0]))
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=1)
arbol.fit(X_train, y_train)
pred_arbol.append(arbol.predict(X_test))
precision_apriori.append(metrics.accuracy_score(y_test, pred_arbol[1]))
X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.3, random_state=1)
arbol.fit(X_train, y_train)
pred_arbol.append(arbol.predict(X_test))
precision_apriori.append(metrics.accuracy_score(y_test, pred_arbol[2]))
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=1)
arbol.fit(X_train, y_train)
pred_arbol.append(arbol.predict(X_test))
precision_apriori.append(metrics.accuracy_score(y_test, pred_arbol[3]))

# Grid search para el arbol de decision
param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
grid = GridSearchCV(DecisionTreeClassifier(), param_grid, refit = True, verbose = 3)
pred_grid = []; precision_grid = []
# Entrenamos el arbol de decision
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=1)
grid.fit(X_train, y_train)
pred_grid.append(grid.predict(X_test))
precision_grid.append(metrics.accuracy_score(y_test, pred_grid[0]))
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=1)
grid.fit(X_train, y_train)
pred_grid.append(grid.predict(X_test))
precision_grid.append(metrics.accuracy_score(y_test, pred_grid[1]))
X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.3, random_state=1)
grid.fit(X_train, y_train)
pred_grid.append(grid.predict(X_test))
precision_grid.append(metrics.accuracy_score(y_test, pred_grid[2]))
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=1)
grid.fit(X_train, y_train)
pred_grid.append(grid.predict(X_test))
precision_grid.append(metrics.accuracy_score(y_test, pred_grid[3]))

# Mostramos la precision del modelo

print('Dataset: \t PrecisionPriori: \t PrecisionGrid: \t')
for i in range(len(datasets)):
    print(datasets[i],'\t',precision_apriori[i],'\t',precision_grid[i])
    
