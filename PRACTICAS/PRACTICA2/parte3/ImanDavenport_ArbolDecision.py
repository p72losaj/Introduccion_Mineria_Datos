# Comparacion de diferencias significativas entre los algoritmos usando Iman-Davenport
# Hipotesis nula: No hay diferencias significativas entre los algoritmos
# Hipotesis alternativa: Hay diferencias significativas entre los algoritmos
# Si el p-value es menor que 0.05, rechazamos la hipotesis nula
# Autor: Jaime Lorenzo Sanchez

# Listado de bibliotecas
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from scipy.stats import mannwhitneyu
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


# Dataset a entrenar y testear
iris = datasets.load_iris()
wine = datasets.load_wine()
breast_cancer = datasets.load_breast_cancer()
digits = datasets.load_digits()

# Variables para almacenar los resultados
datasets = ['iris','wine','breast cancer','digits']
precision_arbol = []
precision_arbol_bagging = []
precision_arbol_adaboost = []
precision_arbol_gradientboost = []
rechazo_BaseBagging = []
rechazo_BaseAdaBoost = []
rechazo_BaseGradientBoost = []
rechazo_BaggingAdaBoost = []
rechazo_BaggingGradientBoost = []
rechazo_AdaBoostGradientBoost = []

# Obtenemos la precision del arbol de decision Base para cada dataset
scores = cross_val_score(DecisionTreeClassifier(), iris.data, iris.target, cv=10, scoring='accuracy')
precision_arbol.append(scores.mean())
scores = cross_val_score(DecisionTreeClassifier(), wine.data, wine.target, cv=10, scoring='accuracy')
precision_arbol.append(scores.mean())
scores = cross_val_score(DecisionTreeClassifier(), breast_cancer.data, breast_cancer.target, cv=10, scoring='accuracy')
precision_arbol.append(scores.mean())
scores = cross_val_score(DecisionTreeClassifier(), digits.data, digits.target, cv=10, scoring='accuracy')
precision_arbol.append(scores.mean())

# Obtenemos la precision del arbol de decision con Bagging para cada dataset
scores = cross_val_score(BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=0.5), iris.data, iris.target, cv=10, scoring='accuracy')
precision_arbol_bagging.append(scores.mean())
scores = cross_val_score(BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=0.5), wine.data, wine.target, cv=10, scoring='accuracy')
precision_arbol_bagging.append(scores.mean())
scores = cross_val_score(BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=0.5), breast_cancer.data, breast_cancer.target, cv=10, scoring='accuracy')
precision_arbol_bagging.append(scores.mean())
scores = cross_val_score(BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=0.5), digits.data, digits.target, cv=10, scoring='accuracy')
precision_arbol_bagging.append(scores.mean())

# Obtenemos la precision del arbol de decision con AdaBoost para cada dataset
scores = cross_val_score(AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200), iris.data, iris.target, cv=10, scoring='accuracy')
precision_arbol_adaboost.append(scores.mean())
scores = cross_val_score(AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200), wine.data, wine.target, cv=10, scoring='accuracy')
precision_arbol_adaboost.append(scores.mean())
scores = cross_val_score(AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200), breast_cancer.data, breast_cancer.target, cv=10, scoring='accuracy')
precision_arbol_adaboost.append(scores.mean())
scores = cross_val_score(AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200), digits.data, digits.target, cv=10, scoring='accuracy')
precision_arbol_adaboost.append(scores.mean())

# Obtenemos la precision del arbol de decision con GradientBoost para cada dataset
scores = cross_val_score(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0), iris.data, iris.target, cv=10, scoring='accuracy')
precision_arbol_gradientboost.append(scores.mean())
scores = cross_val_score(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0), wine.data, wine.target, cv=10, scoring='accuracy')
precision_arbol_gradientboost.append(scores.mean())
scores = cross_val_score(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0), breast_cancer.data, breast_cancer.target, cv=10, scoring='accuracy')
precision_arbol_gradientboost.append(scores.mean())
scores = cross_val_score(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0), digits.data, digits.target, cv=10, scoring='accuracy')
precision_arbol_gradientboost.append(scores.mean())

# Imprimimos los resultados
print('Test de Iman-Davenport - Arbol de Decision')
print('----------------------')
for i in range (0,datasets.__len__()):
    print('Dataset: ', datasets[i])
    # Comparacion Iman-Davenport Arbol Decision Base con Arbol Decision Bagging
    stat, p = mannwhitneyu(precision_arbol[i], precision_arbol_bagging[i])
    if p > 0.05:
        rechazo_BaseBagging.append('No')
    else:
        rechazo_BaseBagging.append('Si')
    print('P-value Base vs Bagging: ', p , '## Rechazo H0: ', rechazo_BaseBagging[i])
    # Comparacion Iman-Davenport Arbol Decision Base con Arbol Decision AdaBoost
    stat, p = mannwhitneyu(precision_arbol[i], precision_arbol_adaboost[i])
    if p > 0.05:
        rechazo_BaseAdaBoost.append('No')
    else:
        rechazo_BaseAdaBoost.append('Si')
    print('P-value Base vs AdaBoost: ', p , '## Rechazo H0: ', rechazo_BaseAdaBoost[i])
    # Comparacion Iman-Davenport Arbol Decision Base con Arbol Decision GradientBoost
    stat, p = mannwhitneyu(precision_arbol[i], precision_arbol_gradientboost[i])
    if p > 0.05:
        rechazo_BaseGradientBoost.append('No')
    else:
        rechazo_BaseGradientBoost.append('Si')
    print('P-value Base vs GradientBoost: ', p , '## Rechazo H0: ', rechazo_BaseGradientBoost[i])
    # Comparacion Iman-Davenport Arbol Decision Bagging con Arbol Decision AdaBoost
    stat, p = mannwhitneyu(precision_arbol_bagging[i], precision_arbol_adaboost[i])
    if p > 0.05:
        rechazo_BaggingAdaBoost.append('No')
    else:
        rechazo_BaggingAdaBoost.append('Si')
    print('P-value Bagging vs AdaBoost: ', p , '## Rechazo H0: ', rechazo_BaggingAdaBoost[i])
    # Comparacion Iman-Davenport Arbol Decision Bagging con Arbol Decision GradientBoost
    stat, p = mannwhitneyu(precision_arbol_bagging[i], precision_arbol_gradientboost[i])
    if p > 0.05:
        rechazo_BaggingGradientBoost.append('No')
    else:
        rechazo_BaggingGradientBoost.append('Si')
    print('P-value Bagging vs GradientBoost: ', p , '## Rechazo H0: ', rechazo_BaggingGradientBoost[i])
    # Comparacion Iman-Davenport Arbol Decision AdaBoost con Arbol Decision GradientBoost
    stat, p = mannwhitneyu(precision_arbol_adaboost[i], precision_arbol_gradientboost[i])
    if p > 0.05:
        rechazo_AdaBoostGradientBoost.append('No')
    else:
        rechazo_AdaBoostGradientBoost.append('Si')
    print('P-value AdaBoost vs GradientBoost: ', p , '## Rechazo H0: ', rechazo_AdaBoostGradientBoost[i])
    print('----------------------')


