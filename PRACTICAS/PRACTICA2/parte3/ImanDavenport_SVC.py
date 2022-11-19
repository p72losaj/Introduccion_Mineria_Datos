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
from sklearn.svm import SVC

# Dataset a entrenar y testear
iris = datasets.load_iris()
wine = datasets.load_wine()
breast_cancer = datasets.load_breast_cancer()
digits = datasets.load_digits()

# Variables para almacenar los resultados
datasets = ['iris','wine','breast cancer','digits']
precision_SVC = []
precision_SVC_bagging = []
precision_SVC_adaboost = []
precision_SVC_gradientboost = []

rechazo_BaseBagging = []
rechazo_BaseAdaBoost = []
rechazo_BaseGradientBoost = []
rechazo_BaggingAdaBoost = []
rechazo_BaggingGradientBoost = []
rechazo_AdaBoostGradientBoost = []

# Procesamiento del dataset iris
scores = cross_val_score(SVC(), iris.data, iris.target, cv=10, scoring='accuracy')
precision_SVC.append(scores.mean())
scores = cross_val_score(BaggingClassifier(SVC(), max_samples=0.5, max_features=0.5), iris.data, iris.target, cv=10, scoring='accuracy')
precision_SVC_bagging.append(scores.mean())
scores = cross_val_score(AdaBoostClassifier(), iris.data, iris.target, cv=10, scoring='accuracy')
precision_SVC_adaboost.append(scores.mean())
scores = cross_val_score(GradientBoostingClassifier(), iris.data, iris.target, cv=10, scoring='accuracy')
precision_SVC_gradientboost.append(scores.mean())

# Procesamiento del dataset wine
scores = cross_val_score(SVC(), wine.data, wine.target, cv=10, scoring='accuracy')
precision_SVC.append(scores.mean())
scores = cross_val_score(BaggingClassifier(SVC(), max_samples=0.5, max_features=0.5), wine.data, wine.target, cv=10, scoring='accuracy')
precision_SVC_bagging.append(scores.mean())
scores = cross_val_score(AdaBoostClassifier(), wine.data, wine.target, cv=10, scoring='accuracy')
precision_SVC_adaboost.append(scores.mean())
scores = cross_val_score(GradientBoostingClassifier(), wine.data, wine.target, cv=10, scoring='accuracy')
precision_SVC_gradientboost.append(scores.mean())

# Procesamiento del dataset breast cancer
scores = cross_val_score(SVC(), breast_cancer.data, breast_cancer.target, cv=10, scoring='accuracy')
precision_SVC.append(scores.mean())
scores = cross_val_score(BaggingClassifier(SVC(), max_samples=0.5, max_features=0.5), breast_cancer.data, breast_cancer.target, cv=10, scoring='accuracy')
precision_SVC_bagging.append(scores.mean())
scores = cross_val_score(AdaBoostClassifier(), breast_cancer.data, breast_cancer.target, cv=10, scoring='accuracy')
precision_SVC_adaboost.append(scores.mean())
scores = cross_val_score(GradientBoostingClassifier(), breast_cancer.data, breast_cancer.target, cv=10, scoring='accuracy')
precision_SVC_gradientboost.append(scores.mean())

# Procesamiento del dataset digits
scores = cross_val_score(SVC(), digits.data, digits.target, cv=10, scoring='accuracy')
precision_SVC.append(scores.mean())
scores = cross_val_score(BaggingClassifier(SVC(), max_samples=0.5, max_features=0.5), digits.data, digits.target, cv=10, scoring='accuracy')
precision_SVC_bagging.append(scores.mean())
scores = cross_val_score(AdaBoostClassifier(), digits.data, digits.target, cv=10, scoring='accuracy')
precision_SVC_adaboost.append(scores.mean())
scores = cross_val_score(GradientBoostingClassifier(), digits.data, digits.target, cv=10, scoring='accuracy')
precision_SVC_gradientboost.append(scores.mean())


# Comparacion de los resultados
print('Test de Iman-Davenport: SVC')
print('----------------------')
for i in range(0, len(datasets)):
    print('Dataset: ' + datasets[i])
    # Comparacion Iman-Davenport SVC Base con Bagging
    stat, p = mannwhitneyu(precision_SVC[i], precision_SVC_bagging[i])
    if (p > 0.05):
        rechazo_BaseBagging.append('No')
    else:
        rechazo_BaseBagging.append('Si')
    print('SVC Base vs SVC Bagging: p-value = ' + str(p) + ' - Rechazo = ' + rechazo_BaseBagging[i])
    # Comparacion Iman-Davenport SVC Base con AdaBoost
    stat, p = mannwhitneyu(precision_SVC[i], precision_SVC_adaboost[i])
    if (p > 0.05):
        rechazo_BaseAdaBoost.append('No')
    else:
        rechazo_BaseAdaBoost.append('Si')
    print('SVC Base vs SVC AdaBoost: p-value = ' + str(p) + ' - Rechazo = ' + rechazo_BaseAdaBoost[i])
    # Comparacion Iman-Davenport SVC Base con GradientBoost
    stat, p = mannwhitneyu(precision_SVC[i], precision_SVC_gradientboost[i])
    if (p > 0.05):
        rechazo_BaseGradientBoost.append('No')
    else:
        rechazo_BaseGradientBoost.append('Si')
    print('SVC Base vs SVC GradientBoost: p-value = ' + str(p) + ' - Rechazo = ' + rechazo_BaseGradientBoost[i])
    # Comparacion Iman-Davenport SVC Bagging con AdaBoost
    stat, p = mannwhitneyu(precision_SVC_bagging[i], precision_SVC_adaboost[i])
    if (p > 0.05):
        rechazo_BaggingAdaBoost.append('No')
    else:
        rechazo_BaggingAdaBoost.append('Si')
    print('SVC Bagging vs SVC AdaBoost: p-value = ' + str(p) + ' - Rechazo = ' + rechazo_BaggingAdaBoost[i])
    # Comparacion Iman-Davenport SVC Bagging con GradientBoost
    stat, p = mannwhitneyu(precision_SVC_bagging[i], precision_SVC_gradientboost[i])
    if (p > 0.05):
        rechazo_BaggingGradientBoost.append('No')
    else:
        rechazo_BaggingGradientBoost.append('Si')
    print('SVC Bagging vs SVC GradientBoost: p-value = ' + str(p) + ' - Rechazo = ' + rechazo_BaggingGradientBoost[i])
    # Comparacion Iman-Davenport SVC AdaBoost con GradientBoost
    stat, p = mannwhitneyu(precision_SVC_adaboost[i], precision_SVC_gradientboost[i])
    if (p > 0.05):
        rechazo_AdaBoostGradientBoost.append('No')
    else:
        rechazo_AdaBoostGradientBoost.append('Si')
    print('SVC AdaBoost vs SVC GradientBoost: p-value = ' + str(p) + ' - Rechazo = ' + rechazo_AdaBoostGradientBoost[i])

