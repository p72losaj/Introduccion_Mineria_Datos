# Realizado por Jaime Lorenzo Sanchez
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

# Datasets a emplear
iris = datasets.load_iris()
wine = datasets.load_wine()
diabetes = datasets.load_diabetes()
cancer = datasets.load_breast_cancer()

# Boxplot de los datasets
plt.figure(1)
plt.boxplot(iris.data)
plt.title('Boxplot Iris')
# Eje X
plt.xticks([1, 2, 3, 4], ['Sepal length', 'Sepal width', 'Petal length', 'Petal width'])
# Eje Y
plt.ylabel('cm')
plt.savefig('boxplot_iris.png')
# Limpiamos la figura
plt.clf()

# Generamos un boxplot del dataset wine
plt.boxplot(wine.data)
plt.title('Boxplot Wine')
# Eje X
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], wine.feature_names)
# Eje Y
plt.ylabel('cm')
plt.savefig('boxplot_wine.png')
plt.clf()

# Generamos un boxplot del dataset diabetes
plt.boxplot(diabetes.data)
plt.title('Boxplot Diabetes')
# Eje X
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9], diabetes.feature_names)
# Eje Y
plt.ylabel('cm')
plt.savefig('boxplot_diabetes.png')
plt.clf()

# Generamos un boxplot del dataset cancer
plt.boxplot(cancer.data)
plt.title('Boxplot Cancer')
# Eje X
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], cancer.feature_names)
# Eje Y
plt.ylabel('cm')
plt.savefig('boxplot_cancer.png')
plt.clf()

# Boxplot de las clases del dataset iris
plt.figure(1)
plt.boxplot(iris.data[iris.target == 0])
plt.title('Boxplot Iris Setosa')
# Eje X
plt.xticks([1, 2, 3, 4], ['Sepal length', 'Sepal width', 'Petal length', 'Petal width'])
# Eje Y
plt.ylabel('cm')
plt.savefig('boxplot_iris_setosa.png')

plt.figure(2)
plt.boxplot(iris.data[iris.target == 1])
plt.title('Boxplot Iris Versicolor')
# Eje X
plt.xticks([1, 2, 3, 4], ['Sepal length', 'Sepal width', 'Petal length', 'Petal width'])
# Eje Y
plt.ylabel('cm')
plt.savefig('boxplot_iris_versicolor.png')

plt.figure(3)
plt.boxplot(iris.data[iris.target == 2])
plt.title('Boxplot Iris Virginica')
# Eje X
plt.xticks([1, 2, 3, 4], ['Sepal length', 'Sepal width', 'Petal length', 'Petal width'])
# Eje Y
plt.ylabel('cm')
plt.savefig('boxplot_iris_virginica.png')