# Realizado por Jaime Lorenzo Sanchez
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns

# Datasets a emplear
iris = datasets.load_iris()
wine = datasets.load_wine()
diabetes = datasets.load_diabetes()
cancer = datasets.load_breast_cancer()

# Matriz de correlacion entre las variables de los datasets usando un mapa de calor
sns.heatmap(pd.DataFrame(iris.data, columns=iris.feature_names).corr(), annot=True)
plt.title("Matriz de correlacion Iris")
plt.savefig('iris_corr.png')
plt.clf()

sns.heatmap(pd.DataFrame(wine.data, columns=wine.feature_names).corr(), annot=True)
plt.title("Matriz de correlacion Wine")
plt.savefig('wine_corr.png')
plt.clf()

sns.heatmap(pd.DataFrame(diabetes.data, columns=diabetes.feature_names).corr(), annot=True)
plt.title("Matriz de correlacion Diabetes")
plt.savefig('diabetes_corr.png')
plt.clf()

sns.heatmap(pd.DataFrame(cancer.data, columns=cancer.feature_names).corr(), annot=True)
plt.title("Matriz de correlacion Cancer")
plt.savefig('cancer_corr.png')
plt.clf()
