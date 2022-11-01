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

# Scatter plot matricial de los datasets
sns.pairplot(pd.DataFrame(iris.data, columns=iris.feature_names))
plt.savefig('iris.png')
plt.clf()
sns.pairplot(pd.DataFrame(wine.data, columns=wine.feature_names))
plt.savefig('wine.png')
plt.clf()
sns.pairplot(pd.DataFrame(diabetes.data, columns=diabetes.feature_names))
plt.savefig('diabetes.png')
plt.clf()
sns.pairplot(pd.DataFrame(cancer.data, columns=cancer.feature_names))
plt.savefig('cancer.png')
plt.clf()

# END