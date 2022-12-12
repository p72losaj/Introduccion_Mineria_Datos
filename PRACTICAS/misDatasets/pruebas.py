# Cargamos 10 dataset
# Dataset iris
import pandas as pd
from sklearn.datasets import load_breast_cancer 

iris = pd.read_csv('iris.data', header=None,names=["sepal length","sepal width","petal length","petal width","class"])
iris_features= ["sepal length","sepal width","petal length","petal width"]
iris_X=iris[iris_features]
iris_y=iris["class"]
# Dataset wine
wine = pd.read_csv('wine.data', header = None,
                   names = ['class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
                            'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315',
                            'Proline']
                   )
wine_features = ['Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols', 
                'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 
                'OD280/OD315', 'Proline']
wine_X=wine[wine_features]
wine_y=wine['class']
# Dataset diabetes
diabetes = pd.read_csv('diabetes.csv', header = None,
                       names = ['preg', 'plas','pres','skin','insu','mass','pedi','age','class'])
diabetes_features= ['preg', 'plas','pres','skin','insu','mass','pedi','age']
diabetes_X = diabetes[diabetes_features]
diabetes_y = diabetes['class']
# Dataset Breast Cancer 
breast_cancer = load_breast_cancer()
breast_cancer_X = breast_cancer.data
breast_cancer_Y = breast_cancer.target
# Dataset cpu
cpu = pd.read_csv('cpu.csv', header = None,
                    names = ['MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX','class'])
cpu_features = ['MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX']
cpu_X = cpu[cpu_features]
cpu_Y = cpu['class']

# Dataset abalone.data


abalone = pd.read_csv('abalone.data', header = None,
                names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings'])

abalone_features = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']
abalone_X = abalone[abalone_features]
abalone_Y = abalone['Rings']
# Transformamos los datos de abalone en numéricos
abalone_X['Sex'] = abalone['Sex'].map({'M': 0, 'F': 1, 'I': 2})


