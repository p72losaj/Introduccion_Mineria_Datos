# Importamos la biblioteca de Flask
from flask import Flask, render_template
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import io
import base64


# Inicio de la aplicacion
app = Flask(__name__)

# Ubicacion de la ruta de la pagina web
@app.route('/')
def principal():
    return render_template('index.html') # Renderizamos el archivo index.html

# Ruta de los lenguajes de programacion
@app.route('/lenguajes')
def lenguajes():
    mislenguajes = ("PHP", "Python")
    return render_template('lenguajes.html', lenguajes=mislenguajes)

# PRACTICA 1
@app.route('/practica1')
def practica1():
    return render_template('practica1.html') # Renderizamos el archivo practica1.html

@app.route('/practica1/ejercicio2')
# Ubicacion del ejercicio1 de la practica 1
def practica1_ejercicio2():
    arbol_size= [];arbol_accuracy = []; arbol_error = []
    img = io.BytesIO()
    # Leemos el dataset
    columnas = ("sepal", "length", "sepal", "width", "petal", "length", "petal", "width", "class")
    pd=datasets.load_iris();x=pd.data; y=pd.target;arbol_size.append(x.size)
    # Entrenamos el dataset usando Arbol de decision
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    modelo = DecisionTreeClassifier();modelo.fit(x_train, y_train);prediccion = modelo.predict(x_test);test = modelo.score(x_test, y_test)
    # Calculamos las metricas del modelo
    accuracy = metrics.accuracy_score(y_test, prediccion);arbol_accuracy.append(accuracy);error = 1 - accuracy;arbol_error.append(error)
    # Graficamos el modelo
    plt.plot(arbol_size,arbol_accuracy,label = 'randomForest_accuracy'); plt.plot(arbol_size,arbol_error,label='randomForest_error'); plt.xlabel('Number of elements of dataset'); plt.ylabel('Rate of the result'); plt.title('Relation Dataset and classifier result'); plt.legend(); plt.savefig("Practica1_Ejercicio2.png"); img.seek(0); plt.close() 

    return render_template('practica1_ejercicio2.html')
    

# Funcion principal
if __name__ == '__main__':
    app.run(debug=True, port=5017)