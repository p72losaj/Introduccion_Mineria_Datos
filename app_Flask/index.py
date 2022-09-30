# Importamos la biblioteca de Flask
from flask import Flask, render_template

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

# Ubicacion de la practica1
@app.route('/practica1')
def practica1():
    return render_template('practica1.html') # Renderizamos el archivo practica1.html

# Funcion principal
if __name__ == '__main__':
    app.run(debug=True, port=5017)