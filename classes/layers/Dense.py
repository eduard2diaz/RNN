from classes.layers.Layer import Layer
import numpy as np

class Dense(Layer):

    def __init__(self, n_neur, act_f):
        """
        Constructor que declara la matriz de pesos y el vector de bias
        :param n_neur: numero de neuronas en la capa
        :param act_f: tupla con la funcion de activacion y derivada de esta ultima
        """
        self.n_neur = n_neur
        self.act_f = act_f  # tupla con la funcion de activacion y su derivada
        # Definiendo un vector columna de bias, con tantos elementos como neuronas
        self.b = np.random.rand(1, self.n_neur) * 2 - 1
        self.W, self.z, self.a = None, None, None

    def forward(self, X):
        """
        Funcionalidad que retorna la suma ponderada y la evaluacion de la funcion de activacion en esta ultima
        :param X: datos de entrada
        :return: tupla compuesta por la suma ponderada y la funcion de activacion evaluada en esta ultima
        """
        if not isinstance(self.W, np.ndarray):
            # Al igual que con el bias inicializamos los pesos asociados a cada neurona
            self.W = np.random.rand(X.shape[1], self.n_neur) * 2 - 1

        z = X @ self.W + self.b  # Calculando la suma ponderada
        a = self.act_f[0](z)  # Aplicando de la funcion de activacion sobre la suma ponderada
        # Guardamos la suma ponderada y la salida de la funcion de activacion para facilitar la homogeneidad del
        # metodo backpropagation
        self.input, self.z, self.a = X, z, a
        return (z, a)

    def backward(self, delta, learning_rate):
        """
        Funcion que evalua la derivada de la funcion de activacion en la suma ponderada, y multiplica este resultado
        por el gradiente recibido de la siguiente capa.
        :param delta: gradiente de la siguiente capa
        :param learning_rate: factor de aprendizaje
        :return: gradiente de salida para la anterior capa
        """
        dA = self.act_f[1](self.z)  # Derivada de la funcion de activacion
        d_curr_layer = delta * dA  # Propagacion del error para actualizar el delta de la actual capa
        # Guardamos el delta que recibira la anterior capa
        delta_before_layer = d_curr_layer @ self.W.T
        # Normalmente backpropagation y gradient descent son dos procesos distintos separados pero podemos unirlos
        # cuando tenemos diferentes tipos de capas

        # ---Inicio del descenso del gradiente
        self.b = self.b - np.mean(d_curr_layer, axis=0, keepdims=True) * learning_rate
        self.W = self.W - self.input.T @ d_curr_layer * learning_rate
        # ---Fin del descenso del grtadiente

        # Continuamos con backpropagation: Actualizamos delta para pasarselo a la capa anterior
        return delta_before_layer

    def __str__(self):
        shape = None
        if isinstance(self.W, np.ndarray):
            shape = self.W.shape
        return f"DenseLayer(bias: {self.b.shape}, W: {shape})"