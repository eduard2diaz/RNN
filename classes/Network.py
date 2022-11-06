class Network:
    def __init__(self):
        self.layers = []

    def appendLayer(self, layer):
        self.layers.append(layer)

    def getModel(self):
        return self.layers

    def forwardPass(self, X):
        out = [(None, X)]

        neural_net = self.getModel()
        for l, layer in enumerate(neural_net):
            output = neural_net[l].forward(out[-1][1])
            if not isinstance(output, tuple):
                output = (None, output)
            out.append(output)
        return out

    def predict(self, X):
        # retorno el resultado de arrojado por la funcion de activacion en la ultima capa
        return self.forwardPass(X)[-1][1]

    # Definiendo la funcion de entrenamiento
    def _train(self, X, y, l2_cost, lr=.5):
        # Forward pass
        out = self.forwardPass(X)

        # Backward pass
        def backwardPass(neural_net, out):
            delta = l2_cost[1](out[-1][1], y)  # Derivada del error respecto a la funcion de costo
            #import numpy as np
            #output_gradient = np.mean(delta, axis=0)
            for l in reversed(range(len(neural_net))):
                delta = neural_net[l].backward(delta, lr)

        backwardPass(self.getModel(), out)

        return out[-1][1]

    def train(self, X, y, cost_function, lr=.5, epochs=1000, error_range=.0000002):
        i = 0
        cost = 1
        while i < epochs and cost > error_range:
            self._train(X, y, cost_function, lr)
            y_predicted = self.predict(X)
            cost = cost_function[0](y_predicted, y)
            print(f"Epoch {i}, cost {cost}")
            i += 1

    def __str__(self):
        print('-----------------Model Structure-----------------')
        for l in self.layers:
            print('\t', l.__str__())
        return '-----------------End Model Structure-------------'
