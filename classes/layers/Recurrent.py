from classes.layers.Layer import Layer
import numpy as np

class Recurrent(Layer):
    _LAST_SEQUENCE_AMOUNT = 5

    def __init__(self, hidden_units, h_act_f, output_act_f, n_outputs = 1):
        """
        :param hidden_units: numero de unidades/estados/neuronas ocultas de la capa
        """
        self.w_x = None
        self.h_act_f = h_act_f  # tupla con la funcion de activacion y su derivada para el contexto
        self.output_act_f = output_act_f  # tupla con la funcion de activacion y su derivada para la salida
        self.head_prev = np.zeros((1,hidden_units)) #Contexto de la capa anterior (es decir, contexto en t-1)
        self.b_y = np.random.rand(1) * 2 - 1 
        self.b_h = np.random.rand(hidden_units) * 2 - 1
        self.W_h =  np.random.rand(hidden_units, hidden_units) * 2 - 1
        self.w_y = np.random.rand(hidden_units, n_outputs) * 2 - 1
        
    def cellForward(self, x, h_prev):
        h_next = np.dot(x, self.w_x) +  np.dot(h_prev, self.W_h) + self.b_h
        h_next_activation = self.h_act_f[0](h_next)
        output = np.dot(h_next_activation, self.w_y) + self.b_y
        output_activation = self.output_act_f[0](output)
        return {
                    'h_prev': h_prev,
                    'h_next': h_next,
                    'h_next_activation': h_next_activation,
                    'output': output,
                    'output_activation': output_activation,
                }

    def forward(self, X):
        if not isinstance(self.w_x, np.ndarray):
            self.w_x =  np.random.rand(X.shape[2], self.head_prev.shape[1]) * 2 - 1

        memory = []
        self.input = X
        out = np.zeros((X.shape[0], self.w_y.shape[1]))

        for i in range(X.shape[0]):
            temp_memory = []
            h_prev = self.head_prev
            for timestep in range(X[i].shape[0]):
                x = X[i][timestep]
                result = self.cellForward(x, h_prev)
                temp_memory.append(result) 
                h_prev = result['h_next_activation']
            out[i] = temp_memory[-1]['output_activation']
            memory.append(temp_memory)
        self.memory = memory
        return out

    def cellBackward(self, da_next, a_next, h_prev, x):
        # compute the gradient of tanh with respect to a_next (≈1 line)
        d_activation_h = self.h_act_f[1](a_next) * da_next

        # compute the gradient of the loss with respect to Wax (≈2 lines)
        dxt = np.dot(self.w_x.T, d_activation_h)
        dWx = np.dot(d_activation_h, x.T)

        # compute the gradient with respect to Waa (≈2 lines)
        da_prev = np.dot(self.W_h.T, d_activation_h)
        dWh = np.dot(d_activation_h, h_prev.T)

        # compute the gradient with respect to b (≈1 line)
        dbh = np.sum(d_activation_h, 1, keepdims=True)

        # Store the gradients in a python dictionary
        gradients = {"dxt": dxt, "da_prev": da_prev, "dWx": dWx, "dWh": dWh, "dbh": dbh}
        
        return gradients

    def backward(self, output_gradient, learning_rate):
        dW_h, dw_x, dw_y = np.zeros_like(self.W_h), np.zeros_like(self.w_x), np.zeros_like(self.w_y)
        db_h, db_y = np.zeros_like(self.b_h), np.zeros_like(self.b_y)

        count = 0
        n_sequences = len(self.memory)
        for i in range(n_sequences):
            n_instances = len(self.memory[i])
            for timestamp in range(n_instances-1, max(n_instances - 1 - self._LAST_SEQUENCE_AMOUNT, 0), -1):
                x = self.input[i][timestamp]
                output = self.memory[i][timestamp]['output']
                h_next_activation = self.memory[i][timestamp]['h_next_activation']
                h_next = self.memory[i][timestamp]['h_next']
                h_prev = self.memory[i][timestamp]['h_prev']
                #Acumulando los gradientes
                db_y_i = self.output_act_f[1](output)
                db_Wy_i = self.output_act_f[1](output) * h_next_activation


                da_next = self.output_act_f[1](output)
                gradients = self.cellBackward(da_next, h_next, h_prev, x)
                dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients["dWaa"], gradients["dba"]
                # Increment global derivatives w.r.t parameters by adding their derivative at time-step t (≈4 lines)
                dW_h = gradients["dWh"]
                dw_x = gradients["dWx"]
                db_h = gradients["dbh"]
                
                count += 1

        #dW_h, dw_x, dw_y, db_h, db_y = dW_h/count, dw_x/count, dw_y/count, db_h/count, db_y/count

        self.w_x -= (dw_x * learning_rate * output_gradient)
        self.W_h -= (dW_h * learning_rate * output_gradient)
        self.w_y -= (dw_y * learning_rate * output_gradient)
        self.b_h -= (db_h * learning_rate * output_gradient)
        self.b_y -= (db_y * learning_rate * output_gradient)

        return 1

    def __str__(self):
        return f"RecurrentLayer(W_h: {self.W_h.shape}, b_h: {self.b_h.shape}, w_x: {self.w_x.shape}, w_y: {self.w_y.shape})"