from classes.layers.Layer import Layer
import numpy as np

class Recurrent(Layer):
    _LAST_SEQUENCE_AMOUNT = 5
    _MAX_CLIP_VAL = 7
    _MIN_CLIP_VAL = -7

    def __init__(self, hidden_units, h_act_f):
        """
        :param hidden_units: numero de unidades/estados/neuronas ocultas de la capa
        """
        self.w_x = None
        self.h_act_f = h_act_f  # tupla con la funcion de activacion y su derivada para el contexto
        self.head_prev = np.zeros((1,hidden_units)) #Contexto de la capa anterior (es decir, contexto en t-1)
        self.b_h = np.random.rand(hidden_units) * 2 - 1
        self.W_h =  np.random.rand(hidden_units, hidden_units) * 2 - 1

    def nextContext(self, x, h_prev):
        #h_next = x * w_x + h_prev * W_h + b_h
        return np.dot(x, self.w_x) +  np.dot(h_prev, self.W_h) + self.b_h

    def cellForward(self, x, h_prev):
        h_next = self.nextContext(x, h_prev)
        #h_next_act = f(x * w_x + h_prev * W_h + b_h) = f(h_next)
        h_next_activation = self.h_act_f[0](h_next)

        return {
                    'h_prev': h_prev,
                    'h_next': h_next,
                    'h_next_activation': h_next_activation,
                }

    def forward(self, X):
        if not isinstance(self.w_x, np.ndarray):
            self.w_x =  np.random.rand(X.shape[2], self.head_prev.shape[1]) * 2 - 1

        memory = []
        self.input = X
        out = np.zeros((X.shape[0], self.W_h.shape[1]))
        for i in range(X.shape[0]):
            temp_memory = []
            h_prev = self.head_prev
            for timestep in range(X[i].shape[0]):
                x = X[i][timestep]
                result = self.cellForward(x, h_prev)
                temp_memory.append(result['h_prev']) 
                h_prev = result['h_next_activation']
                if timestep == X[i].shape[0] - 1 :
                    out[i,:] = result['h_next_activation']               
            memory.append(temp_memory)
        self.memory = memory
        return out

    def cellBackward(self, x, h_prev, da_next):
        dF_dhNext = self.h_act_f[1](self.nextContext(x, h_prev)) #d_h_next_act/d_h_next = d_f(h_next)
        dF_dbh = np.multiply(da_next, dF_dhNext)  
        #d_h_next_act/d_Wx = d_h_next_act/d_h_next * d_h_next/d_W_x,
        # entonces, ya que d_h_next/d_W_x = x
        #d_h_next_act/d_Wx = d_h_next_act/d_h_next * x
        #d_h_next_act/d_Wx = d_f(h_next) * x
        dF_dWx = x.T.dot(dF_dbh)
        #d_h_next_act/d_Wh = d_h_next_act/d_h_next * d_h_next/d_W_h,
        #d_h_next_act/d_Wx = d_f(h_next) * d_h_next/d_W_h
        #d_h_next_act/d_Wx = d_f(h_next) * h_prev,
        dF_dWh = np.multiply(dF_dbh, h_prev)
        #pero a su vez h_prev depende de Wh_prev, es decir Wh_{t-1},  y asi sucesivamente.
        #Como h_prev = f(x * Wx + Wh * h_prev_prevact + bh), entonces:
        # dh_prev/dWh = d_f(x_prev * Wx + Wh * h_prev_prevact + bh) * h_prev_prevact, y asi sucesivamente

        #Con este fin haremos uso de dF_dh el cual almacena la retropropagacion del error con respecto a h_prev
        #desde la evaluacion de la funcion de costo hasta la capa actual y sera lo que le pasaremos a la capa
        # anterior para que calcule su correspondiente dF_dWh
        dF_dh = np.dot(dF_dbh, self.W_h)

        return {"dbh" : dF_dbh, "dWx" : dF_dWx, "dWh" : dF_dWh, "dh" : dF_dh}

    def sequenceBackward(self, sequence, sequence_memory, output_gradient):
        n_timestep = len(sequence_memory)

        dW_x = np.zeros_like(self.w_x)
        dW_h = np.zeros_like(self.W_h)
        db_h= np.zeros_like(self.b_h)

        for timestep in range(n_timestep-1, max(n_timestep - 1 - self._LAST_SEQUENCE_AMOUNT, -1), -1):
            partial_grads = self.cellBackward(sequence[timestep].reshape(1,-1),
            sequence_memory[timestep], output_gradient)

            #Applying gradient clipping
            partial_grads["dWx"] = self.gradientClipping(partial_grads["dWx"])
            partial_grads["dWh"] = self.gradientClipping(partial_grads["dWh"])
            partial_grads["dbh"] = self.gradientClipping(partial_grads["dbh"])
            partial_grads['dh'] = self.gradientClipping(partial_grads['dh'])

            #Adding respective gradients
            dW_x+= partial_grads["dWx"]
            dW_h+= partial_grads["dWh"]
            db_h+= partial_grads["dbh"].reshape(db_h.shape)
            output_gradient = partial_grads['dh']

        dW_x/=n_timestep
        dW_h/=n_timestep
        db_h/=n_timestep

        return {"dbh" : db_h, "dWx" : dW_x, "dWh" : dW_h}, output_gradient

    def gradientClipping(self, value):
        if value.max() > self._MAX_CLIP_VAL:
            value[value > self._MAX_CLIP_VAL] = self._MAX_CLIP_VAL
        
        if value.min() < self._MIN_CLIP_VAL:
            value[value < self._MIN_CLIP_VAL] = self._MIN_CLIP_VAL

        return value 

    def backward(self, output_gradient, learning_rate):
        n_sequences = len(self.memory)
        dw_x = np.zeros_like(self.w_x)
        dW_h = np.zeros_like(self.W_h)
        db_h = np.zeros_like(self.b_h)
        dh_prev = np.zeros_like(self.head_prev)

        for i in range(n_sequences):
            #Backpropagation through sequences
            gradients, da_next = self.sequenceBackward(self.input[i], self.memory[i], output_gradient[i])
            #Acumalating gradients of the sequences
            dw_x += gradients['dWx']
            dW_h += gradients['dWh']
            db_h += gradients['dbh'].reshape(self.b_h.shape)
            dh_prev += da_next
        
        #Averaging gotten gradients through sequences
        dw_x/=n_sequences
        dW_h/=n_sequences
        db_h/=n_sequences
        dh_prev/= n_sequences

        #Gradient descent
        self.w_x -= dw_x * learning_rate
        self.W_h -= dW_h * learning_rate
        self.b_h -= db_h * learning_rate

        return dh_prev

    def __str__(self):
        return f"RecurrentLayer(W_h: {self.W_h.shape}, b_h: {self.b_h.shape}, w_x: {self.w_x.shape})"