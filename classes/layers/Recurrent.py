from classes.layers.Layer import Layer
import numpy as np

class Recurrent(Layer):
    _LAST_SEQUENCE_AMOUNT = 5
    _MAX_CLIP_VAL = 7
    _MIN_CLIP_VAL = -7

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

    def nextContext(self, x, h_prev):
        #h_next = x * w_x + h_prev * W_h + b_h
        return np.dot(x, self.w_x) +  np.dot(h_prev, self.W_h) + self.b_h

    def cellForward(self, x, h_prev):
        h_next = self.nextContext(x, h_prev)
        #h_next_act = f(x * w_x + h_prev * W_h + b_h) = f(h_next)
        h_next_activation = self.h_act_f[0](h_next)
        #output = h_next_act * W_y + b_y
        output = np.dot(h_next_activation, self.w_y) + self.b_y 
        #output_activation = g(h_next_act * W_y + b_y) = g(output)
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
                temp_memory.append(result['h_prev']) 
                h_prev = result['h_next_activation']
                if timestep == X[i].shape[0] -1 :
                    out[i] = result['output_activation']
            memory.append(temp_memory)
        self.memory = memory
        return out

    def cellPartialBackward(self, x, h_prev, da_next):
        dF_dhNext = self.h_act_f[1](self.nextContext(x, h_prev)) #d_h_next_act/d_h_next = d_f(h_next)
        dF_dbh = np.multiply(da_next, dF_dhNext)  
        #d_Y/d_Wx = d_g/d_output * d_output/d_h_next_act * d_h_next_act/d_h_next * d_h_next/d_W_x,
        # entonces, ya que d_h_next/d_W_x = x
        #d_Y/d_Wx = d_g/d_output * d_output/d_h_next_act * d_h_next_act/d_h_next * x
        #d_Y/d_Wx = d_g/d_output * w_y * d_f(h_next) * x
        dF_dWx = x.T.dot(dF_dbh)
        #d_Y/d_Wh = d_g/d_output * d_output/d_h_next_act * d_h_next_act/d_h_next * d_h_next/d_W_h,
        #d_Y/d_Wx = d_g/d_output * w_y * d_f(h_next) * d_h_next/d_W_h
        #d_Y/d_Wx = d_g/d_output * w_y * d_f(h_next) * h_prev,
        dF_dWh = np.multiply(dF_dbh, h_prev)
        #pero a su vez h_prev depende de Wh_prev, es decir Wh_{t-1},  y asi sucesivamente.
        #Como h_prev = f(x * Wx + Wh * h_prev_prevact + bh), entonces:
        # dh_prev/dWh = d_f(x_prev * Wx + Wh * h_prev_prevact + bh) * h_prev_prevact, y asi sucesivamente

        #Con este fin haremos uso de dF_dh el cual almacena la retropropagacion del error con respecto a h_prev
        #desde la evaluacion de la funcion de costo hasta la capa actual y sera lo que le pasaremos a la capa
        # anterior para que calcule su correspondiente dF_dWh
        dF_dh = np.dot(dF_dbh, self.W_h)

        return {"dbh" : dF_dbh, "dWx" : dF_dWx, "dWh" : dF_dWh, "dh" : dF_dh}

    def lastCellBackward(self, x, h_prev, output_gradient):
        parameters = self.cellForward(x, h_prev)
        #d_Y/d_b_Y = d_g/d_output * d_output/d_b_Y, y d_output/d_b_Y = 1, entonces:
        #d_Y/d_b_Y = d_g(output)
        dY_dBy = self.output_act_f[1](parameters['output']) * output_gradient
        #d_Y/d_W_Y = d_g/d_output * d_output/d_W_Y, entonces: 
        #d_Y/d_W_Y = d_g/d_output * h_next_act = d_g(output) * h_next_act
        dY_dWy = parameters['h_next_activation'].T.dot(dY_dBy)
        #Calculando los otros gradientes
        dY_dHnextact = dY_dBy.dot(self.w_y.T)
        partial_gradients = self.cellPartialBackward(x, h_prev, dY_dHnextact)

        return {"dY_dBy": dY_dBy, "dY_dWy": dY_dWy, "dY_dbh": partial_gradients['dbh'],
         "dY_dWx": partial_gradients['dWx'], "dY_dWh": partial_gradients['dWh'],
         "dY_dh": partial_gradients['dh']}
        
    def sequenceBackward(self, sequence, sequence_memory, output_gradient):
        n_timestep = len(sequence_memory)
        last_cell_gradient = self.lastCellBackward(sequence[n_timestep-1].reshape(1,-1),
        sequence_memory[n_timestep-1], output_gradient)

        dWh_acum = last_cell_gradient["dY_dWh"]
        da_next = last_cell_gradient["dY_dh"]

        for timestep in range(n_timestep-2, max(n_timestep - 2 - self._LAST_SEQUENCE_AMOUNT, -1), -1):
                #Obteniendo la derivada de la salida final respecto a los pesos de los contextos para las
                #_LAST_SEQUENCE_AMOUNT ultimas capas predecesoras a la ultima capa de la RNN.
                partial_grads= self.cellPartialBackward(sequence[timestep].reshape(1,-1), 
                sequence_memory[timestep], da_next)
                dWh_acum+= partial_grads["dWh"]
                da_next = partial_grads['dh']

        last_cell_gradient["dY_dWh"] = dWh_acum

        #We return da_next in case of exist other kind of layer, no RNN,
        #and this way we can update its parameters
        return last_cell_gradient, da_next

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
        dw_y = np.zeros_like(self.w_y)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        dh_prev = np.zeros_like(self.head_prev)

        for i in range(n_sequences):
            #Backpropagation through sequences
            gradients, da_next = self.sequenceBackward(self.input[i], self.memory[i], output_gradient[i])
            #Acumalating gradients of the sequences
            dw_x += gradients['dY_dWx']
            dW_h += gradients['dY_dWh']
            dw_y += gradients['dY_dWy']
            db_h += gradients['dY_dbh'].reshape(self.b_h.shape)
            db_y += gradients['dY_dBy'].reshape(self.b_y.shape)
            dh_prev += da_next
        
        #Averaging gotten gradients through sequences
        dw_x/=n_sequences
        dW_h/=n_sequences
        dw_y/=n_sequences
        db_h/=n_sequences
        db_y/=n_sequences
        dh_prev/= n_sequences

        #Applying gradient clipping
        dw_x = self.gradientClipping(dw_x)
        dW_h = self.gradientClipping(dW_h)
        dw_y = self.gradientClipping(dw_y)
        db_h = self.gradientClipping(db_h)
        db_y = self.gradientClipping(db_y)

        #Gradient descent
        self.w_x -= dw_x * learning_rate
        self.W_h -= dW_h * learning_rate
        self.w_y -= dw_y * learning_rate
        self.b_h -= db_h * learning_rate
        self.b_y -= db_y * learning_rate 

        return dh_prev

    def __str__(self):
        return f"RecurrentLayer(W_h: {self.W_h.shape},\
             b_h: {self.b_h.shape}, w_x: {self.w_x.shape}, w_y: {self.w_y.shape})"