from classes.layers.Layer import Layer
import numpy as np

class Recurrent(Layer):
    _LAST_SEQUENCE_AMOUNT = 1

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
        #h_next_act = f(x * w_x + h_prev * W_h + b_h)
        h_next = np.dot(x, self.w_x) +  np.dot(h_prev, self.W_h) + self.b_h
        h_next_activation = self.h_act_f[0](h_next)
        #output = g(h_next * W_y + b_y)
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
                temp_memory.append(result['h_prev']) 
                h_prev = result['h_next_activation']
                if timestep == X[i].shape[0] -1 :
                    out[i] = result['output_activation']
            memory.append(temp_memory)
        self.memory = memory
        return out

    def cellBackward(self, x, h_prev):
        parameters = self.cellForward(x, h_prev)
        h_next = parameters['h_next']
        h_next_activation = parameters['h_next_activation']
        output = parameters['output']
        
        db_y_t = self.output_act_f[1](output)
        d_Wy_t = db_y_t * h_next_activation

        db_h_t = self.h_act_f[1](h_next) * db_y_t
        dw_x_t = np.dot(db_h_t, x.T)
        dW_h_t = np.dot(db_h_t, h_prev.T)

        """
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

        """
        # Store the gradients in a python dictionary
        gradients = {"dW_h_t": dW_h_t, "dw_x_t": dw_x_t, "db_h_t": db_h_t, "db_Wy_t": d_Wy_t, "db_y_t": db_y_t}
        
        return gradients

    def sequenceBackWard(self, sequence, sequence_memory):
        n_timestep = len(sequence_memory)
        for timestep in range(n_timestep-1, max(n_timestep - 1 - self._LAST_SEQUENCE_AMOUNT, 0), -1):
                x = sequence[timestep].reshape(1,-1)
                h_prev = sequence_memory[timestep]
                forward_result = self.cellForward(x, h_prev)
                #d_Y/d_b_Y = d_g/d_output * d_output/d_b_Y, y d_output/d_b_Y = 1, entonces:
                #d_Y/d_b_Y = d_g(output) 
                forward_output = forward_result['output'] 
                dY_dBy_t = self.output_act_f[1](forward_output) 
                #d_Y/d_W_Y = d_g/d_output * d_output/d_W_Y, entonces: 
                #d_Y/d_W_Y = d_g/d_output * h_next_act = d_g(output) * h_next_act
                forward_h_next_activation = forward_result['h_next_activation']
                dY_dWy_t = forward_h_next_activation.T.dot(dY_dBy_t)
                #d_Y/d_b_h = d_g/d_output * d_output/d_h_next_act * d_h_next_act/d_h_next * d_h_next/d_b_h,
                # entonces, ya que d_h_next/d_b_h = 1
                #d_Y/d_b_h = d_g/d_output * d_output/d_h_next_act * d_h_next_act/d_h_next
                #d_Y/d_b_h = d_g/d_output * w_y * d_f(h_next)
                forward_h_next = forward_result['h_next']
                dF_dhNext_t = self.h_act_f[1](forward_h_next) #d_h_next_act/d_h_next = d_f(h_next)
                dY_dHnextact_t = dY_dBy_t.dot(self.w_y.T)
                dY_dbh_t = np.multiply(dY_dHnextact_t, dF_dhNext_t)  
                #d_Y/d_Wx = d_g/d_output * d_output/d_h_next_act * d_h_next_act/d_h_next * d_h_next/d_W_x,
                # entonces, ya que d_h_next/d_W_x = x
                #d_Y/d_Wx = d_g/d_output * d_output/d_h_next_act * d_h_next_act/d_h_next * x
                #d_Y/d_Wx = d_g/d_output * w_y * d_f(h_next) * x
                dY_dWx_t = x.T.dot(dY_dbh_t)
                #d_Y/d_Wh = d_g/d_output * d_output/d_h_next_act * d_h_next_act/d_h_next * d_h_next/d_W_h,
                #d_Y/d_Wx = d_g/d_output * w_y * d_f(h_next) * d_h_next/d_W_h
                #d_Y/d_Wx = d_g/d_output * w_y * d_f(h_next) * h_prev,
                #pero a su vez h_prev depende de Wh_prev, es decir Wh_{t-1},  y asi sucesivamente.
                #Como h_prev = f(x * Wx + Wh * h_prev_prevact + bh), entonces:
                # dh_prev/dWh = d_f(x * Wx + Wh * h_prev_prevact + bh) * h_prev_prevact, y asi sucesivamente
                dY_dWh = np.multiply(dY_dbh_t, h_prev)
                dY_dWh_acumulador = np.zeros_like(dY_dWh)
                for it in range(timestep-1, -1, -1):
                    previous_h_next = sequence_memory[it]
                    previous_result = self.cellForward(sequence[it].reshape(1,-1), previous_h_next)
                    _temp = np.multiply(self.h_act_f[1](previous_result['h_next']), previous_h_next)
                    dY_dWh_acumulador += _temp
                dY_dWh = np.multiply(dY_dWh, dY_dWh_acumulador)
                break

        return {
            "dY_dBy_t" : dY_dBy_t,
            "dY_dWy_t" : dY_dWy_t,
            "dY_dbh_t" : dY_dbh_t,
            "dY_dWx_t" : dY_dWx_t,
            "dY_dWh" : dY_dWh,
        }



    def backward(self, output_gradient, learning_rate):
        n_sequences = len(self.memory)
        w_x = np.zeros_like(self.w_x)
        W_h = np.zeros_like(self.W_h)
        w_y = np.zeros_like(self.w_y)
        b_h = np.zeros_like(self.b_h)
        b_y = np.zeros_like(self.b_y)
        for i in range(n_sequences):
            gradients = self.sequenceBackWard(self.input[i], self.memory[i])
            w_x += gradients['dY_dWx_t'] * output_gradient[i]
            W_h += gradients['dY_dWh'] * output_gradient[i]
            w_y += gradients['dY_dWy_t'] * output_gradient[i]
            b_h += gradients['dY_dbh_t'].reshape(self.b_h.shape) * output_gradient[i]
            b_y += gradients['dY_dBy_t'].reshape(self.b_y.shape) * output_gradient[i]
        
        w_x/=n_sequences
        W_h/=n_sequences
        w_y/=n_sequences
        b_h/=n_sequences
        b_y/=n_sequences

        self.w_x -= w_x * learning_rate
        self.W_h -= W_h * learning_rate
        self.w_y -= w_y * learning_rate
        self.b_h -= b_h * learning_rate
        self.b_y -= b_y * learning_rate 

        return 1

    def __str__(self):
        return f"RecurrentLayer(W_h: {self.W_h.shape}, b_h: {self.b_h.shape}, w_x: {self.w_x.shape}, w_y: {self.w_y.shape})"