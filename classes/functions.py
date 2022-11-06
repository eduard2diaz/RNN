import numpy as np

#Definiendo la funcion de activacion linear y su derivada
linear = (lambda x: x,
          lambda x: np.ones_like(x))

#Definiendo la funcion de activacion tangente hiperbolica y su derivada
def tanhTemplate(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x)+np.exp(-x))

tanh = (lambda x: tanhTemplate(x),
        lambda x: 1 - tanhTemplate(x)**2)

def relu_template(x):
    out = np.copy(x)
    out[out < 0] = 0
    return out

relu = (lambda x: relu_template(x),
        lambda x: (x > 0) * 1) #Derivada de la funcion relu

def softmax_template(vector):
    normalize_vector = (vector - np.min(vector))/(np.max(vector) - np.min(vector))
    e = np.e ** normalize_vector
    return e/e.sum()

def softmax_tensor(X):
    out = np.zeros(X.shape)
    if X.ndim == 2:
        for i in range(X.shape[0]):
            temp = softmax_template(X[i,:])
            out[i,:] = np.array(temp)
    else:
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                temp = softmax_template(X[i,:,j])
                out[i,:,j] = np.array(temp)
    return out

def softmax_derivative_template(softmax):
    s = softmax.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

def softmax_derivative_tensor(softmax):
    salida = []
    
    for obj in softmax:
        dS = softmax_derivative_template(obj)
        salida.append(dS)
        
    salida = np.array(salida) 
    return salida

softmax = (lambda x: softmax_tensor(x), 
           lambda softmax_value: softmax_derivative_tensor(softmax_value))

def crossEntropy(y_true, y_pred):
    eps = np.finfo(float).eps
    return -np.sum(y_true * np.log(y_pred + eps))

def crossEntropyGrad(y_true, y_pred): #Derivada de cross entropy
    eps = np.finfo(float).eps
    return -y_true/(y_pred + eps)

def crossEntropySoftMaxGrad(y_true, z_pred): #Derivada de cross entropy evaluado en la derivada de softmax
    return z_pred - y_true

cross_entropy = (lambda y_true, y_pred: crossEntropy(y_true, y_pred),
                 lambda y_true, y_pred: crossEntropyGrad(y_true, y_pred),
                 lambda y_true, z_pred: crossEntropySoftMaxGrad(y_true, z_pred))

sigm_template = lambda x: 1/ (1 + np.e ** (-x)) #Funcion sigmoide

sigm = (lambda x: sigm_template(x), #Funcion sigmoide
        lambda x: sigm_template(x) * (1 - sigm_template(x))) #Derivada de la funcion sigmoide

#Definiendo la funcion de costo o error y su derivada
l2_cost = (lambda Yp, Yr: np.mean((Yr - Yp) ** 2), #Funcion de coste error cuadratico medio
           lambda Yp, Yr: (Yp - Yr) ) #Derivada de la funcion de coste

def accuracy(y_true, y_pred):
    cont = 0
    for pred, true in zip(y_true, y_pred):
        if np.argmax(pred) == np.argmax(true):
            cont+=1
    return cont/len(y_true)