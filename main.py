from tkinter import E
import numpy as np
import matplotlib.pyplot as plt
import math

sin_wave = np.array([math.sin(x) for x in range(20)])#200

seq_len = 5 #50
X = []
Y = []
num_records = len(sin_wave) - seq_len # 150
# X entries are 50 data points
# Y entries are the 51st data point
for i in range(num_records-seq_len):
    X.append(sin_wave[i:i+seq_len])
    Y.append(sin_wave[i+seq_len])

X = np.expand_dims(np.array(X), axis=2) # 100 x 50 x 1
Y = np.expand_dims(np.array(Y), axis=1) # 100 x 1

#print(f"X.shape {X.shape} Y.shape {Y.shape}")

from classes.layers.Recurrent import Recurrent
from classes.Network import Network
from classes.functions import *

net = Network()
net.appendLayer(Recurrent(2, linear, linear))
net.train(X, Y, l2_cost, lr=.7, epochs= 1000)
