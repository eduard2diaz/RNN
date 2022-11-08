# RNN
Este es un ejercicio de autoaprendizaje para comprender el funcionamiento de la retropropagacion del error en una
RNN. No obstante, como sabemos las RNN son susceptibles a exploding y vanishing gradient. En este sentido, usamos gradient clipping para mitingar el exploding gradient. 

Gradient clipping consiste en reescalar el gradiente si este ultimo se hace muy grande. Especificamente
si ‖g‖ ≥ c, entonces g ↤ c · g/‖g‖, en caso contrario no hacemos nada. Donde c es un hiperparametro, g es el gradiente y ‖g‖ es la norma de g. Asimismo, como resultado de g/‖g‖ obtenemos un vector unitario.

Al igual que en NLP al predecir la salida en un tiempo t, no es necesario considerar todas las entradas y salidas anteriores, basta con considerar x entradas y salidas anterios, donde x > 0, y analizaremos las entradas y salidas de la red desde t-x hasta t-1, con este sentido fue definida la variable _LAST_SEQUENCE_AMOUNT en classes/layers/Recurrent.py


Cualquier contribucion, critica o recomendacion sera bien recibida.

Bibliografia:
https://pythonalgos.com/build-a-recurrent-neural-network-from-scratch-in-python-3/
https://datascience-enthusiast.com/DL/Building_a_Recurrent_Neural_Network-Step_by_Step_v1.html
https://towardsdatascience.com/what-is-gradient-clipping-b8e815cdfb48