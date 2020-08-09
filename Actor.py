#!/usr/bin/env python
"""Provides class Critic with the function approximator (NN) of the Critic

IncrementalModel creates the Neural Network model with Tensorflow and it can train the network online or at the
end of the episode. The user can decide the number of layers, the number of neurons, the batch size and the number
of epochs and activation functions. If trained online, the algorithm trains the Network after the number of collected
data points equals the batch size. This means that if the batch size is 10, then the NN is updated every 10 time steps.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten


"----------------------------------------------------------------------------------------------------------------------"
__author__ = "Jose Ignacio de Alvear Cardenas"
__copyright__ = "Copyright (C) 2020 Jose Ignacio"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Jose Ignacio de Alvear Cardenas"
__email__ = "j.i.dealvearcardenas@student.tudelft.nl"
__status__ = "Production"
"----------------------------------------------------------------------------------------------------------------------"


class Actor:
    def __init__(self, selected_inputs, selected_states, number_time_steps, layers=(10, 1),
                 activations=('lrelu', 'linear'), batch_size=1, epochs=1):
        self.number_inputs = len(selected_inputs)
        self.number_states = len(selected_states)
        self.xt = None
        self.xt_ref = None
        self.ut = 0

        # Attributes related to time
        self.number_time_steps = number_time_steps
        self.time_step = 0

        # Attributes related to the NN
        if layers[-1] != 1:
            raise Exception("The last layer should have a single neuron.")
        elif len(layers) != len(activations):
            raise Exception("The number of layers needs to be equal to the number of activations.")
        self.layers = layers
        self.activations = activations
        self.batch_size = batch_size
        self.epochs = epochs

        self.W = {}
        self.b = {}

    def build_actor_model(self):
        """
        Function that creates the neural network. At the moment, it is a densely connected neural network. The user
        can decide the number of layers, the number of neurons, as well as the activation function.
        :return:
        """
        initializer = tf.keras.initializers.GlorotNormal()
        self.model = tf.keras.Sequential()
        self.model.add(Flatten(input_shape=(self.number_states, 1), name='Flatten_1'))
        self.model.add(Dense(self.layers[0], activation=self.activations[0], kernel_initializer=initializer,
                             name='dense_1'))
        for counter, layer in enumerate(self.layers[1:]):
            self.model.add(Dense(self.layers[counter+1], activation=self.activations[counter+1],
                                 kernel_initializer=initializer, name='dense_'+str(counter+2)))
        self.model.compile(optimizer='adam',
                           loss='mean_squared_error',
                           metrics=['accuracy'])
        trainable_variables = self.model.trainable_variables

        for layer in range(len(trainable_variables/2)):
            self.W['W_' + str(layer+1)] = self.model.trainable_variables[2 * layer]
            self.b['b_' + str(layer+1)] = self.model.trainable_variables[2 * layer + 1]

    def derivative_relu(self, x):
        if x > 0:
            return 1
        else:
            return 0

    def derivative_linear(self, x):
        return 1



if __name__ == '__main__':
    selected_inputs = ['ele']
    selected_states = ['velocity', 'alpha', 'theta', 'q']
    number_time_steps = 500

    actor = Actor(selected_inputs, selected_states, number_time_steps)
    actor.build_critic_model()