#!/usr/bin/env python
"""Provides class Actor with the function approximator (NN) of the Actor

Actor creates the Neural Network model with Tensorflow and it can train the network online.
The user can decide the number of layers, the number of neurons, the batch size and the number
of epochs and activation functions.
"""
# TODO: implement the variable learning rate
# TODO: implement the actor model with two neural networks, exploiting the physical model information
# TODO: implement the code that runs the complete algorithm
# TODO: implement batches
# TODO: implement multiple inputs
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
                 activations=('tanh', 'linear'), batch_size=1, epochs=1, learning_rate=10, WB_limits=30):
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
        self.learning_rate = learning_rate
        self.WB_limits = WB_limits


        # Attributes related to the training of the NN
        self.dJt_dWb = None
        self.dJt_dWb_1 = None

        # Attributes related to the Adam optimizer
        self.Adam_opt = None

    def build_actor_model(self):
        """
        Function that creates the neural network. At the moment, it is a densely connected neural network. The user
        can decide the number of layers, the number of neurons, as well as the activation function.
        :return:
        """
        initializer = tf.keras.initializers.GlorotNormal()
        self.model = tf.keras.Sequential()
        self.model.add(Flatten(input_shape=(self.number_states * 2, 1), name='Flatten_1'))
        self.model.add(Dense(self.layers[0], activation=self.activations[0], kernel_initializer=initializer,
                             name='dense_1'))
        for counter, layer in enumerate(self.layers[1:]):
            self.model.add(Dense(self.layers[counter+1], activation=self.activations[counter+1],
                                 kernel_initializer=initializer, name='dense_'+str(counter+2)))
        self.model.compile(optimizer='SGD',
                           loss='mean_squared_error',
                           metrics=['accuracy'])
        # trainable_variables = self.model.trainable_variables
        #
        # for layer in range(int(len(trainable_variables) / 2)):
        #     self.W['W_' + str(layer+1)] = self.model.trainable_variables[2 * layer].numpy()
        #     self.b['b_' + str(layer+1)] = self.model.trainable_variables[2 * layer + 1].numpy()

    def run_actor_online(self, xt, xt_ref):
        """
        Generate input to the system with the reference and real states.
        :param xt: current time_step states
        :param xt_ref: current time step reference states
        :return: ut --> input to the system and the incremental model
        """
        self.xt = xt
        self.xt_ref = xt_ref

        nn_input = tf.constant(np.array([np.vstack((self.xt, self.xt_ref))]).astype('float32'))
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            ut = self.model(nn_input)
        self.dJt_dWb = tape.gradient(ut, self.model.trainable_variables)
        # ut = self.model(nn_input)

        e0 = self.compute_persistent_excitation()
        self.ut = ut + e0
        self.time_step += 1

        return self.ut.numpy()

    def train_actor_online(self, Jt, critic_derivative, G):
        """
        Obtains the elements of the chain rule, computes the gradient and applies it to the corresponding weights and
        biases.
        :param Jt: dEa/dJ
        :param critic_derivative: dJ/dx
        :param G: dx/du, obtained from the incremental model
        :return:
        """
        Jt = Jt.flatten()[0]
        critic_derivative = np.reshape(critic_derivative, [self.number_states, 1])
        chain_rule = Jt * np.matmul(G.T, critic_derivative)
        chain_rule = chain_rule.flatten()[0]
        for count in range(len(self.dJt_dWb)):
            update = chain_rule * self.dJt_dWb[count]
            self.model.trainable_variables[count].assign_sub(np.reshape(self.learning_rate * update, self.model.trainable_variables[count].shape))

            # Implement WB_limits: the weights and biases can not have values whose absolute value exceeds WB_limits
            WB_variable = self.model.trainable_variables[count].numpy()
            WB_variable[WB_variable > self.WB_limits] = self.WB_limits
            WB_variable[WB_variable < -self.WB_limits] = -self.WB_limits
            self.model.trainable_variables[count].assign(WB_variable)

        self.dJt_dWb_1 = self.dJt_dWb

    def train_actor_online_adam(self, Jt, critic_derivative, G):
        """
        Obtains the elements of the chain rule, computes the gradient and applies it to the corresponding weights and
        biases with the Adam optimizer.
        :param Jt: dEa/dJ
        :param critic_derivative: dJ/dx
        :param G: dx/du, obtained from the incremental model
        :return:
        """
        # Set up the Adam optimizer
        if self.time_step == 0:
            self.Adam_opt = tf.optimizers.Adam(learning_rate=self.learning_rate, decay=1e-6)
        Jt = Jt.flatten()[0]
        critic_derivative = np.reshape(critic_derivative, [self.number_states, 1])
        chain_rule = Jt * np.matmul(G.T, critic_derivative)
        chain_rule = chain_rule.flatten()[0]
        update = [tf.Variable(chain_rule * self.dJt_dWb[i]) for i in range(len(self.dJt_dWb))]
        update = [tf.Variable(np.reshape(update[i].numpy(), [-1, ]))
                  if len(self.model.trainable_variables[i].shape) == 1
                  else np.reshape(update[i].numpy(), [-1, self.model.trainable_variables[i].shape[1]])
                  for i in range(len(update))]

        # Apply the Adam optimizer
        self.Adam_opt.apply_gradients(zip(update, self.model.trainable_variables))

        for count in range(len(self.dJt_dWb)):
            # Implement WB_limits: the weights and biases can not have values whose absolute value exceeds WB_limits
            WB_variable = self.model.trainable_variables[count].numpy()
            WB_variable[WB_variable > self.WB_limits] = self.WB_limits
            WB_variable[WB_variable < -self.WB_limits] = -self.WB_limits
            self.model.trainable_variables[count].assign(WB_variable)

        self.dJt_dWb_1 = self.dJt_dWb

    def compute_persistent_excitation(self):
        """
        Computation of the persistent excitation at each time step. Formula obtained from Pedro's thesis
        :return: e0 --> PE deviation
        """
        t = self.time_step+1
        e0 = 0.3 / t * (np.sin(100 * t) ** 2 * np.cos(100 * t) + np.sin(2 * t) ** 2 * np.cos(0.1 * t) +
                        np.sin(-1.2 * t) ** 2 * np.cos(0.5 * t) + np.sin(t) ** 5 + np.sin(1.12 * t) ** 2 +
                        np.cos(2.4 * t) * np.sin(2.4 * t) ** 3) / 10
        return e0




if __name__ == '__main__':
    selected_inputs = ['ele']
    selected_states = ['velocity', 'alpha', 'theta', 'q']
    number_time_steps = 500

    actor = Actor(selected_inputs, selected_states, number_time_steps)
    actor.build_actor_model()

    xt = np.array([[1], [2], [3], [4]])
    xt_ref = np.array([[2], [3], [4], [5]])
    actor.run_actor_online(xt, xt_ref)
    # actor.model.optimizer.learning_rate = tf.Variable(name='learning_rate:0', dtype='float32', shape=(),
    #                                                   initial_value=10)

