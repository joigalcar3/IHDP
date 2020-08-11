#!/usr/bin/env python
"""Provides class Critic with the function approximator (NN) of the Critic

Critic creates the Neural Network model with Tensorflow and it can train the network online or at the
end of the episode. The user can decide the number of layers, the number of neurons, the batch size and the number
of epochs and activation functions. If trained online, the algorithm trains the Network after the number of collected
data points equals the batch size. This means that if the batch size is 10, then the NN is updated every 10 time steps.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from datetime import datetime
import keras
import tensorboard


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


class Critic:

    def __init__(self, Q_weights, selected_states, number_time_steps, gamma=0.8, layers=(10, 1),
                 activations=("tanh", "linear"), batch_size=1, epochs=1, activate_tensorboard=False):
        # Declaration of attributes regarding the states and rewards
        self.number_states = len(selected_states)
        self.xt = None
        self.xt_1 = np.zeros((self.number_states, 1))
        self.xt_ref = None
        self.ct = 0
        self.ct_1 = 0
        self.Jt = 0

        if len(Q_weights)<self.number_states:
            raise Exception("The size of Q_weights needs to equal the number of states")
        self.Q = np.zeros((self.number_states, self.number_states))
        np.fill_diagonal(self.Q, Q_weights)
        self.number_time_steps = number_time_steps
        self.time_step = 0

        # Store the states
        self.store_states = np.zeros((self.number_time_steps, self.number_states, 1))

        # Declaration of attributes related to the neural network
        if layers[-1] != 1:
            raise Exception("The last layer should have a single neuron.")
        elif len(layers) != len(activations):
            raise Exception("The number of layers needs to be equal to the number of activations.")
        self.layers = layers
        self.activations = activations
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.dJt_dxt = None
        self.tensorboard_callback = None
        self.activate_tensorboard = activate_tensorboard

        # Declaration of attributes related to the cost function
        if not(0 <= gamma <= 1):
            raise Exception("The forgetting factor should be in the range [0,1]")
        self.gamma = gamma
        self.store_J = np.zeros((1, self.number_time_steps))
        self.store_c = np.zeros((1, self.number_time_steps))

        # Declaration of storage arrays for online training
        self.store_inputs = np.zeros((self.batch_size, self.number_states, 1))
        self.store_targets = np.zeros((self.batch_size, 1))

    def build_critic_model(self):
        """
        Function that creates the neural network. At the moment, it is a densely connected neural network. The user
        can decide the number of layers, the number of neurons, as well as the activation function.
        :return:
        """
        initializer = tf.keras.initializers.GlorotNormal()
        self.model = tf.keras.Sequential()
        self.model.add(Flatten(input_shape=(self.number_states, 1), name='Flatten_1'))
        self.model.add(Dense(self.layers[0], activation=self.activations[0], kernel_initializer=initializer,
                             name='dense_0'))
        for counter, layer in enumerate(self.layers[1:]):
            self.model.add(Dense(self.layers[counter+1], activation=self.activations[counter+1],
                                 kernel_initializer=initializer, name='dense_'+str(counter+1)))
        self.model.compile(optimizer='adam',
                           loss='mean_squared_error',
                           metrics=['accuracy'])

        if self.activate_tensorboard:
            logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    def run_train_critic_online(self, xt, xt_ref):
        """
        Function that evaluates once the critic neural network and returns the value of J(xt). At the same
        time, it trains the function approximator every number of time steps equal to the chosen batch size.
        :param xt: current time step states
        :param xt_ref: current time step reference states for the computation of the one-step cost function
        :return: Jt --> evaluation of the critic at the current time step
                dJt_dxt --> gradient of dJt/dxt necessary for actor weight update
        """
        self.xt = xt
        self.xt_ref = xt_ref
        self.ct = self.c_computation()

        nn_input = tf.constant(np.array([self.xt]).astype('float32'))
        with tf.GradientTape() as tape:
            tape.watch(nn_input)
            prediction = self.model(nn_input)

        self.Jt = prediction.numpy()
        self.dJt_dxt = tape.gradient(prediction, nn_input).numpy()

        target = self.targets_computation_online()
        inputs = np.reshape(self.xt_1, [1, self.number_states, 1])
        self.store_targets[self.time_step % self.batch_size, :] = target
        self.store_inputs[self.time_step % self.batch_size, :, :] = inputs

        if (self.time_step+1) % self.batch_size == 0:
            if self.activate_tensorboard:
                self.model.fit(self.store_inputs, self.store_targets, batch_size=self.batch_size, epochs=self.epochs,
                               verbose=2, callbacks=[self.tensorboard_callback])
            else:
                self.model.fit(self.store_inputs, self.store_targets, batch_size=self.batch_size, epochs=self.epochs,
                               verbose=2)
        self.time_step += 1
        self.ct_1 = self.ct
        self.xt_1 = self.xt
        return self.Jt, self.dJt_dxt

    def evaluate_critic(self, xt):
        """
        Function that evaluates once the critic neural network and returns the value of J(xt).
        :param xt: current time step states
        :return: Jt --> evaluation of the critic at the current time step
        """
        nn_input = tf.constant(np.array([xt]).astype('float32'))
        with tf.GradientTape() as tape:
            tape.watch(nn_input)
            prediction = self.model(nn_input)

        Jt = prediction.numpy()
        dJt_dxt = tape.gradient(prediction, nn_input).numpy()
        return Jt, dJt_dxt

    def run_critic(self, xt, xt_ref):
        """
        Function which evaluates the critic only once and returns the cost function at the current time step
        :param xt: current time step states
        :param xt_ref: current time step reference states for the computation of the one-step cost function
        :return: Jt --> evaluation of the critic at the current time step
        """
        self.xt = xt
        self.store_states[self.time_step, :, 0] = np.reshape(self.xt, [4,])
        self.xt_ref = xt_ref
        self.c_computation()

        nn_input = np.array([self.xt])
        Jt = self.model.predict(nn_input)
        self.store_J[0, self.time_step] = Jt

        self.time_step += 1
        return Jt

    def train_critic_end(self):
        """
        Function which trains the model at the end of a full episode with the stored one-step cost function
        and the cost function
        :return:
        """
        targets = self.targets_computation_end()
        inputs = self.store_states[:-1, :, :]

        self.model.fit(inputs, targets, batch_size=self.batch_size, epochs=self.epochs, verbose=2)

    def c_computation(self):
        """
        Computation of the one-step cost function with the received real and reference states.
        :return: ct --> current time step one-step cost function
        """
        ct = np.matmul(np.matmul((self.xt - self.xt_ref).T, self.Q), (self.xt - self.xt_ref))
        self.store_c[0, self.time_step] = ct[0]
        return ct

    def targets_computation_end(self):
        """
        Computes the targets at the end of an episode with the stored one-step cost function (ct_1) and
        the stored cost functions (Jt).
        :return: targets --> targets of the complete episode
        """
        targets = np.reshape(self.store_c[0, :-1] + self.gamma * self.store_J[0, 1:], [-1, 1])
        return targets

    def targets_computation_online(self):
        """
        Computes the target at the current time step with the one-step cost function of the previous
        time step and the current cost function.
        :return: target --> the target of the previous time step.
        """
        target = np.reshape(self.ct_1 + self.gamma * self.Jt, [-1, 1])
        return target


if __name__ == "__main__":
    Q_weights = [1,1,1,1]
    selected_states = ['velocity', 'alpha', 'theta', 'q']
    number_time_steps = 500
    critic = Critic(Q_weights, selected_states, number_time_steps)
    a = np.array([[[1], [2], [3], [4]]])
    # a = np.array([[1], [2], [3], [4]])
    critic.critic_model()

    xt = np.array([[[1], [2], [3], [4]]])
    xt = tf.constant(xt.astype('float32'))
    with tf.GradientTape() as tape:
        tape.watch(xt)
        y = self.model(xt)