#!/usr/bin/env python
"""Provides class Actor with the function approximator (NN) of the Actor

Actor creates the Neural Network model with Tensorflow and it can train the network online.
The user can decide the number of layers, the number of neurons, the batch size and the number
of epochs and activation functions.
"""
# TODO: implement the variable learning rate --> DONE
# TODO: implement the actor model with two neural networks, exploiting the physical model information
# TODO: implement the code that runs the complete algorithm --> DONE
# TODO: implement batches
# TODO: implement multiple inputs
# Try to update the actor before updating the critic
# Try to change the activation functions of the hidden layers --> did not work
# Try removing the doubling of the learning rate
# Ask what is taken as input to the NNs
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
    def __init__(self, selected_inputs, selected_states, tracking_states, indices_tracking_states,
                 number_time_steps, start_training, layers=(6, 1), activations=('tanh', 'linear'), batch_size=1, epochs=1,
                 learning_rate=10, learning_rate_exponent_limit=10, only_track_xt_input=False, WB_limits=30):
        self.number_inputs = len(selected_inputs)
        self.number_states = len(selected_states)
        self.number_tracking_states = len(tracking_states)
        self.indices_tracking_states = indices_tracking_states
        self.xt = None
        self.xt_ref = None
        self.ut = 0

        # Attributes related to time
        self.number_time_steps = number_time_steps
        self.time_step = 0
        self.start_training = start_training

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
        self.learning_rate_0 = learning_rate
        self.learning_rate_exponent_limit = learning_rate_exponent_limit
        self.only_track_xt_input = only_track_xt_input
        self.WB_limits = WB_limits


        # Attributes related to the training of the NN
        self.dut_dWb = None
        self.dut_dWb_1 = None

        # Attributes related to the Adam optimizer
        self.Adam_opt = None

        # Attributes related to the momentum
        self.momentum_dict = {}
        self.beta_momentum = 0.9

        # Attributes related to RMSprop
        self.rmsprop_dict = {}
        self.beta_rmsprop = 0.999
        self.epsilon = 1e-8

    def build_actor_model(self):
        """
        Function that creates the neural network. At the moment, it is a densely connected neural network. The user
        can decide the number of layers, the number of neurons, as well as the activation function.
        :return:
        """
        # initializer = tf.keras.initializers.GlorotNormal()
        initializer = tf.keras.initializers.VarianceScaling(
            scale=0.04, mode='fan_in', distribution='truncated_normal', seed=None)
        self.model = tf.keras.Sequential()
        if self.only_track_xt_input:
            self.model.add(Flatten(input_shape=(len(self.indices_tracking_states) + self.number_tracking_states, 1), name='Flatten_1'))
        else:
            self.model.add(Flatten(input_shape=(self.number_states + self.number_tracking_states, 1), name='Flatten_1'))
        self.model.add(Dense(self.layers[0], activation=self.activations[0], kernel_initializer=initializer,
                             name='dense_1'))
        for counter, layer in enumerate(self.layers[1:]):
            self.model.add(Dense(self.layers[counter+1], activation=self.activations[counter+1],
                                 kernel_initializer=initializer, name='dense_'+str(counter+2)))
        self.model.compile(optimizer='SGD',
                           loss='mean_squared_error',
                           metrics=['accuracy'])

        for count in range(len(self.model.trainable_variables)):
            self.momentum_dict[count] = 0
            self.rmsprop_dict[count] = 0

    def run_actor_online(self, xt, xt_ref):
        """
        Generate input to the system with the reference and real states.
        :param xt: current time_step states
        :param xt_ref: current time step reference states
        :return: ut --> input to the system and the incremental model
        """
        if self.only_track_xt_input:
            self.xt = np.reshape(xt[self.indices_tracking_states, :], [-1, 1])
        else:
            self.xt = xt
        self.xt_ref = xt_ref

        nn_input = tf.constant(np.array([np.vstack((self.xt, self.xt_ref))]).astype('float32'))
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            ut = self.model(nn_input)
        self.dut_dWb = tape.gradient(ut, self.model.trainable_variables)
        # ut = self.model(nn_input)

        e0 = self.compute_persistent_excitation()
        self.ut = ut + e0

        return self.ut.numpy()

    def train_actor_online(self, Jt1, dJt1_dxt1, G):
        """
        Obtains the elements of the chain rule, computes the gradient and applies it to the corresponding weights and
        biases.
        :param Jt1: dEa/dJ
        :param dJt1_dxt1: dJ/dx
        :param G: dx/du, obtained from the incremental model
        :return:
        """
        Jt1 = Jt1.flatten()[0]
        chain_rule = Jt1 * np.matmul(G.T, dJt1_dxt1)
        chain_rule = chain_rule.flatten()[0]
        for count in range(len(self.dut_dWb)):
            update = chain_rule * self.dut_dWb[count]
            self.model.trainable_variables[count].assign_sub(np.reshape(self.learning_rate * update,
                                                                        self.model.trainable_variables[count].shape))

            # Implement WB_limits: the weights and biases can not have values whose absolute value exceeds WB_limits
            WB_variable = self.model.trainable_variables[count].numpy()
            WB_variable[WB_variable > self.WB_limits] = self.WB_limits
            WB_variable[WB_variable < -self.WB_limits] = -self.WB_limits
            self.model.trainable_variables[count].assign(WB_variable)

    def train_actor_online_adaptive_alpha(self, Jt1, dJt1_dxt1, G, incremental_model, critic, xt_ref1):
        """
        Train the actor with an adaptive alpha depending on the sign and magnitude of the network errors
        :param Jt1: the evaluation of the critic with the next time step prediction of the incremental model
        :param dJt1_dxt1: the gradient of the critic network with respect to the next time prediction of the incremental model
        :param G: the input distribution matrix
        :param indices_tracking_states: the states of the system that are being tracked
        :param incremental_model: the incremental model
        :param critic: the critic
        :return:
        """
        Ec_actor_before = 0.5 * np.square(Jt1)
        print("ACTOR LOSS xt1 before= ", Ec_actor_before)
        weight_cache = [tf.Variable(self.model.trainable_variables[i].numpy()) for i in
                        range(len(self.model.trainable_variables))]
        network_improvement = False
        n_reductions = 0
        while not network_improvement and self.time_step > self.start_training:
            # Train the actor
            self.train_actor_online(Jt1, dJt1_dxt1, G)

            # Code for checking if the actor NN error with the new weights has changed sign
            ut_after = self.evaluate_actor()
            xt1_est_after = incremental_model.evaluate_incremental_model(ut_after)
            Jt1_after, _ = critic.evaluate_critic(xt1_est_after, xt_ref1)
            Ec_actor_after = 0.5 * np.square(Jt1_after)
            print("ACTOR LOSS xt1 after= ", Ec_actor_after)

            # Code for checking whether the learning rate of the actor should be halved
            if Ec_actor_after <= Ec_actor_before or n_reductions > 10:
                network_improvement = True
                if np.sign(Jt1) == np.sign(Jt1_after):
                    self.learning_rate = min(2 * self.learning_rate,
                                             self.learning_rate_0 * 2**self.learning_rate_exponent_limit)
                    print("ACTOR LEARNING_RATE = ", self.learning_rate)
            else:
                n_reductions += 1
                self.learning_rate = max(self.learning_rate / 2,
                                         self.learning_rate_0/2**self.learning_rate_exponent_limit)
                for WB_count in range(len(self.model.trainable_variables)):
                    self.model.trainable_variables[WB_count].assign(weight_cache[WB_count].numpy())
                print("ACTOR LEARNING_RATE = ", self.learning_rate)

    def train_actor_online_momentum(self, Jt1, dJt1_dxt1, G, incremental_model, critic, xt_ref1):
        """
        Train the actor with an adaptive alpha depending on the sign and magnitude of the network errors
        :param Jt1: the evaluation of the critic with the next time step prediction of the incremental model
        :param dJt1_dxt1: the gradient of the critic network with respect to the next time prediction of the incremental model
        :param G: the input distribution matrix
        :param indices_tracking_states: the states of the system that are being tracked
        :param incremental_model: the incremental model
        :param critic: the critic
        :return:
        """
        Ec_actor_before = 0.5 * np.square(Jt1)
        print("ACTOR LOSS xt1 before= ", Ec_actor_before)

        # Train the actor
        Jt1 = Jt1.flatten()[0]
        chain_rule = Jt1 * np.matmul(G.T, dJt1_dxt1)
        chain_rule = chain_rule.flatten()[0]
        if self.time_step > self.start_training:
            for count in range(len(self.dut_dWb)):
                gradient = chain_rule * self.dut_dWb[count]
                update = self.beta_momentum * self.momentum_dict[count] + (1-self.beta_momentum) * gradient
                self.momentum_dict[count] = update
                self.model.trainable_variables[count].assign_sub(np.reshape(self.learning_rate * update,
                                                                            self.model.trainable_variables[count].shape))

                # Implement WB_limits: the weights and biases can not have values whose absolute value exceeds WB_limits
                WB_variable = self.model.trainable_variables[count].numpy()
                WB_variable[WB_variable > self.WB_limits] = self.WB_limits
                WB_variable[WB_variable < -self.WB_limits] = -self.WB_limits
                self.model.trainable_variables[count].assign(WB_variable)

        # Code for checking if the actor NN error with the new weights has changed sign
        ut_after = self.evaluate_actor()
        xt1_est_after = incremental_model.evaluate_incremental_model(ut_after)
        Jt1_after, _ = critic.evaluate_critic(xt1_est_after, xt_ref1)
        Ec_actor_after = 0.5 * np.square(Jt1_after)
        print("ACTOR LOSS xt1 after= ", Ec_actor_after)

    def train_actor_online_adam(self, Jt1, dJt1_dxt1, G, incremental_model, critic, xt_ref1):
        """
        Train the actor with an adaptive alpha depending on the sign and magnitude of the network errors
        :param Jt1: the evaluation of the critic with the next time step prediction of the incremental model
        :param dJt1_dxt1: the gradient of the critic network with respect to the next time prediction of the incremental model
        :param G: the input distribution matrix
        :param indices_tracking_states: the states of the system that are being tracked
        :param incremental_model: the incremental model
        :param critic: the critic
        :return:
        """
        Ec_actor_before = 0.5 * np.square(Jt1)
        print("ACTOR LOSS xt1 before= ", Ec_actor_before)

        # Train the actor
        Jt1 = Jt1.flatten()[0]
        chain_rule = Jt1 * np.matmul(G.T, dJt1_dxt1)
        chain_rule = chain_rule.flatten()[0]
        if self.time_step > self.start_training:
            for count in range(len(self.dut_dWb)):
                gradient = chain_rule * self.dut_dWb[count]
                momentum = self.beta_momentum * self.momentum_dict[count] + (1-self.beta_momentum) * gradient
                self.momentum_dict[count] = momentum
                momentum_corrected = momentum/(1-self.beta_momentum**(self.time_step + 1))

                rmsprop = self.beta_rmsprop * self.rmsprop_dict[count] + \
                          (1 - self.beta_rmsprop) * np.multiply(gradient, gradient)
                self.rmsprop_dict[count] = rmsprop
                rmsprop_corrected = rmsprop/(1 - self.beta_rmsprop**(self.time_step + 1))

                update = momentum_corrected/(np.sqrt(rmsprop_corrected) + self.epsilon)
                self.model.trainable_variables[count].assign_sub(np.reshape(self.learning_rate * update,
                                                                            self.model.trainable_variables[count].shape))

                # Implement WB_limits: the weights and biases can not have values whose absolute value exceeds WB_limits
                WB_variable = self.model.trainable_variables[count].numpy()
                WB_variable[WB_variable > self.WB_limits] = self.WB_limits
                WB_variable[WB_variable < -self.WB_limits] = -self.WB_limits
                self.model.trainable_variables[count].assign(WB_variable)

        # Code for checking if the actor NN error with the new weights has changed sign
        ut_after = self.evaluate_actor()
        xt1_est_after = incremental_model.evaluate_incremental_model(ut_after)
        Jt1_after, _ = critic.evaluate_critic(xt1_est_after, xt_ref1)
        Ec_actor_after = 0.5 * np.square(Jt1_after)
        print("ACTOR LOSS xt1 after= ", Ec_actor_after)


    # def train_actor_online_adam(self, Jt, critic_derivative, G):
    #     """
    #     Obtains the elements of the chain rule, computes the gradient and applies it to the corresponding weights and
    #     biases with the Adam optimizer.
    #     :param Jt: dEa/dJ
    #     :param critic_derivative: dJ/dx
    #     :param G: dx/du, obtained from the incremental model
    #     :return:
    #     """
    #     # Set up the Adam optimizer
    #     if self.time_step == 0:
    #         self.Adam_opt = tf.optimizers.Adam(learning_rate=self.learning_rate, decay=1e-6)
    #     Jt = Jt.flatten()[0]
    #     critic_derivative = np.reshape(critic_derivative[:self.number_states, :], [self.number_states, 1])
    #     chain_rule = Jt * np.matmul(G.T, critic_derivative)
    #     chain_rule = chain_rule.flatten()[0]
    #     update = [tf.Variable(chain_rule * self.dut_dWb[i]) for i in range(len(self.dut_dWb))]
    #     update = [tf.Variable(np.reshape(update[i].numpy(), [-1, ]))
    #               if len(self.model.trainable_variables[i].shape) == 1
    #               else np.reshape(update[i].numpy(), [-1, self.model.trainable_variables[i].shape[1]])
    #               for i in range(len(update))]
    #
    #     # Apply the Adam optimizer
    #     self.Adam_opt.apply_gradients(zip(update, [self.model.trainable_variables[i].numpy() for i in range(len(self.model.trainable_variables))]))
    #
    #     for count in range(len(self.dut_dWb)):
    #         # Implement WB_limits: the weights and biases can not have values whose absolute value exceeds WB_limits
    #         WB_variable = self.model.trainable_variables[count].numpy()
    #         WB_variable[WB_variable > self.WB_limits] = self.WB_limits
    #         WB_variable[WB_variable < -self.WB_limits] = -self.WB_limits
    #         self.model.trainable_variables[count].assign(WB_variable)

    def compute_persistent_excitation(self):
        """
        Computation of the persistent excitation at each time step. Formula obtained from Pedro's thesis
        :return: e0 --> PE deviation
        """
        t = self.time_step + 1
        # e0 = 0.3 * (np.sin(100 * t) ** 2 * np.cos(100 * t) + np.sin(2 * t) ** 2 * np.cos(0.1 * t) +
        #                 np.sin(-1.2 * t) ** 2 * np.cos(0.5 * t) + np.sin(t) ** 5 + np.sin(1.12 * t) ** 2 +
        #                 np.cos(2.4 * t) * np.sin(2.4 * t) ** 3)

        e0 = 0.3 / t * (np.sin(100 * t) ** 2 * np.cos(100 * t) + np.sin(2 * t) ** 2 * np.cos(0.1 * t) +
                        np.sin(-1.2 * t) ** 2 * np.cos(0.5 * t) + np.sin(t) ** 5 + np.sin(1.12 * t) ** 2 +
                        np.cos(2.4 * t) * np.sin(2.4 * t) ** 3) / 10
        return e0

    def update_actor_attributes(self):
        """
        The attributes that change with every time step are updated
        :return:
        """
        self.time_step += 1
        self.dut_dWb_1 = self.dut_dWb

    def evaluate_actor(self, *args):
        """
        Evaluation of the actor NN given an input or attributes stored in the object
        :param args: the real and reference states could be provided as input for the evaluation, or not if already stored
        :return: ut --> input to the system and the incremental model
        """
        if len(args) == 0:
            nn_input = tf.constant(np.array([np.vstack((self.xt, self.xt_ref))]).astype('float32'))
            ut = self.model(nn_input).numpy()
        elif len(args) == 2:
            xt = args[0]
            xt_ref = args[1]
            nn_input = tf.constant(np.array([np.vstack((xt, xt_ref))]).astype('float32'))
            ut = self.model(nn_input).numpy()
        else:
            raise Exception("THERE SHOULD BE AN OUTPUT in the evaluate_actor function.")
        return ut

    def restart_actor(self):
        """
        Restart the actor attributes
        :return:
        """
        self.time_step = 0
        self.xt = None
        self.xt_ref = None
        self.ut = 0

        # Attributes related to the training of the NN
        self.dut_dWb = None
        self.dut_dWb_1 = None
        self.learning_rate = self.learning_rate_0

        # Attributes related to the Adam optimizer
        self.Adam_opt = None

        # Restart momentum and rmsprop
        for count in range(len(self.model.trainable_variables)):
            self.momentum_dict[count] = 0
            self.rmsprop_dict[count] = 0



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

