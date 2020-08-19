#!/usr/bin/env python
"""Provides class Critic with the function approximator (NN) of the Critic

Critic creates the Neural Network model with Tensorflow and it can train the network online or at the
end of the episode. The user can decide the number of layers, the number of neurons, the batch size and the number
of epochs and activation functions. If trained online, the algorithm trains the Network after the number of collected
data points equals the batch size. This means that if the batch size is 10, then the NN is updated every 10 time steps.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout
from datetime import datetime
import keras
import tensorboard
import random


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

    def __init__(self, Q_weights, selected_states, tracking_states, indices_tracking_states, number_time_steps,
                 start_training, gamma=1, learning_rate=20, learning_rate_exponent_limit=10, layers=(6, 1),
                 activations=("tanh", "linear"), batch_size=1, epochs=1, activate_tensorboard=False,
                 input_include_reference=True, WB_limits=30):
        # Declaration of attributes regarding the states and rewards
        self.number_states = len(selected_states)
        self.number_tracking_states = len(tracking_states)
        self.indices_tracking_states = indices_tracking_states
        self.xt = None
        self.xt_1 = np.zeros((self.number_states, 1))
        self.xt_ref = None
        self.xt_ref_1 = np.zeros((self.number_tracking_states, 1))
        self.ct = 0
        self.ct_1 = 0
        self.Jt = 0
        self.Jt_1 = 1

        if len(Q_weights)<self.number_tracking_states:
            raise Exception("The size of Q_weights needs to equal the number of states")
        self.Q = np.zeros((self.number_tracking_states, self.number_tracking_states))
        np.fill_diagonal(self.Q, Q_weights)
        self.number_time_steps = number_time_steps
        self.time_step = 0
        self.start_training = start_training

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
        self.input_include_reference = input_include_reference

        # Declaration of attributes related to the cost function
        if not(0 <= gamma <= 1):
            raise Exception("The forgetting factor should be in the range [0,1]")
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.learning_rate_0 = learning_rate
        self.learning_rate_exponent_limit = learning_rate_exponent_limit
        self.WB_limits = WB_limits
        self.store_J = np.zeros((1, self.number_time_steps))
        self.store_c = np.zeros((1, self.number_time_steps))

        # Declaration of storage arrays for online training
        self.store_inputs = np.zeros((self.batch_size, self.number_states, 1))
        self.store_targets = np.zeros((self.batch_size, 1))

        # Attributes related to the momentum
        self.momentum_dict = {}
        self.beta_momentum = 0.9

        # Attributes related to RMSprop
        self.rmsprop_dict = {}
        self.beta_rmsprop = 0.999
        self.epsilon = 1e-8

        # Attributes related to experience replay
        self.replay = []

    def build_critic_model(self):
        """
        Function that creates the neural network. At the moment, it is a densely connected neural network. The user
        can decide the number of layers, the number of neurons, as well as the activation function.
        :return:
        """
        initializer = tf.keras.initializers.GlorotNormal()
        # initializer = tf.keras.initializers.VarianceScaling(
        #     scale=0.04, mode='fan_in', distribution='truncated_normal', seed=None)
        self.model = tf.keras.Sequential()
        if self.input_include_reference:
            self.model.add(Flatten(input_shape=(self.number_states + self.number_tracking_states, 1), name='Flatten_1'))
        else:
            self.model.add(Flatten(input_shape=(self.number_states, 1), name='Flatten_1'))
        self.model.add(Dense(self.layers[0], activation=self.activations[0], kernel_initializer=initializer,
                             name='dense_0'))
        # self.model.add(Dropout(0.1, name='Dropout_0'))
        for counter, layer in enumerate(self.layers[1:]):
            self.model.add(Dense(self.layers[counter+1], activation=self.activations[counter+1],
                                 kernel_initializer=initializer, name='dense_'+str(counter+1)))
        self.model.compile(optimizer='adam',
                           loss='mean_squared_error',
                           metrics=['accuracy'])

        for count in range(len(self.model.trainable_variables)):
            self.momentum_dict[count] = 0
            self.rmsprop_dict[count] = 0

        if self.activate_tensorboard:
            logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    # def run_train_critic_online(self, xt, xt_ref):
    #     """
    #     Function that evaluates once the critic neural network and returns the value of J(xt). At the same
    #     time, it trains the function approximator every number of time steps equal to the chosen batch size.
    #     :param xt: current time step states
    #     :param xt_ref: current time step reference states for the computation of the one-step cost function
    #     :return: Jt --> evaluation of the critic at the current time step
    #             dJt_dxt --> gradient of dJt/dxt necessary for actor weight update
    #     """
    #     self.xt = xt
    #     self.xt_ref = xt_ref
    #     self.ct = self.c_computation()
    #
    #     nn_input = tf.constant(np.array([self.xt]).astype('float32'))
    #     with tf.GradientTape() as tape:
    #         tape.watch(nn_input)
    #         prediction = self.model(nn_input)
    #
    #     self.Jt = prediction.numpy()
    #     self.dJt_dxt = tape.gradient(prediction, nn_input).numpy()
    #
    #     target = self.targets_computation_online()
    #     inputs = np.reshape(self.xt_1, [1, self.number_states, 1])
    #     self.store_targets[self.time_step % self.batch_size, :] = target
    #     self.store_inputs[self.time_step % self.batch_size, :, :] = inputs
    #
    #     if (self.time_step+1) % self.batch_size == 0:
    #         if self.activate_tensorboard:
    #             self.model.fit(self.store_inputs, self.store_targets, batch_size=self.batch_size, epochs=self.epochs,
    #                            verbose=2, callbacks=[self.tensorboard_callback])
    #         else:
    #             self.model.fit(self.store_inputs, self.store_targets, batch_size=self.batch_size, epochs=self.epochs,
    #                            verbose=2)
    #     self.time_step += 1
    #     self.ct_1 = self.ct
    #     self.xt_1 = self.xt
    #     return self.Jt, self.dJt_dxt

    def run_train_critic_online_adaptive_alpha(self, xt, xt_ref):
        """
        Function that evaluates once the critic neural network and returns the value of J(xt). At the same
        time, it trains the function approximator every number of time steps equal to the chosen batch size.
        :param xt: current time step states
        :param xt_ref: current time step reference states for the computation of the one-step cost function
        :return: Jt --> evaluation of the critic at the current time step
        """
        nn_input, dJt_dW = self.compute_forward_pass(xt, xt_ref)
        dE_dJ, ec_critic_before, EC_critic_before= self.compute_loss_derivative()
        weight_cache = [tf.Variable(self.model.trainable_variables[i].numpy())
                        for i in range(len(self.model.trainable_variables))]

        network_improvement = False
        n_reductions = 0
        while not network_improvement and self.time_step > self.start_training:
            for count in range(len(dJt_dW)):
                update = dE_dJ * dJt_dW[count]
                self.model.trainable_variables[count].assign_sub(np.reshape(self.learning_rate * update, self.model.trainable_variables[count].shape))

                # Implement WB_limits: the weights and biases can not have values whose absolute value exceeds WB_limits
                self.check_WB_limits(count)
            updated_Jt = self.model(nn_input)
            ec_critic_after = np.reshape(-self.ct_1 - self.gamma * updated_Jt.numpy(), [-1, 1]) + self.Jt_1
            Ec_critic_after = 0.5 * np.square(ec_critic_after)
            print("CRITIC LOSS xt after= ", Ec_critic_after)

            # In the case that the error is not decreased, the time step is repeated with half the learning rate
            if Ec_critic_after <= EC_critic_before or n_reductions > 10:
                network_improvement = True
                # The learning rate is doubled if the network errors have the same signs
                if np.sign(ec_critic_before) == np.sign(ec_critic_after):
                    self.learning_rate = min(2 * self.learning_rate,
                                             self.learning_rate_0 * 2**self.learning_rate_exponent_limit)
            else:
                n_reductions += 1
                self.learning_rate = max(self.learning_rate / 2,
                                         self.learning_rate_0/2**self.learning_rate_exponent_limit)
                for WB_count in range(len(self.model.trainable_variables)):
                    self.model.trainable_variables[WB_count].assign(weight_cache[WB_count].numpy())

        return self.Jt

    def run_train_critic_online_momentum(self, xt, xt_ref):
        """
        Function that evaluates once the critic neural network and returns the value of J(xt). At the same
        time, it trains the function approximator every number of time steps equal to the chosen batch size.
        :param xt: current time step states
        :param xt_ref: current time step reference states for the computation of the one-step cost function
        :return: Jt --> evaluation of the critic at the current time step
        """
        nn_input, dJt_dW = self.compute_forward_pass(xt, xt_ref)
        dE_dJ, _, _ = self.compute_loss_derivative()
        if self.time_step > self.start_training:
            for count in range(len(dJt_dW)):
                gradient = dE_dJ * dJt_dW[count]
                update = self.beta_momentum * self.momentum_dict[count] + (1-self.beta_momentum) * gradient
                self.momentum_dict[count] = update
                self.model.trainable_variables[count].assign_sub(
                    np.reshape(self.learning_rate * update, self.model.trainable_variables[count].shape))

                # Implement WB_limits: the weights and biases can not have values whose absolute value exceeds WB_limits
                self.check_WB_limits(count)
        updated_Jt = self.model(nn_input)
        ec_critic_after = np.reshape(-self.ct_1 - self.gamma * updated_Jt.numpy(), [-1, 1]) + self.Jt_1
        Ec_critic_after = 0.5 * np.square(ec_critic_after)
        print("CRITIC LOSS xt after= ", Ec_critic_after)

        return self.Jt

    def run_train_critic_online_adam(self, xt, xt_ref, iteration):
        """
        Function that evaluates once the critic neural network and returns the value of J(xt). At the same
        time, it trains the function approximator every number of time steps equal to the chosen batch size.
        :param xt: current time step states
        :param xt_ref: current time step reference states for the computation of the one-step cost function
        :return: Jt --> evaluation of the critic at the current time step
        """
        # Safe the information in the replay attribute
        if self.input_include_reference:
            self.replay.append((self.xt_1, self.xt_ref_1, xt, xt_ref, self.ct_1))
        else:
            self.replay.append((self.xt_1, xt, self.ct_1))

        # Obtain the forward pass of the critic and the derivatives of the output with respect to the weights and biases
        nn_input, dJt_dW = self.compute_forward_pass(xt, xt_ref)

        # Obtain the derivative of the loss with respect to the critic NN output (Jt)
        dE_dJ, _, _ = self.compute_loss_derivative()

        # Run the Adam optimizer given the gradients
        self.adam_iteration(dJt_dW, dE_dJ, iteration)

        # Check the impact of the update on the critic loss function
        updated_Jt = self.model(nn_input)
        ec_critic_after = np.reshape(-self.ct_1 - self.gamma * updated_Jt.numpy(), [-1, 1]) + self.Jt_1
        Ec_critic_after = 0.5 * np.square(ec_critic_after)
        print("CRITIC LOSS xt after= ", Ec_critic_after)

        return self.Jt

    def train_critic_replay_adam(self, replay_size, iteration):
        """
        Function that trains the critic with values stored in the replay.
        :param xt: current time step states
        :param xt_ref: current time step reference states for the computation of the one-step cost function
        :return: Jt --> evaluation of the critic at the current time step
        """
        # Compute the number of data points used in the replay training
        replay_size = min(replay_size, len(self.replay))

        # Define the data points that are going to be used in the replay training
        indices = list(range(len(self.replay)))
        random.shuffle(indices)
        for i in range(replay_size):
            # Extract the data point information
            index = indices[i]
            replay = self.replay[index]
            if self.input_include_reference:
                xt_1, xt_ref_1, xt, xt_ref, ct_1 = replay
                nn_input_1 = tf.constant(np.array([np.vstack((xt_1, xt_ref_1))]).astype('float32'))
            else:
                xt_1, xt, ct_1 = replay
                xt_ref = 0
                nn_input_1 = tf.constant(np.array([(xt_1)]).astype('float32'))

            # Obtain the forward pass of xt and the derivative of the output with respect to weights and biases
            nn_input, dJt_dW, Jt = self.compute_forward_pass(xt, xt_ref, replay=True)

            # Obtain the forward pass of xt_1
            Jt_1 = self.model(nn_input_1).numpy()

            # Obtain the derivative of the critic cost function with respect to the critic output
            dE_dJ, _, _ = self.compute_loss_derivative(Jt_1, Jt, ct_1)

            # Carry out the Adam optimisation
            self.adam_iteration(dJt_dW, dE_dJ, iteration)

            # Check the impact of the training to the loss function of the critic
            updated_Jt = self.model(nn_input)
            ec_critic_after = self.targets_computation_online(updated_Jt, ct_1) + Jt_1
            Ec_critic_after = 0.5 * np.square(ec_critic_after)
            print("CRITIC LOSS xt after= ", Ec_critic_after)

    def adam_iteration(self, dJt_dW, dE_dJ, iteration):
        """
        Adam update to all the weights and biases given the derivative of the loss function with respect to the NN
        output and the derivative of the neural network output with respect to the weights and biases.
        :param dJt_dW: derivative of the NN output with respect to the weights and biases
        :param dE_dJ: derivative of the loss function with respect to the NN output
        :return:
        """
        if self.time_step > self.start_training:
            for count in range(len(dJt_dW)):
                gradient = dE_dJ * dJt_dW[count]
                momentum = self.beta_momentum * self.momentum_dict[count] + (1 - self.beta_momentum) * gradient
                self.momentum_dict[count] = momentum
                momentum_corrected = momentum / (1 - self.beta_momentum ** (self.time_step + 1))
                # momentum_corrected = momentum / (1 - self.beta_momentum ** (iteration + 1))

                rmsprop = self.beta_rmsprop * self.rmsprop_dict[count] + \
                          (1 - self.beta_rmsprop) * np.multiply(gradient, gradient)
                self.rmsprop_dict[count] = rmsprop
                rmsprop_corrected = rmsprop / (1 - self.beta_rmsprop ** (self.time_step + 1))
                # rmsprop_corrected = rmsprop / (1 - self.beta_rmsprop ** (iteration + 1))

                update = momentum_corrected / (np.sqrt(rmsprop_corrected) + self.epsilon)

                self.model.trainable_variables[count].assign_sub(
                    np.reshape(self.learning_rate * update, self.model.trainable_variables[count].shape))

                # Implement WB_limits: the weights and biases can not have values whose absolute value exceeds WB_limits
                self.check_WB_limits(count)

    def compute_forward_pass(self, xt, xt_ref, replay=False):
        """
        Compute the output of the critic, as well as the derivative of Jt with respect to the network weights and biases
        :param xt: states
        :param xt_ref: reference states
        :return: nn_input --> formatted input to the neural network
                dJt_dW --> derivative of the loss function with respect to the weights and biases
        """
        # If it is online, safe the input in the object
        if not replay:
            self.xt = xt
            self.xt_ref = xt_ref
            self.ct = self.c_computation()

        # Define the input to the critic NN
        if self.input_include_reference:
            nn_input = tf.constant(np.array([np.vstack((xt, xt_ref))]).astype('float32'))
            # nn_input_1 = tf.constant(np.array([np.vstack((xt_1, xt_ref_1))]).astype('float32'))     # TEMPORAL
        else:
            nn_input = tf.constant(np.array([(xt)]).astype('float32'))
            # nn_input_1 = tf.constant(np.array([(xt_1)]).astype('float32'))     # TEMPORAL

        # Run the input through the network watching the weights and biases for later derivatives
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            prediction = self.model(nn_input)
        # Comment: the gradient of the tanh activation function is 1-tanh(Z)**2. As a result, if the values of Z are
        # very high or very low, tanh(Z) will lead to either -1 or 1, since the output values of the tanh(Z) activation
        # function are constrained to the range [-1,1]. Consequently, the derivative will be always be zero and the
        # derivatives will not be propagated beyond this activation function. The weight and bias of Z = W.T*X + b
        # will not be updated and the NN only relies in low input (X) values to change the weights and biases. The NN
        # arrives at a stagnated point.

        # Obtain the derivative of the output with respect to the weights and biases
        dJt_dW = tape.gradient(prediction, self.model.trainable_variables)

        # In the case that it is online, safe the output in the object; otherwise provide as function output
        if not replay:
            self.Jt = prediction.numpy()
            self.store_J[:, self.time_step] = np.reshape(self.Jt, [-1])
            # self.Jt_1 = self.model(nn_input_1).numpy()     # TEMPORAL
            return nn_input, dJt_dW
        else:
            Jt = prediction.numpy()
            return nn_input, dJt_dW, Jt


    def compute_loss_derivative(self, *args):
        """
        Computes the derivative of the loss function with respect to Jt
        :return: dE_dJ --> derivative of the loss function with respect to Jt
                ec_critic_before --> error of the network before the training
                EC_critic_before --> loss function of the network before the training
        """
        # In the case that there are no inputs, obtain data from the object attributes
        if len(args) == 0:
            Jt_1 = self.Jt_1
            Jt = self.Jt
            target = self.targets_computation_online()
        elif len(args) == 3:
            Jt_1 = args[0]
            Jt = args[1]
            ct_1 = args[2]
            target = self.targets_computation_online(Jt, ct_1)
        else:
            Jt_1 = 0
            Jt = 0
            target = 0
            Exception("Unexpected number of arguments.")

        # Compute the network error
        ec_critic_before = target + Jt_1

        # Compute the derivative of the loss function with respect to the critic network output (Jt)
        dE_dJ = -self.gamma * ec_critic_before

        # Check what is the critic and actor loss values before the critic network update.
        EC_critic_before = 0.5 * np.square(ec_critic_before)
        Ec_actor_before = 0.5 * np.square(Jt)
        print("CRITIC LOSS xt before= ", EC_critic_before)
        print("ACTOR LOSS xt = ", Ec_actor_before)

        return dE_dJ, ec_critic_before, EC_critic_before

    def check_WB_limits(self, count):
        """
        Check whether any of the weights and biases exceed the limit imposed (WB_limits) and saturate the values
        :param count: index within the model.trainable_variables being analysed
        :return:
        """
        WB_variable = self.model.trainable_variables[count].numpy()
        WB_variable[WB_variable > self.WB_limits] = self.WB_limits
        WB_variable[WB_variable < -self.WB_limits] = -self.WB_limits
        self.model.trainable_variables[count].assign(WB_variable)

    # def run_train_critic_online(self, xt, xt_ref):
    #     """
    #     Function that evaluates once the critic neural network and returns the value of J(xt). At the same
    #     time, it trains the function approximator every number of time steps equal to the chosen batch size.
    #     :param xt: current time step states
    #     :param xt_ref: current time step reference states for the computation of the one-step cost function
    #     :return: Jt --> evaluation of the critic at the current time step
    #             dJt_dxt --> gradient of dJt/dxt necessary for actor weight update
    #     """
    #     self.xt = xt
    #     self.xt_ref = xt_ref
    #     self.ct = self.c_computation()
    #
    #     nn_input = tf.constant(np.array([np.vstack((self.xt, self.xt_ref))]).astype('float32'))
    #     with tf.GradientTape() as tape:
    #         tape.watch(self.model.trainable_variables)
    #         prediction = self.model(nn_input)
    #
    #     self.Jt = prediction.numpy()
    #     dJt_dW = tape.gradient(prediction, self.model.trainable_variables)
    #
    #     target = self.targets_computation_online()
    #     ec_critic_before = target - self.Jt_1
    #     dE_dJ = self.gamma * ec_critic_before
    #
    #     EC_critic_before = 0.5 * np.square(ec_critic_before)
    #     Ec_actor_before = 0.5 * np.square(self.Jt)
    #     print("CRITIC LOSS = ", EC_critic_before)
    #     print("ACTOR LOSS = ", Ec_actor_before)
    #     for count in range(len(dJt_dW)):
    #         update = dE_dJ * dJt_dW[count]
    #         self.model.trainable_variables[count].assign_sub(np.reshape(self.learning_rate * update,
    #                                                                     self.model.trainable_variables[count].shape))
    #
    #         # Implement WB_limits: the weights and biases can not have values whose absolute value exceeds WB_limits
    #         WB_variable = self.model.trainable_variables[count].numpy()
    #         WB_variable[WB_variable > self.WB_limits] = self.WB_limits
    #         WB_variable[WB_variable < -self.WB_limits] = -self.WB_limits
    #         self.model.trainable_variables[count].assign(WB_variable)
    #
    #     return self.Jt

    def evaluate_critic(self, xt, xt_ref):
        """
        Function that evaluates once the critic neural network and returns the value of J(xt).
        :param xt: current time step states
        :return: Jt --> evaluation of the critic at the current time step
                dJt_dxt --> gradient of the cost function with respect to the input (xt)
        """
        if self.input_include_reference:
            nn_input = tf.constant(np.array([np.vstack((xt, xt_ref))]).astype('float32'))
        else:
            nn_input = tf.constant(np.array([(xt)]).astype('float32'))
        with tf.GradientTape() as tape:
            tape.watch(nn_input)
            prediction = self.model(nn_input)

        Jt = prediction.numpy()
        dJt_dxt = tape.gradient(prediction, nn_input).numpy()
        if self.input_include_reference:
            dJt_dxt = np.reshape(dJt_dxt, [-1, 1])
            dJt_dxt = np.reshape(dJt_dxt[:self.number_states, :], [-1, 1])

        return Jt, dJt_dxt



    # def run_critic(self, xt, xt_ref):
    #     """
    #     Function which evaluates the critic only once and returns the cost function at the current time step
    #     :param xt: current time step states
    #     :param xt_ref: current time step reference states for the computation of the one-step cost function
    #     :return: Jt --> evaluation of the critic at the current time step
    #     """
    #     self.xt = xt
    #     self.store_states[self.time_step, :, 0] = np.reshape(self.xt, [4,])
    #     self.xt_ref = xt_ref
    #     self.c_computation()
    #
    #     nn_input = np.array([self.xt])
    #     Jt = self.model.predict(nn_input)
    #     self.store_J[0, self.time_step] = Jt
    #
    #     self.time_step += 1
    #     return Jt
    #
    # def train_critic_end(self):
    #     """
    #     Function which trains the model at the end of a full episode with the stored one-step cost function
    #     and the cost function
    #     :return:
    #     """
    #     targets = self.targets_computation_end()
    #     inputs = self.store_states[:-1, :, :]
    #
    #     self.model.fit(inputs, targets, batch_size=self.batch_size, epochs=self.epochs, verbose=2)

    def c_computation(self):
        """
        Computation of the one-step cost function with the received real and reference states.
        :return: ct --> current time step one-step cost function
        """
        ct = np.matmul(np.matmul((np.reshape(self.xt[self.indices_tracking_states, :], [-1, 1]) - self.xt_ref).T,
                                 self.Q), (np.reshape(self.xt[self.indices_tracking_states, :], [-1, 1]) - self.xt_ref))
        self.store_c[0, self.time_step] = ct[0]
        return ct

    # def targets_computation_end(self):
    #     """
    #     Computes the targets at the end of an episode with the stored one-step cost function (ct_1) and
    #     the stored cost functions (Jt).
    #     :return: targets --> targets of the complete episode
    #     """
    #     targets = np.reshape(self.store_c[0, :-1] + self.gamma * self.store_J[0, 1:], [-1, 1])
    #     return targets

    def targets_computation_online(self, *args):
        """
        Computes the target at the current time step with the one-step cost function of the previous
        time step and the current cost function.
        :return: target --> the target of the previous time step.
        """
        if len(args) == 0:
            target = np.reshape(-self.ct_1 - self.gamma * self.Jt, [-1, 1])
        elif len(args) == 2:
            Jt = args[0]
            ct_1 = args[1]
            target = np.reshape(-ct_1 - self.gamma * Jt, [-1, 1])
        else:
            Exception("Unexpected number of arguments")
            target = 0
        return target

    def update_critic_attributes(self):
        """
        The attributes that change with every time step are updated
        :return:
        """
        self.time_step += 1
        self.ct_1 = self.ct
        self.xt_1 = self.xt
        self.Jt_1 = self.Jt
        self.xt_ref_1 = self.xt_ref

    def restart_critic(self):
        """
        Restart the Critic.
        :return:
        """
        # Declaration of attributes regarding the states and rewards
        self.time_step = 0
        self.xt = None
        self.xt_1 = np.zeros((self.number_states, 1))
        self.xt_ref = None
        self.xt_ref_1 = np.zeros((self.number_tracking_states, 1))
        self.ct = 0
        self.ct_1 = 0
        self.Jt = 0
        self.Jt_1 = 0
        self.learning_rate = self.learning_rate_0

        # Store the states
        self.store_states = np.zeros((self.number_time_steps, self.number_states, 1))

        # Declaration of attributes related to the neural network
        self.dJt_dxt = None

        # Declaration of attributes related to the cost function
        self.store_J = np.zeros((1, self.number_time_steps))
        self.store_c = np.zeros((1, self.number_time_steps))

        # Declaration of storage arrays for online training
        self.store_inputs = np.zeros((self.batch_size, self.number_states, 1))
        self.store_targets = np.zeros((self.batch_size, 1))

        # Restart momentum and rmsprop
        for count in range(len(self.model.trainable_variables)):
            self.momentum_dict[count] = 0
            self.rmsprop_dict[count] = 0


if __name__ == "__main__":
    Q_weights = [1,1,1,1]
    selected_states = ['velocity', 'alpha', 'theta', 'q']
    number_time_steps = 500
    critic = Critic(Q_weights, selected_states, number_time_steps)
    a = np.array([[[1], [2], [3], [4]]])
    # a = np.array([[1], [2], [3], [4]])
    critic.critic_model()

