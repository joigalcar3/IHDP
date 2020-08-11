#!/usr/bin/env python
"""Provides class Critic with the function approximator (NN) of the Critic

Critic creates the Neural Network model with Tensorflow and it can train the network online or at the
end of the episode. The user can decide the number of layers, the number of neurons, the batch size and the number
of epochs and activation functions. If trained online, the algorithm trains the Network after the number of collected
data points equals the batch size. This means that if the batch size is 10, then the NN is updated every 10 time steps.
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from datetime import datetime
import keras
import tensorboard
from Actor import Actor
from Critic import Critic
from System import System, F16System
from Incremental_model import IncrementalModel
import matplotlib.pyplot as plt
import numpy as np
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


class Simulation:
    def __init__(self, iterations, selected_inputs, selected_states, selected_outputs, number_time_steps, Q_weights,
                 folder, initial_states, reference_signals, discretization_time=0.5, tracking_states=['alpha']):
        # Attributes regarding the simulation
        self.iterations = iterations
        self.number_time_steps = number_time_steps
        self.time_step = 0
        self.discretization_time = discretization_time
        self.time = list(np.arange(0, self.number_time_steps * self.discretization_time, self.discretization_time))

        # Attributes regarding the system
        self.folder = folder
        self.selected_inputs = selected_inputs
        self.selected_states = selected_states
        self.selected_outputs = selected_outputs
        self.initial_states = initial_states
        self.tracking_states = tracking_states
        self.indices_tracking_states = [self.selected_states.index(self.tracking_states[i])
                                        for i in range(len(self.tracking_states))]

        self.reference_signals = reference_signals

        # Attributes regarding the cost function
        self.Q_weights = Q_weights

        # Initialise all the elements of the simulation
        self.actor = Actor(self.selected_inputs, self.tracking_states, self.number_time_steps)
        self.critic = Critic(self.Q_weights, self.tracking_states, self.number_time_steps)
        self.critic_incremental = Critic(self.Q_weights, self.tracking_states, self.number_time_steps)
        self.system = F16System(self.folder, self.selected_states, self.selected_outputs, self.selected_inputs)
        self.incremental_model = IncrementalModel(self.selected_states, self.selected_inputs, self.number_time_steps)

        # Cyclic parameters
        self.xt = self.initial_states
        self.xt_track = np.reshape(self.xt[self.indices_tracking_states, self.time_step], [-1, 1])
        self.xt_ref = np.reshape(self.reference_signals[:, self.time_step], [-1, 1])

        # Prepare system
        self.system.import_linear_system()
        self.system.simplify_system()
        self.system.initialise_system(self.xt, self.number_time_steps)

        # Initialise the NN
        self.actor.build_actor_model()
        self.critic.build_critic_model()

    def run_iteration(self):
        while self.time_step < self.number_time_steps:
            print(self.time_step)
            # Obtain the input from the actor
            ut = self.actor.run_actor_online(self.xt_track, self.xt_ref)

            # Run the system
            xt1, _ = self.system.run_step(ut)

            # Identify the incremental model
            G = self.incremental_model.identify_incremental_model_LS(self.xt, ut)

            # Run the incremental model
            xt1_est = self.incremental_model.evaluate_incremental_model()

            # Run and train the critic model
            _, _ = self.critic.run_train_critic_online(self.xt_track, self.xt_ref)

            # Update the actor
            Jt1, dJt1_dxt1 = self.critic.evaluate_critic(np.reshape(xt1_est[self.indices_tracking_states, :], [-1, 1]))
            self.actor.train_actor_online(Jt1, dJt1_dxt1, G[self.indices_tracking_states, :])

            self.time_step += 1
            self.xt = xt1
            self.xt_track = np.reshape(xt1[self.indices_tracking_states, :], [-1, 1])
            self.xt_ref = np.reshape(self.reference_signals[:, self.time_step], [-1, 1])
        self.plot_state_results()

    def plot_state_results(self):
        plt.figure(1)
        n_rows = min(len(self.tracking_states), 3)
        if len(self.tracking_states) > 3:
            n_cols = 2
        else:
            n_cols = 1

        for i in range(len(self.tracking_states)):
            plt.subplot(n_rows, n_cols, i)
            plt.plot(self.time, self.reference_signals[i, :], 'r', label='Reference state')
            plt.plot(self.time, self.system.store_states[i, :], 'b', label='Real state')
            plt.legend()
            plt.grid(True)
            plt.show()

if __name__ == "__main__":
    random.seed(1)
    iterations = 50
    selected_inputs = ['ele']
    selected_states = ['velocity', 'alpha', 'theta', 'q']
    selected_outputs = ['alpha']
    number_time_steps = 500
    Q_weights = [1]
    folder = "Linear_system"
    initial_states = np.array([[0], [0], [0], [0]])
    discretization_time = 0.5
    time = np.arange(0, number_time_steps * discretization_time, discretization_time)
    reference_signals = np.reshape(5 * np.sin(0.04*time), [1, -1])

    simulation = Simulation(iterations, selected_inputs, selected_states, selected_outputs, number_time_steps, Q_weights,
                 folder, initial_states, reference_signals, discretization_time=discretization_time)
    simulation.run_iteration()
    print('hola')

