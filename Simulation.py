#!/usr/bin/env python
"""Provides class Simulation that initialises all the elements of the controllers and structures the execution of the
different components.

Simulation initialises all the components of the controller, namely the System, the Incremental Model, the Actor and the
Critic, as well as building the required Neural Networks. It counts with a method that executes in the required order
each of the controller elements during a complete iteration. Also, the "run_simulation" method runs the required
iterations, as well as restarting the simulation parameters.
"""


from Actor import Actor
from Critic import Critic
from System import System, F16System
from Incremental_model import IncrementalModel
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib as mpl
mpl.use('TkAgg')  # or can use 'TkAgg', whatever you have/prefer


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
                 folder, initial_states, reference_signals, discretisation_time=0.5, tracking_states=['alpha']):
        # Attributes regarding the simulation
        self.iterations = iterations
        self.number_time_steps = number_time_steps
        self.time_step = 0
        self.discretisation_time = discretisation_time
        self.time = list(np.arange(0, self.number_time_steps * self.discretisation_time, self.discretisation_time))
        self.iteration = 0

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
        self.actor = Actor(self.selected_inputs, self.selected_states, self.tracking_states, self.number_time_steps)
        self.critic = Critic(self.Q_weights, self.selected_states, self.tracking_states, self.indices_tracking_states,
                             self.number_time_steps)
        self.system = F16System(self.folder, self.selected_states, self.selected_outputs, self.selected_inputs,
                                discretisation_time=discretisation_time)
        self.incremental_model = IncrementalModel(self.selected_states, self.selected_inputs, self.number_time_steps,
                                                  discretisation_time=discretisation_time)

        # Cyclic parameters
        self.xt = self.initial_states
        self.xt_track = np.reshape(self.xt[self.indices_tracking_states, self.time_step], [-1, 1])
        self.xt_ref = np.reshape(self.reference_signals[:, self.time_step], [-1, 1])

        # Prepare system
        self.system.initialise_system(self.xt, self.number_time_steps)

        # Initialise the NN
        self.actor.build_actor_model()
        self.critic.build_critic_model()

    def run_simulation(self):
        """
        Runs the complete simulation by executing each iteration, restarting the controller components, as well as
        the simulation attributes
        :return:
        """
        while self.iteration < self.iterations:
            # Run a complete iteration
            self.run_iteration()

            # Restart the elements of an iteration
            self.restart_iteration()
            self.time_step = 0

            # Restart cyclic parameters
            self.xt = self.initial_states
            self.xt_track = np.reshape(self.xt[self.indices_tracking_states, self.time_step], [-1, 1])

            self.iteration += 1

    def run_iteration(self):
        """
        Core of the program that runs a complete iteration, evaluating and training the controller components in the
        correct order.
        :return:
        """
        while self.time_step < self.number_time_steps:
            print(self.time_step)

            # Retrieve the reference signal
            self.xt_ref = np.reshape(self.reference_signals[:, self.time_step], [-1, 1])

            # Obtain the input from the actor
            ut = self.actor.run_actor_online(self.xt, self.xt_ref)

            # Run the system
            xt1 = self.system.run_step(ut)

            # Identify the incremental model
            G = self.incremental_model.identify_incremental_model_LS(self.xt, ut)

            # Run the incremental model
            xt1_est = self.incremental_model.evaluate_incremental_model()

            # Run and train the critic model
            _ = self.critic.run_train_critic_online_adaptive_alpha(self.xt, self.xt_ref)

            # Evaluate the critic
            xt_ref1 = np.reshape(self.reference_signals[:, self.time_step + 1], [-1, 1])
            Jt1, dJt1_dxt1 = self.critic.evaluate_critic(np.reshape(xt1_est, [-1, 1]), xt_ref1)

            # Train the actor
            # if self.time_step > 20:
            self.actor.train_actor_online_adaptive_alpha(Jt1, dJt1_dxt1, G,
                                                         self.incremental_model, self.critic, xt_ref1)

            # Update models attributes
            self.system.update_system_attributes()
            self.incremental_model.update_incremental_model_attributes()
            self.critic.update_critic_attributes()
            self.actor.update_actor_attributes()

            self.time_step += 1
            self.xt = xt1
            self.xt_track = np.reshape(xt1[self.indices_tracking_states, :], [-1, 1])

            max_x_frame = 0
            self.plot_state_results(max_x_frame)
            self.plot_input_results(max_x_frame)
            self.plot_training_critic(max_x_frame)

    def plot_state_results(self, max_x_frame):
        """
        Plots the desired real and reference states to be tracked to assess the performance of the complete controller.
        :return:
        """
        fig = plt.figure(self.iteration)
        plt.clf()
        n_rows = min(len(self.tracking_states), 3)
        if len(self.tracking_states) > 3:
            n_cols = 2
        else:
            n_cols = 1

        for i in range(len(self.indices_tracking_states)):
            plt.subplot(n_rows, n_cols, i+1)
            plt.plot(self.time[:self.time_step + max_x_frame], self.reference_signals[i, :min(len(self.time), self.time_step + max_x_frame)], 'r', label='Reference state')
            plt.plot(self.time[:self.time_step + max_x_frame], self.system.store_states[self.indices_tracking_states[i], :min(len(self.time), self.time_step + max_x_frame)], 'b', label='Real state')
            plt.legend()
            plt.grid(True)
            # plt.pause(0.001)
            plt.show()

    def plot_input_results(self, max_x_frame):
        """
        Plots the input to the system to verify that the limits of the platform are not exceeded.
        :return:
        """
        plt.figure(self.iterations + self.iteration)
        plt.clf()
        n_rows = min(len(self.selected_inputs), 3)
        if len(self.selected_inputs) > 3:
            n_cols = 2
        else:
            n_cols = 1

        for i in range(len(self.selected_inputs)):
            plt.subplot(n_rows, n_cols, i+1)
            plt.plot(self.time[:self.time_step + max_x_frame], self.system.store_input[i, :min(len(self.time), self.time_step + max_x_frame)], 'g', label='Input system')
            plt.plot(self.time[:self.time_step + max_x_frame], self.system.store_input[i, :min(len(self.time), self.time_step + max_x_frame)], 'y', label='Input incremental model')
            plt.legend()
            plt.grid(True)
            # plt.pause(0.001)
            plt.show()

    def plot_training_critic(self, max_x_frame):
        """
        Plots the evolution of the target vs Jt_1, the evolution of c and the evolution of the loss function Ec
        :return:
        """
        c = np.hstack((np.zeros((1, 1)), self.critic.store_c[:, :-1]))[:, :min(len(self.time), self.time_step + max_x_frame)]
        targets = c + self.critic.gamma * self.critic.store_J[:, :min(len(self.time), self.time_step + max_x_frame)]
        targets = targets.T
        c = c.T
        Jt_1 = np.hstack((np.zeros((1, 1)), self.critic.store_J[:, :-1]))[:, :min(len(self.time), self.time_step + max_x_frame)].T
        time = np.reshape(self.time[:self.time_step + max_x_frame], [1, -1]).T

        plt.figure(2 * self.iterations + self.iteration)
        plt.clf()
        plt.subplot(3, 1, 1)
        plt.plot(time, targets, 'b', label='Targets')
        plt.plot(time, Jt_1, 'r', label='Jt_1')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(time, c, 'b', label='c')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(time, 0.5 * np.square(targets - Jt_1), 'b', label='Ec')
        plt.legend()
        plt.grid(True)
        plt.pause(0.0001)
        plt.show()

    def restart_iteration(self):
        """
        Restarts the different components of the controller in order to start a new iteration
        :return:
        """
        # Prepare system
        self.system.initialise_system(self.initial_states, self.number_time_steps)

        # Restart incremental system
        self.incremental_model.restart_incremental_model()

        # Restart the Critic
        self.critic.restart_critic()

        # Restart the Actor
        self.actor.restart_actor()


if __name__ == "__main__":
    random.seed(1)
    iterations = 50
    selected_inputs = ['ele']
    selected_states = ['velocity', 'alpha', 'theta', 'q']
    selected_outputs = ['alpha']
    number_time_steps = 500
    Q_weights = [1]
    folder = "Linear_system"
    initial_states = np.array([[0], [np.deg2rad(-1)], [0], [0]])
    discretisation_time = 0.1
    time = np.arange(0, number_time_steps * discretisation_time, discretisation_time)
    reference_signals = np.reshape(np.deg2rad(5 * np.sin(0.04*time)), [1, -1])

    simulation = Simulation(iterations, selected_inputs, selected_states, selected_outputs, number_time_steps, Q_weights,
                 folder, initial_states, reference_signals, discretisation_time=discretisation_time)
    simulation.run_simulation()


