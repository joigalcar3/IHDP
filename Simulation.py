#!/usr/bin/env python
"""Provides class Simulation that initialises all the elements of the controllers and structures the execution of the
different components.

Simulation initialises all the components of the controller, namely the System, the Incremental Model, the Actor and the
Critic, as well as building the required Neural Networks. It counts with a method that executes in the required order
each of the controller elements during a complete iteration.
"""


from Actor import Actor
from Critic import Critic
from System import F16System
from Incremental_model import IncrementalModel
from user_input import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')  # or can use 'TkAgg', whatever you have/prefer


"----------------------------------------------------------------------------------------------------------------------"
__author__ = "Jose Ignacio de Alvear Cardenas"
__copyright__ = "Copyright (C) 2020 Jose Ignacio"
__credits__ = []
__license__ = "MIT"
__version__ = "2.0.1"
__maintainer__ = "Jose Ignacio de Alvear Cardenas"
__email__ = "j.i.dealvearcardenas@student.tudelft.nl"
__status__ = "Production"
"----------------------------------------------------------------------------------------------------------------------"


class Simulation:
    def __init__(self, iterations, selected_inputs, selected_states, selected_outputs, number_time_steps,
                 initial_states, reference_signals, actor, critic, system, incremental_model,
                 discretisation_time=0.5, tracking_states=['alpha']):
        # Attributes regarding the simulation
        self.iterations = iterations
        self.number_time_steps = number_time_steps
        self.time_step = 0
        self.discretisation_time = discretisation_time
        self.time = list(np.arange(0, self.number_time_steps * self.discretisation_time, self.discretisation_time))
        self.iteration = 0

        # Attributes regarding the system
        self.selected_inputs = selected_inputs
        self.selected_states = selected_states
        self.selected_outputs = selected_outputs
        self.initial_states = initial_states
        self.tracking_states = tracking_states
        self.indices_tracking_states = [self.selected_states.index(self.tracking_states[i])
                                        for i in range(len(self.tracking_states))]

        self.reference_signals = reference_signals

        # Initialise all the elements of the simulation
        self.actor = actor
        self.critic = critic
        self.system = system
        self.incremental_model = incremental_model

        # Cyclic parameters
        self.xt = self.initial_states
        self.xt_track = np.reshape(self.xt[self.indices_tracking_states, self.time_step], [-1, 1])
        self.xt_ref = np.reshape(self.reference_signals[:, self.time_step], [-1, 1])
        self.store_xt1 = np.zeros((len(self.selected_states), self.number_time_steps))

        # Prepare system
        self.system.initialise_system(self.xt, self.number_time_steps)

        # Initialise the NN
        self.actor.build_actor_model()
        self.critic.build_critic_model()

        # Initialise the error
        self.RMSE = 0

    def run_simulation(self):
        """
        Runs the complete simulation by executing each iteration, restarting the controller components, as well as
        the simulation attributes
        :return:
        """
        self.run_iteration()

        self.compute_performance()
        print(self.RMSE)
        return self.RMSE

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
            xt_ref1 = np.reshape(self.reference_signals[:, self.time_step + 1], [-1, 1])
            # _ = self.critic.run_train_critic_online_adaptive_alpha(self.xt, self.xt_ref)
            # _ = self.critic.run_train_critic_online_adam(self.xt, self.xt_ref, self.iteration)
            _ = self.critic.run_train_critic_online_alpha_decay(self.xt, self.xt_ref)

            # Evaluate the critic
            # self.critic.train_critic_replay_adam(10, self.iteration)
            Jt1, dJt1_dxt1 = self.critic.evaluate_critic(np.reshape(xt1_est, [-1, 1]), xt_ref1)

            # Train the actor
            # self.actor.train_actor_online_adaptive_alpha(Jt1, dJt1_dxt1, G,
            #                                    self.incremental_model, self.critic, xt_ref1)
            # self.actor.train_actor_online_adam(Jt1, dJt1_dxt1, G,
            #                                              self.incremental_model, self.critic, xt_ref1)
            self.actor.train_actor_online_alpha_decay(Jt1, dJt1_dxt1, G,
                                                      self.incremental_model, self.critic, xt_ref1)

            # Update models attributes
            self.system.update_system_attributes()
            self.incremental_model.update_incremental_model_attributes()
            self.critic.update_critic_attributes()
            self.actor.update_actor_attributes()

            self.time_step += 1
            self.xt = xt1
            self.xt_track = np.reshape(xt1[self.indices_tracking_states, :], [-1, 1])

            if self.time_step % 500 == 0:
                max_x_frame = 0
                self.plot_state_results(max_x_frame)
                self.plot_input_results(max_x_frame)
                self.plot_training_critic(max_x_frame)
                self.plot_weights_critic(max_x_frame)
                self.plot_weights_actor(max_x_frame)
                self.plot_q_difference(max_x_frame)
                self.compute_performance()

    def plot_state_results(self, max_x_frame):
        """
        Plots the desired real and reference states to be tracked to assess the performance of the complete controller.
        :return:
        """
        plt.figure(self.iteration)
        plt.clf()
        n_rows = min(len(self.tracking_states), 3)
        if len(self.tracking_states) > 3:
            n_cols = 2
        else:
            n_cols = 1

        for i in range(len(self.indices_tracking_states)):
            plt.subplot(n_rows, n_cols, i+1)
            plt.plot(self.time[:self.time_step + max_x_frame], self.reference_signals[i, :min(len(self.time), self.time_step + max_x_frame)], 'r', label='$x_t^{ref}$')
            plt.plot(self.time[:self.time_step + max_x_frame], self.system.store_states[self.indices_tracking_states[i], :min(len(self.time), self.time_step + max_x_frame)], 'b', label='$x_t$')
            plt.xlabel("Time [s]")
            plt.ylabel("Angle of attack [rad]")
            plt.legend()
            plt.grid(True)
            plt.show()

    def plot_input_results(self, max_x_frame):
        """
        Plots the input to the system to verify that the limits of the platform are not exceeded.
        :param max_x_frame: the number of time steps ahead of the current time-step to be plotted.
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
            plt.plot(self.time[:self.time_step + max_x_frame], self.system.store_input[i, :min(len(self.time), self.time_step + max_x_frame)], 'g', label='$u_t$')
            plt.xlabel("Time [s]")
            plt.ylabel("Elevator deflection [deg]")
            plt.legend()
            plt.grid(True)
            plt.show()

    def plot_training_critic(self, max_x_frame):
        """
        Plots the evolution of the target vs Jt_1, the evolution of c and the evolution of the loss function Ec
        :param max_x_frame: the number of time steps ahead of the current time-step to be plotted.
        :return:
        """
        c = np.hstack((np.zeros((1, 1)), self.critic.store_c[:, :-1]))[:, :min(len(self.time), self.time_step + max_x_frame)]
        targets = c + self.critic.gamma * self.critic.store_J[:, :min(len(self.time), self.time_step + max_x_frame)]
        targets = -targets.T
        c = c.T
        Jt_1 = self.critic.store_J_1[:, :min(len(self.time), self.time_step + max_x_frame)].T
        time = np.reshape(self.time[:self.time_step + max_x_frame], [1, -1]).T

        plt.figure(2 * self.iterations + self.iteration)
        plt.clf()
        plt.subplot(3, 1, 1)
        plt.plot(time, -targets, 'b', label='$J_{t} + c_{t-1}$')
        plt.plot(time, Jt_1, 'r', label='$J_{t-1}$')
        plt.xlabel("Time [s]")
        plt.ylabel("True cost-to-go [-],  Target [-]")
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(time, c, 'b')
        plt.xlabel("Time [s]")
        plt.ylabel("One-step cost function [-]")
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(time, 0.5 * np.square(targets + Jt_1), 'b')
        plt.xlabel("Time [s]")
        plt.ylabel("Critic loss function [-]")
        plt.legend()
        plt.grid(True)
        plt.pause(0.0001)
        plt.show()

    def plot_weights_critic(self, max_x_frame):
        """
        Plots the evolution of the weights of the critic with respect to time.
        :param max_x_frame: the number of time steps ahead of the current time-step to be plotted.
        :return:
        """
        n_rows =len(self.critic.store_weights.keys())
        time = np.reshape(self.time[:self.time_step + max_x_frame], [1, -1]).T

        plt.figure(3 * self.iterations + self.iteration)
        plt.clf()
        for weight in range(n_rows):
            W = self.critic.store_weights['W' + str(weight+1)]
            plt.subplot(n_rows, 1, weight+1)
            n_weights = W.shape[0]
            for i in range(n_weights):
                plt.plot(time, W[i, :min(len(self.time), self.time_step + max_x_frame)].T)
            plt.grid(True)
            plt.xlabel("Time [s]")
            plt.ylabel("Critic weights layer " + str(weight + 1) + "-" + str(weight + 2) + " [-]")
            plt.pause(0.0001)
        plt.tight_layout()
        plt.show()

    def plot_weights_actor(self, max_x_frame):
        """
        Plots the evolution of the weights of the actor with respect to time.
        :param max_x_frame: the number of time steps ahead of the current time-step to be plotted.
        :return:
        """
        n_rows =len(self.actor.store_weights.keys())
        time = np.reshape(self.time[:self.time_step + max_x_frame], [1, -1]).T

        plt.figure(4 * self.iterations + self.iteration)
        plt.clf()
        for weight in range(n_rows):
            W = self.actor.store_weights['W' + str(weight+1)]
            plt.subplot(n_rows, 1, weight+1)
            n_weights = W.shape[0]
            for i in range(n_weights):
                plt.plot(time, W[i, :min(len(self.time), self.time_step + max_x_frame)].T)
            plt.grid(True)
            plt.xlabel("Time [s]")
            plt.ylabel("Actor 1 weights layer " + str(weight + 1) + "-" + str(weight + 2) + " [-]")
        plt.tight_layout()

        if self.actor.cascaded_actor:
            plt.figure(5 * self.iterations + self.iteration)
            plt.clf()
            for weight in range(n_rows):
                W = self.actor.store_weights_q['W' + str(weight + 1)]
                plt.subplot(n_rows, 1, weight + 1)
                n_weights = W.shape[0]
                for i in range(n_weights):
                    plt.plot(time, W[i, :min(len(self.time), self.time_step + max_x_frame)].T)
                plt.grid(True)
                plt.xlabel("Time [s]")
                plt.ylabel("Actor 2 weights layer " + str(weight + 1) + "-" + str(weight + 2) + " [-]")
            plt.tight_layout()
        plt.show()

    def plot_q_difference(self, max_x_frame):
        """
        Plos the vehicle pitch rate and the reference pitch rate generated by the cascaded actor network
        :param max_x_frame: the number of time steps ahead of the current time-step to be plotted.
        :return:
        """
        time = np.reshape(self.time[:self.time_step + max_x_frame], [1, -1]).T
        plt.figure(6 * self.iterations + self.iteration)
        plt.clf()
        q = np.reshape(self.system.store_states[1, :min(len(self.time), self.time_step + max_x_frame)], [-1, 1])
        q_ref = np.reshape(self.actor.store_q[0, :min(len(self.time), self.time_step + max_x_frame)], [-1, 1])
        plt.plot(time, q, label='$q_t$')
        plt.plot(time, q_ref, label='$q_t^{ref}$')
        plt.xlabel("Time [s]")
        plt.ylabel("Pitch rate [rad/s]")
        plt.legend()
        plt.grid(True)
        plt.pause(0.0001)
        plt.show()

    def compute_performance(self):
        """
        Compute the Root Mean Square error of the tracking task
        :return:
        """
        x = self.system.store_states[self.indices_tracking_states[0], :min(len(self.time), self.time_step)]
        x_ref = self.reference_signals[0, :min(len(self.time), self.time_step)]

        self.RMSE = np.sqrt(np.sum(np.power((x-x_ref), 2)/x.shape[0]))
        print("The current RMSE is ", self.RMSE)

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
    # Initialise all the elements of the simulation
    # RMSE_lst = []
    # actor_learning_rate_range = np.arange(1, 20, 1)
    # for actor_learning_rate in actor_learning_rate_range:
    actor = Actor(selected_input, selected_states, tracking_states, indices_tracking_states,
                 number_time_steps, actor_start_training, actor_layers, actor_activations,
                 actor_learning_rate, actor_learning_rate_cascaded, actor_learning_rate_exponent_limit,
                  type_PE, amplitude_3211, pulse_length_3211, WB_limits,
                 maximum_input, maximum_q_rate, cascaded_actor, NN_initial)

    critic = Critic(Q_weights, selected_states, tracking_states, indices_tracking_states, number_time_steps,
                 critic_start_training, gamma, critic_learning_rate, critic_learning_rate_exponent_limit, critic_layers,
                 critic_activations, WB_limits, NN_initial)

    system = F16System(folder, selected_states, selected_output, selected_input, discretisation_time,
                 input_magnitude_limits, input_rate_limits)

    incremental_model = IncrementalModel(selected_states, selected_input, number_time_steps, discretisation_time,
                 input_magnitude_limits, input_rate_limits)

    # Initialise the simulation
    simulation = Simulation(iterations, selected_input, selected_states, selected_output, number_time_steps,
                 initial_states, reference_signals, actor, critic, system, incremental_model,
                 discretisation_time, tracking_states)

    # Run the simulation
    RMSE = simulation.run_simulation()
        # RMSE_lst.append(RMSE)
        # print(RMSE_lst)

    # plt.figure(100000)
    # plt.plot(actor_learning_rate_range, RMSE_lst)
    # plt.xlabel("Actor learning rate [-]")
    # plt.ylabel("RMSE [-]")
    # plt.grid(True)
    # plt.show()



