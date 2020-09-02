#!/usr/bin/env python
"""Provides class Simulation that initialises all the elements of the controllers and structures the execution of the
different components.

Simulation initialises all the components of the controller, namely the System, the Incremental Model, the Actor and the
Critic, as well as building the required Neural Networks. It counts with a method that executes in the required order
each of the controller elements during a complete iteration.
"""


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

    def compute_performance(self):
        """
        Compute the Root Mean Square error of the tracking task
        :return:
        """
        x = self.system.store_states[self.indices_tracking_states[0], :min(len(self.time), self.time_step)]
        x_ref = self.reference_signals[0, :min(len(self.time), self.time_step)]

        self.RMSE = np.sqrt(np.sum(np.power((x-x_ref), 2)/x.shape[0]))
        print("The current RMSE is ", self.RMSE)


