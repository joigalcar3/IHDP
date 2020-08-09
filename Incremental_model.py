#!/usr/bin/env python
"""Provides class IncrementalModel in order to identify the system

IncrementalModel computes the A and x matrices required for the system identification,
computes the F and G matrices required for the incremental model and evaluates the
identified model in order to provide the estimates states at the next time step.
"""

import numpy as np


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


class IncrementalModel:

    def __init__(self, selected_states, selected_input, number_time_steps):
        # Define the inputs to the incremental model
        self.xt_1 = None
        self.xt = None
        self.ut_1 = None
        self.ut = None
        self.delta_xt = None
        self.delta_ut = None
        self.xt1_est = None

        # Define the data window size
        self.number_time_steps = number_time_steps
        self.number_states = len(selected_states)
        self.number_inputs = len(selected_input)
        self.L = 2 * (self.number_inputs+self.number_states)
        self.store_delta_xt = np.zeros((self.number_states, self.number_time_steps))
        self.store_delta_ut = np.zeros((self.number_inputs, self.number_time_steps))

        # Define the system identification matrices
        self.F = None
        self.G = None

        # Define the time variable
        self.time_step = 0

    def build_A_LS_matrix(self):
        """
        Builds the A matrix required in the Least Squares online identification method
        :return: A_LS_matrix --> A matrix of the LS
        """
        if self.time_step >= self.L:
            x_component = np.flip(self.store_delta_xt[:, self.time_step-self.L:self.time_step], 1).T
            u_component = np.flip(self.store_delta_ut[:, self.time_step-self.L:self.time_step], 1).T
        else:
            x_component_1 = np.flip(self.store_delta_xt[:, :self.time_step], 1).T
            x_component_2 = np.zeros((self.L-self.time_step, self.number_states))
            x_component = np.vstack((x_component_1, x_component_2))

            u_component_1 = np.flip(self.store_delta_ut[:, :self.time_step], 1).T
            u_component_2 = np.zeros((self.L - self.time_step, self.number_inputs))
            u_component = np.vstack((u_component_1, u_component_2))
        A_LS_matrix = np.hstack((x_component, u_component))
        return A_LS_matrix

    def build_x_LS_vector(self):
        """
        Builds the x vector required in the Least Squares of the online system identification
        :return: x_LS_vector--> x vector required in the LS
        """
        # Computation and storage of the gradients
        self.delta_xt = self.xt - self.xt_1
        self.delta_ut = self.ut - self.ut_1
        self.store_delta_xt[:, self.time_step] = np.reshape(self.delta_xt, [self.delta_xt.shape[0]])
        self.store_delta_ut[:, self.time_step] = np.reshape(self.delta_ut, [self.delta_ut.shape[0]])

        x_LS_vector = np.flip(self.store_delta_xt[:, self.time_step-self.L+1:], 1).T
        return x_LS_vector

    def identify_incremental_model_LS(self, xt, ut):
        """
        Computes the F and G matrices of the system identification
        :param xt: current time step states
        :param ut: current time step input
        :return:
        """
        # Store the input variables
        self.xt = xt
        self.ut = ut

        # Obtain the A matrix and the x vector
        A_LS_matrix = self.build_A_LS_matrix()
        x_LS_vector = self.build_x_LS_vector()
        identified_matrices = np.matmul(np.matmul(np.linalg.inv(np.matmul(A_LS_matrix.T, A_LS_matrix)), A_LS_matrix.T),
                                        x_LS_vector).T
        self.F = identified_matrices[:, :self.number_states]
        self.G = identified_matrices[:, self.number_states:]

    def evaluate_incremental_model(self):
        """
        Estimates the next time step states
        :return: xt1_est --> next time step state estimation
        """
        # Estimate the next time step states
        self.xt1_est = self.xt + np.matmul(self.F, self.delta_xt) + np.matmul(self.G, self.delta_ut)

        # Update the object state and input variables
        self.xt_1 = self.xt
        self.ut_1 = self.ut
        self.time_step += 1
        return self.xt1_est


