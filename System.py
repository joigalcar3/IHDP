#!/usr/bin/env python
"""Provides classes System and F16System classes to simulate the external system

System provides a simple template and common attributes among all the possible systems.
F16System provides the additional attributes and methods in order to use the linear F-16
aircraft system. It simplifies the system matrices according to the chosen states, inputs and
outputs, as well as implementing the methods required for initialisation and running the simulation.
"""

from scipy.io import loadmat
from scipy.signal import *
import json
import numpy as np
from abc import abstractmethod

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


class System:
    def __init__(self):
        # Storing arrays
        self.store_states = None
        self.store_input = None

        # Current state, input and output
        self.x0 = None
        self.xt = None
        self.xt1 = None
        self.yt = None
        self.ut = None

        # Time information
        self.number_time_steps = None
        self.time_step = None

    @abstractmethod
    def initialise_system(self, x0, number_steps):
        pass

    @abstractmethod
    def run_step(self, ut):
        pass


class F16System(System):
    """
    System block of the F-16 aircraft that acts as the real system. At the moment, it only takes the matrices of a
    linear system.
    """

    def __init__(self, folder, selected_states, selected_output, selected_input, discretisation_time=0.5,
                 input_magnitude_limits=25, input_rate_limits=60):
        super().__init__()
        self.folder = folder  # Folder where the data is located
        self.discretisation_time = discretisation_time

        # Selected data for the system
        self.selected_states = selected_states
        self.selected_output = selected_output
        self.selected_input = selected_input

        # Store the number of inputs, states and outputs
        self.number_inputs = len(self.selected_input)
        self.number_outputs = len(self.selected_output)
        self.number_states = len(self.selected_states)

        # Original matrices of the system
        self.A = None
        self.B = None
        self.C = None
        self.D = None

        # Processed matrices of the system
        self.filt_A = None
        self.filt_B = None
        self.filt_C = None
        self.filt_D = None

        # Limitations of the system
        self.input_magnitude_limits = input_magnitude_limits
        self.input_rate_limits = input_rate_limits

    def import_linear_system(self):
        """
        Retrieves the stored linearised matrices obtained from Matlab
        :return:
        """
        x = loadmat(self.folder + '/A.mat')
        self.A = x['A_lo']

        x = loadmat(self.folder + '/B.mat')
        self.B = x['B_lo']

        x = loadmat(self.folder + '/C.mat')
        self.C = x['C_lo']

        x = loadmat(self.folder + '/D.mat')
        self.D = x['D_lo']

    def simplify_system(self):
        """
        Function which simplifies the F-16 matrices. The filtered matrices are stored as part of the object
        :return:
        """

        # Create dictionaries with the information from the system
        states_rows = self.create_dictionary('states')
        selected_rows_states = np.array([states_rows[state] for state in self.selected_states])
        output_rows = self.create_dictionary('output')
        selected_rows_output = np.array([output_rows[output] for output in self.selected_output])
        input_rows = self.create_dictionary('input')
        selected_rows_input = np.array([input_rows[input_var] for input_var in self.selected_input])

        # Create the new system and initial condition
        self.filt_A = self.A[selected_rows_states[:, None], selected_rows_states]
        self.filt_B = self.A[selected_rows_states[:, None], 12 + selected_rows_input] + \
                      self.B[selected_rows_states[:, None], selected_rows_input]
        self.filt_C = self.C[selected_rows_output[:, None], selected_rows_states]
        self.filt_D = self.C[selected_rows_output[:, None], 12 + selected_rows_input] + \
                      self.D[selected_rows_output[:, None], selected_rows_input]

    def create_dictionary(self, file_name):
        """
        Creates dictionaries from the available states, inputs and outputs
        :param file_name: name of the file to be read
        :return: rows --> dictionary with the rows used of the input/state/output vectors
        """
        full_name = self.folder + '/keySet_' + file_name + '.txt'
        with open(full_name, 'r') as f:
            keySet = json.loads(f.read())
        rows = dict(zip(keySet, range(len(keySet))))
        return rows

    def initialise_system(self, x0, number_time_steps):
        """
        Initialises the F-16 aircraft dynamics
        :param x0: the initial states
        :param number_time_steps: the number of time steps within an iteration
        :return:
        """
        # Import the stored system
        self.import_linear_system()

        # Simplify the system with the chosen states
        self.simplify_system()

        # Store the number of time steps
        self.number_time_steps = number_time_steps
        self.time_step = 0

        # Discretise the system according to the discretisation time
        (self.filt_A, self.filt_B, self.filt_C, self.filt_D, _) = cont2discrete((self.filt_A, self.filt_B, self.filt_C,
                                                                                 self.filt_D),
                                                                                self.discretisation_time)

        self.store_states = np.zeros((self.number_states, self.number_time_steps + 1))
        self.store_input = np.zeros((self.number_inputs, self.number_time_steps))

        self.x0 = x0
        self.xt = x0
        self.store_states[:, self.time_step] = np.reshape(self.xt, [-1, ])

    def run_step(self, ut_0):
        """
        Runs one time step of the iteration.
        :param ut: input to the system
        :return: xt1 --> the next time step state
        """
        if self.time_step != 0:
            ut_1 = self.store_input[:, self.time_step - 1]
        else:
            ut_1 = ut_0
        ut = max(min(max(min(ut_0,
                             np.reshape(np.array([ut_1 + self.input_rate_limits * self.discretisation_time]), [-1, 1])),
                         np.reshape(np.array([ut_1 - self.input_rate_limits * self.discretisation_time]), [-1, 1])),
                     np.array([[self.input_magnitude_limits]])),
                 - np.array([[self.input_magnitude_limits]]))

        self.xt1 = np.matmul(self.filt_A, np.reshape(self.xt, [-1, 1])) + np.matmul(self.filt_B, np.reshape(ut, [-1, 1]))

        self.store_input[:, self.time_step] = np.reshape(ut, [ut.shape[0]])
        self.store_states[:, self.time_step + 1] = np.reshape(self.xt1, [self.xt1.shape[0]])

        return self.xt1

    def update_system_attributes(self):
        """
        The attributes that change with every time step are updated
        :return:
        """
        self.xt = self.xt1
        self.time_step += 1
