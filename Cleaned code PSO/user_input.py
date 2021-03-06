#!/usr/bin/env python
"""Provides class Simulation that initialises all the elements of the controllers and structures the execution of the
different components.

Simulation initialises all the components of the controller, namely the System, the Incremental Model, the Actor and the
Critic, as well as building the required Neural Networks. It counts with a method that executes in the required order
each of the controller elements during a complete iteration. Also, the "run_simulation" method runs the required
iterations, as well as restarting the simulation parameters.
"""

import random
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

# Inputs concerning the F16 system
folder = "Linear_system"
selected_states = ['velocity', 'alpha', 'theta', 'q']
# selected_states = ['alpha', 'q']
selected_output = ['alpha']
selected_input = ['ele']
discretisation_time = 0.01
input_magnitude_limits = 25
input_rate_limits = 60

# Inputs concerning the incremental model
number_time_steps = 4001

# Inputs concerning the critic
Q_weights = [10]
tracking_states = ['alpha']
indices_tracking_states = [selected_states.index(tracking_states[i]) for i in range(len(tracking_states))]
critic_start_training = -1
gamma = 0.99
critic_learning_rate = 5
critic_learning_rate_exponent_limit = 10
critic_layers = (20, 1)
critic_activations = ("tanh", "linear")
WB_limits = 30

# Inputs concerning the actor
actor_start_training = 5
actor_layers = (20, 1)
actor_activations = ('tanh', 'tanh')
actor_learning_rate = 1
actor_learning_rate_cascaded = 0.9
actor_learning_rate_exponent_limit = 10
type_PE = 'combined'
amplitude_3211 = 15
pulse_length_3211 = 5/discretisation_time
maximum_input = 25
maximum_q_rate = np.deg2rad(20)
cascaded_actor = False

# Inputs to the simulation
random.seed(1)
iterations = 1
# initial_states = np.array([[np.deg2rad(0)], [0]])
initial_states = np.array([[np.deg2rad(0)], [np.deg2rad(0)], [0], [0]])
time = np.arange(0, (number_time_steps + 1) * discretisation_time, discretisation_time)
reference_signals = np.reshape(np.deg2rad(10 * np.sin(0.5 * time)), [1, -1])
NN_initial = 130


# Tests
if len(selected_states) != initial_states.shape[0]:
    raise Exception("The number of states is not equal to the initial states.")
