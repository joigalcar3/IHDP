#!/usr/bin/env python
"""Provides the optimisation via Particle Swarm Optimisation in order to tune the most critical parameters for the
performance.
"""


from Actor import Actor
from Critic import Critic
from System import F16System
from Incremental_model import IncrementalModel
from Simulation import Simulation
from user_input import *
import pygmo as pg
from datetime import datetime


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


class PSO_Problem:
    def __init__(self, dim):
        self.dim = dim

    def fitness(self, x):
        critic_learning_rate = x[0]
        actor_learning_rate = x[1]
        actor_learning_rate_cascaded = x[2]
        Q_weights = [x[3]]
        gamma = x[4]

        # Initialise all the elements of the simulation
        actor = Actor(selected_input, selected_states, tracking_states, indices_tracking_states,
                      number_time_steps, actor_start_training, actor_layers, actor_activations,
                      actor_learning_rate, actor_learning_rate_cascaded, actor_learning_rate_exponent_limit,
                      type_PE, amplitude_3211, pulse_length_3211, WB_limits,
                      maximum_input, maximum_q_rate, cascaded_actor, NN_initial)

        critic = Critic(Q_weights, selected_states, tracking_states, indices_tracking_states, number_time_steps,
                        critic_start_training, gamma, critic_learning_rate, critic_learning_rate_exponent_limit,
                        critic_layers,
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
        f = simulation.run_simulation()

        return [f]

    def get_bounds(self):
        alpha_critic_l = 0.5
        alpha_critic_u = 10.

        alpha_actor_l = 0.5
        alpha_actor_u = 10.

        alpha_actor_cascaded_l = 0.5
        alpha_actor_cascaded_u = 10.

        Q_l = 1.
        Q_u = 1000.

        gamma_l = 0.5
        gamma_u = 1.

        lower_limits = [alpha_critic_l, alpha_actor_l, alpha_actor_cascaded_l, Q_l, gamma_l]
        upper_limits = [alpha_critic_u, alpha_actor_u, alpha_actor_cascaded_u, Q_u, gamma_u]

        return (lower_limits, upper_limits)

    def get_name(self):
        return "PSO Function"

if __name__ == "__main__":
    b_type = pg.mp_bfe()
    b_type.resize_pool(6)

    b = pg.bfe(b_type)
    prob = pg.problem(PSO_Problem(5))
    uda = pg.pso_gen(gen=10)
    uda.set_bfe(b)
    algo = pg.algorithm(uda)
    algo.set_verbosity(1)

    pop = pg.population(prob, size=100)
    pop = algo.evolve(pop)

    # Log all the information
    x = pop.champion_x
    f = pop.champion_f
    now = datetime.now()
    date = now.strftime("%d/%m/%Y %H:%M:%S")
    separation = "\n------------------------------------------------------------------------------------------------------\n"
    with open('logs.txt', 'a') as file:
        intro = separation + date + separation
        file.write(intro)
        file.write("\nalpha_critic = " + str(x[0]))
        file.write("\nalpha_actor = " + str(x[1]))
        file.write("\nalpha_actor_cascaded = " + str(x[2]))
        file.write("\nQ = " + str(x[3]))
        file.write("\ngamma = " + str(x[4]))
        file.write("\n\n feval = " + str(f))