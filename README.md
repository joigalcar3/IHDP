# Intelliggent Adaptive Control Using IHDP Applied to F-16 Aircraft
Implementation of Incremental Heuristic Dynamic Programming with Neural Networks as Actor and Critic function approximators.
The algorithm was applied to the longitudinal model of the F-16 aircraft linearised at 5000 [ft] and 300 [ft/s]. The paper
presenting the theory, the implementation, the results, as well as a sensitivity analysis on the architecture of the controller, the initial conditions,
the Artificial Neural Network optimizer and the actor/critic learning rates will soon be uploaded to 
[ResearchGate](https://www.researchgate.net/profile/Jose_Ignacio_De_Alvear_Cardenas).

# Code structure
Within this repository you will find 3 other subdirectories, as well as 6 Python 
files. First, as short description of the folders:

- Linear System: contains the matrices of the F-16 aircraft model linerised at
5000 [ft] and 00 [ft/s].
- Cloned code PSO: contains the Linear System folder and the 6 Python files, as 
well as another Python file that carries out the PSO optimisation. The 6 
afore-mentioned Python files have been cleaned of printing statements and
plotting.
- Tests: contains tests of the methods of the different classes that shape the controller architecture. (**WIP**)

Next, a short description of the 6 Python files:

- Actor.py: Provides class Actor with the function approximator (NN) of the Actor.
Actor creates the Neural Network model with Tensorflow and it can train the network online.
The user can decide the number of layers, the number of neurons and activation functions.

- Critic.py: Provides class Critic with the function approximator (NN) of the Critic.
Critic creates the Neural Network model with Tensorflow and it can train the network online. 
The user can decide the number of layers, the number of neurons and activation functions. 

- System.py: Provides classes System and F16System classes to simulate the external system.
System provides a simple template and common attributes among all the possible systems.
F16System provides the additional attributes and methods in order to use the linear F-16
aircraft system. It simplifies the system matrices according to the chosen states, inputs and
outputs, as well as implementing the methods required for initialisation and running the simulation.

- Incremental_model.py: Provides class IncrementalModel in order to identify the system.
IncrementalModel computes the A and x matrices required for the system identification,
computes the F and G matrices required for the incremental model and evaluates the
identified model in order to provide the estimates states at the next time step.

- Simulation.py: Provides class Simulation that initialises all the elements of the controllers and structures the execution of the
different components. Simulation initialises all the components of the controller, namely the System, the Incremental Model, the Actor and the
Critic, as well as building the required Neural Networks. It counts with a method that executes in the required order
each of the controller elements during a complete iteration. **MAIN FILE**

- User_input.py: this file compiles all the parameters that can be tuned in the
Actor, Critic, System, IncrementalModel and Simulation classes.

Hope you enjoy the code!! For any questions, comments or suggestions, please reach out to me at 
[j.i.dealvearcardenas@student.tudelft.nl](j.i.dealvearcardenas@student.tudelft.nl).