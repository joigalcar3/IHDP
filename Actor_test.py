import unittest
from Actor import Actor
import numpy as np


class TestActor(unittest.TestCase):
    def setUp(self):
        selected_inputs = ['ele']
        selected_states = ['velocity', 'alpha', 'theta', 'q']
        number_time_steps = 500
        self.actor = Actor(selected_inputs, selected_states, number_time_steps)

    def test_build_actor_model(self):
        self.actor.build_actor_model()
        trainable_variables = self.actor.model.trainable_variables
        total = 0
        for matrix in trainable_variables:
            if len(matrix.shape) != 2:
                total += matrix.shape[0]
            else:
                total += matrix.shape[0] * matrix.shape[1]

        self.assertEqual(total, 101)

    def test_run_actor_online(self):
        time_steps = 500
        dt = 0.5
        time = np.arange(0, time_steps * dt, dt)

        store_xt = np.array([5 * np.sin(0.04*time), 6 * np.sin(0.04*time), 7 * np.sin(0.04*time), 8 * np.sin(0.04*time)])
        store_xt_ref = np.array([1 * np.sin(0.2*time), 2 * np.sin(0.2*time), 3 * np.sin(0.2*time), 4 * np.sin(0.2*time)])

        self.actor.build_actor_model()

        W1 = self.actor.model.trainable_variables[0].numpy()
        W2 = self.actor.model.trainable_variables[2].numpy()
        for count in range(store_xt.shape[1]):
            hidden = np.matmul(np.reshape(np.vstack((store_xt[:, count], store_xt_ref[:, count])), [8, 1]).T, W1)
            hidden[hidden < 0] = 0
            output = np.matmul(hidden, W2)
            e0 = self.actor.compute_persistent_excitation()
            output = output + e0
            ut = self.actor.run_actor_online(np.reshape(store_xt[:, count], [4, 1]), np.reshape(store_xt_ref[:, count], [4, 1]))
            if count % 20 == 0:
                self.assertAlmostEqual(ut[0, 0], output[0, 0], 5)



if __name__ == '__main__':
    unittest.main()
