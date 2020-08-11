import unittest
from Actor import Actor
import numpy as np
import tensorflow as tf


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
        cache = {}
        for count in range(store_xt.shape[1]):
            # Forward pass
            Z1 = np.matmul(W1.T, np.reshape(np.vstack((store_xt[:, count], store_xt_ref[:, count])), [8, 1])) + \
                     np.reshape(self.actor.model.trainable_variables[1].numpy(), [-1, 1])
            cache['Z1'] = Z1

            A1 = Z1
            A1[Z1 < 0] = 0
            cache['A1'] = A1

            Z2 = np.matmul(W2.T, A1) + np.reshape(self.actor.model.trainable_variables[3].numpy(), [-1, 1])
            cache['Z2'] = Z2

            A2 = Z2
            cache['A2'] = A2

            e0 = self.actor.compute_persistent_excitation()
            output = A2 + e0

            ut = self.actor.run_actor_online(np.reshape(store_xt[:, count], [4, 1]),
                                             np.reshape(store_xt_ref[:, count], [4, 1]))
            if count % 20 == 0:
                self.assertAlmostEqual(ut[0, 0], output[0, 0], 5)

            # Backard pass
            dZ2 = np.ones(Z2.shape)
            dW2= np.matmul(A1, dZ2.T)
            db2 = dZ2

            dA2 = np.matmul(W2, dZ2)
            filter = np.ones(Z1.shape)
            filter[Z1 <= 0] = 0
            dZ1 = np.multiply(dA2, filter)
            dW1 = np.matmul(np.reshape(np.vstack((store_xt[:, count], store_xt_ref[:, count])), [8, 1]), dZ1.T)
            db1 = dZ1
            self.assertAlmostEqual(db2[0, 0], np.reshape(self.actor.dJt_dWb[3].numpy(), [1, 1])[0, 0], 5)
            self.assertAlmostEqual(dW2[3, 0], self.actor.dJt_dWb[2].numpy()[3, 0], 5)
            self.assertAlmostEqual(db1[4, 0], np.reshape(self.actor.dJt_dWb[1].numpy(), [-1, 1])[4, 0], 5)
            self.assertAlmostEqual(dW1[3, 3], self.actor.dJt_dWb[0].numpy()[3, 3], 5)

    def test_train_actor_online(self):
        selected_inputs = ['ele']
        selected_states = ['velocity', 'alpha', 'theta', 'q']
        number_time_steps = 500
        layers = [2, 1]
        learning_rate = 1
        self.actor = Actor(selected_inputs, selected_states, number_time_steps, layers=layers,
                           learning_rate=learning_rate)

        Jt = np.array([1])
        critic_derivative = np.array([[1], [1], [1], [1]])
        G = np.array([[0], [1], [0], [2]])

        self.actor.build_actor_model()
        W1 = self.actor.model.trainable_variables[0].numpy()
        W2 = self.actor.model.trainable_variables[2].numpy()
        b1 = self.actor.model.trainable_variables[1].numpy()
        b2 = self.actor.model.trainable_variables[3].numpy()

        dW1 = tf.Variable(np.array([[1, 2], [3, 4], [5, 6], [7, 8], [1, 2], [3, 4], [5, 6], [7, 11]], dtype='float32'))
        db1 = tf.Variable(np.array([[1], [3]], dtype='float32'))
        dW2 = tf.Variable(np.array([[2], [4]], dtype='float32'))
        db2 = tf.Variable(np.array([[5]], dtype='float32'))
        gradients = [dW1, db1, dW2, db2]
        self.actor.dJt_dWb = gradients

        general_update = learning_rate * Jt.flatten() * np.matmul(G.T, critic_derivative).flatten()[0]  # should be 3
        # updates = [i * general_update for i in gradients]
        # new_weights = [updates[i].numpy() - self.actor.model.trainable_variables[i].numpy() for i in range(len(gradients))]

        self.actor.train_actor_online(Jt, critic_derivative, G)

        self.assertEqual(general_update, 3)
        self.assertAlmostEqual(self.actor.model.trainable_variables[0].numpy()[5, 0], W1[5, 0] - 9, 5)
        self.assertAlmostEqual(self.actor.model.trainable_variables[0].numpy()[2, 1], W1[2, 1] - 18, 5)
        self.assertAlmostEqual(self.actor.model.trainable_variables[0].numpy()[7, 1], max(W1[7, 1] - 33, -30), 5)
        self.assertAlmostEqual(self.actor.model.trainable_variables[0].numpy()[5, 1], W1[5, 1] - 12, 5)

        self.assertAlmostEqual(np.reshape(self.actor.model.trainable_variables[1].numpy(), [-1, 1])[0, 0],
                               np.reshape(b1, [-1, 1])[0, 0] - 3, 5)
        self.assertAlmostEqual(np.reshape(self.actor.model.trainable_variables[1].numpy(), [-1, 1])[1, 0],
                               np.reshape(b1, [-1, 1])[1, 0] - 9, 5)

        self.assertAlmostEqual(np.reshape(self.actor.model.trainable_variables[2].numpy(), [-1, 1])[0, 0],
                               np.reshape(W2, [-1, 1])[0, 0] - 6, 5)
        self.assertAlmostEqual(np.reshape(self.actor.model.trainable_variables[2].numpy(), [-1, 1])[1, 0],
                               np.reshape(W2, [-1, 1])[1, 0] - 12, 5)

        self.assertAlmostEqual(np.reshape(self.actor.model.trainable_variables[3].numpy(), [-1, 1])[0, 0],
                               np.reshape(b2, [-1, 1])[0, 0] - 15, 5)



if __name__ == '__main__':
    unittest.main()
