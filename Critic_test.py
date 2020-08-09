import unittest
from Critic import Critic
import numpy as np
import tensorflow as tf
import random


class TestCritic(unittest.TestCase):
    def setUp(self):
        Q_weights = [1, 1, 1, 1]
        selected_states = ['velocity', 'alpha', 'theta', 'q']
        number_time_steps = 500
        self.critic = Critic(Q_weights, selected_states, number_time_steps)

    def test_c_computation(self):
        self.critic.xt = np.array([[1, 2, 3, 4]]).T
        self.critic.xt_ref = np.array([[-1, -2, -3, -4]]).T
        ct = self.critic.c_computation()

        result = np.array([[120]])
        self.assertEqual(ct.shape, (1, 1))
        self.assertEqual(ct, result)

    def test_targets_computation_end(self):
        self.critic.store_c = np.reshape(np.array(range(0, 5)), [1, 5])
        self.critic.store_J = np.reshape(np.array(range(0, 5)), [1, 5])
        self.critic.gamma = 1
        targets = self.critic.targets_computation_end()

        result = np.reshape(np.array([[1, 3, 5, 7]]), [4, 1])
        self.assertEqual(targets.shape, (4, 1))
        self.assertEqual(targets[0, 0], result[0, 0])
        self.assertEqual(targets[1, 0], result[1, 0])
        self.assertEqual(targets[2, 0], result[2, 0])
        self.assertEqual(targets[3, 0], result[3, 0])

    def test_build_critic_model(self):
        self.critic.build_critic_model()
        trainable_variables = self.critic.model.trainable_variables
        total = 0
        for matrix in trainable_variables:
            if len(matrix.shape) != 2:
                total += matrix.shape[0]
            else:
                total += matrix.shape[0]*matrix.shape[1]

        self.assertEqual(total, 61)

    def test_run_critic(self):
        time_steps = 500
        dt = 0.5
        time = np.arange(0, time_steps * dt, dt)

        store_xt = np.array([5 * np.sin(0.04*time), 6 * np.sin(0.04*time), 7 * np.sin(0.04*time), 8 * np.sin(0.04*time)])
        store_xt_ref = np.array([1 * np.sin(0.2*time), 2 * np.sin(0.2*time), 3 * np.sin(0.2*time), 4 * np.sin(0.2*time)])

        self.critic.build_critic_model()

        W1 = self.critic.model.trainable_variables[0].numpy()
        W2 = self.critic.model.trainable_variables[2].numpy()
        for count in range(store_xt.shape[1]):
            hidden = np.matmul(np.reshape(store_xt[:, count], [4, 1]).T, W1)
            hidden[hidden < 0] = 0
            output = np.matmul(hidden, W2)
            Jt = self.critic.run_critic(np.reshape(store_xt[:, count], [4, 1]), np.reshape(store_xt_ref[:, count], [4, 1]))
            if count % 20 == 0:
                self.assertAlmostEqual(Jt[0, 0], output[0, 0], 5)

    def test_train_critic_end(self):
        Q_weights = [1, 1, 1, 1]
        selected_states = ['velocity', 'alpha', 'theta', 'q']
        number_time_steps = 500
        layers = [20, 10, 1]
        activations = ['relu', 'relu', 'linear']
        self.critic1 = Critic(Q_weights, selected_states, number_time_steps, layers=layers, activations=activations)

        time_steps = 500
        dt = 0.5
        time = np.arange(0, time_steps * dt, dt)

        store_xt = np.array([5 * np.sin(0.04*time), 6 * np.sin(0.04*time), 7 * np.sin(0.04*time), 8 * np.sin(0.04*time)])
        store_xt_ref = np.array([1 * np.sin(0.2*time), 2 * np.sin(0.2*time), 3 * np.sin(0.2*time), 4 * np.sin(0.2*time)])

        self.critic1.build_critic_model()
        for count in range(store_xt.shape[1]):
            self.critic1.run_critic(np.reshape(store_xt[:, count], [4, 1]),
                                        np.reshape(store_xt_ref[:, count], [4, 1]));
        self.critic1.train_critic_end()

    def test_run_train_critic_online(self):
        random.seed(1)
        Q_weights = [1, 1, 1, 1]
        selected_states = ['velocity', 'alpha', 'theta', 'q']
        number_time_steps = 500
        layers = [20, 10, 1]
        activations = ['relu', 'relu', 'linear']
        batch_size = 10
        self.critic1 = Critic(Q_weights, selected_states, number_time_steps, layers=layers, activations=activations,
                              batch_size=batch_size)
        time_steps = 500
        dt = 0.5
        time = np.arange(0, time_steps * dt, dt)

        store_xt = np.array(
            [5 * np.sin(0.04 * time), 6 * np.sin(0.04 * time), 7 * np.sin(0.04 * time), 8 * np.sin(0.04 * time)])
        store_xt_ref = np.array(
            [1 * np.sin(0.2 * time), 2 * np.sin(0.2 * time), 3 * np.sin(0.2 * time), 4 * np.sin(0.2 * time)])

        self.critic1.build_critic_model()

        # Part of the algorithm required to test the gradient: definition of W and b
        trainable_variables = self.critic1.model.trainable_variables
        self.critic1.W = {}
        self.critic1.b = {}
        self.critic1.cache = {}
        for layer in range(int(len(trainable_variables) / 2)):
            self.critic1.W['W_' + str(layer + 1)] = trainable_variables[2 * layer].numpy()
            self.critic1.b['b_' + str(layer + 1)] = trainable_variables[2 * layer + 1].numpy()

        for count in range(store_xt.shape[1]):
            Jt, dJt_dxt = self.critic1.run_train_critic_online(np.reshape(store_xt[:, count], [4, 1]),
                                    np.reshape(store_xt_ref[:, count], [4, 1]))

            if (self.critic1.time_step-1) % self.critic1.batch_size == 0:
                # Part of the algorithm required to test the gradient: definition of W and b
                trainable_variables = self.critic1.model.trainable_variables

                for layer in range(int(len(trainable_variables) / 2)):
                    self.critic1.W['W_' + str(layer + 1)] = trainable_variables[2 * layer].numpy()
                    self.critic1.b['b_' + str(layer + 1)] = trainable_variables[2 * layer + 1].numpy()

        # Part of the algorithm required to test the gradient: Forward pass
            a = self.critic1.xt
            for layer in range(len(self.critic1.layers)):
                Z = np.matmul(self.critic1.W['W_' + str(layer+1)].T, a) + \
                    np.reshape(self.critic1.b['b_' + str(layer+1)], [-1, 1])
                try:
                    a = tf.keras.activations.deserialize(self.critic1.activations[layer])(Z.astype('float32')).numpy()
                except:
                    a = tf.keras.activations.deserialize(self.critic1.activations[layer])(Z.astype('float32'))
                self.critic1.cache['Z_' + str(layer + 1)] = Z
                self.critic1.cache['A_' + str(layer + 1)] = a

            # Testing whether the prediction is correct given the weights and biases
            self.assertAlmostEqual(Jt[0,0], a[0,0], 5)

            # Part of the algorithm required to test the gradient: Backward pass
            dA = 1
            for i in range(len(self.critic1.layers)):
                layer = len(self.critic1.layers) - i
                if self.critic1.activations[layer - 1] == 'linear':
                    dZ = np.ones(self.critic1.cache['Z_' + str(layer)].shape)
                elif self.critic1.activations[layer - 1] == 'relu':
                    dZ = np.ones(self.critic1.cache['Z_' + str(layer)].shape)
                    dZ_or = self.critic1.cache['Z_' + str(layer)]
                    dZ[dZ_or <= 0] = 0
                dZ = np.multiply(dA, dZ)
                dA = np.matmul(self.critic1.W['W_' + str(layer)], dZ)

            # Testing whether the gradients are correctly computed
            self.assertAlmostEqual(dJt_dxt[0, 0, 0], dA[0, 0], 5)
            self.assertAlmostEqual(dJt_dxt[0, 1, 0], dA[1, 0], 5)
            self.assertAlmostEqual(dJt_dxt[0, 2, 0], dA[2, 0], 5)
            self.assertAlmostEqual(dJt_dxt[0, 3, 0], dA[3, 0], 5)


if __name__ == '__main__':
    unittest.main()
