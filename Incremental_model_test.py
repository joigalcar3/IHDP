import unittest
from Incremental_model import IncrementalModel
import numpy as np


class TestIncrementalModel(unittest.TestCase):
    def setUp(self):
        selected_states = ['alpha', 'q']
        selected_input = ['ele']
        number_time_steps = 10
        self.incremental_model = IncrementalModel(selected_states, selected_input, number_time_steps)

    def test_build_A_LS_matrix(self):
        self.incremental_model.time_step = 9
        self.incremental_model.store_delta_xt = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
        self.incremental_model.store_delta_ut = np.reshape(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                                                           [self.incremental_model.number_inputs,
                                                            self.incremental_model.number_time_steps])

        A_LS_matrix = self.incremental_model.build_A_LS_matrix()

        self.assertEqual(A_LS_matrix.shape, (6, 3))
        self.assertEqual(A_LS_matrix[0, 0], 9)
        self.assertEqual(A_LS_matrix[1, 1], 3)
        self.assertEqual(A_LS_matrix[2, 2], 7)
        self.assertEqual(A_LS_matrix[5, 2], 4)

    def test_build_x_LS_vector(self):
        self.incremental_model.time_step = 9
        self.incremental_model.store_delta_xt = np.array(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
        self.incremental_model.store_delta_ut = np.reshape(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                                                           [self.incremental_model.number_inputs,
                                                            self.incremental_model.number_time_steps])
        self.incremental_model.xt = np.array([[10], [1]])
        self.incremental_model.xt_1 = np.array([[0], [0]])
        self.incremental_model.ut = np.array([[10]])
        self.incremental_model.ut_1 = np.array([[0]])

        x_LS_vector = self.incremental_model.build_x_LS_vector()

        self.assertEqual(x_LS_vector.shape, (6, 2))
        self.assertEqual(x_LS_vector[0, 0], 10)
        self.assertEqual(x_LS_vector[1, 1], 2)
        self.assertEqual(x_LS_vector[5, 1], 6)
        self.assertEqual(x_LS_vector[3, 0], 7)

    def test_identify_incremental_model_LS(self):
        selected_states = ['alpha']
        selected_input = ['ele']
        number_time_steps = 10
        self.incremental_model1 = IncrementalModel(selected_states, selected_input, number_time_steps)
        self.incremental_model1.time_step = 9
        self.incremental_model1.store_delta_xt = np.array([[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
        self.incremental_model1.store_delta_ut = np.reshape(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                                                           [self.incremental_model1.number_inputs,
                                                            self.incremental_model1.number_time_steps])
        self.incremental_model1.xt = np.array([[1]])
        self.incremental_model1.xt_1 = np.array([[0]])
        self.incremental_model1.ut = np.array([[10]])
        self.incremental_model1.ut_1 = np.array([[0]])

        self.incremental_model1.identify_incremental_model_LS(np.array([[1]]), np.array([[10]]))
        A = np.array([[2, 3, 4, 5, 6, 7], [9, 8, 7, 6, 5, 4]]).T
        x = np.array([[1, 2, 3, 4, 5, 6]]).T
        id_sys = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T,A)), A.T), x).T
        F = id_sys[:, :1]
        G = id_sys[:, 1:]

        self.assertEqual(self.incremental_model1.F.shape, (1, 1))
        self.assertEqual(self.incremental_model1.G.shape, (1, 1))
        self.assertAlmostEqual(self.incremental_model1.F[0, 0], F[0,0])
        self.assertAlmostEqual(self.incremental_model1.G[0, 0], G[0, 0])

    def test_evaluate_incremental_model(self):
        self.incremental_model.xt = np.array([[1], [2]])
        self.incremental_model.delta_xt = np.array([[3], [4]])
        self.incremental_model.delta_ut = np.array([[5]])
        self.incremental_model.F = np.array([[2, 3], [3, 2]])
        self.incremental_model.G = np.array([[2], [3]])

        xt1_est = self.incremental_model.evaluate_incremental_model()

        self.assertEqual(xt1_est.shape, (2, 1))
        self.assertEqual(xt1_est[0, 0], 29)
        self.assertEqual(xt1_est[1, 0], 34)


if __name__ == '__main__':
    unittest.main()
