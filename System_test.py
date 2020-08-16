import unittest
from System import F16System, System
from scipy.io import loadmat
import numpy as np


class TestSystem(unittest.TestCase):
    def setUp(self):
        self.selected_states = ['velocity', 'alpha', 'theta', 'q']
        self.selected_output = ['alpha', 'q']
        self. selected_input = ['ele']
        self.discretisation_time = 0.5
        self.folder = "Testing"
        self.folder1 = "Linear_system"
        self.sys = F16System(self.folder, self.selected_states, self.selected_output, self.selected_input,
                             self.discretisation_time)
        self.sys1 = F16System(self.folder1, self.selected_states, self.selected_output, self.selected_input,
                              self.discretisation_time)

    def test_import_linear_system(self):
        self.sys.import_linear_system()
        self.assertEqual(self.sys.A[0, 0], 1)
        self.assertEqual(self.sys.B[0, 1], 2)
        self.assertEqual(self.sys.C[1, 0], 7)
        self.assertEqual(self.sys.D[1, 1], 8)

    def test_create_dictionary(self):
        file_name = 'test_create_dictionary'
        expected_result = {'Hello': 0, 'World': 1, 'is': 2, 'my': 3, 'first': 4, 'code': 5}
        self.assertEqual(self.sys.create_dictionary(file_name), expected_result)

    def test_simplify_system(self):
        self.sys1.import_linear_system()
        self.sys1.simplify_system()

        # Obtain the pre-computed results with Matlab
        x = loadmat('Testing/filt_A.mat')
        filt_A = x['filt_A']
        x = loadmat('Testing/filt_B.mat')
        filt_B = x['filt_B']
        x = loadmat('Testing/filt_C.mat')
        filt_C = x['filt_C']
        x = loadmat('Testing/filt_D.mat')
        filt_D = x['filt_D']

        self.assertEqual(self.sys1.filt_A[3,3], filt_A[3,3])
        self.assertEqual(self.sys1.filt_B[2,0], filt_B[2,0])
        self.assertEqual(self.sys1.filt_C[1,1], filt_C[1,1])
        self.assertEqual(self.sys1.filt_D[0,0], filt_D[0,0])

    def test_initialise_system(self):
        # Computing the filtered matrices
        x0 = np.transpose(np.array([0,0,0,0]))
        number_time_steps = 500
        self.sys1.initialise_system(x0, number_time_steps)

        # Obtain the pre-computed results with Matlab
        x = loadmat('Testing/A_disc.mat')
        A_disc = x['A_disc']
        x = loadmat('Testing/B_disc.mat')
        B_disc = x['B_disc']
        x = loadmat('Testing/C_disc.mat')
        C_disc = x['C_disc']
        x = loadmat('Testing/D_disc.mat')
        D_disc = x['D_disc']

        self.assertAlmostEqual(self.sys1.filt_A[3, 3], A_disc[3, 3])
        self.assertAlmostEqual(self.sys1.filt_B[2, 0], B_disc[2, 0])
        self.assertAlmostEqual(self.sys1.filt_C[1, 1], C_disc[1, 1])
        self.assertAlmostEqual(self.sys1.filt_D[0, 0], D_disc[0, 0])
        self.assertEqual(self.sys1.store_states.shape, (4, number_time_steps+1))

    def test_run_step(self):
        self.sys1.number_states = 2
        self.sys1.number_inputs = 1
        self.sys1.number_outputs = 1

        x0 = np.transpose(np.array([0, 0]))
        number_time_steps = 500
        self.sys1.initialise_system(x0, number_time_steps)

        self.sys1.filt_A = np.array([[1, 1], [1, 1]])
        self.sys1.filt_B = np.array([[2], [3]])
        self.sys1.filt_C = np.array([1, 2])
        self.sys1.filt_D = np.array([4])
        self.sys1.xt = np.array([[2],[2]])
        ut = np.reshape(np.array([10]), [1,1])

        xt1 = self.sys1.run_step(ut)

        self.assertEqual(xt1[0, 0], 24)
        self.assertEqual(xt1[1, 0], 34)


if __name__ == '__main__':
    unittest.main()
