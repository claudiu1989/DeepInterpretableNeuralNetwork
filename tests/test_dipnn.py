'''
   Unit and integration tests
'''

import sys
import numpy as np
from numpy.testing import assert_almost_equal
sys.path.append('../')
from  DeepInterpretablePolynomialNeuralNetwork.src.deep_interpretable_polynomial_neural_network import DeepInterpretablePolynomialNeuralNetwork, GrowthPolicy

import unittest
from parameterized import parameterized

class TestDipnn(unittest.TestCase):
    
    @staticmethod
    def create_test_data():
        X_train = np.array([[1.0,0.0,1.0,0.0],[0.0,1.0,1.0,1.0]])
        Y_train = np.array([1.0,-1.0])
        return X_train, Y_train

    @parameterized.expand([
        [2, 5.0, 4.0,True,3.0,0.5,0.4,100,1000,GrowthPolicy.SELECT_BY_DERIVATIVE
    ]])
    def test_init(self,d_max, lambda_param, balance, fixed_margin, ro, derivative_magnitude_th, coeff_magnitude_th, 
                 max_no_terms_per_iteration, max_no_terms, growth_policy):
        dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, 
                                                       derivative_magnitude_th=derivative_magnitude_th, coeff_magnitude_th=coeff_magnitude_th, 
                                                       max_no_terms_per_iteration=max_no_terms_per_iteration, max_no_terms=max_no_terms, growth_policy=growth_policy)
        self.assertEqual(dipnn.d_max,d_max)
        self.assertEqual(dipnn.lambda_param,lambda_param)
        self.assertEqual(dipnn.balance,balance)
        self.assertEqual(dipnn.fixed_margin,fixed_margin)
        self.assertEqual(dipnn.ro,ro)
        self.assertEqual(dipnn.derivative_magnitude_th,derivative_magnitude_th)
        self.assertEqual(dipnn.coeff_magnitude_th,coeff_magnitude_th)
        self.assertEqual(dipnn.max_no_terms_per_iteration,max_no_terms_per_iteration)
        self.assertEqual(dipnn.max_no_terms,max_no_terms)
        self.assertEqual(dipnn.growth_policy,growth_policy)

    @parameterized.expand([
        [np.array([1.0,1.0,1.0,1.0]), np.array([[1.0,0.0,1.0,0.0],[0.0,1.0,1.0,1.0]]), np.array([1.0,-1.0]), 2, 4, [4], 1, 1.0,  True, 0.25, 1.0, 16.365],
        [np.array([1.0,1.0,1.0,1.0]), np.array([[1.0,0.0,1.0,0.0],[0.0,1.0,1.0,1.0]]), np.array([1.0,-1.0]), 2, 4, [4], 1, 1.0,  True, 1.0, 1.0, 206.745],
        [np.array([1.0,1.0,1.0,1.0]), np.array([[1.0,0.0,1.0,0.0],[0.0,1.0,1.0,1.0]]), np.array([1.0,-1.0]), 2, 4, [4], 1, 1.0,  False, 0.25, 1.0, 16.365]])
    def test_compute_objective_function(self,w, X_train_cr, Y_train, m, n, cr_degrees_limits, cr_degree, lambda_param, fixed_margin, ro, balance, expected_value):        
        obj_value = DeepInterpretablePolynomialNeuralNetwork.objective_function(w, X_train_cr, Y_train, m, n, cr_degrees_limits, cr_degree, lambda_param, fixed_margin, ro, balance)
        assert_almost_equal(obj_value, expected_value,decimal=3)

    @parameterized.expand([
        [np.array([[1.0,1.0,1.0,1.0],[0.0,0.0,1.0,1.0],[0.0,0.0,1.0,1.0]]), [[1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0],[0.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0],[0.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0]]]])
    def test_add_negated_variables(self, X, expected_result):
        d_max = 1
        balance = 1.5
        lambda_param = 1.0
        ro = 1.0
        fixed_margin = False
        dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max, lambda_param, balance, fixed_margin, ro)
        X_with_negates = dipnn.add_negated_variables(X)
        are_identical = np.array_equal(np.array(X_with_negates), np.array(expected_result))
        self.assertTrue(are_identical)