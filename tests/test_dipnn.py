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

def mock_compute_exp_factors_derivative():
        return np.array([-1.0,1.0])

def mock_compute_derivative(new_term, next_degree, data_exp_factors):
    return -1
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

    @parameterized.expand([
        [np.array([[1.0,1.0,1.0,1.0],[1.0,1.0,0.0,0.0]]), np.array([1.0,-1.0]), np.array([1.0,0.0,0.0,0.0]),np.array([2.718,2.718])],
        [np.array([[1.0,1.0,1.0,1.0],[1.0,1.0,0.0,0.0]]), np.array([1.0,-1.0]), np.array([0.0,0.0,0.0,0.0]),np.array([54.598,0.367])]])   
    def test_compute_exp_factors_derivative(self, X_train_cr, Y_train, beta_optimal, expected_result):
        d_max = 1
        balance = 1.5
        lambda_param = 1.0
        ro = 0.5
        fixed_margin = True
        dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max, lambda_param, balance, fixed_margin, ro)
        dipnn.beta_optimal = beta_optimal
        dipnn.X_train_cr = X_train_cr
        dipnn.Y_train = Y_train
        exp_factors = dipnn.compute_exp_factors_derivative()
        for expected, computed in zip(exp_factors, expected_result):
           assert_almost_equal(expected, computed,decimal=3)

    @parameterized.expand([
    [np.array([[1.0,1.0,1.0,1.0],[1.0,0.0,0.0,0.0]]), np.array([1.0,-1.0]), np.array([1.0,0.0]),[1,0], 2, 0.384],
    [np.array([[1.0,1.0,1.0,1.0],[1.0,0.0,1.0,0.0]]), np.array([-1.0,-1.0]), np.array([1.0,0.0]),[1,0], 2, 2.384]])   
    def test_compute_derivative(self, X_train, Y_train, data_exp_factors, term, next_degree, expected_result):
        d_max = 1
        balance = 1.5
        lambda_param = 1.0
        ro = 0.5
        fixed_margin = True
        dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max, lambda_param, balance, fixed_margin, ro)
        dipnn.X_train = X_train
        dipnn.Y_train = Y_train
        dipnn.m = 2
        dipnn.n = 2
        derivative_value = dipnn.compute_derivative(term, next_degree, data_exp_factors)
        assert_almost_equal(derivative_value, expected_result, decimal=3)

    
    @parameterized.expand([
    [np.array([[1.0,1.0,1.0,1.0],[1.0,0.0,0.0,0.0]]), np.array([1.0,-1.0]), np.array([1.0,1.0,1.0,0.2]),  np.array([1.0,1.0,1.0])],
    [np.array([[1.0,1.0,1.0,1.0],[1.0,0.0,1.0,0.0]]), np.array([-1.0,-1.0]), np.array([1.0,0.1,2.0,0.2]), np.array([1.0,2.0])]])   
    def test_prune_terms_and_features(self, X_train_cr, Y_train, beta, expected_result):
        d_max = 1
        balance = 1.5
        lambda_param = 1.0
        ro = 0.5
        fixed_margin = True
        dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max, lambda_param, balance, fixed_margin, ro)
        dipnn.X_train_cr = X_train_cr
        dipnn.Y_train = Y_train
        dipnn.coeff_magnitude_th = 0.5
        dipnn.beta_optimal = beta
        dipnn.w_optimal = beta
        dipnn.m = 2
        dipnn.n = 2
        dipnn.terms = [[0],[1,1],[2,2],[1,2]]
        dipnn.prune_terms_and_features()
        are_identical = np.array_equal(np.array(dipnn.beta_optimal), np.array(expected_result))
        self.assertTrue(are_identical)
        are_identical = np.array_equal(np.array(dipnn.beta_optimal), np.array(expected_result))
        self.assertTrue(are_identical)

    @parameterized.expand([
    [[-0.1,-5.0, -0.3,-1.0],  [[1,1],[1,2]]],
    [[-10.1,-2.0, -0.3,-1.0], [[0],[1,1]]]])   
    def test_get_top_terms_by_derivative(self, derivative_values, expected_result):
        d_max = 1
        balance = 1.5
        lambda_param = 1.0
        ro = 0.5
        fixed_margin = True
        dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max, lambda_param, balance, fixed_margin, ro)
        new_terms = [[0],[1,1],[2,2],[1,2]]
        no_terms_to_return = 2
        top_terms = dipnn.get_top_terms_by_derivative(new_terms, derivative_values, no_terms_to_return)
        are_identical = np.array_equal(np.array(top_terms), np.array(expected_result))
        self.assertTrue(are_identical)
    
    @parameterized.expand([
    [[[0],[1],[2],[3]],  [[0],[1],[2],[3],[0,0],[0,1],[0,2],[0,3],[1,1],[1,2],[1,3],[2,2],[2,3],[3,3]]],
    [[[0],[2],[3]],  [[0],[2],[3],[0,0],[0,1],[0,2],[0,3],[1,2],[2,2],[2,3],[1,3],[3,3]]]])   
    def test_add_terms_and_features_of_next_degree1(self, terms, expected_result):
        d_max = 1
        balance = 1.5
        lambda_param = 1.0
        ro = 0.5
        fixed_margin = True
        dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max, lambda_param, balance, fixed_margin, ro)
        dipnn.n = 2
        dipnn.terms = terms
        dipnn.beta_optimal = np.array([1.0,0.1,2.0,0.2])
        dipnn.w_optimal = np.array([1.0,0.1,2.0,0.2])
        dipnn.compute_exp_factors_derivative = mock_compute_exp_factors_derivative
        dipnn.compute_derivative= mock_compute_derivative
        dipnn.max_no_terms_per_iteration = 100
        dipnn.max_no_terms = 100
        dipnn.add_terms_and_features_of_next_degree(2)
        are_identical = np.array_equal(np.array(dipnn.terms), np.array(expected_result))
        self.assertTrue(are_identical)
    
    @parameterized.expand([
    [[[0],[2],[0,2]],  [[0], [2], [0,2], [0,0,2],[0,1,2],[0,2,2],[0,2,3],]]])   
    def test_add_terms_and_features_of_next_degree2(self, terms, expected_result):
        d_max = 1
        balance = 1.5
        lambda_param = 1.0
        ro = 0.5
        fixed_margin = True
        dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max, lambda_param, balance, fixed_margin, ro)
        dipnn.n = 2
        dipnn.terms = terms
        dipnn.beta_optimal = np.array([1.0,0.1,2.0,0.2])
        dipnn.w_optimal = np.array([1.0,0.1,2.0,0.2])
        dipnn.compute_exp_factors_derivative = mock_compute_exp_factors_derivative
        dipnn.compute_derivative= mock_compute_derivative
        dipnn.max_no_terms_per_iteration = 100
        dipnn.max_no_terms = 100
        dipnn.cr_degrees_limits = [2,4]
        dipnn.add_terms_and_features_of_next_degree(2)
        are_identical = np.array_equal(np.array(dipnn.terms), np.array(expected_result))
        self.assertTrue(are_identical)

    @parameterized.expand([
    [np.array([1.0,0.1,2.0,0.2]),  np.array([1.0,0.1,2.0,0.2,0.0,0.0,0.0,0.0])]])   
    def test_add_terms_and_features_of_next_degree3(self, beta_optimal, expected_result):
        d_max = 1
        balance = 1.5
        lambda_param = 1.0
        ro = 0.5
        fixed_margin = True
        dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max, lambda_param, balance, fixed_margin, ro)
        dipnn.n = 2
        dipnn.terms = [[0],[1],[2],[3]]
        dipnn.beta_optimal = beta_optimal
        dipnn.w_optimal = beta_optimal
        dipnn.X_train = np.array([[1.0,1.0,1.0,1.0],[1.0,0.0,0.0,0.0]])
        dipnn.X_train_cr = np.array([[1.0,1.0,1.0,1.0],[1.0,0.0,0.0,0.0]])
        dipnn.compute_exp_factors_derivative = mock_compute_exp_factors_derivative
        dipnn.compute_derivative= mock_compute_derivative
        dipnn.max_no_terms_per_iteration = 4
        dipnn.max_no_terms = 100
        dipnn.cr_degrees_limits = [4]
        dipnn.add_terms_and_features_of_next_degree(2)
        are_identical = np.array_equal(np.array(dipnn.beta_optimal), np.array(expected_result))
        self.assertTrue(are_identical)
        self.assertEqual(dipnn.no_features,8)

    @parameterized.expand([
    [2,  [[0],[1],[2],[3],[0,0],[0,1],[0,2],[0,3],[1,1],[1,2],[1,3],[2,2],[2,3],[3,3]]]])   
    def test_generate_all_terms1(self, d_max, expected_result):
        balance = 1.5
        lambda_param = 1.0
        ro = 0.5
        fixed_margin = True
        dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max, lambda_param, balance, fixed_margin, ro)
        dipnn.terms = [[0],[1],[2],[3]]
        dipnn.n = 2
        terms = dipnn.generate_all_terms(d_max)
        are_identical = np.array_equal(np.array(terms), np.array(expected_result))
        self.assertTrue(are_identical)

    @parameterized.expand([
    [2,  [4,14]]])   
    def test_generate_all_terms2(self, d_max, expected_result):
        balance = 1.5
        lambda_param = 1.0
        ro = 0.5
        fixed_margin = True
        dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max, lambda_param, balance, fixed_margin, ro)
        dipnn.terms = [[0],[1],[2],[3]]
        dipnn.n = 2
        dipnn.generate_all_terms(d_max)
        are_identical = np.array_equal(np.array(dipnn.cr_degrees_limits), np.array(expected_result))
        self.assertTrue(are_identical)

    @parameterized.expand([
    [[[1,1],[0,2]],  np.array([[1.0,2.0,1.0,1.0,4.0,1.0],[1.0,0.0,0.0,0.0, 0.0, 0.0]])]])   
    def test_add_new_features(self, terms, expected_result):
        balance = 1.5
        lambda_param = 1.0
        ro = 0.5
        fixed_margin = True
        d_max = 2 
        dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max, lambda_param, balance, fixed_margin, ro)
        dipnn.n = 2
        dipnn.X_train_cr = np.array([[1.0,2.0,1.0,1.0],[1.0,0.0,0.0,0.0]])
        dipnn.X_train = np.array([[1.0,2.0,1.0,1.0],[1.0,0.0,0.0,0.0]])
        dipnn.add_new_features(terms)
        are_identical = np.array_equal(np.array(dipnn.X_train_cr), np.array(expected_result))
        self.assertTrue(are_identical)

    @parameterized.expand([
    [np.array([1.0,0.1,2.0,0.2]),  np.array([1.0,0.1,0.2,0.1,4.0])]])   
    def test_compute_features_data_point(self, x, expected_result):
        balance = 1.5
        lambda_param = 1.0
        ro = 0.5
        fixed_margin = True
        d_max = 2 
        dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max, lambda_param, balance, fixed_margin, ro)
        dipnn.n = 2
        dipnn.terms = [[0],[1],[3], [0,1], [2,2]]
        features = dipnn.compute_features_data_point(x)
        are_identical = np.array_equal(features, np.array(expected_result))
        self.assertTrue(are_identical)

    @parameterized.expand([
    [np.array([[1.0,0.1,2.0,0.2],[1.0,0.1,0.5,0.2]]),  np.array([[1.0,0.1,0.2,0.1,4.0],[1.0,0.1,0.2,0.1,0.25]])]])   
    def test_compute_features(self, X, expected_result):
        balance = 1.5
        lambda_param = 1.0
        ro = 0.5
        fixed_margin = True
        d_max = 2 
        dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max, lambda_param, balance, fixed_margin, ro)
        dipnn.n = 2
        dipnn.terms = [[0],[1],[3], [0,1], [2,2]]
        features = dipnn.compute_features(X)
        are_identical = np.array_equal(features, np.array(expected_result))
        self.assertTrue(are_identical)