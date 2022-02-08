import csv
import numpy as np
import sys
sys.path.append('../')

from DeepInterpretablePolynomialNeuralNetwork.src.deep_interpretable_polynomial_neural_network import DeepInterpretablePolynomialNeuralNetwork, GrowthPolicy
from DeepInterpretablePolynomialNeuralNetwork.src.evaluation_tools import EvaluationTools

def read_data(path):
    with open(path) as csvfile:
        data = list(csv.reader(csvfile))
    X = []
    Y = []
    for row in data[1:]:
        X.append([float(value) for value in row[:-1]])
        if float(row[-1]) > 0.0:
            Y.append(1.0)
        else:
            Y.append(0.0)
    return np.array(X), np.array(Y)

def convert_to_logical_values(X):
    max = X[0].copy()
    min = X[0].copy()
    for row in X:
        for i, value in enumerate(row):
            if value > max[i]:
                max[i] = value
            if value < min[i]:
                min[i] = value
    new_X = []
    for row in X:
        new_row = []
        for i, value in enumerate(row):
            new_row.append(float(value - min[i])/(max[i]-min[i]))
        new_X.append(new_row)
    return np.array(new_X)

def experiment_no_growth_best_results():                          
    X, Y = read_data('./data/BloodTransfusion/transfusion.data')
    X = convert_to_logical_values(X)
    # Model
    d_max = 2
    balance = 3.0
    lambda_param = 10.0
    ro = 1.0
    fixed_margin = False
    growth_policy = GrowthPolicy.ALL_TERMS
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=0.0, 
                                        max_no_terms_per_iteration=20, max_no_terms=200, growth_policy=growth_policy)
    
    # Evaluation
    no_runs = 1
    test_size = 0.2
    coefficient_threshold = 0.01
    precision = 2
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
    print(f'The margin: {dipnn.ro}')


def experiment_experiments_with_different_degrees():                          
    X, Y = read_data('./data/BloodTransfusion/transfusion.data')
    X = convert_to_logical_values(X)
    # Model
    d_max = 3
    balance = 1.25
    lambda_param = 0.25
    ro = 1.0
    fixed_margin = True
    growth_policy = GrowthPolicy.GROW
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=0.0, 
                                        max_no_terms_per_iteration=20, max_no_terms=200, growth_policy=growth_policy)
    
    # Evaluation
    no_runs = 100
    test_size = 0.2
    coefficient_threshold = 0.01
    precision = 2
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
    print(f'The margin: {dipnn.ro}')

if __name__ == '__main__':
    experiment_experiments_with_different_degrees()
