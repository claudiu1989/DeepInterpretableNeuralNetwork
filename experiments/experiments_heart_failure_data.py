'''
This module contains experiments with the dataset https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records. 
See section 7.1 Experiment 1: accuracy and sensitivity-specificity trade-off.

Before running the experiments, be sure you download and copy the CSV file in the "data" folder.
'''

import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../')
from DeepInterpretablePolynomialNeuralNetwork.src.deep_interpretable_polynomial_neural_network import DeepInterpretablePolynomialNeuralNetwork, GrowthPolicy
from DeepInterpretablePolynomialNeuralNetwork.src.evaluation_tools import EvaluationTools

def read_data(path):
    with open(path) as csvfile:
        data = list(csv.reader(csvfile))
    X = []
    Y = []
    for row in data[1:]:
        X.append([float(value) for value in row[:-2]])
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

 # EXPERIMENTS REPORTED IN THE PAPER

def experiment_with_growth():
    # Data
    X, Y = read_data('.\data\heart_failure_clinical_records_dataset.csv')
    X = convert_to_logical_values(X)
    # Model
    d_max = 3
    balance = 1.55
    lambda_param = 0
    ro = 1.0
    fixed_margin = False
    for d_max in [1,2]:
        for balance in [1.0,1.5,2.0]:
            growth_policy = GrowthPolicy.GROW
            dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=0.0, 
                                                max_no_terms_per_iteration=20, max_no_terms=200, growth_policy=growth_policy)
            print(f'Results for d_max={d_max} and b={balance}')
            # Evaluation
            no_runs = 100
            test_size = 0.2
            coefficient_threshold = 0.01
            precision = 2
            EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)

def experiment_different_b_values():                          
    # Data
    X, Y = read_data('.\data\heart_failure_clinical_records_dataset.csv')
    X = convert_to_logical_values(X)
    b_param_values = np.linspace(0,10,43)
    avg_acc_list = list()
    avg_tp_list = list()
    avg_tn_list = list()
    for b_param in b_param_values:
        print(f'Run with balance: {b_param}')
        # Model
        d_max = 1
        balance = b_param
        lambda_param = 1.0
        ro = 1.0
        fixed_margin = False
        growth_policy = GrowthPolicy.ALL_TERMS
        dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=0.0, 
                                            max_no_terms_per_iteration=20, max_no_terms=200, growth_policy=growth_policy)
        # Evaluation
        no_runs = 100
        test_size = 0.2
        coefficient_threshold = 0.01
        precision = 2
        avg_acc, avg_training_time, var_training_time, avg_test_time, var_test_time, avg_tp_rate, avg_tn_rate = EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
        avg_acc_list.append(avg_acc)
        avg_tp_list.append(avg_tp_rate)
        avg_tn_list.append(avg_tn_rate)
    fig1, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.plot(b_param_values, avg_acc_list, color=color, label="Accuracy")
    plt.title('Accuracy, true positive rate and true negative rate as functions of "b" parameter')
    ax1.set_xlabel('b')
    ax1.set_ylabel('')

    color = 'tab:red'
    ax1.plot(b_param_values, avg_tp_list, color=color, label="TP rate")
    ax1.tick_params(axis='y', labelcolor=color)
    
    color = 'tab:green'
    ax1.plot(b_param_values, avg_tn_list, color=color, label="TN rate")
    ax1.tick_params(axis='y', labelcolor=color)

    leg = plt.legend()
    plt.show()

# OTHER EXPERIMENTS

def basic_experiment_no_growth():
    # Data
    X, Y = read_data('.\data\heart_failure_clinical_records_dataset.csv')
    X = convert_to_logical_values(X)
    # Model
    d_max = 1
    balance = 1.55
    lambda_param = 1
    ro = 1.0
    fixed_margin = False
    growth_policy = GrowthPolicy.ALL_TERMS
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=0.0, 
                                        max_no_terms_per_iteration=20, max_no_terms=200, growth_policy=growth_policy)
    
    # Evaluation
    no_runs = 100
    test_size = 0.2
    coefficient_threshold = 0.01
    precision = 2
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)

if __name__ == '__main__':
    experiment_with_growth()
    experiment_different_b_values()
