'''
This module contains experiments with the dataset https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic) (Accessed 14.02.2022)
See section 7.4 Experiment 4: the role of hyperparameters.

Before running the experiments, be sure you download and copy the CSV file in the "data" folder.
'''

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
sys.path.append('../')

from DeepInterpretablePolynomialNeuralNetwork.src.deep_interpretable_polynomial_neural_network import DeepInterpretablePolynomialNeuralNetwork, GrowthPolicy
from DeepInterpretablePolynomialNeuralNetwork.src.evaluation_tools import EvaluationTools

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
def experiment_results_different_degrees():                          
    data = load_breast_cancer()
    X = convert_to_logical_values(data.data)
    Y = np.array(data.target)
    X = convert_to_logical_values(X)
   
    for d_max in [4]:
        # Model
        balance = 1.0
        lambda_param = 0.01
        ro = 1.0
        fixed_margin = False
        growth_policy = GrowthPolicy.GROW
        dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=0.0, 
                                            max_no_terms_per_iteration=20, max_no_terms=200, growth_policy=growth_policy)
        
        # Evaluation
        no_runs = 100
        test_size = 0.2
        coefficient_threshold = 0.01
        precision = 2
        EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
       
# OTHER EXPERIMENTS

def experiment_results_different_lambda_values():                          
    data = load_breast_cancer()
    X = convert_to_logical_values(data.data)
    Y = np.array(data.target)
    X = convert_to_logical_values(X)
    lambda_param_values = range(0,101)
    avg_acc_list = list()
    n_terms_list = list()
    for lambda_param in lambda_param_values:
        # Model
        d_max = 3
        balance = 1.0
        ro = 1.0
        fixed_margin = False
        growth_policy = GrowthPolicy.GROW
        dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=0.0, 
                                            max_no_terms_per_iteration=20, max_no_terms=200, growth_policy=growth_policy)
        # Evaluation
        no_runs = 10
        test_size = 0.2
        coefficient_threshold = 0.01
        precision = 2
        avg_acc, avg_training_time, var_training_time, avg_test_time, var_test_time, avg_tp_rate, avg_tn_rate = EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
        n_terms = len(dipnn.get_the_model_representation(coefficient_threshold, precision).split('+'))
        avg_acc_list.append(avg_acc)
        n_terms_list.append(n_terms)
    fig1, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.plot(lambda_param_values, avg_acc_list, color=color)
    plt.title('Accuracy and number of terms as a functon of labda parameter')
    ax1.set_xlabel('\u03BB')
    ax1.set_ylabel('Accuracy', color=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('No. of terms', color=color)  # we already handled the x-label with ax1
    ax2.plot(lambda_param_values, n_terms_list, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
   
    fig1.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

def experiment_results_different_b_values():                          
    data = load_breast_cancer()
    X = convert_to_logical_values(data.data)
    Y = np.array(data.target)
    X = convert_to_logical_values(X)
    b_param_values = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0,4.0,5.0,6.0,7.0,8.0, 9.0, 10.0]
    avg_acc_list = list()
    avg_tp_list = list()
    avg_tn_list = list()
    for b_param in b_param_values:
        print(f'Run with balance: {b_param}')
        # Model
        d_max = 1
        balance = b_param
        lambda_param = 0.01
        ro = 1.0
        fixed_margin = False
        growth_policy = GrowthPolicy.ALL_TERMS
        dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=0.0, 
                                            max_no_terms_per_iteration=20, max_no_terms=200, growth_policy=growth_policy)
        # Evaluation
        no_runs = 10
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

if __name__ == '__main__':
    experiment_results_different_degrees()