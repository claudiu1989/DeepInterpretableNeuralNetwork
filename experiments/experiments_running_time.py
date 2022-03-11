'''
This module contains experiments regarding the running time. 
See 7.7 Experiment 7: the running time analysis.

Before running the experiments, be sure you download and copy the CSV file located at https://archive.ics.uci.edu/ml/datasets/adult (Accessed 22.01.2022)
in the "data" folder.
'''

import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../')

from DeepInterpretablePolynomialNeuralNetwork.src.generate_synthetic_data import boolean_concept_uniform_distribution
from DeepInterpretablePolynomialNeuralNetwork.src.deep_interpretable_polynomial_neural_network import DeepInterpretablePolynomialNeuralNetwork, GrowthPolicy
from DeepInterpretablePolynomialNeuralNetwork.src.evaluation_tools import EvaluationTools

def read_data(path):
    X = []
    Y = []
    with open(path) as csvfile:
        all_rows = list(csv.reader(csvfile))
        numerical_features = [0,2,4,10,11,12]
        features_values = {i:[] for i in range(len(all_rows[0]))}
        features_minimum = {i:float('inf') for i in range(len(all_rows[0]))}
        features_maximum = {i:0.0 for i in range(len(all_rows[0]))}
        
        for row in all_rows:
            for i, feature in enumerate(row[:-1]):
                if i not in numerical_features:
                    if feature.strip().lower() not in features_values[i]:
                        features_values[i].append(feature.strip().lower())
                if i in numerical_features:
                    numerical_value = float(row[i])
                    if numerical_value < features_minimum[i]:
                        features_minimum[i] = numerical_value
                    if numerical_value > features_maximum[i]:
                        features_maximum[i] = numerical_value       
        features_names = []
        for i in range(len(all_rows[0])):
            if i == 0:
                features_names.append("age")
            if i == 2:
                features_names.append("fnlwgt")
            if i == 4:
                features_names.append("education-num")
            if i == 10:
                features_names.append("capital-gain")
            if i == 11:
                features_names.append("capital-loss")
            if i == 12:
                features_names.append("hours-per-week")
            if len(features_values[i]) > 0:
                for feature_value in features_values[i]:
                    features_names.append(feature_value)    
        
        # set labels and features
        for row in all_rows:
            # set the features
            cr_instance_features = []
            for i, value in enumerate(row):
                if i not in numerical_features:
                    cr_feature_values = features_values[i]
                    for feature_value in cr_feature_values:
                        if value.strip().lower() in feature_value:
                            cr_instance_features.append(1.0)
                        else:
                            cr_instance_features.append(0.0)
                else:
                    cr_instance_features.append((float(value)- features_minimum[i])/(features_maximum[i]- features_minimum[i]))
            X.append(cr_instance_features)
            # set the label
            if '<' in row[-1]:
                Y.append(0.0)
            else:
                Y.append(1.0)   
        '''
        X_header =  [features_names] + X
        with open("X_features.csv","w+") as my_csv:
            csvWriter = csv.writer(my_csv,delimiter=',')
            csvWriter.writerows(X_header)
        '''
        return np.array(X), np.array(Y), features_names

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
def experiment_running_time_vs_number_of_features():
    # Data
    X_original, Y, features_names = read_data('./data/adult.data')
    print(len(X_original))
    # Model
    d_max = 1
    balance = 2.0
    lambda_param = 10.0
    ro = 1.0
    fixed_margin = True
    growth_policy = GrowthPolicy.ALL_TERMS
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=0.0, 
                                        max_no_terms_per_iteration=20, max_no_terms=200, growth_policy=growth_policy)
    no_runs = 10
    test_size = 0.2
    coefficient_threshold = 0.01
    precision = 2
    n_features_values = range(5, 100, 5)
    avg_training_time_list = list()
    var_training_time_list = list()
    avg_test_time_list = list()
    var_test_time_list = list()
    for n_features in n_features_values:
        print(f'Experiment with {n_features} features')
        X = X_original[:,range(n_features)]
        # Evaluation
        avg_acc, avg_training_time, var_training_time, avg_test_time, var_test_time, avg_tp_rate, avg_tn_rate = EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
        avg_training_time_list.append(avg_training_time)
        var_training_time_list.append(np.sqrt(var_training_time))
        avg_test_time_list.append(avg_test_time)
        var_test_time_list.append(np.sqrt(var_test_time))
    avg_training_time_list = np.array(avg_training_time_list)
    var_training_time_list = np.array(var_training_time_list)
    avg_test_time_list = np.array(avg_test_time_list)
    var_test_time_list = np.array(var_test_time_list)
    fig1, ax1 = plt.subplots()
    ax1.plot(n_features_values,avg_training_time_list)
    ax1.fill_between(n_features_values, (avg_training_time_list-var_training_time_list), (avg_training_time_list+var_training_time_list), color='b', alpha=.1)
    plt.title('Evolution of training time with the number of features')
    plt.xlabel('Number of features')
    plt.ylabel('Average time (s)')

    fig2, ax2 = plt.subplots()
    ax2.plot(n_features_values,avg_test_time_list)
    ax2.fill_between(n_features_values, (avg_test_time_list-var_test_time_list), (avg_test_time_list+var_test_time_list), color='b', alpha=.1)
    plt.title('Evolution of test time with the number of features')
    plt.xlabel('Number of features')
    plt.ylabel('Average time (s)')
    plt.show()

def experiment_running_time_vs_number_of_data_points():
    # Data
    X_original, Y_original, features_names = read_data('./data/adult.data')
    print(len(X_original[1]))
    # Model
    d_max = 1
    balance = 2.0
    lambda_param = 10.0
    ro = 1.0
    fixed_margin = True
    growth_policy = GrowthPolicy.ALL_TERMS
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=0.0, 
                                        max_no_terms_per_iteration=20, max_no_terms=200, growth_policy=growth_policy)
    no_runs = 10
    test_size = 0.2
    coefficient_threshold = 0.01
    precision = 2
    n_data_points_values = range(50, 30000, 5000)
    avg_training_time_list = list()
    var_training_time_list = list()
    avg_test_time_list = list()
    var_test_time_list = list()
    for n_points in n_data_points_values:
        print(f'Experiment with {n_points} data points')
        X = X_original[range(n_points),:]
        Y = Y_original[range(n_points)]
        # Evaluation
        avg_acc, avg_training_time, var_training_time, avg_test_time, var_test_time, avg_tp_rate, avg_tn_rate = EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
        avg_training_time_list.append(avg_training_time)
        var_training_time_list.append(np.sqrt(var_training_time))
        avg_test_time_list.append(avg_test_time)
        var_test_time_list.append(np.sqrt(var_test_time))
    avg_training_time_list = np.array(avg_training_time_list)
    var_training_time_list = np.array(var_training_time_list)
    avg_test_time_list = np.array(avg_test_time_list)
    var_test_time_list = np.array(var_test_time_list)
    fig1, ax1 = plt.subplots()
    ax1.plot(n_data_points_values,avg_training_time_list)
    ax1.fill_between(n_data_points_values, (avg_training_time_list-var_training_time_list), (avg_training_time_list+var_training_time_list), color='b', alpha=.1)
    plt.title('Evolution of training time with the number of data points')
    plt.xlabel('Number of data points')
    plt.ylabel('Average time (s)')

    fig2, ax2 = plt.subplots()
    ax2.plot(n_data_points_values,avg_test_time_list)
    ax2.fill_between(n_data_points_values, (avg_test_time_list-var_test_time_list), (avg_test_time_list+var_test_time_list), color='b', alpha=.1)
    plt.title('Evolution of test time with the number of data points')
    plt.xlabel('Number of data points')
    plt.ylabel('Average time (s)')
    plt.show()

def experiment_running_time_vs_max_degree_grow():
    # Data
    X_original, Y, features_names = read_data('./data/adult.data')
    X = X_original[:, :3]
    print(len(X))
    # Parameters
    balance = 1.0
    lambda_param = 0.01
    ro = 1.0
    fixed_margin = True
    growth_policy = GrowthPolicy.GROW
    no_runs = 10
    test_size = 0.2
    coefficient_threshold = 0.01
    precision = 2
    d_max_range = range(1,6,1)
    avg_training_time_list = list()
    var_training_time_list = list()
    avg_test_time_list = list()
    var_test_time_list = list()
    for d_max in d_max_range:
        print(f'Experiment with max degree {d_max}')
        dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=0.0, 
                                        max_no_terms_per_iteration=20, max_no_terms=200, growth_policy=growth_policy)
        # Evaluation
        avg_acc, avg_training_time, var_training_time, avg_test_time, var_test_time, avg_tp_rate, avg_tn_rate = EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
        avg_training_time_list.append(avg_training_time)
        var_training_time_list.append(np.sqrt(var_training_time))
        avg_test_time_list.append(avg_test_time)
        var_test_time_list.append(np.sqrt(var_test_time))
    avg_training_time_list = np.array(avg_training_time_list)
    var_training_time_list = np.array(var_training_time_list)
    avg_test_time_list = np.array(avg_test_time_list)
    var_test_time_list = np.array(var_test_time_list)
    fig1, ax1 = plt.subplots()
    ax1.plot(d_max_range,avg_training_time_list, marker='o')
    ax1.fill_between(d_max_range, (avg_training_time_list-var_training_time_list), (avg_training_time_list+var_training_time_list), color='b', alpha=.1)
    plt.title('Evolution of training time with the maximum degree')
    plt.xlabel('d_max')
    plt.ylabel('Average time (s)')

    fig2, ax2 = plt.subplots()
    ax2.plot(d_max_range,avg_test_time_list, marker='o')
    ax2.fill_between(d_max_range, (avg_test_time_list-var_test_time_list), (avg_test_time_list+var_test_time_list), color='b', alpha=.1)
    plt.title('Evolution of test time with the maximum degree')
    plt.xlabel('d_max')
    plt.ylabel('Average time (s)')
    plt.show()

# OTHER EXPERIMENTS
def experiment_running_time_vs_max_degree_all_terms():
    # Data
    X_original, Y, features_names = read_data('./data/adult.data')
    X = X_original[:, :3]
    # Parameters
    balance = 1.0
    lambda_param = 10.0
    ro = 1.0
    fixed_margin = True
    growth_policy = GrowthPolicy.ALL_TERMS
    no_runs = 10
    test_size = 0.2
    coefficient_threshold = 0.01
    precision = 2
    d_max_range = range(1,6,1)
    avg_training_time_list = list()
    var_training_time_list = list()
    avg_test_time_list = list()
    var_test_time_list = list()
    for d_max in d_max_range:
        print(f'Experiment with max degree {d_max}')
        dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=0.0, 
                                        max_no_terms_per_iteration=20, max_no_terms=200, growth_policy=growth_policy)
        # Evaluation
        avg_acc, avg_training_time, var_training_time, avg_test_time, var_test_time, avg_tp_rate, avg_tn_rate = EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
        avg_training_time_list.append(avg_training_time)
        var_training_time_list.append(np.sqrt(var_training_time))
        avg_test_time_list.append(avg_test_time)
        var_test_time_list.append(np.sqrt(var_test_time))
    avg_training_time_list = np.array(avg_training_time_list)
    var_training_time_list = np.array(var_training_time_list)
    avg_test_time_list = np.array(avg_test_time_list)
    var_test_time_list = np.array(var_test_time_list)
    fig1, ax1 = plt.subplots()
    ax1.plot(d_max_range,avg_training_time_list, marker='o')
    ax1.fill_between(d_max_range, (avg_training_time_list-var_training_time_list), (avg_training_time_list+var_training_time_list), color='b', alpha=.1)
    plt.title('Evolution of training time with the maximum degree')
    plt.xlabel('d_max')
    plt.ylabel('Average time (s)')

    fig2, ax2 = plt.subplots()
    ax2.plot(d_max_range,avg_test_time_list, marker='o')
    ax2.fill_between(d_max_range, (avg_test_time_list-var_test_time_list), (avg_test_time_list+var_test_time_list), color='b', alpha=.1)
    plt.title('Evolution of test time with the maximum degree')
    plt.xlabel('d_max')
    plt.ylabel('Average time (s)')
    plt.show()
if __name__ == '__main__':
    experiment_running_time_vs_number_of_data_points()
    experiment_running_time_vs_max_degree_grow()
    experiment_running_time_vs_number_of_features()