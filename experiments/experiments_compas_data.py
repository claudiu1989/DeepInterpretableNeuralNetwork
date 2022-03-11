'''
This module contains experiments regarding with the COMPAS dataset (https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis). 
See 7.5 Experiment 5: comparison with Neural Additive Models method and 7.6 Experiment 6: Comparison with Learning Certifiably Optimal Rule Lists.

Two preprocessed versions of the dataset are used. One can be downloaded using the code available at https://github.com/google-research/google-research/blob/master/neural_additive_models/data_utils.py (Accessed 25.02.2022)- see the function "load_recidivism_data()".
The second one is available at https://corels.eecs.harvard.edu/corels/compas.csv .
Before running the experiments, be sure you download and copy the CSV files in the "data" folder.
'''

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import corels

sys.path.append('../')

from DeepInterpretablePolynomialNeuralNetwork.src.deep_interpretable_polynomial_neural_network import DeepInterpretablePolynomialNeuralNetwork, GrowthPolicy
from DeepInterpretablePolynomialNeuralNetwork.src.evaluation_tools import EvaluationTools

def read_data(path):
    data = pd.read_csv(path, delimiter=';')
    race_data = pd.get_dummies(data['race'], prefix='race')
    sex_data = pd.get_dummies(data['sex'], prefix='sex')
    charge_data = pd.get_dummies(data['c_charge_degree'], prefix='c_charge_degree')
    data = pd.concat([data,race_data,sex_data,charge_data],axis=1)
    data = data.reindex(columns=['age','sex','race','priors_count','length_of_stay','c_charge_degree','two_year_recid','race_1','race_2','race_3','race_4','race_5','race_6','sex_1','sex_2','c_charge_degree_1','c_charge_degree_2'])
    data.drop(['race', 'sex', 'c_charge_degree'], axis=1, inplace=True)
    data.to_csv('final_data_compas.csv')
    normalized_data = (data-data.min())/(data.max()-data.min())
    print(list(zip(normalized_data.loc[:, normalized_data.columns != 'two_year_recid'].columns, range(len(data.columns)))))
    Y = normalized_data['two_year_recid'].to_numpy()
    X = normalized_data.loc[:, normalized_data.columns != 'two_year_recid'].to_numpy()
    return X, Y

def read_data_male(path):
    data = pd.read_csv(path, delimiter=';')
    data = data[data['sex']==2]
    race_data = pd.get_dummies(data['race'], prefix='race')
    charge_data = pd.get_dummies(data['c_charge_degree'], prefix='c_charge_degree')
    data = pd.concat([data,race_data,charge_data],axis=1)
    data = data.reindex(columns=['age','sex','race','priors_count','length_of_stay','c_charge_degree','two_year_recid','race_1','race_2','race_3','race_4','race_5','race_6','c_charge_degree_1','c_charge_degree_2'])
    data.drop(['race', 'sex', 'c_charge_degree'], axis=1, inplace=True)
    data.to_csv('final_data_compas.csv')
    normalized_data = (data-data.min())/(data.max()-data.min())
    print(list(zip(normalized_data.loc[:, normalized_data.columns != 'two_year_recid'].columns, range(len(data.columns)))))
    Y = normalized_data['two_year_recid'].to_numpy()
    X = normalized_data.loc[:, normalized_data.columns != 'two_year_recid'].to_numpy()
    return X, Y

def read_data_female(path):
    data = pd.read_csv(path, delimiter=';')
    data = data[data['sex']==1]
    race_data = pd.get_dummies(data['race'], prefix='race')
    charge_data = pd.get_dummies(data['c_charge_degree'], prefix='c_charge_degree')
    data = pd.concat([data,race_data,charge_data],axis=1)
    data = data.reindex(columns=['age','sex','race','priors_count','length_of_stay','c_charge_degree','two_year_recid','race_1','race_2','race_3','race_4','race_5','race_6','c_charge_degree_1','c_charge_degree_2'])
    data.drop(['race', 'sex', 'c_charge_degree'], axis=1, inplace=True)
    data.to_csv('final_data_compas.csv')
    normalized_data = (data-data.min())/(data.max()-data.min())
    print(list(zip(normalized_data.loc[:, normalized_data.columns != 'two_year_recid'].columns, range(len(data.columns)))))
    Y = normalized_data['two_year_recid'].to_numpy()
    X = normalized_data.loc[:, normalized_data.columns != 'two_year_recid'].to_numpy()
    return X, Y

# EXPERIMENTS REPORTED IN THE PAPER

def experiment_growth_cross_validation_binary_all():                          
    path = "./data/compas-binary.csv"
    X, Y, features, prediction = corels.load_from_csv(path)
    # Model
    ro = 1.0
    n_folds = 5
    coefficient_threshold = 0.01
    precision = 2
    seed = 101 #100
    fixed_margin = False
    growth_policy = GrowthPolicy.GROW
    balance = 1.0
    lambda_param = 0.1
    for d_max in [6]:
        dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=0.0, 
                                            max_no_terms_per_iteration=20, max_no_terms=1000, growth_policy=growth_policy)
        
        # Evaluation
        avg_acc, avg_training_time, var_training_time, avg_test_time, var_test_time, avg_tp_rate, avg_tn_rate, avg_roc = EvaluationTools.evaluate_k_fold(dipnn, X, Y, n_folds, coefficient_threshold, precision, new_threshold=0.5, seed=seed)

def experiment_growth_cross_validation_all():                          
    path = './data/compas.csv'
    X, Y = read_data(path)
    # Model
    ro = 1.0
    fixed_margin = False
    growth_policy = GrowthPolicy.GROW
    balance = 1.0
    lambda_param = 0.1
    seed = 30
    for d_max in [6]:
        dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=0.0, 
                                            max_no_terms_per_iteration=500, max_no_terms=1000, growth_policy=growth_policy)
        # Evaluation
        n_folds = 5
        coefficient_threshold = 0.01
        precision = 2
        avg_acc, avg_training_time, var_training_time, avg_test_time, var_test_time, avg_tp_rate, avg_tn_rate, avg_roc = EvaluationTools.evaluate_k_fold(dipnn, X, Y, n_folds, coefficient_threshold, precision, new_threshold=0.5, seed=seed)

# OTHER EXPERIMENTS
def experiment_growth_best_results_female():                          
    path = './data/compas.csv'
    X, Y = read_data_female(path)
    # Model
    d_max = 3
    ro = 1.0
    fixed_margin = False
    growth_policy = GrowthPolicy.GROW
    balance = 1.0
    lambda_param = 0.1
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=0.0, 
                                        max_no_terms_per_iteration=100, max_no_terms=1000, growth_policy=growth_policy)
    
    # Evaluation
    no_runs = 100
    test_size = 0.2
    coefficient_threshold = 0.01
    precision = 2
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision, new_threshold=0.5)
    print(f'The margin: {dipnn.ro}')

def experiment_growth_best_results_male():                          
    path = './data/compas.csv'
    X, Y = read_data_male(path)
    # Model
    d_max = 3
    ro = 1.0
    fixed_margin = False
    growth_policy = GrowthPolicy.GROW
    balance = 1.0
    lambda_param = 0.1
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=0.0, 
                                        max_no_terms_per_iteration=100, max_no_terms=1000, growth_policy=growth_policy)
    
    # Evaluation
    no_runs = 10
    test_size = 0.2
    coefficient_threshold = 0.01
    precision = 2
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision, new_threshold=0.5)
    print(f'The margin: {dipnn.ro}')

def experiment_growth_best_results():                          
    path = './data/compas.csv'
    X, Y = read_data(path)
    # Model
    d_max = 4
    ro = 1.0
    fixed_margin = False
    growth_policy = GrowthPolicy.GROW
    balance = 1.0
    lambda_param = 0.1
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=0.0, 
                                        max_no_terms_per_iteration=100, max_no_terms=1000, growth_policy=growth_policy)
    
    # Evaluation
    no_runs = 100
    test_size = 0.2
    coefficient_threshold = 0.01
    precision = 2
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision, new_threshold=0.5)
    print(f'The margin: {dipnn.ro}')        

def experiment_growth_cross_validation_male():                          
    path = './data/compas.csv'
    X, Y = read_data_male(path)
    # Model
    d_max = 5
    ro = 1.0
    fixed_margin = False
    growth_policy = GrowthPolicy.GROW
    balance = 1.0
    lambda_param = 0.0
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=0.0, 
                                        max_no_terms_per_iteration=50, max_no_terms=1000, growth_policy=growth_policy)
    
    # Evaluation
    n_folds = 5
    coefficient_threshold = 0.01
    precision = 2
    EvaluationTools.evaluate_k_fold(dipnn, X, Y, n_folds, coefficient_threshold, precision, new_threshold=0.5)
    print(f'The margin: {dipnn.ro}')


def experiment_growth_cross_validation_female():                          
    path = './data/compas.csv'
    X, Y = read_data_female(path)
    # Model
    d_max = 3
    ro = 1.0
    fixed_margin = False
    growth_policy = GrowthPolicy.GROW
    balance = 1.0
    lambda_param = 0.1
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=0.0, 
                                        max_no_terms_per_iteration=50, max_no_terms=1000, growth_policy=growth_policy)
    
    # Evaluation
    n_folds = 5
    coefficient_threshold = 0.01
    precision = 2
    EvaluationTools.evaluate_k_fold(dipnn, X, Y, n_folds, coefficient_threshold, precision, new_threshold=0.5)
    print(f'The margin: {dipnn.ro}')


if __name__ == '__main__':
    experiment_growth_cross_validation_all()