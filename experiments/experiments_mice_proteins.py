import xlrd
import csv
import sys
import numpy as np
sys.path.append('../')

from DeepInterpretablePolynomialNeuralNetwork.src.generate_synthetic_data import boolean_concept_uniform_distribution
from DeepInterpretablePolynomialNeuralNetwork.src.deep_interpretable_polynomial_neural_network import DeepInterpretablePolynomialNeuralNetwork, GrowthPolicy
from DeepInterpretablePolynomialNeuralNetwork.src.evaluation_tools import EvaluationTools

classes_encoding = {}
classes_encoding['c-CS-m'] = 0
classes_encoding['c-SC-m'] = 1
classes_encoding['c-CS-s'] = 2
classes_encoding['c-SC-s'] = 3
classes_encoding['t-CS-m'] = 4
classes_encoding['t-SC-m'] = 5
classes_encoding['t-CS-s'] = 6
classes_encoding['t-SC-s'] = 7

def csv_from_xls():
    xlsx_file = xlrd.open_workbook('./data/MiceProteins/Data_Cortex_Nuclear.xls')
    data = xlsx_file.sheet_by_name('Hoja1')
    with open('./data/MiceProteins/Data_Cortex_Nuclear.csv', 'w', newline='') as csv_file:
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        for rownum in range(data.nrows):
            wr.writerow(data.row_values(rownum))

def read_data(path):
    with open(path) as csvfile:
        data = list(csv.reader(csvfile))
    X = []
    Y = []
    for row in data[1:]:
        try:
          X.append([float(value) for value in row[1:-4]])
          Y.append(classes_encoding[row[-1].strip()])
        except:
            pass
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

def convert_dataset_for_binary_classification(Y, class_1):
    return np.array([1.0 if y==class_1 else 0.0 for y in Y])

def basic_experiment_no_growth(X,Y):
    d_max = 1
    balance = 10.0
    lambda_param = 10.0
    ro = 1.0
    fixed_margin = True
    sip = DeepInterpretablePolynomialNeuralNetwork(d_max, lambda_param, balance, fixed_margin, ro)
    no_runs = 1
    sip.evaluate_multiple_times(X, Y, no_runs)
    print('Optimal w:')
    w_optim_trimed = [w if w>0.01 else 0.0 for w in sip.w_optimal]
    print(w_optim_trimed)
    print(f'Ro:{sip.ro}')

def experiment_no_growth_degree1_fixed_margin():
    # Data
    X, Y = read_data('./data/MiceProteins/Data_Cortex_Nuclear.csv')
    X = convert_to_logical_values(X)
    Y = convert_dataset_for_binary_classification(Y, 0)

    # Model
    d_max = 1
    balance = 10.0
    lambda_param = 10.0
    ro = 1.0
    fixed_margin = True
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

def experiment_growth_degree2_no_fixed_margin():
    # Data
    X, Y = read_data('./data/MiceProteins/Data_Cortex_Nuclear.csv')
    X = convert_to_logical_values(X)
    Y = convert_dataset_for_binary_classification(Y, 0)

    # Model
    d_max = 2
    balance = 10.0
    lambda_param = 0.0
    ro = 1.0
    fixed_margin = False
    growth_policy = GrowthPolicy.GROW
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=0.0, 
                                        max_no_terms_per_iteration=20, max_no_terms=200, growth_policy=growth_policy)
    
    # Evaluation
    no_runs = 1
    test_size = 0.2
    coefficient_threshold = 0.01
    precision = 2
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
    print(f'The margin: {dipnn.ro}')

if __name__ == '__main__':
   '''
   X, Y = read_data('./data/MiceProteins/Data_Cortex_Nuclear.csv')
   X = convert_to_logical_values(X)
   Y = convert_dataset_for_binary_classification(Y, 0.0)
   basic_experiment_no_growth(X,Y)
   '''
   experiment_growth_degree2_no_fixed_margin()