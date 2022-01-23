import xlrd
import csv
import sys
import numpy as np
sys.path.append('../')

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
    return np.array(X), np.array(Y), data[0][1:-4]

def read_data_imputation(path_original, path_imputation):
    with open(path_original) as csvfile:
        data = list(csv.reader(csvfile))
        Y = []
        for row in data[1:]:
            try:
                Y.append(classes_encoding[row[-1].strip()])
            except:
                pass
    with open(path_imputation) as csvfile:
        data = list(csv.reader(csvfile))
        X = []
        for row in data[1:]:
            X.append([float(value) for value in row])

    return np.array(X), np.array(Y)

def create_dataset_with_data_imputation(input_path, output_path):
    X,_,_ = read_data(input_path)
    averages = np.average(X, axis=0)
    X_imputed = []
    with open(input_path) as csvfile:
        data = list(csv.reader(csvfile))
        for row in data[1:]:
            new_row = []
            for i,value in enumerate(row[1:-4]):
                try:
                    new_value = float(value)
                except:
                    new_value = averages[i]
                new_row.append(new_value)
            X_imputed.append(new_row)
    with open(output_path, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        for row in X_imputed:
            wr.writerow(row)   


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

def experiment_no_growth_degree1_no_fixed_margin_c_CS_sm_vs_c_SC_sm():
    ''' 
    c-CS-s,m (classes 0 and 2) vs c-SC-s,m (classes 1 and 3)

    c-CS-s: control mice, stimulated to learn, injected with saline (9 mice)
    c-CS-m: control mice, stimulated to learn, injected with memantine (10 mice)
    c-SC-s: control mice, not stimulated to learn, injected with saline (9 mice)
    c-SC-m: control mice, not stimulated to learn, injected with memantine (10 mice) 
    '''

    # Data
    _, _, proteins_names = read_data('./data/MiceProteins/Data_Cortex_Nuclear.csv')
    print(list(zip(proteins_names, range(len(proteins_names)))))
    X, Y = read_data_imputation('./data/MiceProteins/Data_Cortex_Nuclear.csv', './data/MiceProteins/Data_Cortex_Nuclear_imputed_X.csv')
    X = convert_to_logical_values(X)
    Z = zip(X,Y)
    Z = [(x,y) for x,y in Z if y<=3]
    X = [x for x,y in Z ]
    Y = np.array([1.0 if y==0 or y==2 else 0.0 for x,y in Z])
    # Model
    d_max = 1
    balance = 1.0
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
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
    print(f'The margin: {dipnn.ro}')

def experiment_growth_degree2_no_fixed_margin_c_CS_sm_vs_c_SC_sm():
    ''' 
    c-CS-s,m (classes 0 and 2) vs c-SC-s,m (classes 1 and 3)

    c-CS-s: control mice, stimulated to learn, injected with saline (9 mice)
    c-CS-m: control mice, stimulated to learn, injected with memantine (10 mice)
    c-SC-s: control mice, not stimulated to learn, injected with saline (9 mice)
    c-SC-m: control mice, not stimulated to learn, injected with memantine (10 mice) 
    '''

    # Data
    _, _, proteins_names = read_data('./data/MiceProteins/Data_Cortex_Nuclear.csv')
    print(list(zip(proteins_names, range(len(proteins_names)))))
    X, Y = read_data_imputation('./data/MiceProteins/Data_Cortex_Nuclear.csv', './data/MiceProteins/Data_Cortex_Nuclear_imputed_X.csv')
    X = convert_to_logical_values(X)
    Z = zip(X,Y)
    Z = [(x,y) for x,y in Z if y<=3]
    X = [x for x,y in Z ]
    Y = np.array([1.0 if y==0 or y==2 else 0.0 for x,y in Z])
    # Model
    d_max = 2
    balance = 1.0
    lambda_param = 1.0
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

def experiment_growth_degree3_no_fixed_margin_c_CS_sm_vs_c_SC_sm():
    ''' 
    c-CS-s,m (classes 0 and 2) vs c-SC-s,m (classes 1 and 3)

    c-CS-s: control mice, stimulated to learn, injected with saline (9 mice)
    c-CS-m: control mice, stimulated to learn, injected with memantine (10 mice)
    c-SC-s: control mice, not stimulated to learn, injected with saline (9 mice)
    c-SC-m: control mice, not stimulated to learn, injected with memantine (10 mice) 
    '''

    # Data
    _, _, proteins_names = read_data('./data/MiceProteins/Data_Cortex_Nuclear.csv')
    print(list(zip(proteins_names, range(len(proteins_names)))))
    X, Y = read_data_imputation('./data/MiceProteins/Data_Cortex_Nuclear.csv', './data/MiceProteins/Data_Cortex_Nuclear_imputed_X.csv')
    X = convert_to_logical_values(X)
    Z = zip(X,Y)
    Z = [(x,y) for x,y in Z if y<=3]
    X = [x for x,y in Z ]
    Y = np.array([1.0 if y==0 or y==2 else 0.0 for x,y in Z])
    # Model
    d_max = 3
    balance = 1.0
    lambda_param = 1.0
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

def experiment_no_growth_degree1_no_fixed_margin_c_SC_m_vs_c_SC_s():
    ''' 
    c-SC-m (class 1) vs c-SC-s (classes 3)

    c-SC-s: control mice, not stimulated to learn, injected with saline (9 mice)
    c-SC-m: control mice, not stimulated to learn, injected with memantine (10 mice) 
    '''

    # Data
    _, _, proteins_names = read_data('./data/MiceProteins/Data_Cortex_Nuclear.csv')
    print(list(zip(proteins_names, range(len(proteins_names)))))
    X, Y = read_data_imputation('./data/MiceProteins/Data_Cortex_Nuclear.csv', './data/MiceProteins/Data_Cortex_Nuclear_imputed_X.csv')
    X = convert_to_logical_values(X)
    Z = zip(X,Y)
    Z = [(x,y) for x,y in Z if y==1 or y==3]
    X = [x for x,y in Z ]
    Y = np.array([1.0 if y==1 else 0.0 for x,y in Z])
    # Model
    d_max = 1
    balance = 1.0
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
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
    print(f'The margin: {dipnn.ro}')

def experiment_growth_degree2_no_fixed_margin_c_SC_m_vs_c_SC_s():
    ''' 
    c-SC-m (class 1) vs c-SC-s (classes 3)

    c-SC-s: control mice, not stimulated to learn, injected with saline (9 mice)
    c-SC-m: control mice, not stimulated to learn, injected with memantine (10 mice) 
    '''

    # Data
    _, _, proteins_names = read_data('./data/MiceProteins/Data_Cortex_Nuclear.csv')
    print(list(zip(proteins_names, range(len(proteins_names)))))
    X, Y = read_data_imputation('./data/MiceProteins/Data_Cortex_Nuclear.csv', './data/MiceProteins/Data_Cortex_Nuclear_imputed_X.csv')
    X = convert_to_logical_values(X)
    Z = zip(X,Y)
    Z = [(x,y) for x,y in Z if y==1 or y==3]
    X = [x for x,y in Z ]
    Y = np.array([1.0 if y==1 else 0.0 for x,y in Z])
    # Model
    d_max = 2
    balance = 1.0
    lambda_param = 1.0
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
    print(f'The margin: {dipnn.ro}')

def experiment_growth_degree3_no_fixed_margin_c_SC_m_vs_c_SC_s():
    ''' 
    c-SC-m (class 1) vs c-SC-s (classes 3)

    c-SC-s: control mice, not stimulated to learn, injected with saline (9 mice)
    c-SC-m: control mice, not stimulated to learn, injected with memantine (10 mice) 
    '''

    # Data
    _, _, proteins_names = read_data('./data/MiceProteins/Data_Cortex_Nuclear.csv')
    print(list(zip(proteins_names, range(len(proteins_names)))))
    X, Y = read_data_imputation('./data/MiceProteins/Data_Cortex_Nuclear.csv', './data/MiceProteins/Data_Cortex_Nuclear_imputed_X.csv')
    X = convert_to_logical_values(X)
    Z = zip(X,Y)
    Z = [(x,y) for x,y in Z if y==1 or y==3]
    X = [x for x,y in Z ]
    Y = np.array([1.0 if y==1 else 0.0 for x,y in Z])
    # Model
    d_max = 3
    balance = 1.0
    lambda_param = 1.0
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
    print(f'The margin: {dipnn.ro}')

def experiment_no_growth_degree1_no_fixed_margin_c_SC_m_vs_c_SC_s():
    ''' 
    c-CS-m (class 0) vs c-CS-s (class 2)

    c-CS-s: control mice, stimulated to learn, injected with saline (9 mice) 
    c-CS-m: control mice, stimulated to learn, injected with memantine (10 mice)  
    '''

    # Data
    _, _, proteins_names = read_data('./data/MiceProteins/Data_Cortex_Nuclear.csv')
    print(list(zip(proteins_names, range(len(proteins_names)))))
    X, Y = read_data_imputation('./data/MiceProteins/Data_Cortex_Nuclear.csv', './data/MiceProteins/Data_Cortex_Nuclear_imputed_X.csv')
    X = convert_to_logical_values(X)
    Z = zip(X,Y)
    Z = [(x,y) for x,y in Z if y==0 or y==2]
    X = [x for x,y in Z ]
    Y = np.array([1.0 if y==0 else 0.0 for x,y in Z])
    # Model
    d_max = 1
    balance = 1.0
    lambda_param = 1.0
    ro = 1.0
    fixed_margin = False
    growth_policy = GrowthPolicy.ALL_TERMS
    dipnn = DeepInterpretablePolynomialNeuralNetwork(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=0.0, 
                                        max_no_terms_per_iteration=20, max_no_terms=200, growth_policy=growth_policy)
    
    # Evaluation
    no_runs = 3
    test_size = 0.2
    coefficient_threshold = 0.01
    precision = 2
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
    print(f'The margin: {dipnn.ro}')

def experiment_no_growth_degree2_no_fixed_margin_c_SC_m_vs_c_SC_s():
    ''' 
    c-CS-m (class 0) vs c-CS-s (class 2)

    c-CS-s: control mice, stimulated to learn, injected with saline (9 mice) 
    c-CS-m: control mice, stimulated to learn, injected with memantine (10 mice)  
    '''

    # Data
    _, _, proteins_names = read_data('./data/MiceProteins/Data_Cortex_Nuclear.csv')
    print(list(zip(proteins_names, range(len(proteins_names)))))
    X, Y = read_data_imputation('./data/MiceProteins/Data_Cortex_Nuclear.csv', './data/MiceProteins/Data_Cortex_Nuclear_imputed_X.csv')
    X = convert_to_logical_values(X)
    Z = zip(X,Y)
    Z = [(x,y) for x,y in Z if y==0 or y==2]
    X = [x for x,y in Z ]
    Y = np.array([1.0 if y==0 else 0.0 for x,y in Z])
    # Model
    d_max = 2
    balance = 1.0
    lambda_param = 0.5
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

def experiment_no_growth_degree1_no_fixed_margin_t_CS_m_vs_t_SC_s_m():
    ''' 
    t-CS-m (class 4) vs t-SC-s,m (class 5 and class 7)

    t-CS-m: trisomy mice, stimulated to learn, injected with memantine (9 mice)  
    t-SC-s: trisomy mice, not stimulated to learn, injected with saline (9 mice) 
    t-SC-m: trisomy mice, not stimulated to learn, injected with memantine (9 mice)   
    '''

    # Data
    _, _, proteins_names = read_data('./data/MiceProteins/Data_Cortex_Nuclear.csv')
    print(list(zip(proteins_names, range(len(proteins_names)))))
    X, Y = read_data_imputation('./data/MiceProteins/Data_Cortex_Nuclear.csv', './data/MiceProteins/Data_Cortex_Nuclear_imputed_X.csv')
    X = convert_to_logical_values(X)
    Z = zip(X,Y)
    Z = [(x,y) for x,y in Z if y==4 or y==5 or y == 7]
    X = [x for x,y in Z ]
    Y = np.array([1.0 if y==4 else 0.0 for x,y in Z])
    # Model
    d_max = 1
    balance = 1.0
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
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
    print(f'The margin: {dipnn.ro}')

def experiment_no_growth_degree1_no_fixed_margin_t_SC_m_vs_t_SC_s():
    ''' 
    t-SC-m (class 5) vs t-SC-s (class 7)

    t-SC-s: trisomy mice, not stimulated to learn, injected with saline (9 mice) 
    t-SC-m: trisomy mice, not stimulated to learn, injected with memantine (9 mice)   
    '''

    # Data
    _, _, proteins_names = read_data('./data/MiceProteins/Data_Cortex_Nuclear.csv')
    print(list(zip(proteins_names, range(len(proteins_names)))))
    X, Y = read_data_imputation('./data/MiceProteins/Data_Cortex_Nuclear.csv', './data/MiceProteins/Data_Cortex_Nuclear_imputed_X.csv')
    X = convert_to_logical_values(X)
    Z = zip(X,Y)
    Z = [(x,y) for x,y in Z if  y==5 or y == 7]
    X = [x for x,y in Z ]
    Y = np.array([1.0 if y==5 else 0.0 for x,y in Z])
    # Model
    d_max = 1
    balance = 1.0
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
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
    print(f'The margin: {dipnn.ro}')

def experiment_no_growth_degree1_no_fixed_margin_t_CS_m_vs_t_CS_s():
    ''' 
    t-CS-m (class 4) vs t-CS-s (class 6)

    t-CS-m: trisomy mice, stimulated to learn, injected with memantine (9 mice)  
    t-CS-s: trisomy mice, stimulated to learn, injected with saline (7 mice)    
    '''

    # Data
    _, _, proteins_names = read_data('./data/MiceProteins/Data_Cortex_Nuclear.csv')
    print(list(zip(proteins_names, range(len(proteins_names)))))
    X, Y = read_data_imputation('./data/MiceProteins/Data_Cortex_Nuclear.csv', './data/MiceProteins/Data_Cortex_Nuclear_imputed_X.csv')
    X = convert_to_logical_values(X)
    Z = zip(X,Y)
    Z = [(x,y) for x,y in Z if  y==4 or y == 6]
    X = [x for x,y in Z ]
    Y = np.array([1.0 if y==4 else 0.0 for x,y in Z])
    # Model
    d_max = 1
    balance = 1.0
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
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
    print(f'The margin: {dipnn.ro}')

def experiment_no_growth_degree1_no_fixed_margin_t_CS_s_vs_c_CS_s_m():
    ''' 
    t-CS-s (class 6) vs c-CS-s,m (classes 0, 2)
    
    t-CS-s: trisomy mice, stimulated to learn, injected with saline (7 mice) 
    c-CS-m: control mice, stimulated to learn, injected with memantine (10 mice)  
    c-CS-s: control mice, stimulated to learn, injected with saline (9 mice)     
    '''

    # Data
    _, _, proteins_names = read_data('./data/MiceProteins/Data_Cortex_Nuclear.csv')
    print(list(zip(proteins_names, range(len(proteins_names)))))
    X, Y = read_data_imputation('./data/MiceProteins/Data_Cortex_Nuclear.csv', './data/MiceProteins/Data_Cortex_Nuclear_imputed_X.csv')
    X = convert_to_logical_values(X)
    Z = zip(X,Y)
    Z = [(x,y) for x,y in Z if  y==0 or y==2 or y == 6]
    X = [x for x,y in Z ]
    Y = np.array([1.0 if y == 6 else 0.0 for x,y in Z])
    # Model
    d_max = 1
    balance = 2.0
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
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
    print(f'The margin: {dipnn.ro}')

def experiment_no_growth_degree1_no_fixed_margin_t_SC_s_vs_t_SC_m_and_c_SC_s_m():
    ''' 
    t-SC-s (class 7) vs t-SC-m, c-SC-s,m(classes 5, 1, 3)
    
    t-SC-s: trisomy mice, not stimulated to learn, injected with saline (9 mice) 
    t-SC-m: trisomy mice, not stimulated to learn, injected with memantine (9 mice)   
    c-SC-s: control mice, not stimulated to learn, injected with saline (9 mice)   
    c-SC-m: control mice, not stimulated to learn, injected with memantine (10 mice)    
    '''

    # Data
    _, _, proteins_names = read_data('./data/MiceProteins/Data_Cortex_Nuclear.csv')
    print(list(zip(proteins_names, range(len(proteins_names)))))
    X, Y = read_data_imputation('./data/MiceProteins/Data_Cortex_Nuclear.csv', './data/MiceProteins/Data_Cortex_Nuclear_imputed_X.csv')
    X = convert_to_logical_values(X)
    Z = zip(X,Y)
    Z = [(x,y) for x,y in Z if  y==7 or y==5 or y == 1 or y == 3]
    X = [x for x,y in Z ]
    Y = np.array([1.0 if y == 7 else 0.0 for x,y in Z])
    # Model
    d_max = 1
    balance = 2.0
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
    EvaluationTools.evaluate_multiple_times(dipnn, X, Y, no_runs, test_size, coefficient_threshold, precision)
    print(f'The margin: {dipnn.ro}')

if __name__ == '__main__':
   '''
   X, Y = read_data('./data/MiceProteins/Data_Cortex_Nuclear.csv')
   X = convert_to_logical_values(X)
   Y = convert_dataset_for_binary_classification(Y, 0.0)
   basic_experiment_no_growth(X,Y)
   '''
   #experiment_no_growth_degree1_no_fixed_margin_c_CS_sm_vs_c_SC_sm()
   #experiment_growth_degree2_no_fixed_margin_c_CS_sm_vs_c_SC_sm()
   #experiment_growth_degree3_no_fixed_margin_c_CS_sm_vs_c_SC_sm()

   #experiment_no_growth_degree1_no_fixed_margin_c_SC_m_vs_c_SC_s()
   #experiment_growth_degree2_no_fixed_margin_c_SC_m_vs_c_SC_s()
   #experiment_growth_degree3_no_fixed_margin_c_SC_m_vs_c_SC_s()

   #experiment_no_growth_degree1_no_fixed_margin_t_CS_m_vs_t_SC_s_m()

   #experiment_no_growth_degree1_no_fixed_margin_t_SC_m_vs_t_SC_s()

   #experiment_no_growth_degree1_no_fixed_margin_t_CS_m_vs_t_CS_s()

   #experiment_no_growth_degree1_no_fixed_margin_t_CS_s_vs_c_CS_s_m()

   experiment_no_growth_degree1_no_fixed_margin_t_SC_s_vs_t_SC_m_and_c_SC_s_m()