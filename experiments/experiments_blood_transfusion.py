import csv
import numpy as np
from simple_interpretable_classifier import GrowthPolicy

from simple_interpretable_classifier import SimpleInterpretableClassifier

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
            Y.append(-1.0)
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

def basic_experiment_no_growth():
    d_max = 2
    balance = 2.0
    lambda_param = 10.0
    ro = 1.0
    fixed_margin = False
    sip = SimpleInterpretableClassifier(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.0, coeff_magnitude_th=0.0, 
                                        max_no_terms_per_iteration=10, max_no_terms=200, growth_policy=GrowthPolicy.ALL_TERMS)
    X, Y = read_data('./data/BloodTransfusion/transfusion.data')
    X = convert_to_logical_values(X)
    no_runs = 1
    sip.evaluate_multiple_times(X, Y, no_runs)
    print('Optimal w:')
    w_optim_trimed = [w if w>0.01 else 0.0 for w in sip.w_optimal]
    print(w_optim_trimed)
    print(f'Ro:{sip.ro}')


if __name__ == '__main__':
    basic_experiment_no_growth()
