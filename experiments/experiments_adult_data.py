import csv
import numpy as np

from simple_interpretable_classifier import SimpleInterpretableClassifier, GrowthPolicy

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
                Y.append(-1.0)
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

def basic_test_all_terms():
    X, Y, features_names = read_data('./Data/adult.data')
    d_max = 1
    balance = 1.0
    lambda_param = 10.0
    ro = 1.0
    fixed_margin = True
    sip = SimpleInterpretableClassifier(d_max, lambda_param, balance, fixed_margin, ro, derivative_magnitude_th=0.0, growth_policy=GrowthPolicy.ALL_TERMS)
    no_runs = 1
    sip.evaluate_multiple_times(X, Y, no_runs)
    w_optim_trimed = [w if w>0.01 else 0.0 for w in sip.w_optimal]
    print('Optimal w:')
    print(w_optim_trimed)
    '''
    indexes_non_zero = [i for i, e in enumerate(w_optim_trimed) if e > 0.0]
    features_names.extend(['neg_' + feature for feature in features_names])
    important_features = [(features_names[i],w_optim_trimed[i]) for i in  indexes_non_zero]
    print(w_optim_trimed)
    print('Important features:')
    print(important_features)
    '''
    print(f'Ro:{sip.ro}')

def basic_test_growth():
    X, Y, features_names = read_data('./Data/adult.data')
    d_max = 1
    balance = 1.0
    lambda_param = 1.0
    ro = 1.0
    fixed_margin = True
    sip = SimpleInterpretableClassifier(d_max=d_max, lambda_param=lambda_param, balance=balance, fixed_margin=fixed_margin, ro=ro, derivative_magnitude_th=0.02, coeff_magnitude_th=0.02, 
                                        max_no_terms_per_iteration=20, max_no_terms=300, growth_policy=GrowthPolicy.SELECT_BY_DERIVATIVE)
    no_runs = 1
    sip.evaluate_multiple_times(X, Y, no_runs)
    print(f'Ro:{sip.ro}')

if __name__ == '__main__':
   basic_test_growth()
   
    