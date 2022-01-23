import sys
sys.path.append('../')
from DeepInterpretablePolynomialNeuralNetwork.src.deep_learning_reference_classifier import DeepLearningReferenceClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import csv

def read_data_adult(path):
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

def read_data_heart_failure(path):
    with open(path) as csvfile:
        data = list(csv.reader(csvfile))
    X = []
    Y = []
    for row in data[1:]:
        X.append([float(value) for value in row[:-2]])
        if float(row[-1]) > 0.0:
            Y.append(1)
        else:
            Y.append(0)
    return np.array(X), np.array(Y)

def experiment_on_heart_attack_dataset_reference_classifier():
    X, Y = read_data_heart_failure('.\Data\heart_failure_clinical_records_dataset.csv')
    #X, Y, _ = read_data_adult('.\Data\\adult.csv')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)
    epochs = 1
    no_classes = 2
    no_hidden_layers = 10
    size_of_hidden_layers = 5 #len(X_train[0])/2
    dlc = DeepLearningReferenceClassifier()
    no_runs = 100
    test_size = 0.2
    class_weights = {0: 1.0,
                1: 1.1}
    dlc.evaluate_multiple_times(X, Y, no_runs, test_size, epochs, no_classes, no_hidden_layers, size_of_hidden_layers, class_weights)

if __name__ == '__main__': 
    experiment_on_heart_attack_dataset_reference_classifier()
    