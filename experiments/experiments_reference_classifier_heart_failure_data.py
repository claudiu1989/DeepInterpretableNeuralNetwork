import sys
sys.path.append('../')
from DeepInterpretablePolynomialNeuralNetwork.src.deep_learning_reference_classifier import DeepLearningReferenceClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import csv

def read_data(path):
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

def evaluate():
    pass

if __name__ == '__main__': 
    X, Y = read_data('.\Data\heart_failure_clinical_records_dataset.csv')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)
    epochs = 10
    no_classes = 2
    no_hidden_layers = 3
    size_of_hidden_layers = len(X_train)/2
    dlc = DeepLearningReferenceClassifier()
    #dlc.train(X_train, Y_train, epochs, no_classes, no_hidden_layers, size_of_hidden_layers)
    no_runs = 1
    test_size = 0.2
    dlc.evaluate_multiple_times(X, Y, no_runs, test_size, epochs, no_classes, no_hidden_layers, size_of_hidden_layers)
