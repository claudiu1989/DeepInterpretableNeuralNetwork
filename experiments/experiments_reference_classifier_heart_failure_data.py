import sys
sys.path.append('../')
from DeepInterpretablePolynomialNeuralNetwork.src.deep_learning_reference_classifier import DeepLearningReferenceClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import csv

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
    epochs = 50
    dlc = DeepLearningReferenceClassifier()
    no_runs = 100
    test_size = 0.2
    class_weights = {0: 1.0,
                1: 2.0}
    dlc.evaluate_multiple_times(X, Y, no_runs, test_size, epochs, class_weights)

if __name__ == '__main__': 
    experiment_on_heart_attack_dataset_reference_classifier()
    