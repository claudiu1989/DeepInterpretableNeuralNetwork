'''
This module contains experiments with the CORELS algorithm on the COMPAS dataset (https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis). 
See 7.6 Experiment 6: Comparison with Learning Certifiably Optimal Rule Lists.

The dataset is available at https://corels.eecs.harvard.edu/corels/compas.csv .
Before running the experiments, be sure you download and copy the CSV file in the "data" folder.
'''

import time
import numpy as np
import corels
from sklearn.model_selection import KFold


def experiment_corels():
    n_folds = 5
    X, Y, features, prediction = corels.load_from_csv("./data/compas-binary.csv")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=30)
    kf.get_n_splits(X)
    for max_card in  [1,2,3,4]:
        for c in[0.005,0.01]:
            acc_list = list()
            time_list = list()
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                start = time.time()
                corels_model = corels.CorelsClassifier(c=c, n_iter=10000000, max_card=max_card, min_support=0.01)
                train_acc = corels_model.fit(X_train, Y_train).score(X_train, Y_train)
                end = time.time()
                test_acc = corels_model.score(X_test, Y_test)
                acc_list.append(test_acc)
                time_list.append(end-start)
            avg_acc = sum(acc_list)/float(n_folds)
            var_acc = (np.sum((np.array(acc_list) - avg_acc)**2))/float(n_folds) 
            avg_time = sum(time_list)/float(n_folds)
            print(f'Average accuracy: {avg_acc}')
            print(f'Variance of accuracy: {var_acc}')
            print(f'Average training time: {avg_time}')

if __name__ == '__main__':
    experiment_corels()