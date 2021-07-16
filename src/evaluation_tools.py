
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np

class EvaluationTools:
    
    @staticmethod
    def test(dipnn, X_test, Y_test):
        Y_predicted_binary,Y_predicted = dipnn.predict(X_test)
        no_errors = 0.0 
        N = 0.0 
        P = 0.0 
        TN = 0.0
        TP = 0.0
        for y_p,y in zip(Y_predicted_binary,Y_test):
            if y_p != y:
                no_errors += 1.0
            if y == 1.0:
                P += 1.0
                if y_p == 1.0:
                    TP += 1.0
            if y == 0.0:
                N += 1.0
                if y_p == 0.0:
                    TN += 1.0
        if P == 0.0:
            print('No positive examples were found!')
            TP_rate = 1.0
        else:
            TP_rate = TP/P
        if N == 0.0:
            print('No negative examples were found!')
            TN_rate = 1.0
        else:
            TN_rate = TN/N
        roc_auc_score_value = roc_auc_score([1.0 if y >  0.5 else 0.0 for y in Y_test], Y_predicted)

        return 1.0 - no_errors/float(len(Y_predicted_binary)), TP_rate, TN_rate, roc_auc_score_value
    
    @staticmethod
    def evaluate_multiple_times(dipnn, X, Y, no_runs):
        sum_acc = 0.0
        sum_TP_rate = 0.0
        sum_TN_rate = 0.0
        sum_roc_auc = 0.0
        accuracy_list = []
        for k in range(no_runs):
            # re-init
            dipnn.set_to_default()
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)
            dipnn.train(X_train,Y_train)
            acc, TP_rate, TN_rate, roc_auc = EvaluationTools.test(dipnn, X_test, Y_test)
            accuracy_list.append(acc)
            sum_acc += acc
            sum_TP_rate += TP_rate
            sum_TN_rate += TN_rate
            sum_roc_auc += roc_auc
        avg_acc = sum_acc/float(no_runs)
        var_acc = (np.sum((np.array(accuracy_list) - avg_acc)**2))/float(no_runs) 
        indices_to_remove = dipnn.beta_optimal >= dipnn.coeff_magnitude_th
        dipnn.w_optimal = dipnn.w_optimal[indices_to_remove]
        dipnn.terms = np.array(dipnn.terms)[indices_to_remove]
        terms_w = zip(dipnn.terms, dipnn.w_optimal)
        print('The terms and their coeffcients in the last iteration:')
        print(list(terms_w))
        print(f'\nAvg Acc:{avg_acc}')
        print(f'Var Acc:{var_acc}')
        print(f'Avg TP_rate:{sum_TP_rate/float(no_runs)}')
        print(f'Avg TN_rate:{sum_TN_rate/float(no_runs)}')
        print(f'ROC AUC score:{sum_roc_auc/float(no_runs)}')