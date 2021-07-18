
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


class DeepLearningReferenceClassifier:
   
    def train(self, X_train, Y_train,  epochs, no_classes, no_hidden_layers, size_of_hidden_layers):
        no_features = len(X_train[0])
        layers = [tf.keras.layers.Dense(no_features, activation='relu')]
        for i in range(no_hidden_layers):
            layers.append(tf.keras.layers.Dense(size_of_hidden_layers, activation='relu'))
        layers.append(tf.keras.layers.Dense(no_classes))
        self.model = tf.keras.Sequential(layers)
        if no_classes < 2:
            raise Exception("At least two classes are expected (no_classes>=2)") 
        elif no_classes == 2:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        else:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer='adam',
                    loss=loss,
                    metrics=['accuracy'])
        
        self.model.fit(X_train, Y_train, epochs=epochs)
        
    
    def predict(self, X_test):
        probability_model = tf.keras.Sequential([self.model, 
                                         tf.keras.layers.Softmax()])
        predictions_probabilities = probability_model.predict(X_test)
        predictions = np.argmax(predictions_probabilities, axis=1)
        return predictions, predictions_probabilities


    def test(self, X_test, Y_test):
        Y_predicted_binary,Y_predicted = self.predict(X_test)
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
        roc_auc_score_value = roc_auc_score([1.0 if y >  0.5 else 0.0 for y in Y_test], Y_predicted[:,0])

        return 1.0 - no_errors/float(len(Y_predicted_binary)), TP_rate, TN_rate, roc_auc_score_value

    def evaluate_multiple_times(self, X, Y, no_runs, test_size, epochs, no_classes, no_hidden_layers, size_of_hidden_layers):
        sum_acc = 0.0
        sum_TP_rate = 0.0
        sum_TN_rate = 0.0
        sum_roc_auc = 0.0
        accuracy_list = []
        training_time = []
        test_time = []
        for k in range(no_runs):
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
            start_training = time.time()
            self.train(X_train,Y_train, epochs, no_classes, no_hidden_layers, size_of_hidden_layers)
            end_training = time.time()
            training_time.append(end_training-start_training)
            start_test = time.time()
            acc, TP_rate, TN_rate, roc_auc = self.test(X_test, Y_test)
            end_test = time.time()
            test_time.append(end_test-start_test) # Include some additional processing 
            accuracy_list.append(acc)
            sum_acc += acc
            sum_TP_rate += TP_rate
            sum_TN_rate += TN_rate
            sum_roc_auc += roc_auc
        avg_acc = sum_acc/float(no_runs)
        var_acc = (np.sum((np.array(accuracy_list) - avg_acc)**2))/float(no_runs) 
        avg_training_time = sum(training_time)/float(no_runs)
        var_training_time = (np.sum((np.array(training_time) - avg_training_time)**2))/float(no_runs) 
        avg_test_time = sum(test_time)/float(no_runs)
        var_test_time = (np.sum((np.array(test_time) - avg_test_time)**2))/float(no_runs) 
        print(f'Average accuracy: {avg_acc}')
        print(f'Variance of accuracy: {var_acc}')
        print(f'Average true positive rate: {sum_TP_rate/float(no_runs)}')
        print(f'Averagevg true negative rate: {sum_TN_rate/float(no_runs)}')
        print(f'Area Under the Receiver Operating Characteristic Score: {sum_roc_auc/float(no_runs)}')
        print(f'Average training time: {avg_training_time}')
        print(f'Variance of training time: {var_training_time}')
        print(f'Average test time: {avg_test_time}')
        print(f'Variance of test time: {var_test_time}')