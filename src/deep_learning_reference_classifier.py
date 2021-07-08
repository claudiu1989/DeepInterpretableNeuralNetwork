# TensorFlow
import tensorflow as tf
import numpy as np

class DeepLearningReferenceClassifier:
    @staticmethod
    def train_and_validate(X_train, Y_train, X_test, Y_test, epochs, no_classes, no_hidden_layers, size_of_hidden_layers):
        no_features = len(X_train[0])
        layers = [tf.keras.layers.Dense(no_features, activation='relu')]
        for i in range(no_hidden_layers):
            layers.append(tf.keras.layers.Dense(size_of_hidden_layers, activation='relu'))
        layers.append(tf.keras.layers.Dense(no_classes))
        model = tf.keras.Sequential(layers)
        if no_classes < 2:
            raise Exception("At least two classes are expected (no_classes>=2)") 
        elif no_classes == 2:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        else:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer='adam',
                    loss=loss,
                    metrics=['accuracy'])
        
        model.fit(X_train, Y_train, epochs=epochs)
        test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=1)
        probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
        predictions_probabilities = probability_model.predict(X_test)
        predictions = np.argmax(predictions_probabilities, axis=1)
        return test_acc, predictions
