import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras import backend as K
from keras.layers import Dense
from keras.optimizers import SGD
from imblearn.over_sampling import SMOTE
from keras import regularizers
from keras.layers import Dropout
from sklearn.metrics import classification_report, f1_score

import warnings
warnings.filterwarnings('ignore')

class neural_net:
    def __init__(self, x,y):
        self.x = x
        self.y = y

    def train_test_split(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2,
                                                                                random_state=1)
        self.y_train.value_counts().plot(kind='bar', figsize=(8, 4), color='pink')
        plt.title('Original Data Before Oversampling')

    def oversample(self, type1, type2):
        """
        This method uses SMOTE oversampling to account for imbalanced data
        """
        os = SMOTE(random_state=0)
        columns = self.x_train.columns
        os_data_X, os_data_y = os.fit_sample(self.x_train, self.y_train)
        self.os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
        self.os_data_y = pd.DataFrame(data=os_data_y, columns=['y'])

        # check the numbers of our data
        print("length of oversampled data is ", len(self.os_data_X))
        print("Number of", type1, "in oversampled data", len(self.os_data_y[self.os_data_y['y'] == type1]))
        print("Number of", type2, len(self.os_data_y[self.os_data_y['y'] == type2 ]))
        print("Proportion of", type1, "data in oversampled data is ", len(self.os_data_y[self.os_data_y['y'] == type1 ]) / len(self.os_data_X))
        print("Proportion of", type2, "data in oversampled data is ", len(self.os_data_y[self.os_data_y['y'] == type2 ]) / len(self.os_data_X))

    def sgd(self, learn_rate, decay, momentum):
        self.sgd = SGD(lr=learn_rate, momentum=momentum, decay=decay, nesterov=True)

    def build_model(self):
        """
        This adds layers to our neural net. Since we have 11 variables, input_dim = 11.
        Output layer has sigmoid activation because we are working with binary classification.
        """
        # build model
        self.model = Sequential()

        # hidden layer 1, rule of thumb: num of nodes = mean(input layer + output layer)
        self.model.add(Dense(6, activation='relu', kernel_initializer='random_normal', input_dim=11,
                             kernel_regularizer=regularizers.l2(0.0001)))
        self.model.add(Dropout(0.2))

        # hidden layer 2, decide whether or not to keep this.
        self.model.add(Dense(6, activation='relu', kernel_initializer='random_normal',
                             kernel_regularizer=regularizers.l2(0.0001)))
        self.model.add(Dropout(0.2))

        # ouput layer
        self.model.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

    def compile(self):
        def f1(y_true, y_pred):
            """"
             Code sourced from https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
             In order to use f1 score as a metric in keras, we must implement a custom metric. We want to use
             F1 scoring because most of the data is imbalanced, and accuracy alone is not a good metric in this case.
            """
            def recall(y_true, y_pred):
                """Recall metric.

                Only computes a batch-wise average of recall.

                Computes the recall, a metric for multi-label classification of
                how many relevant items are selected.
                """
                true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
                recall = true_positives / (possible_positives + K.epsilon())
                return recall

            def precision(y_true, y_pred):
                """Precision metric.

                Only computes a batch-wise average of precision.

                Computes the precision, a metric for multi-label classification of
                how many selected items are relevant.
                """
                true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
                precision = true_positives / (predicted_positives + K.epsilon())
                return precision

            precision = precision(y_true, y_pred)
            recall = recall(y_true, y_pred)
            return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

        self.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=[f1, 'accuracy'])

    def fit(self, batch_size, epochs):
        self.os_data_y = self.os_data_y['y'].map(dict(E=1, I=0))
        self.y_test = self.y_test.map(dict(E=1, I=0))
        self.model_history = self.model.fit(self.os_data_X, self.os_data_y, batch_size = batch_size,
                                            epochs = epochs,verbose=1, validation_data = (self.x_test, self.y_test))

    def plot_loss(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(np.sqrt(self.model_history.history['loss']), 'r', label='train')
        ax.plot(np.sqrt(self.model_history.history['val_loss']), 'b', label='val')
        ax.set_xlabel(r'Epoch', fontsize=20)
        ax.set_ylabel(r'Loss', fontsize=20)
        ax.legend()
        ax.tick_params(labelsize=20)

    def plot_acc(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(np.sqrt(self.model_history.history['acc']), 'r', label='train')
        ax.plot(np.sqrt(self.model_history.history['val_acc']), 'b', label='val')
        ax.set_xlabel(r'Epoch', fontsize=20)
        ax.set_ylabel(r'Accuracy', fontsize=20)
        ax.legend()

    def plot_f1(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(np.sqrt(self.model_history.history['f1']), 'r', label='train')
        ax.plot(np.sqrt(self.model_history.history['val_f1']), 'b', label='val')
        ax.set_xlabel(r'Epoch', fontsize=20)
        ax.set_ylabel(r'F1', fontsize=20)
        ax.legend()
        ax.tick_params(labelsize=20)

    def eval(self):
        eval_model = self.model.evaluate(self.os_data_X, self.os_data_y)
        print(eval_model)

    def predict(self):
        self.y_pred = self.model.predict(self.x_test)
        self.y_pred = (self.y_pred > 0.5)

        print(classification_report(self.y_test, self.y_pred))
