import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score

class random_forest:
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
        This method uses SMOTE oversampling to account for imbalanced data. Use if data is over 30/70 imbalance
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

        self.x_train = self.os_data_X
        self.y_train = self.os_data_y

    def grid_search(self, type1, oversample):
        if oversample:
            self.y_train = self.y_train['y'].eq(type1).mul(1)
        else:
            self.y_train = self.y_train.eq(type1).mul(1)
        self.y_test = self.y_test.eq(type1).mul(1)

        param_grid = [
            {'n_estimators': [3,10,30], 'max_features':[2,4,6,8]},
            {'bootstrap':[False], 'n_estimators': [3,10], 'max_features':[2,3,4]}
        ]

        rf_class = RandomForestClassifier()

        grid_search = GridSearchCV(rf_class, param_grid, cv=5, scoring='f1', return_train_score=True)
        grid_search.fit(self.x_train, self.y_train)
        print(grid_search.best_estimator_)

        self.tuned_model = grid_search.best_estimator_

    def predict(self):
        self.y_pred = self.tuned_model.predict(self.x_test)

    def scores(self):
        print(classification_report(self.y_test, self.y_pred))


