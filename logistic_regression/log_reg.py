from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

class log_reg:
    def __init__(self, x,y):
        self.x = x
        self.y = y

    def train_test_split(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=1)
        self.y_train.value_counts().plot(kind='bar', figsize=(8, 4), color='pink')
        plt.title('Original Data Before Oversampling')

    def oversample(self, type1, type2):
        os = SMOTE(random_state=0)
        columns = self.x_train.columns
        os_data_X, os_data_y = os.fit_sample(self.x_train, self.y_train)
        os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
        os_data_y = pd.DataFrame(data=os_data_y, columns=['y'])

        # check the numbers of our data
        print("length of oversampled data is ", len(os_data_X))
        print("Number of", type1, "in oversampled data", len(os_data_y[os_data_y['y'] == type1]))
        print("Number of", type2, len(os_data_y[os_data_y['y'] == type2 ]))
        print("Proportion of", type1, "data in oversampled data is ", len(os_data_y[os_data_y['y'] == type1 ]) / len(os_data_X))
        print("Proportion of", type2, "data in oversampled data is ", len(os_data_y[os_data_y['y'] == type2 ]) / len(os_data_X))

    def model(self):
        self.logreg = LogisticRegression()
        self.logreg.fit(self.x_train, self.y_train)

    def predict(self):
        self.y_pred = self.logreg.predict(self.x_test)

    def scores(self):
        print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(self.logreg.score(self.x_test, self.y_test)))
        print(classification_report(self.y_test, self.y_pred))