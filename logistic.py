import warnings
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import unittest

warnings.filterwarnings("ignore")

data = pd.read_csv("tested_patients.csv")
data = np.array(data)
print(data)

# X = data[1:, 1:-1]
# y = data[1:, -1]
# y = y.astype('str')
# X = X.astype('str')
# print(X,y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# log_reg = LogisticRegression()

# log_reg.fit(X_train, y_train)


tested_patients = {
    'sore_throat': [1,0,1,0,1,0,1,1,1,0,0,1,0,1,0,0],
    'sense_of_taste': [1,0,1,1,0,1,0,1,0,1,0,1,0,1,0,1],
    'contact_indication': [1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0],
    'corona_result': [1,0,1,1,0,0,0,1,0,1,0,1,0,0,1,0]
    }

df = pd.DataFrame(tested_patients, columns=['sore_throat', 'sense_of_taste', 'contact_indication', 'corona_result'])

print(df)

X = df[['sore_throat', 'sense_of_taste', 'contact_indication']]
y = df['corona_result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)

y_pred = logistic_regression.predict(X_test)

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)

print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
plt.show()
Testcount = 0;
class TestAccuracy(unittest.TestCase):

    def test_predicted_acccuracy(self):
        print("Test Case Name: Predicted Accuracy")
        print("Test Case Number: ", (Testcount+1))
        self.assertEqual(metrics.accuracy_score(y_test, y_pred), 0.75 , "Should be 0.75")

    def test_accuracy(self):
        print("Test Case Name: Actual Accuracy")
        print("Test Case Number: ", (Testcount + 2))
        self.assertEqual(metrics.accuracy_score(y_test, y_pred)-0.15, 0.75, "Should be 0.75")

if __name__ == '__main__':
    unittest.main()

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(logistic_regression, open(filename, 'wb'))