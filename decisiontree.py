import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split#for decision tree object
from sklearn.tree import DecisionTreeClassifier#for checking testing results
from sklearn.metrics import classification_report, confusion_matrix#for visualizing tree
import seaborn as sns

df = pd.read_csv('tested_patients.csv')
print("Dataset",df)
print("Datatset info",df.info())

d = {'negative': 0 , 'positive': 1 , 'other': 1, 'None': 0}
df['corona_result'] = df['corona_result'].map(d)
y= df['corona_result']
print("Output Data",y)

features = ['cough','fever', 'sore_throat','shortness_of_breath','head_ache']
X = df[features]
print("Input Data",X)

X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.2, random_state = 42)
print("Training split input- ", X_train.shape)
print("Testing split input- ", X_test.shape)

# Defining the decision tree algorithm
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
print('Decision Tree Classifier Created')

# Predicting the values of test data
y_pred = dtree.predict(X_test)
print("Classification report - \n", classification_report(y_test,y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,square = True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(dtree.score(X_test, y_test))
plt.title(all_sample_title, size = 15)
plt.show()