import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('../00-Prerequisites/resources/kyphosis.csv')
feature_list = ['Kyphosis', 'Age', 'Number', 'Start']
df[feature_list[0]] = pd.get_dummies(df[feature_list[0]], drop_first=True)

X = df.drop(feature_list[0], axis=1)
y = df[feature_list[0]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(predictions)
print('##### Classification Report')
print(classification_report(y_true=y_test, y_pred=predictions))
print('##### Confusion Metrices')
print(confusion_matrix(y_true=y_test, y_pred=predictions))



