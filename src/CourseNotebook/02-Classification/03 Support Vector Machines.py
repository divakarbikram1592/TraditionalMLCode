import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

cancer = load_breast_cancer()
key_list = ['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename']

data = pd.DataFrame(cancer[key_list[0]], columns=cancer[key_list[4]])
target = cancer[key_list[1]]

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=101)

model = SVC()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("#### Classification Report")
print(classification_report(y_true=y_test, y_pred=predictions))
print("#### Confusion Report")
print(confusion_matrix(y_true=y_test, y_pred=predictions))

