import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('../00-Prerequisites/resources/titanic_train.csv')
# df_test = pd.read_csv('/Users/divakar/Documents/PythonProject/PyAnaProject/src/courses/resources/titanic_test.csv')
feature_list = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin',
                'Embarked']

df.drop(feature_list[len(feature_list) - 2], axis=1, inplace=True)


def impute_age(cols):
    age = cols[0]
    pclass = cols[1]

    if pd.isnull(age):
        if pclass == 1:
            return 37
        elif pclass == 2:
            return 29
        else:
            return 24
    else:
        return age


df[feature_list[5]] = df[[feature_list[5], feature_list[2]]].apply(impute_age, axis=1)
sex_dummy = pd.get_dummies(df[feature_list[4]], drop_first=True)
emb_dummy = pd.get_dummies(df[feature_list[len(feature_list) - 1]], drop_first=True)
df = pd.concat([df, sex_dummy, emb_dummy], axis=1)
df.drop([feature_list[4], feature_list[len(feature_list) - 1], feature_list[0], feature_list[3], feature_list[8]], axis=1, inplace=True)

X = df.drop(feature_list[1], axis=1)
y = df[feature_list[1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

model = LogisticRegression(max_iter=600)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# sns.heatmap(df.isnull(),cbar=False)
# sns.boxenplot(x=feature_list[2], y=feature_list[5], data=df)
# sns.distplot(predictions, kde=False, bins=30)
# sns.countplot(x=predictions)
# plt.show()

print('####Classification Report')
print(classification_report(y_true=y_test, y_pred=predictions))
print('####Confusion Matrix')
print(confusion_matrix(y_true=y_test, y_pred=predictions))
