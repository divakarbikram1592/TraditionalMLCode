import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv('../00-Prerequisites/resources/Classified Data')
feature_list = ['Unnamed: 0', 'WTT', 'PTI', 'EQW', 'SBI', 'LQE', 'QWG', 'FDJ', 'PJF', 'HQE', 'NXJ', 'TARGET CLASS']
df.drop(feature_list[0], axis=1, inplace=True)

X = df.drop(feature_list[11], axis=1)
y = df[feature_list[11]]

scaler = StandardScaler()
scaler.fit(X,y)
scaled_data = scaler.transform(X)

X = pd.DataFrame(scaled_data, columns=df.columns[:-1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print('##########Classification Report')
print(classification_report(y_true=y_test, y_pred=predictions))
print('##########Confusion Metrices')
print(confusion_matrix(y_true=y_test, y_pred=predictions))

###### We need to adjust n_neighbors to get best result, to find out best n_neighbors
###### Let's calculate the error rate
error_rate = []

for i in range(1,40):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    error_rate.append(np.mean(predict != y_test))


######## BEST-FIT as K=12
model = KNeighborsClassifier(n_neighbors=12)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print('##########Classification Report')
print(classification_report(y_true=y_test, y_pred=predictions))
print('##########Confusion Metrices')
print(confusion_matrix(y_true=y_test, y_pred=predictions))



sns.jointplot(range(1,40), error_rate, )


# sns.countplot(df[feature_list[len(feature_list)-1]])
# sns.heatmap(df.isnull(), cbar=False)
# plt.show()