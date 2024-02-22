import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv('../00-Prerequisites/resources/USA_Housing.csv')

feature_list = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population', 'Price', 'Address']

df.drop(feature_list[6], axis=1, inplace=True)
# sns.distplot(df[feature_list[5]])
# sns.jointplot(feature_list[4], feature_list[5], data=df)
# sns.pairplot(df, aspect=2.5, height=1)
# sns.lmplot(feature_list[0], feature_list[5], data=df, markers='*')
# plt.show()

X = df.drop(feature_list[5], axis=1)
y = df[feature_list[5]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

model = LinearRegression()
model.fit(X_train, y_train)

####Intercept and Coefficient
# print(f"Intercept : {model.intercept_}")
# print(f"Coeff(Slope of model) : {model.coef_}")
# intrcpt_df = pd.DataFrame({'Columns':'Intercept', 'Coeff(Except Last)':[model.intercept_]})
# coeff_df = pd.DataFrame({'Columns':df.columns[:-1], 'Coeff(Except Last)':model.coef_})
# intrcpt_coeff_df = pd.concat([coeff_df, intrcpt_df])

predictions = model.predict(X_test)
###Let's find out max difference
# act_pred_df = pd.DataFrame({'Actual':y_test, 'Prediction':predictions, 'Diff':(y_test-predictions)})
# print(act_pred_df)
# print(np.max(y_test-predictions))
# imax = np.argmax(y_test-predictions)
# print(f"Max value index : {imax}")
# print(f'Actual:{y_test[imax]}, Prediction:{predictions[imax]}, Diff:{(y_test[imax-1]-predictions[imax-1])}')

###Visualize, diff between actual and prediction
sns.distplot((y_test-predictions))

###Evaluate
print(f"Mean Absolute Error : {metrics.mean_absolute_error(y_true=y_test, y_pred=predictions)}")
print(f"Mean Square Error : {metrics.mean_absolute_error(y_true=y_test, y_pred=predictions)}")
print(f"Root Mean Square Error : {metrics.mean_absolute_error(y_true=y_test, y_pred=predictions)}")

plt.show()
