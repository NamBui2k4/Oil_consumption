import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

df0 = pd.read_csv('data/Oil Consumption by Country 1965 to 2023.csv')

# normalize
df0.index = df0.iloc[:,0]
df = df0.T[1:]


target = 'Africa'
Africa_df = df[[target]].copy()

window_size = 5
print()
def create_recursive_data(data, window_size, feature_name):
    for i in range(1, window_size + 1):
        data[feature_name + ' {}'.format(i+1)] = data[feature_name].shift(-i)
    return data

Africa_df = create_recursive_data(Africa_df, window_size, target)
Africa_df = Africa_df.iloc[0:-window_size]

X = Africa_df.drop(['Africa 6'], axis=1) # X.shape = (54, 5)
y = Africa_df[['Africa 6']] # y.shape = (54, 1)

train_size = 0.8

X_train = X[:int(train_size*len(Africa_df))]
y_train = y[:int(train_size*len(Africa_df))]

X_test = X[int(train_size*len(Africa_df)):]
y_test = y[int(train_size*len(Africa_df)):]
y_test= y_test.values.flatten()

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

with open('Result/prediction.txt', 'w') as f:
    for i, j in zip(y_pred, y_test):
        f.write('Predicted: {}, Actual: {} \n'.format(i, j))









