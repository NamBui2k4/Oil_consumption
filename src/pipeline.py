import pandas as pd
import numpy as np
from training import Dataset, Training
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    # read data
    df = pd.read_csv('data/Oil Consumption by Country 1965 to 2023.csv')

    # preprocessing
    df = Dataset(df)
    df.normalize()

    columns = df.columns().drop(['Africa',
                                 'Africa (EI)',
                                 'Asia',
                                 'Asia Pacific (EI)',
                                 'Australia',
                                 'Eastern Africa (EI)',
                                 'Non-OECD (EI)',
                                 'North America',
                                 'North America (EI)',
                                 'OECD (EI)',
                                 'Oceania',
                                 'South Africa',
                                 'South America',
                                 'South and Central America (EI)',
                                 'Western Africa (EI)',
                                 'CIS (EI)','Central America (EI)',
                                 'Middle Africa (EI)',
                                 'Lower-middle-income countries',
                                 'Middle East (EI)',
                                 'USSR','Europe','Upper-middle-income countries',
                                 'Europe (EI)','European Union (27)',
                                 'High-income countries',
                                 'World'])
    with open('Result/prediction.txt', 'w') as f:
        f.write('| Country | 2024 |\n')
        f.write('|---------|------|\n')
        for col in columns:
            data = df.create_recursive_data(5,col)
            data = data.dropna()
            # split data
            X_train, y_train, X_test, y_test = df.train_test_split(data,0.8)

            # training
            lg = Training('lg')
            lg.fit(X_train, y_train)
            pred = lg.predict(data.tail(1).values[-6:][:,-5:])

            # write result
            f.write(f'| {col} | {pred[0]:.3f} |\n')