import pandas as pd
import numpy as np
from jedi.inference import param
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error, median_absolute_error, r2_score, explained_variance_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV,KFold
class Dataset:
    def __init__(self, data):
        if isinstance(data, pd.DataFrame):
            self.df = data
        else:
            self.df = pd.DataFrame(data)

    def dtype(self):
        return self.df.dtypes
    def __len__(self):
        return len(self.df)
    def columns(self):
        return self.df.columns
    def __str__(self):
        return str(self.df.head())
    def normalize(self):
        df0 = self.df.copy()
        df0.index = df0.iloc[:,0]
        self.df = df0.T[1:]

    def check_nan(self):
        return self.df.isna().sum()

    # def drop_na(self):
    #     self.df.

    def create_recursive_data(self, window_size,feature_name):
        country = self.df[[feature_name]].copy()
        for i in range(1, window_size + 1):
            country[feature_name + ' {}'.format(i+1)] = country[feature_name].shift(-i)
        return country.iloc[0:-window_size]
    def train_test_split(self, data, train_size):
        X = data.values[:,:-1].astype('float64')
        y = data.values[:,-1].astype('float64')

        X_train = X[:int(train_size * len(data))]
        y_train = y[:int(train_size * len(data))]

        X_test = X[int(train_size * len(data)):]
        y_test = y[int(train_size * len(data)):]

        return X_train, y_train, X_test, y_test

class Training:
    def __init__(self, model_name):
        self.params = param
        self.model_name = model_name
        self.model = None
        if model_name == 'SGDRegressor':
            self.model = SGDRegressor(random_state=42)
        elif model_name == 'Decision Tree Regressor':
            self.model = DecisionTreeRegressor(random_state=42)
        elif model_name == 'Random Forest Regressor':
            self.model = RandomForestRegressor(random_state=42)
        else:
            self.model = LinearRegression()


    def fit(self, X, y):
        self.model.fit(X,y)
    def predict(self, X):
        return self.model.predict(X)
    def show_results(self, y_test, y_pred):
        for i, j in zip(y_test, y_pred):
            print(f'\t Actual: {i}, Predicted: {j} \n')

    def regression_report(self,y_true, y_pred, f):

        error = y_true - y_pred
        percentil = [5, 25, 50, 75, 95]
        percentil_value = np.percentile(error, percentil)

        metrics = [
            ('mean squared error', mean_squared_error(y_true, y_pred)),
            ('mean absolute error', mean_absolute_error(y_true, y_pred)),
            ('r2 score', r2_score(y_true, y_pred)),
            ('max error', max_error(y_true, y_pred)),
            ('explained variance score', explained_variance_score(y_true, y_pred))
        ]

        # print(f'Metrics for {self.model_name}:')\
        f.write(f'|{self.model_name}|')
        for metric_name, metric_value in metrics:
            f.write(f'{metric_value: >20.3f}|')
        f.write(f'\n')
        # print('\nPercentiles:')
        # for p, pv in zip(percentil, percentil_value):
        #     print(f'{p: 25d}: {pv:>20.3f}')

    def visualize(self,y_pred,y_test):
        plt.plot(y_test, color='red', label='ground true')
        plt.plot(y_pred, color='blue', label='Prediction')
        plt.title(f'{self.model_name}')
        plt.xlabel('Time')
        plt.ylabel('consumption')
        plt.legend()
        plt.show()

    def Grid(self):
        if isinstance(self.model, SGDRegressor):
            param = {
                'loss': ['squared_error', 'huber'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'shuffle': [True],
                'learning_rate': ['invscaling', 'adaptive'''],
            }
            self.model = GridSearchCV(estimator=self.model,param_grid= param,cv=5, scoring='r2', n_jobs=-1, return_train_score=True)
        elif isinstance(self.model, RandomForestRegressor):
            param = {
                'n_estimators': [2,3,5,7],
                'max_depth': [1, 2, 3, 4, 5],
                'min_samples_split': [2, ],
                'min_samples_leaf': [1, 2, 3],
                'max_features': ['sqrt', 'log2']
            }
            self.model = GridSearchCV(estimator=self.model,param_grid= param,cv=5, scoring='r2', n_jobs=-1, return_train_score=True)

    def check_Grid(self):
        if isinstance(self.model, GridSearchCV):
            cv_results = pd.DataFrame(self.model.cv_results_)

            # Hiển thị độ chính xác trên tập huấn luyện và kiểm tra cho từng fold
            for i in range(5):
                train_score = cv_results[f'split{i}_train_score']
                test_score = cv_results[f'split{i}_test_score']
                print(f"Fold {i + 1}:")
                print(f"Train Accuracy: {train_score.mean()}")
                print(f"Test Accuracy: {test_score.mean()}")
                print()



if __name__ == '__main__':
    # read data
    df = pd.read_csv('data/Oil Consumption by Country 1965 to 2023.csv')

    # preprocessing
    df = Dataset(df)
    df.normalize()
    Uzbekistan_df = df.create_recursive_data(5,'Uzbekistan')
    Uzbekistan_df.dropna(inplace=True)
    # split data
    X_train, y_train, X_test, y_test = df.train_test_split(Uzbekistan_df,0.8)

    # training
    # Random Forest Regressor
    # RFR = Training('Random Forest Regressor')
    # RFR.Grid()
    # RFR.fit(X_train, y_train)
    # pred = RFR.predict(X_test)
    # RFR.show_results(y_test,pred)
    # RFR.visualize(pred,y_test)
    # RFR.regression_report(y_test,pred)

    # sgd
    # sgd = Training('SGDRegressor')
    # sgd.Grid()
    # sgd.fit(X_train, y_train)
    # y_pred = sgd.predict(X_test)
    # sgd.regression_report(y_test,y_pred)
    # sgd.visualize(y_pred,y_test)
    # sgd.show_results(y_test,y_pred)


    # Tree.show_results(y_test,y_pred)
    # Tree.visualize(y_test,y_pred)

    with open('Result/prediction.txt', 'w') as f:
        f.write('|	   Model      |   MSE  |   MAE	|   R2  | Max error | variance  | \n')
        f.write('|----------------|--------|--------|-------|-----------|-----------| \n')
        for name in ['LinearRegression','Random Forest Regressor','Decision Tree Regressor', 'SGDRegressor']:
            model = Training(model_name=name)
            if name == 'SGDRegressor' or name == 'Random Forest Regressor' or name == 'Decision Tree Regressor':
                model.Grid()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            model.regression_report(y_test, y_pred, f)
