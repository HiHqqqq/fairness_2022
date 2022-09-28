import pandas as pd
import sklearn
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

import joblib


train_data = pd.read_csv(r'E:/Data/train_data.csv')
model_filename = './model.pkl'
imputer_filename = './imputer.pkl'
scaler_filename = './scaler.pkl'

def preprocess_data(data, imputer=None, scaler=None):
    column_name = ['Year', 'Life expectancy ', 'infant deaths', 'Alcohol',
                   'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ', 'under-five deaths ',
                   'Polio', 'Total expenditure', 'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population',
                   ' thinness  1-19 years', ' thinness 5-9 years', 'Income composition of resources',
                   'Schooling']
    data = data.drop(["Country", "Status"], axis=1)

    if imputer == None:
        imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
        imputer = imputer.fit(data[column_name])
    data[column_name] = imputer.transform(data[column_name])

    if scaler == None:
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
    data_norm = pd.DataFrame(scaler.transform(data), columns=data.columns)

    data_norm = data_norm.drop(['Year'], axis=1)

    return data_norm, imputer, scaler



train_y = train_data.iloc[:, -1].values
train_data = train_data.drop(["Adult Mortality"], axis=1)
train_data_norm, imputer, scaler = preprocess_data(train_data)

train_x = train_data_norm.values


def gridsearch_cv(train_x,train_y):
    # 需要网格搜索的参数
    n_estimators = [i for i in range(200, 401, 10)]
    max_depth = [i for i in range(5, 11)]
    min_samples_split = [i for i in range(2, 8)]
    min_samples_leaf = [i for i in range(1, 7)]
    max_samples = [i / 100 for i in range(95, 100)]
    parameters = {'n_estimators': n_estimators,
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf,
                  'max_samples': max_samples}
    regressor = RandomForestRegressor(bootstrap=True, oob_score=True, random_state=1)
    gs = GridSearchCV(regressor, parameters, refit=True, cv=5, verbose=1, n_jobs=-1)

    gs.fit(train_x, train_y)

    joblib.dump(gs, model_filename)
    joblib.dump(imputer, imputer_filename)
    joblib.dump(scaler, scaler_filename)

    return gs

gs = gridsearch_cv(train_x, train_y)
print('最优参数: ',gs.best_params_)
print('最佳性能: ', gs.best_score_)


print('movement 1')