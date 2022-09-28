import pandas as pd
import sklearn
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

import joblib

train_data = pd.read_csv(r'E:/Data/train_data.csv')
model_filename = './results/model.pkl'
imputer_filename = './results/imputer.pkl'
scaler_filename = './results/scaler.pkl'


def preprocess_data(data, imputer=None, scaler=None):
    column_name = ['Year', 'Life expectancy ', 'infant deaths', 'Alcohol',
                   'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ', 'under-five deaths ',
                   'Polio', 'Total expenditure', 'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population',
                   ' thinness  1-19 years', ' thinness 5-9 years', 'Income composition of resources',
                   'Schooling']
    data = data.drop(["Country", "Status"], axis=1)

    # if imputer == None:
    #     imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
    #     imputer = imputer.fit(data[column_name])

    from sklearn.impute import KNNImputer
    from sklearn.experimental import enable_iterative_imputer

    from sklearn.impute import IterativeImputer
    #imputer = IterativeImputer(RandomForestRegressor(),max_iter=10, random_state=0)
    imputer = KNNImputer(n_neighbors=4, weights="uniform")
    data[column_name] = imputer.fit_transform(data[column_name])

    if scaler == None:
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
    data_norm = pd.DataFrame(scaler.transform(data), columns=data.columns)

    data_norm = data_norm.drop(['Year'], axis=1)

    return data_norm, imputer, scaler

def model_fit(train_data):

    train_y = train_data.iloc[:,-1].values
    train_data = train_data.drop(["Adult Mortality"], axis=1)



    train_data_norm, imputer, scaler = preprocess_data(train_data)

    train_x = train_data_norm.values

    #regressor = GradientBoostingRegressor() #LinearRegression()
    regressor = RandomForestRegressor(bootstrap=True, oob_score=True, random_state=1,
                                      n_estimators=450, max_depth=8, max_features=18,
                                      max_samples=0.96, min_samples_leaf=1, min_samples_split=3)

    regressor.fit(train_x, train_y)

    joblib.dump(regressor, model_filename)
    joblib.dump(imputer, imputer_filename)
    joblib.dump(scaler, scaler_filename)

    return regressor


def predict(test_data, filename):
    loaded_model = joblib.load(model_filename)
    imputer = joblib.load(imputer_filename)
    scaler = joblib.load(scaler_filename)

    test_data_norm, _, _ = preprocess_data(test_data, imputer, scaler)
    test_x = test_data_norm.values
    predictions = loaded_model.predict(test_x)

    return predictions





model = model_fit(train_data)

label = train_data.loc[:,'Adult Mortality']
data = train_data.iloc[:,:-1]
y_pred = predict(data, './model.pkl')
r2 = r2_score(label, y_pred)
mse = mean_squared_error(label, y_pred)
print("MSE is {}".format(mse))
print("R2 score is {}".format(r2))