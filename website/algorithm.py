from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import *
from math import sqrt
from tensorflow.keras.models import load_model


date_format_str = '%Y-%m-%d %H:%M:%S'

data = requests.get(
    "https://api.thingspeak.com/channels/1576707/feeds.json?results=100")


def pull_data():
    element = data.json()
    row = element['feeds']
    LPG = []
    CO = []
    date = []
    for i in row:
        date.append(i['created_at'])
        LPG.append(i['field1'])
        CO.append(i['field2'])
    arr = np.stack((date, LPG, CO), axis=1)
    df = pd.DataFrame(arr)
    df = df.rename({0: 'date', 1: 'LPG', 2: 'CO'}, axis=1)
    # df = df.astype({'LPG': float, 'CO': float})
    return df


def Data(DataFrame):
    data = DataFrame
    print(data)
    data['date'] = data['date'].str[:-1]
    for i in range(len(data['date'])):
        data['date'][i] = data['date'][i].replace("T", " ")
    print(data)

    data1 = data
    length = len(data1.date)-1
    # print(length)

    for i in range(length):
        start = datetime.strptime(data1.date[i], date_format_str)
        end = datetime.strptime(data1.date[i+1], date_format_str)
        diff = end - start
        if (diff.total_seconds() > 330):

            add = start + timedelta(seconds=319)
            add = add.strftime(date_format_str)
            line = pd.DataFrame({"date": add, "LPG": float(
                "NaN"), "CO": float("NaN"), }, index=[i+1])
            data1 = pd.concat(
                [data1.iloc[:i+1], line, data1.iloc[i+1:]]).reset_index(drop=True)
            length += 1
            continue

    data1.index = data1['date']
    data1.index.name = None
    data = data1
    return data


# fill missing (NaN) by Interpolation method
def FillMissingData(DataFrame):
    data = DataFrame
    data = data.astype({'LPG': float, 'CO': float})
    data = data.interpolate(method='linear', limit_direction='forward')
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    return data


def justOneMinute(data, num_prediction):
    li = []
    d = data['date'].dt.date[-1]
    print(d)
    dt = datetime.combine(d, data['date'].dt.time[-1])
    for i in range(num_prediction):
        li.append(dt + timedelta(minutes=i))
    return li


def ARIMA_agorithm(CleanData, column='LPG'):
    data = CleanData
    X = data[column].values
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    print("train here: ", train)
    history = [x for x in train]
    predictions = list()
    OBS = []
    YHAT = []
    # walk-forward validation
    for t in range(len(test)):
        model = ARIMA(history, order=(1, 1, 1))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        OBS.append(obs)
        YHAT.append(yhat)
        df = pd.DataFrame(np.stack((YHAT, OBS), axis=1),
                          columns=['predicted', 'expected'])
        print('predicted=%f, expected=%f' % (yhat, obs))
    # evaluate forecasts
    rmse = sqrt(mean_squared_error(test, predictions))
    print('Mean Absolute Error:', mean_absolute_error(test, predictions))
    print('Mean Squared Error:', mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)

    return df


def time_predict(filldata, algorithm):
    data = filldata
    size = int(len(data) * 0.66)
    time_data = data[0:size]
    index_future_minutes = justOneMinute(time_data, 34)
    predict_LPG = algorithm
    predict_LPG.index = index_future_minutes
    predict_LPG['date'] = predict_LPG.index
    return predict_LPG


def LR_Algo_for_CO(column, Fill_data):
    # data forecast from ARIMA
    LPG_pred = column['predicted'].values

    # data from fill missing data
    lpg = Fill_data['LPG'].values.reshape(
        len(Fill_data), 1)
    co = Fill_data['CO'].values

    # split data 80-20
    X_train, X_test, y_train, y_test = train_test_split(
        lpg, co, test_size=0.2, random_state=0)
    regressor = LinearRegression()

    # fit model
    regressor.fit(X_train, y_train)
    print('intercept:', regressor.intercept_)
    print('slope:', regressor.coef_)

    # the regression model
    CO_pred = regressor.intercept_ + regressor.coef_ * LPG_pred

    print('coefficient of determination:', regressor.score(X_train, y_train))
    print(CO_pred)
    return CO_pred

def LSTM_predict(data, attr, model_name):
    # data should be a list of 4 numbers with shape (1, 4).
    # For example, data = [[1,2,3,4]] is valid while data = [1,2,3,4] is not valid
    size = int(len(data) * 0.66)
    time_data = data[0:size]
    index_future_minutes = justOneMinute(time_data, 34)
    model = load_model(model_name)
    init_data = data.loc[:, attr].values[-4:]
    predicted_values = []
    
    for _ in range(len(index_future_minutes)):
        init_data = init_data.reshape(-1, 4, 1)
        new_data = model.predict(init_data)
        init_data[:3] = init_data[-3:]
        init_data[-1] = float(new_data[0])
        predicted_values.append(float(new_data[0]))
    predicted_df = pd.DataFrame({"predicted": predicted_values})
    predicted_df['date'] = index_future_minutes
    predicted_df.index = index_future_minutes
    return predicted_df


if __name__ == "__main__":

    data = Data(DataFrame=pull_data())
    print('raw data: ', data)
    fill = FillMissingData(data)
    print(' clean data: ', fill)
    LPG = ARIMA_agorithm(fill)
    print('prediction LPG: ', LPG)
    predict_data_for_LPG = time_predict(filldata=fill, algorithm=LPG)
    print('predict with time for LPG: ', predict_data_for_LPG)
    # print(predict_data_for_LPG['date'])
    print('Prediction for CO: ', LR_Algo_for_CO(LPG, fill))
