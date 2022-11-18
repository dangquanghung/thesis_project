from flask import Flask, render_template, request
import requests
import numpy as np
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from algorithm import FillMissingData, Data, ARIMA_agorithm, LR_Algo_for_CO, time_predict, LSTM_predict
from tensorflow.keras.models import load_model


app = Flask(__name__)
MYBUG = True


def callAPI():
    response = requests.get(
        "https://api.thingspeak.com/channels/1576707/feeds.json?results=100")
    element = response.json()
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


scheduler = BackgroundScheduler()


# Hence, in the front-end, the form should call something like 
# http://<your-domain-uri>?algorithm=<the algorithm you want to see>
@app.route('/')
def getInfo():
    # preprocessing
    df = callAPI()
    data = Data(df)
    fill_data = FillMissingData(data)
        

    # predict LPG
    if request.args.get("algorithm") is None or request.args.get("algorithm") == "" or request.args.get("algorithm") == "ARIMA":
        LPG_pred = ARIMA_agorithm(fill_data)
        LPG_pred = time_predict(filldata=fill_data, algorithm=LPG_pred)
    # add more option by adding condition for algorithm
    # example:
    if request.args.get("algorithm") == "lstm":
        LPG_pred = LSTM_predict(fill_data, "LPG", "../training/model/lpg_model.h5")
    
    CO_pred = LR_Algo_for_CO(LPG_pred, fill_data)

    # New record
    co = df['CO'].values[-1]
    lpg = df['LPG'].values[-1]

    # For chart
    labels = [date for date in df['date'].values[-20:]]
    LPG_data = [lpg for lpg in df['LPG'].values[-20:]]
    CO_data = [co for co in df['CO'].values[-20:]]
    time = [str(date_predict).split("T", 1)[1].rpartition(":")[0]
            for date_predict in LPG_pred['date'].values[-20:]]

    if any(LPG_data) >= 400:
        return 'Dangerous Gas! CHECK the gas contains'

    if any(CO_data) >= 10:
        return 'Dangerous level of the air, Please check your pump!'

    return render_template('index.html', co=co, lpg=lpg,
                           LPG_pred=LPG_pred['predicted'].values,
                           CO_pred=CO_pred,
                           labels=labels,
                           time=time,
                           LPG_data=LPG_data, CO_data=CO_data, zip=zip)


if __name__ == '__main__':
    scheduler.add_job(callAPI, 'interval', seconds=12)
    scheduler.start()
    app.run(debug=True)
