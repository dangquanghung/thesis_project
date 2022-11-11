from data import TimeSeriesData
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential

from transform import make_tabular_ts
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--column", default="CO", type=str)
parser.add_argument("--model_name", type=str, default="model.h5")
args = vars(parser.parse_args())


ts = TimeSeriesData()
df = ts.get_data()
print(len(df))

col = args["column"]

data = df.loc[:, col]
train = data.loc[:len(data) * 0.6].dropna()
test = data.loc[int(len(data) * 0.6):].dropna()

train_data = make_tabular_ts(train.tolist())
test_data = make_tabular_ts(test.tolist())

X_train, y_train = train_data[:, :-1].astype(np.float32), train_data[:, -1].flatten().astype(np.float32)
X_test, y_test = test_data[:, :-1].astype(np.float32), test_data[:, -1].flatten().astype(np.float32)

X_train = X_train.reshape(-1, 4, 1)
X_test = X_test.reshape(-1, 4, 1)

model = Sequential()
model.add(LSTM(32, activation="relu", return_sequences=True))
model.add(LSTM(1, activation=None))

model.compile(loss="mse", optimizer="adam", metrics=["mse"])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

model.save(args["model_name"])
