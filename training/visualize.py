from data import TimeSeriesData
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model

from transform import make_tabular_ts
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--column_name", type=str, default="CO")
parser.add_argument("--model_name", type=str, default="model/co_model.h5")
args = vars(parser.parse_args())

ts = TimeSeriesData()
df = ts.get_data()

col = args["column_name"]

rolling_mean = df.loc[:, col].dropna()
train = rolling_mean.loc[:len(rolling_mean) * 0.6]
test = rolling_mean.loc[int(len(rolling_mean) * 0.6):]

train_data = make_tabular_ts(train.tolist())
test_data = make_tabular_ts(test.tolist())

X_train, y_train = train_data[:, :-1], train_data[:, -1].flatten()
X_test, y_test = test_data[:, :-1], test_data[:, -1].flatten()

X_train = X_train.reshape(-1, 4, 1)
X_test = X_test.reshape(-1, 4, 1)

model = load_model(args["model_name"])
model.evaluate(X_test, y_test)


prediction = list(X_test[0]) + list(model.predict(X_test))
prediction = pd.DataFrame(prediction, index = test.index[:-1])

plt.plot(train, color="b")
plt.plot(test, color="r")
plt.plot(prediction, color="black")
plt.show()
