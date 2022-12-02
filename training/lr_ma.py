from data import TimeSeriesData
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

from transform import make_tabular_ts
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--column_name", type=str, default="CO")
args = vars(parser.parse_args())

ts = TimeSeriesData()
df = ts.get_data2()

col = args["column_name"]

rolling_mean = df.loc[:, col].rolling(window=10).mean().dropna()
train = rolling_mean.loc[:len(rolling_mean) * 0.6]
test = rolling_mean.loc[int(len(rolling_mean) * 0.6):]

train_data = make_tabular_ts(train.tolist())
test_data = make_tabular_ts(test.tolist())

X_train, y_train = train_data[:, :-1], train_data[:, -1].flatten()
X_test, y_test = test_data[:, :-1], test_data[:, -1].flatten()

model = LinearRegression()
model.fit(X_train, y_train)
print("score lnma: ",model.score(X_test, y_test))


prediction = list(X_test[0]) + list(model.predict(X_test))
prediction = pd.DataFrame(prediction, index = test.index[:-1])
print(prediction)

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(train, color="b")
ax1.plot(test, color="r")
ax2.plot(train, color="b")
ax2.plot(prediction, color="black")
plt.show()
