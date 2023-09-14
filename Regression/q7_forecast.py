import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
from os import path
import numpy as np
if not path.exists('Plots/Question7/'):
    os.makedirs('Plots/Question7/')
# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv', header=0, index_col=0)

split_date = '1989-06-27'
split_index = df.index.get_loc(split_date)

# Create lagged variables
lags = 30
for i in range(1, lags+1):
    df[f't-{i}'] = df['Temp'].shift(i)

# Remove missing values
df.dropna(inplace=True)

split_ratio = 0.8
split_index = int(len(df) * split_ratio)
# Split into train and test sets
# train_size = int(len(df) * 0.8)
train, test = df.iloc[:split_index], df.iloc[split_index:]

# Fit linear regression model
X_train, y_train = train.iloc[:, 1:], train['Temp']
X_test, y_test = test.iloc[:, 1:], test['Temp']

model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)
# Calculate RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('RMSE:', (rmse))
# Plot predictions vs true values
plt.plot(y_test.values, label='true')
plt.plot(y_pred, label='predicted')
plt.legend()
plt.savefig('./Plots/Question7/q7_forecast.png')