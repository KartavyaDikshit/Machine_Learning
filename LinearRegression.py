import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# dataframe Collection
symbol = "AAPL" 
start_date = "2020-01-01"
end_date = "2021-12-31"
dataframe = yf.download(symbol, start=start_date, end=end_date)
# print(dataframe)


# dataframe Preprocessing
dataframe = dataframe[['Close']]
# print(dataframe)

dataframe['Next_Close'] = dataframe['Close'].shift(-1)
# print(dataframe)

dataframe = dataframe.dropna() 
# print(dataframe)


# Splitting dataframe
X = dataframe[['Close']]
y = dataframe['Next_Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

# model_obj Selection and Training
model_obj = LinearRegression()
model_obj.fit(X_train, y_train)

# model_obj Evaluation
y_pred = model_obj.predict(X_test)
ans=model_obj.score(X_test,y_test)
ans=ans*100
print(f"Accuracy % is: {ans}%")

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# model_obj Testing
predicted_price = model_obj.predict(X_test.tail(1).values.reshape(1, -1))
print(f"Predicted Price for the Next Day: {predicted_price[0]:.2f}")

# Visualizing Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Linear Regression')
plt.xlabel('Close Price')
plt.ylabel('Next Close Price')
plt.legend()
plt.title('Linear Regression Model')
plt.show()
