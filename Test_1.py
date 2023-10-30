import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Data Collection
symbol = "AAPL" 
start_date = "2020-01-01"
end_date = "2021-12-31"
data = yf.download(symbol, start=start_date, end=end_date)
print(data)


# Data Preprocessing
data = data[['Close']]
print(data)

data['Next_Close'] = data['Close'].shift(-1)
print(data)

data = data.dropna() # Make a copy to avoid SettingWithCopyWarning
print(data)


# Splitting Data
X = data[['Close']]
y = data['Next_Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Model Selection and Training
model = LinearRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
ans=model.score(X_test,y_test)
print(ans)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Model Testing
predicted_price = model.predict(X_test.tail(1).values.reshape(1, -1))
print(f"Predicted Price for the Next Day: {predicted_price[0]:.2f}")
