import ccxt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Define your API keys and other configuration
api_key = "bR6HU6prPodBQxpjILCjcMZB1NuivBxVIHxxJHuIe6KzZtse1GNJPTMfXdW09pKb"
api_secret = '8ixbQAuugy4qhMGV3AugAx7uNt2oZLe4axUYpZ1jhJ3S9pVsYHzhcohcZAiqXet5'
symbol = 'BTC/USDT'
interval = '1h'
quantity_to_buy = 0.001  # Buy 0.1 BTC for each buy signal
# 2. Assuming you have access to your account balance through the exchange API
#account_balance = get_account_balance()  # Replace with your code to fetch account balance
percentage_to_buy = 10  # 10% of the account balance
#uantity_to_buy = (percentage_to_buy / 100) * account_balance
# 3.  Implement dynamic logic to calculate quantity_to_buy based on market conditions
# quantity_to_buy = calculate_dynamic_quantity()
# Create the Binance exchange object
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
})
# Define your stop_loss_percent (e.g., 2%)
stop_loss_percent = 2  # 2% stop-loss


# Calculate the stop-loss price based on the buy order price
def calculate_stop_loss_price(buy_price, stop_loss_percent):
    return buy_price * (1 - stop_loss_percent / 100)


# Place a stop-loss order after buying
def place_stop_loss_order(symbol, quantity_to_sell, stop_loss_price):
    try:
        order = exchange.create_limit_sell_order(symbol, quantity_to_sell, stop_loss_price)
        print(f"Stop-loss order placed at {stop_loss_price}")
        return order
    except Exception as e:
        print(f"An error occurred while placing the stop-loss order: {str(e)}")
        return None


# Define your take_profit_percent (e.g., 5%)
take_profit_percent = 5  # 5% take-profit


# Calculate the take-profit price based on the buy order price
def calculate_take_profit_price(buy_price, take_profit_percent):
    return buy_price * (1 + take_profit_percent / 100)


# Place a take-profit order after buying
def place_take_profit_order(symbol, quantity_to_sell, take_profit_price):
    try:
        order = exchange.create_limit_sell_order(symbol, quantity_to_sell, take_profit_price)
        print(f"Take-profit order placed at {take_profit_price}")
        return order
    except Exception as e:
        print(f"An error occurred while placing the take-profit order: {str(e)}")
        return None


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def fetch_and_prepare_data(symbol, interval):
    # Fetch historical data
    ohlcv = exchange.fetch_ohlcv(symbol, interval)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # Feature engineering: Calculate moving averages
    df['sma_fast'] = df['close'].rolling(window=10).mean()
    df['sma_slow'] = df['close'].rolling(window=30).mean()

    # Labeling data based on strategy (example: SMA crossover)
    df['signal'] = np.where(df['sma_fast'] > df['sma_slow'], 1, -1)
    df['signal'].fillna(0, inplace=True)

    # Drop rows with NaN values and reset index
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['close'], label='Close Price', color='blue')
    plt.plot(df['timestamp'], df['sma_fast'], label='SMA Fast', color='orange')
    plt.plot(df['timestamp'], df['sma_slow'], label='SMA Slow', color='green')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.title('Price and Moving Averages')
    plt.legend()
    plt.show()

    return df


# Function to fetch historical data and prepare features
# def fetch_and_prepare_data(symbol, interval):
#     ohlcv = exchange.fetch_ohlcv(symbol, interval)
#     df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
#
#     # Feature engineering: Calculate moving averages
#     df['sma_fast'] = df['close'].rolling(window=10).mean()
#     df['sma_slow'] = df['close'].rolling(window=30).mean()
#
#     # Labeling data based on strategy (example: SMA crossover)
#     df['signal'] = np.where(df['sma_fast'] > df['sma_slow'], 1, -1)
#     df['signal'].fillna(0, inplace=True)
#
#     # Drop rows with NaN values and reset index
#     df.dropna(inplace=True)
#     df.reset_index(drop=True, inplace=True)
#
#     return df

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_and_evaluate_model(df):
    # Features and labels
    X = df[['sma_fast', 'sma_slow']].values
    y = df['signal'].values

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train a machine learning model (Random Forest Classifier)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")

    # Plot the data with buy/sell signals
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['close'], label='Close Price', color='blue')
    plt.plot(df['timestamp'], df['sma_fast'], label='SMA Fast', color='orange')
    plt.plot(df['timestamp'], df['sma_slow'], label='SMA Slow', color='green')

    # Adding buy signals (green) and sell signals (red)
    buy_signals = np.where(df['signal'] == 1, df['close'], None)
    sell_signals = np.where(df['signal'] == -1, df['close'], None)
    plt.scatter(df['timestamp'], buy_signals, marker='^', color='green', label='Buy Signal')
    plt.scatter(df['timestamp'], sell_signals, marker='v', color='red', label='Sell Signal')

    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.title('Price and Moving Averages with Buy/Sell Signals')
    plt.legend()
    plt.show()

    return model


# # Function to train and evaluate the machine learning model
# def train_and_evaluate_model(df):
#     # Features and labels
#     X = df[['sma_fast', 'sma_slow']].values
#     y = df['signal'].values
#
#     # Split data into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # Create and train a machine learning model (Random Forest Classifier)
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#
#     # Predict on the test set
#     y_pred = model.predict(X_test)
#
#     # Evaluate model performance
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Model Accuracy: {accuracy}")
#
#     return model


# Function to place a buy order
def place_buy_order(symbol, quantity):
    try:
        # Fetch the current market price
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['ask']  # Use the ask price for buying

        # Calculate the total cost of the buy order
        total_cost = current_price * quantity

        # Place the buy order
        order = exchange.create_limit_buy_order(symbol, quantity, current_price)
        print(f"Buy order placed: {order}")

        return order
    except Exception as e:
        print(f"An error occurred while placing the buy order: {str(e)}")
        return None


# Function to track and adjust the buy order
def track_buy_order(order_id):
    while True:
        try:
            # Check the status of the buy order
            order_status = exchange.fetch_order(order_id, symbol)

            # If the order is filled, exit the loop
            if order_status['status'] == 'closed':
                print("Buy order filled.")
                break

            # Implement stop-loss logic
            if order_status['price'] > 0:
                stop_loss_price = order_status['price'] * (1 - stop_loss_percent / 100)
                exchange.create_limit_sell_order(symbol, quantity_to_buy, stop_loss_price)
                print(f"Stop-loss order placed at {stop_loss_price}")

            # Implement take-profit logic
            if order_status['price'] > 0:
                take_profit_price = order_status['price'] * (1 + take_profit_percent / 100)
                exchange.create_limit_sell_order(symbol, quantity_to_buy, take_profit_price)
                print(f"Take-profit order placed at {take_profit_price}")

            time.sleep(600)  # Check the order status every 10 minutes

        except Exception as e:
            print(f"An error occurred while tracking the buy order: {str(e)}")
            time.sleep(600)  # Retry after 10 minutes


# Main trading logic
def main_trading_logic():
    while True:
        try:
            # Fetch and prepare data
            df = fetch_and_prepare_data(symbol, interval)

            # Train and evaluate the machine learning model
            model = train_and_evaluate_model(df)

            # Predict the current market situation
            current_data = df.iloc[-1][['sma_fast', 'sma_slow']].values.reshape(1, -1)
            prediction = model.predict(current_data)[0]
            print("Prediction : ", prediction)
            # Implement trading logic based on model prediction (buy/sell signals)
            if prediction == 1:
                print("Buy signal detected. Placing buy order.")
                buy_order = place_buy_order(symbol, quantity_to_buy)
                if buy_order:
                    stop_loss_price = calculate_stop_loss_price(buy_order['price'], stop_loss_percent)
                    place_stop_loss_order(symbol, quantity_to_buy, stop_loss_price)
            elif prediction == -1:
                print("Buy signal detected. Placing buy order.")
                buy_order = place_buy_order(symbol, quantity_to_buy)
                print(buy_order)
                if buy_order:
                    take_profit_price = calculate_take_profit_price(buy_order['price'], take_profit_percent)
                    place_take_profit_order(symbol, quantity_to_buy, take_profit_price)

        except Exception as e:
            print("An error occurred:", str(e))

        time.sleep(0.5)  # Sleep for 1 hour before checking again


if __name__ == "__main__":
    main_trading_logic()
