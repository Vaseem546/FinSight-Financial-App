import os
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import joblib
import datetime

nifty_50_symbols = [
    'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'LT', 'SBIN', 'AXISBANK',
    'KOTAKBANK', 'ITC', 'HINDUNILVR', 'BHARTIARTL', 'ASIANPAINT', 'MARUTI', 'WIPRO',
    'TECHM', 'TITAN', 'ULTRACEMCO', 'BAJAJFINSV', 'BAJFINANCE', 'NTPC', 'POWERGRID',
    'HCLTECH', 'ONGC', 'SUNPHARMA', 'CIPLA', 'DIVISLAB', 'NESTLEIND', 'JSWSTEEL',
    'TATASTEEL', 'COALINDIA', 'HINDALCO', 'GRASIM', 'BPCL', 'BRITANNIA', 'ADANIENT',
    'ADANIPORTS', 'HEROMOTOCO', 'BAJAJ-AUTO', 'M&M', 'DRREDDY', 'EICHERMOT', 'SHREECEM',
    'SBILIFE', 'INDUSINDBK', 'UPL', 'APOLLOHOSP', 'HDFCLIFE'
]

os.makedirs("models", exist_ok=True)

end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=3 * 365)

for symbol in nifty_50_symbols:
    try:
        stock_symbol = symbol + ".NS"
        df = yf.download(stock_symbol, start=start_date, end=end_date)
        if df.empty:
            print(f"[SKIP] No data for {symbol}")
            continue

        data = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        x_train, y_train = [], []
        for i in range(60, len(scaled_data)):
            x_train.append(scaled_data[i-60:i, 0])
            y_train.append(scaled_data[i, 0])

        x_train = np.array(x_train).reshape(-1, 60, 1)
        y_train = np.array(y_train)

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=3, batch_size=64, verbose=0)

        model.save(f"models/{symbol}_model.h5")
        joblib.dump(scaler, f"models/{symbol}_scaler.save")

        print(f"[DONE] Trained and saved model for {symbol}")

    except Exception as e:
        print(f"[ERROR] Failed for {symbol}: {e}")
