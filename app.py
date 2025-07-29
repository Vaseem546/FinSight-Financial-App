import os
import sqlite3
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request, redirect, session, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from plotly.utils import PlotlyJSONEncoder
import plotly.graph_objs as go
import json
from alpha_vantage.timeseries import TimeSeries
import joblib

# Load Alpha Vantage API key
ALPHA_KEY = os.getenv("ALPHA_VANTAGE_KEY")
ts = TimeSeries(key=ALPHA_KEY, output_format='pandas')

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# -------------------- DB SETUP --------------------
def init_sqlite():
    db_path = os.getenv("DATABASE_PATH", "users.db")
    if not os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                symbol TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

init_sqlite()

def get_db():
    db_path = os.getenv("DATABASE_PATH", "users.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

# -------------------- ROUTES --------------------
@app.route('/', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login():
    error, success = None, None
    show = request.form.get("action") or request.args.get("show") or "login"

    if request.method == 'POST':
        action = request.form['action']
        email = request.form['email']
        password = request.form['password']
        conn = get_db()
        cursor = conn.cursor()

        if action == 'login':
            cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
            user = cursor.fetchone()
            if user and check_password_hash(user['password'], password):
                session['user'] = email
                return redirect(url_for('index'))
            else:
                error = "Invalid credentials"

        elif action == 'register':
            cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
            if cursor.fetchone():
                error = "User already exists"
            else:
                hashed_pw = generate_password_hash(password)
                cursor.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, hashed_pw))
                conn.commit()
                success = "Registered successfully! Please log in."
                show = 'login'

        conn.close()

    return render_template('login_register.html', error=error, success=success, show=show)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/index')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', user=session['user'])

# -------------------- MARKET ANALYSIS --------------------
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'user' not in session:
        return redirect(url_for('login'))

    symbol = request.form['symbol'].upper()
    try:
        data, _ = ts.get_daily_adjusted(symbol=symbol, outputsize='compact')
        if data.empty:
            raise ValueError("No data found")

        current_price = float(data.iloc[-1]['4. close'])
        high_52w = data['2. high'].max()
        low_52w = data['3. low'].min()

        return render_template("index.html", user=session['user'],
                               analysis={"symbol": symbol, "price": current_price,
                                         "high": high_52w, "low": low_52w})
    except Exception as e:
        return render_template("index.html", user=session['user'], error=f"Failed to fetch stock info: {str(e)}")

# -------------------- STOCK SCREENER --------------------
top_stocks = ['RELIANCE.BSE', 'TCS.BSE', 'INFY.BSE', 'ICICIBANK.BSE', 'HDFCBANK.BSE']

@app.route('/screener')
def screener():
    if 'user' not in session:
        return redirect(url_for('login'))

    results = []
    for symbol in top_stocks:
        try:
            quote, _ = ts.get_quote_endpoint(symbol=symbol)
            price = float(quote['05. price'])
            change = float(quote['10. change percent'].replace('%', ''))
            results.append({"symbol": symbol, "price": price, "change": change})
        except Exception:
            continue

    return render_template("index.html", user=session['user'], screener=results)

# -------------------- STOCK PREDICTOR --------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    symbol = request.form['predict_symbol'].upper()

    try:
        # Correct file paths
        model_path = f"models/{symbol}.h5"
        scaler_path = f"models/{symbol}.scaler.save"

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Model or Scaler not found for this symbol.")

        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)

        data, _ = ts.get_daily_adjusted(symbol=symbol, outputsize='compact')
        df = data.sort_index()
        close_prices = df['4. close'].values[-60:]

        if len(close_prices) < 60:
            raise ValueError("Not enough data")

        scaled = scaler.transform(close_prices.reshape(-1, 1))
        X_input = np.array([scaled])
        predicted = model.predict(X_input)
        predicted_price = scaler.inverse_transform(predicted)[0][0]

        return render_template("index.html", user=session['user'],
                               prediction={"symbol": symbol, "price": round(predicted_price, 2)})

    except Exception as e:
        return render_template("index.html", user=session['user'],
                               error=f"No data to predict for {symbol}: {str(e)}")

# -------------------- PORTFOLIO --------------------
@app.route('/portfolio')
def portfolio():
    if 'user' not in session:
        return redirect(url_for('login'))

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT symbol FROM portfolio WHERE user_email = ?", (session['user'],))
    symbols = [row['symbol'] for row in cursor.fetchall()]
    stocks = []

    for sym in symbols:
        try:
            data, _ = ts.get_daily_adjusted(symbol=sym, outputsize='compact')
            current_price = float(data.iloc[-1]['4. close'])
            high = data['2. high'].max()
            low = data['3. low'].min()
            stocks.append({"symbol": sym, "price": current_price, "high": high, "low": low})
        except:
            continue

    return render_template("portfolio.html", stocks=stocks, user=session['user'])

@app.route('/add_to_portfolio', methods=['POST'])
def add_to_portfolio():
    if 'user' not in session:
        return redirect(url_for('login'))

    symbol = request.form['symbol'].upper()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO portfolio (user_email, symbol) VALUES (?, ?)", (session['user'], symbol))
    conn.commit()
    return redirect(url_for('portfolio'))

@app.route('/remove_from_portfolio', methods=['POST'])
def remove_from_portfolio():
    if 'user' not in session:
        return redirect(url_for('login'))

    symbol = request.form['symbol'].upper()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM portfolio WHERE user_email = ? AND symbol = ?", (session['user'], symbol))
    conn.commit()
    return redirect(url_for('portfolio'))

if __name__ == '__main__':
    app.run(debug=True)
