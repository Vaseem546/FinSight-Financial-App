import os
import sqlite3
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from flask import Flask, render_template, request, redirect, session, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import joblib

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

    symbol = request.form['analysis_symbol'].upper()
    exchange = request.form['analysis_exchange']
    suffix = 'NS' if exchange == 'NSE' else 'BO'
    full_symbol = f"{symbol}.{suffix}"

    try:
        data = yf.download(full_symbol, period='5d', interval='1d')
        if data.empty:
            raise ValueError("No data found")

        latest = data.iloc[-1]
        info = yf.Ticker(full_symbol).info

        return render_template("index.html", user=session['user'],
                               analysis={
                                   "symbol": full_symbol,
                                   "price": latest['Close'],
                                   "open": latest['Open'],
                                   "previousClose": info.get("previousClose", "-"),
                                   "dayHigh": latest['High'],
                                   "dayLow": latest['Low'],
                                   "volume": latest['Volume'],
                                   "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh", "-"),
                                   "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow", "-")
                               })
    except Exception as e:
        return render_template("index.html", user=session['user'], analysis_error=f"Failed to fetch stock info: {str(e)}")

# -------------------- STOCK SCREENER --------------------
@app.route('/screener', methods=['POST'])
def screener():
    if 'user' not in session:
        return redirect(url_for('login'))

    symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'ICICIBANK.NS', 'HDFCBANK.NS']
    results = []

    for sym in symbols:
        try:
            data = yf.Ticker(sym).info
            results.append({
                "symbol": sym,
                "price": data.get("currentPrice", 0),
                "marketCap": data.get("marketCap", 0) // 10**7,
                "volume": data.get("volume", 0),
                "pe": data.get("trailingPE", 0),
                "pb": data.get("priceToBook", 0),
                "dividendYield": round((data.get("dividendYield", 0) or 0) * 100, 2),
                "bookValue": data.get("bookValue", 0),
                "high": data.get("fiftyTwoWeekHigh", 0),
                "low": data.get("fiftyTwoWeekLow", 0)
            })
        except:
            continue

    return render_template("index.html", user=session['user'], screener_data=results)

# -------------------- STOCK PREDICTOR --------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    symbol = request.form['predict_symbol'].upper()
    exchange = request.form['exchange']
    suffix = 'NS' if exchange == 'NSE' else 'BO'
    full_symbol = f"{symbol}.{suffix}"

    try:
        model_path = f"models/{symbol}_model.h5"
        scaler_path = f"models/{symbol}_scaler.save"

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Model or Scaler not found for this symbol.")

        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)

        data = yf.download(full_symbol, period='90d', interval='1d')
        if data.empty or len(data) < 60:
            raise ValueError("Not enough data")

        close_prices = data['Close'].values[-60:]
        scaled = scaler.transform(close_prices.reshape(-1, 1))
        X_input = np.array([scaled])
        predicted = model.predict(X_input)
        predicted_price = scaler.inverse_transform(predicted)[0][0]

        forecast = [(datetime.today().strftime("%Y-%m-%d"), predicted_price)]

        return render_template("index.html", user=session['user'], forecast=forecast, symbol=full_symbol)

    except Exception as e:
        return render_template("index.html", user=session['user'], error=f"No data to predict for {symbol}: {str(e)}")

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
            data = yf.download(sym, period='5d')
            if not data.empty:
                latest = data.iloc[-1]
                high = data['High'].max()
                low = data['Low'].min()
                stocks.append({
                    "symbol": sym,
                    "price": latest['Close'],
                    "high": high,
                    "low": low
                })
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
