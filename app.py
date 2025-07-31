from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import sqlite3
import os
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from keras.models import load_model
import pickle
from flask import send_from_directory
from werkzeug.utils import secure_filename
import plotly.graph_objs as go

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'models'

# ========== DATABASE SETUP ==========
def init_sqlite():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS portfolio (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_email TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    exchange TEXT)''')
    conn.commit()
    conn.close()

init_sqlite()

def get_db():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

# ========== ROUTES ==========

@app.route('/')
def index():
    if 'user' in session:
        return render_template('index.html', user=session['user'])
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE email = ? AND password = ?', (email, password)).fetchone()
        conn.close()

        if user:
            session['user'] = email
            return redirect(url_for('index'))
        else:
            return render_template('login_register.html', error="Invalid credentials", show_login=True)

    return render_template('login_register.html', show_login=True)

@app.route('/register', methods=['POST'])
def register():
    email = request.form['email']
    password = request.form['password']

    try:
        conn = get_db()
        conn.execute('INSERT INTO users (email, password) VALUES (?, ?)', (email, password))
        conn.commit()
        conn.close()
        session['user'] = email
        return redirect(url_for('index'))
    except sqlite3.IntegrityError:
        return render_template('login_register.html', error="Email already registered", show_login=False)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# ========== ANALYSIS ==========
@app.route('/analyze', methods=['POST'])
def analyze():
    symbol = request.form['analysis_symbol'].upper()
    exchange = request.form['analysis_exchange']
    full_symbol = f"{symbol}.NS" if exchange == "NSE" else f"{symbol}.BO"

    try:
        data = yf.Ticker(full_symbol).history(period="7d", interval="1d")
        info = yf.Ticker(full_symbol).info

        if data.empty:
            return render_template("index.html", analysis_error="No data found", scroll_to="analysis", user=session.get("user"))

        candlestick = go.Figure(data=[go.Candlestick(
            x=data.index.strftime('%Y-%m-%d'),
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close']
        )])
        candlestick.update_layout(title=f"{symbol} Candlestick Chart", xaxis_title="Date", yaxis_title="Price")

        output = {
            "symbol": symbol,
            "price": round(info.get("currentPrice", 0), 2),
            "open": round(info.get("open", 0), 2),
            "previousClose": round(info.get("previousClose", 0), 2),
            "dayHigh": round(info.get("dayHigh", 0), 2),
            "dayLow": round(info.get("dayLow", 0), 2),
            "volume": info.get("volume", 0),
            "fiftyTwoWeekHigh": round(info.get("fiftyTwoWeekHigh", 0), 2),
            "fiftyTwoWeekLow": round(info.get("fiftyTwoWeekLow", 0), 2),
        }

        return render_template("index.html", analysis=output, candlestick=candlestick.to_json(), scroll_to="analysis", user=session.get("user"))

    except Exception as e:
        return render_template("index.html", analysis_error=f"Failed to fetch stock info: {e}", scroll_to="analysis", user=session.get("user"))

# ========== SCREENER ==========
@app.route('/screener', methods=['POST'])
def screener():
    filters = {
        'min_marketcap': request.form.get('min_marketcap'),
        'min_volume': request.form.get('min_volume'),
        'max_pe': request.form.get('max_pe'),
        'max_pb': request.form.get('max_pb'),
        'min_dividend': request.form.get('min_dividend'),
        'min_bookvalue': request.form.get('min_bookvalue'),
    }

    try:
        # Placeholder: Replace with actual logic or API
        results = [
            {'symbol': 'RELIANCE', 'price': 2900, 'marketCap': 1800000, 'volume': 1200000, 'pe': 24.5, 'pb': 3.5, 'dividendYield': 1.2, 'bookValue': 650, 'high': 3200, 'low': 2600},
            {'symbol': 'TCS', 'price': 3600, 'marketCap': 1500000, 'volume': 1000000, 'pe': 29.5, 'pb': 5.2, 'dividendYield': 1.5, 'bookValue': 720, 'high': 3900, 'low': 3100}
        ]
        return render_template("index.html", screener_data=results, scroll_to="screener", user=session.get("user"))

    except Exception as e:
        return render_template("index.html", screener_error=f"Error: {e}", scroll_to="screener", user=session.get("user"))

# ========== PREDICTION ==========
@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form['predict_symbol'].upper()
    exchange = request.form['exchange']
    full_symbol = f"{symbol}.NS" if exchange == "NSE" else f"{symbol}.BO"

    try:
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{symbol}_model.h5")
        scaler_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{symbol}_scaler.save")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return render_template("index.html", error=f"No data to predict for {symbol}: Model or Scaler not found for this symbol.", scroll_to="predictor", user=session.get("user"))

        model = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        df = yf.download(full_symbol, period="60d", interval="1d")
        if df.empty or len(df) < 30:
            return render_template("index.html", error=f"No data to predict for {symbol}: Not enough data", scroll_to="predictor", user=session.get("user"))

        close_data = df['Close'].values.reshape(-1, 1)
        scaled_data = scaler.transform(close_data)
        last_60 = scaled_data[-60:]
        x_input = last_60.reshape(1, 60, 1)

        predictions = []
        for _ in range(7):
            pred = model.predict(x_input)[0][0]
            predictions.append(pred)
            x_input = np.append(x_input[:, 1:, :], [[[pred]]], axis=1)

        forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        future_dates = [(datetime.now() + timedelta(days=i + 1)).strftime('%Y-%m-%d') for i in range(7)]

        return render_template("index.html", forecast=zip(future_dates, forecast), symbol=symbol, scroll_to="predictor", user=session.get("user"))

    except Exception as e:
        return render_template("index.html", error=f"Prediction failed: {e}", scroll_to="predictor", user=session.get("user"))

# ========== PORTFOLIO ==========
@app.route('/portfolio')
def portfolio():
    if 'user' not in session:
        return redirect(url_for('login'))

    conn = get_db()
    stocks = conn.execute('SELECT symbol, exchange FROM portfolio WHERE user_email = ?', (session['user'],)).fetchall()
    conn.close()
    return render_template('portfolio.html', stocks=stocks)

@app.route('/add_to_portfolio', methods=['POST'])
def add_to_portfolio():
    if 'user' not in session:
        return redirect(url_for('login'))

    symbol = request.form['symbol']
    exchange = 'NSE'

    conn = get_db()
    conn.execute('INSERT INTO portfolio (user_email, symbol, exchange) VALUES (?, ?, ?)', (session['user'], symbol, exchange))
    conn.commit()
    conn.close()

    return redirect(url_for('index', scroll_to='analysis'))

# ========== MAIN ==========
if __name__ == '__main__':
    app.run(debug=True)
