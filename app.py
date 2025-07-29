from flask import Flask, render_template, request, redirect, session, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import yfinance as yf
import pandas as pd
import os
import json
import datetime
import numpy as np
import joblib
from plotly.utils import PlotlyJSONEncoder
from tensorflow.keras.models import load_model
import plotly.graph_objs as go

# --- App Setup ---
app = Flask(__name__)
app.secret_key = "finsight_secret_key"

# Ensure users.db exists
def init_db():
    with sqlite3.connect("users.db") as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            symbol TEXT NOT NULL,
            added_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        conn.commit()


init_db()

# Load CSV for screener
stocks_df = pd.read_csv("all_stocks.csv")

# Load model names
NIFTY_50 = [f.replace('_model.h5', '') for f in os.listdir("models") if f.endswith('_model.h5')]

# Cache
model_cache = {}
scaler_cache = {}

def get_model(symbol):
    if symbol in model_cache:
        return model_cache[symbol]
    path = f"models/{symbol}_model.h5"
    if os.path.exists(path):
        model = load_model(path)
        model_cache[symbol] = model
        return model
    return None

def get_scaler(symbol):
    if symbol in scaler_cache:
        return scaler_cache[symbol]
    path = f"models/{symbol}_scaler.save"
    if os.path.exists(path):
        scaler = joblib.load(path)
        scaler_cache[symbol] = scaler
        return scaler
    return None

# ----------- Routes -----------

@app.route('/')
def index():
    if 'user' in session:
        return render_template('index.html', user=session['user'])
    return render_template('login_register.html', show='login')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']

        with sqlite3.connect("users.db") as conn:
            cur = conn.cursor()
            cur.execute("SELECT password FROM users WHERE email = ?", (email,))
            result = cur.fetchone()
            if result and check_password_hash(result[0], password):
                session['user'] = email
                return redirect(url_for("dashboard"))
            else:
                return render_template("login_register.html", show="login", error="Invalid credentials")

    return render_template("login_register.html", show="login")

@app.route('/register', methods=["POST"])
def register():
    email = request.form['email']
    password = request.form['password']

    with sqlite3.connect("users.db") as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email = ?", (email,))
        if cur.fetchone():
            return render_template("login_register.html", show="register", error="Email already exists")

        hashed = generate_password_hash(password)
        cur.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, hashed))
        conn.commit()
        session['user'] = email
        return redirect(url_for("dashboard"))

@app.route("/dashboard")
def dashboard():
    if 'user' not in session:
        return redirect(url_for("login"))
    return render_template("index.html", user=session['user'])

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# -------- Stock Analysis --------
@app.route("/analyze_stock", methods=["POST"])
def analyze_stock():
    symbol = request.form.get("analysis_symbol", "").upper()
    exchange = request.form.get("analysis_exchange", "")

    symbol_full = symbol + (".NS" if exchange == "NSE" else ".BO" if exchange == "BSE" else "")
    if not symbol_full:
        return render_template("index.html", analysis_error="Invalid exchange", scroll_to="analysis", user=session.get('user'))

    try:
        stock = yf.Ticker(symbol_full)
        hist = stock.history(period="1mo")
        if hist.empty:
            raise Exception("No historical data")

        fig = go.Figure(data=[go.Candlestick(
            x=hist.index.strftime('%Y-%m-%d'),
            open=hist['Open'], high=hist['High'],
            low=hist['Low'], close=hist['Close']
        )])
        fig.update_layout(title=f"{symbol} Chart", template="plotly_dark")
        candlestick_json = json.dumps(fig, cls=PlotlyJSONEncoder)

        info = stock.info
        stock_info = {
            'symbol': symbol,
            'price': round(info.get('currentPrice', 0), 2),
            'open': round(info.get('open', 0), 2),
            'previousClose': round(info.get('previousClose', 0), 2),
            'dayHigh': round(info.get('dayHigh', 0), 2),
            'dayLow': round(info.get('dayLow', 0), 2),
            'volume': info.get('volume', 0),
            'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh'),
            'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow')
        }

        return render_template("index.html", analysis=stock_info, candlestick=candlestick_json, scroll_to="analysis", user=session.get('user'))
    except Exception as e:
        return render_template("index.html", analysis_error="Failed to fetch stock info.", scroll_to="analysis", user=session.get('user'))

# -------- Screener --------
@app.route("/screener", methods=["POST"])
def screener():
    if 'user' not in session:
        return redirect(url_for('login'))

    form = request.form
    min_marketcap = float(form.get("min_marketcap") or 0)
    min_volume = float(form.get("min_volume") or 0)
    max_pe = float(form.get("max_pe") or 1000)
    max_pb = float(form.get("max_pb") or 1000)
    min_dividend = float(form.get("min_dividend") or 0)
    min_bookvalue = float(form.get("min_bookvalue") or 0)

    results = []
    for _, row in stocks_df.iterrows():
        try:
            ticker = yf.Ticker(row['Symbol'] + ".NS")
            info = ticker.info
            market_cap = info.get('marketCap', 0) / 1e7
            volume = info.get('volume', 0)
            pe = info.get('trailingPE', 0)
            pb = info.get('priceToBook', 0)
            dividend = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
            book_value = info.get('bookValue', 0)
            price = info.get('currentPrice', 0)
            high = info.get('fiftyTwoWeekHigh', 0)
            low = info.get('fiftyTwoWeekLow', 0)

            if (
                market_cap >= min_marketcap and volume >= min_volume and pe <= max_pe
                and pb <= max_pb and dividend >= min_dividend and book_value >= min_bookvalue
            ):
                results.append({
                    "symbol": row['Symbol'],
                    "price": price,
                    "marketCap": round(market_cap, 2),
                    "volume": volume,
                    "pe": pe,
                    "pb": pb,
                    "dividendYield": round(dividend, 2),
                    "bookValue": book_value,
                    "high": high,
                    "low": low
                })
        except Exception:
            continue

    return render_template("index.html", screener_data=results, scroll_to="screener", user=session.get('user'))

# -------- Prediction --------
@app.route("/predict_stock", methods=["POST"])
def predict_stock():
    symbol = request.form.get('stock_symbol', '').strip().upper()
    exchange = request.form.get('exchange', '').strip()

    symbol_ext = symbol + (".NS" if exchange == "NSE" else ".BO" if exchange == "BSE" else "")
    if not symbol_ext:
        return render_template("index.html", error="Invalid exchange", scroll_to="predictor", user=session.get('user'))

    model = get_model(symbol)
    scaler = get_scaler(symbol)

    if not model or not scaler:
        return render_template("index.html", error="Model or scaler not found. Please train first.", scroll_to="predictor", user=session.get('user'))

    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=365 * 3)

    df = yf.download(symbol_ext, start=start, end=end)
    if df.empty or 'Close' not in df:
        return render_template("index.html", error=f"No data to predict for {symbol}", scroll_to="predictor", user=session.get('user'))

    data = df['Close'].values.reshape(-1, 1)
    scaled = scaler.transform(data)

    input_seq = scaled[-60:]
    preds, dates = [], []

    for i in range(7):
        pred = model.predict(input_seq.reshape(1, 60, 1), verbose=0)[0][0]
        preds.append(pred)
        input_seq = np.append(input_seq, [[pred]], axis=0)[1:]
        dates.append((end + datetime.timedelta(days=i + 1)).strftime('%d %b %Y'))

    final = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    forecast = list(zip(dates, final))

    return render_template("index.html", forecast=forecast, symbol=symbol, scroll_to="predictor", user=session.get('user'))


@app.route("/portfolio")
def portfolio():
    if 'user' not in session:
        return redirect(url_for('login'))

    user_email = session['user']
    with sqlite3.connect("users.db") as conn:
        cur = conn.cursor()
        cur.execute("SELECT symbol FROM portfolio WHERE user_email = ?", (user_email,))
        symbols = cur.fetchall()

    stocks = []
    for (symbol,) in symbols:
        try:
            ticker = yf.Ticker(symbol + ".NS")
            info = ticker.info
            stocks.append({
                'symbol': symbol,
                'price': info.get('currentPrice', 0),
                'high': info.get('fiftyTwoWeekHigh', 0),
                'low': info.get('fiftyTwoWeekLow', 0)
            })
        except Exception:
            continue

    return render_template("portfolio.html", stocks=stocks, user=user_email)

@app.route("/add_to_portfolio", methods=["POST"])
def add_to_portfolio():
    if 'user' not in session:
        return redirect(url_for('login'))

    user_email = session['user']
    symbol = request.form.get('symbol', '').upper()

    with sqlite3.connect("users.db") as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM portfolio WHERE user_email = ? AND symbol = ?", (user_email, symbol))
        if not cur.fetchone():
            cur.execute("INSERT INTO portfolio (user_email, symbol) VALUES (?, ?)", (user_email, symbol))
            conn.commit()
    return redirect(url_for('portfolio'))

@app.route("/remove_from_portfolio", methods=["POST"])
def remove_from_portfolio():
    if 'user' not in session:
        return redirect(url_for('login'))

    user_email = session['user']
    symbol = request.form.get('symbol', '').upper()

    with sqlite3.connect("users.db") as conn:
        conn.execute("DELETE FROM portfolio WHERE user_email = ? AND symbol = ?", (user_email, symbol))
        conn.commit()

    return redirect(url_for('portfolio'))

# -------- Run Server --------
if __name__ == "__main__":
    app.run(debug=True)
