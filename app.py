from flask import Flask, render_template, request, redirect, url_for, session, jsonify
# --- NEW IMPORTS FOR MYSQL AND ENV ---
from dotenv import load_dotenv # Used to load the .env file
import pymysql.cursors       # The MySQL driver
from werkzeug.security import generate_password_hash, check_password_hash 
# -------------------------------------
import os
import json
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from keras.models import load_model
import pickle
from flask import send_from_directory
from werkzeug.utils import secure_filename
import plotly
import plotly.graph_objs as go
import joblib
from functools import lru_cache
# --- END IMPORTS ---


load_dotenv() # Load environment variables from .env

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'models'

# ========== DATABASE SETUP (MYSQL) ==========

# Configuration pulled from .env
DB_CONFIG = {
    'host': os.environ.get('MYSQL_HOST'),
    'port': int(os.environ.get('MYSQL_PORT', 3306)),
    'user': os.environ.get('MYSQL_USER'),
    'password': os.environ.get('MYSQL_PASSWORD'),
    'db': os.environ.get('MYSQL_DB'),
    'cursorclass': pymysql.cursors.DictCursor # Returns rows as dictionaries (like SQLite's Row factory)
}

def get_db():
    """Establishes a connection to the MySQL database."""
    try:
        return pymysql.connect(**DB_CONFIG)
    except Exception as e:
        print(f"Database connection failed: {e}")
        # In a real app, you'd log and handle this gracefully
        raise

# === Utility function to swap SQLite '?' placeholders with MySQL '%s' ===
def convert_query(sql):
    return sql.replace('?', '%s')



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
        with conn.cursor() as cursor:
          
            cursor.execute(convert_query('SELECT * FROM users WHERE email = ?'), (email,))
            user = cursor.fetchone()
        conn.close()

        # Check password using dictionary key from DictCursor
        if user and check_password_hash(user['password'], password):
            session['user'] = email
            return redirect(url_for('index'))
        else:
            return render_template('login_register.html', error="Invalid credentials", show="login")

    return render_template('login_register.html', show="login")


@app.route('/register', methods=['POST'])
def register():
    email = request.form['email']
    password = request.form['password']

    hashed_password = generate_password_hash(password)

    try:
        conn = get_db()
        with conn.cursor() as cursor:
           
            cursor.execute(convert_query('INSERT INTO users (email, password) VALUES (?, ?)'), (email, hashed_password))
        conn.commit()
        conn.close()
        session['user'] = email
        return redirect(url_for('index'))
    # Use the specific PyMySQL exception for integrity errors (e.g., duplicate email)
    except pymysql.err.IntegrityError: 
        return render_template('login_register.html', error="Email already registered", show="register")

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


@lru_cache(maxsize=100)
def fetch_stock_history(full_symbol):
    print(f"[CACHE] Fetching data for {full_symbol}")
    try:
        stock = yf.Ticker(full_symbol)
        hist = stock.history(period="10d")
        if hist.empty:
            raise Exception("No data returned from yfinance")
        return hist
    except Exception as e:
        print(f"[ERROR] yfinance failed for {full_symbol}: {e}")
        return None
# ========== ANALYSIS ==========
@app.route('/analyze', methods=['POST'])
def analyze():
    symbol = request.form.get('analysis_symbol', '').upper()
    exchange = request.form.get('analysis_exchange')

    try:
        if not symbol or not exchange:
            raise Exception("Symbol or Exchange not provided")

        full_symbol = symbol + '.NS' if exchange == 'NSE' else symbol + '.BO'

        hist = fetch_stock_history(full_symbol)
        if hist is None:
            raise Exception("Stock data could not be fetched (API may be rate-limited).")

        hist.reset_index(inplace=True)
        hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')

        candlestick_data = [go.Candlestick(
            x=hist['Date'].tolist(),
            open=hist['Open'].tolist(),
            high=hist['High'].tolist(),
            low=hist['Low'].tolist(),
            close=hist['Close'].tolist()
        )]

        layout = go.Layout(
            title=f'{symbol} Candlestick Chart',
            plot_bgcolor='#111111',
            paper_bgcolor='#111111',
            font=dict(color='white'),
            xaxis=dict(title='Date', color='white', showgrid=False),
            yaxis=dict(title='Price', color='white', showgrid=False)
        )

        fig = go.Figure(data=candlestick_data, layout=layout)
        candlestick_json = fig.to_plotly_json()

        return render_template('index.html',
            candlestick=candlestick_json,
            analysis={
                'symbol': symbol,
                'price': round(float(hist['Close'].iloc[-1]), 2),
                'open': round(float(hist['Open'].iloc[-1]), 2),
                'previousClose': round(float(hist['Close'].iloc[-2]), 2) if len(hist) > 1 else round(float(hist['Close'].iloc[-1]), 2),
                'dayHigh': round(float(hist['High'].iloc[-1]), 2),
                'dayLow': round(float(hist['Low'].iloc[-1]), 2),
                'volume': int(hist['Volume'].iloc[-1]),
                'fiftyTwoWeekHigh': round(float(hist['High'].max()), 2),
                'fiftyTwoWeekLow': round(float(hist['Low'].min()), 2)
            },
            analysis_symbol=symbol,
            exchange=exchange
        )

    except Exception as e:
        return render_template('index.html', analysis_error=str(e))


# ========== SCREENER ==========

NIFTY_50_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "ADANIENT.NS",
    "KOTAKBANK.NS", "SBIN.NS", "ITC.NS", "BHARTIARTL.NS", "LT.NS", "AXISBANK.NS",
    "WIPRO.NS", "HCLTECH.NS", "SUNPHARMA.NS", "BAJFINANCE.NS", "MARUTI.NS", "HINDUNILVR.NS"
]

@app.route('/screener', methods=['POST'])
def screener():
    try:
        filters = {
            'min_marketcap': float(request.form.get('min_marketcap') or 0),
            'min_volume': float(request.form.get('min_volume') or 0),
            'max_pe': float(request.form.get('max_pe') or float('inf')),
            'max_pb': float(request.form.get('max_pb') or float('inf')),
            'min_dividend': float(request.form.get('min_dividend') or 0),
            'min_bookvalue': float(request.form.get('min_bookvalue') or 0),
        }

        filtered_stocks = []

        for symbol in NIFTY_50_SYMBOLS:
            try:
                stock = yf.Ticker(symbol)
                info = stock.info

                data = {
                    "symbol": symbol.replace(".NS", ""),
                    "price": info.get("currentPrice"),
                    "marketCap": (info.get("marketCap") or 0) / 1e7,  # to Cr
                    "volume": info.get("volume") or 0,
                    "pe": info.get("trailingPE") or float('inf'),
                    "pb": info.get("priceToBook") or float('inf'),
                    "dividendYield": (info.get("dividendYield") or 0) * 100,
                    "bookValue": info.get("bookValue") or 0,
                    "high": info.get("fiftyTwoWeekHigh"),
                    "low": info.get("fiftyTwoWeekLow")
                }

                if (
                    data["marketCap"] >= filters["min_marketcap"] and
                    data["volume"] >= filters["min_volume"] and
                    data["pe"] <= filters["max_pe"] and
                    data["pb"] <= filters["max_pb"] and
                    data["dividendYield"] >= filters["min_dividend"] and
                    data["bookValue"] >= filters["min_bookvalue"]
                ):
                    filtered_stocks.append(data)

            except Exception as stock_error:
                print(f"Error fetching {symbol}: {stock_error}")
                continue

        return render_template(
            "index.html",
            screener_data=filtered_stocks,
            scroll_to="screener",
            user=session.get("user")
        )

    except Exception as e:
        return render_template(
            "index.html",
            screener_error=f"Error: {e}",
            scroll_to="screener",
            user=session.get("user")
        )

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
        scaler = joblib.load(scaler_path)  # ✅ FIXED HERE

        df = yf.download(full_symbol, period="90d", interval="1d")
        if df.empty or len(df) < 60:
            return render_template("index.html", error=f"No data to predict for {symbol}: Not enough data", scroll_to="predictor", user=session.get("user"))

        close_data = df['Close'].values.reshape(-1, 1)
        scaled_data = scaler.transform(close_data)
        last_60 = scaled_data[-60:]
        x_input = last_60.reshape(1, 60, 1)

        predictions = []
        for _ in range(7):
            pred = model.predict(x_input, verbose=0)[0][0]
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

    user_email = session['user']

    conn = get_db()
    with conn.cursor() as cursor:
        # SQL Query updated for MySQL placeholder
        cursor.execute(convert_query("SELECT symbol, exchange FROM portfolio WHERE user_email = ?"), (user_email,))
        rows = cursor.fetchall()
    conn.close()

    stocks = []

    # Iterate through the rows (which are dictionaries due to DictCursor)
    for row in rows:
        try:
            symbol = row['symbol']
            exchange = row['exchange']
            
            full_symbol = f"{symbol}.NS" if exchange == "NSE" else f"{symbol}.BO"
            data = yf.Ticker(full_symbol).info

            # Ensure data contains expected keys
            if 'currentPrice' not in data:
                raise ValueError("Missing currentPrice in yfinance info")

            stock_info = {
                'symbol': symbol,
                'exchange': exchange,
                'current_price': round(data['currentPrice'], 2),
                'fifty_two_week_high': round(data.get('fiftyTwoWeekHigh', 0), 2),
                'fifty_two_week_low': round(data.get('fiftyTwoWeekLow', 0), 2)
            }
            stocks.append(stock_info)

        except Exception as e:
            print(f"⚠️ Failed to fetch stock info for {symbol}: {e}")
            continue

    return render_template('portfolio.html', stocks=stocks, user=user_email)

# --- Add to Portfolio ---
@app.route('/add_to_portfolio', methods=['POST'])
def add_to_portfolio():
    if 'user' not in session:
        return redirect(url_for('login'))

    symbol = request.form.get('symbol')
    exchange = request.form.get('exchange')
    user_email = session['user']

    print(f"📥 Received: {symbol}  {user_email} | Exchange: {exchange}")

    # Skip if any field is missing
    if not symbol or not exchange or not user_email:
        print("⚠️ Missing data - insertion skipped")
        return redirect(url_for('portfolio'))

    conn = get_db()
    with conn.cursor() as cursor:
        # SQL Query updated for MySQL placeholder
        cursor.execute(convert_query('INSERT INTO portfolio (user_email, symbol, exchange) VALUES (?, ?, ?)'), 
                     (user_email, symbol, exchange))
    conn.commit()
    conn.close()

    return redirect(url_for('portfolio'))

# --- Remove from Portfolio ---
@app.route('/remove_from_portfolio', methods=['POST'])
def remove_from_portfolio():
    if 'user' not in session:
        return redirect(url_for('login'))

    symbol = request.form.get('symbol')
    email = session['user']

    conn = get_db()
    with conn.cursor() as cursor:
        # SQL Query updated for MySQL placeholder
        cursor.execute(convert_query('DELETE FROM portfolio WHERE user_email = ? AND symbol = ?'), (email, symbol))
    conn.commit()
    conn.close()

    return redirect(url_for('portfolio'))


# ========== MAIN ==========
if __name__ == '__main__':
    app.run(debug=True)