from flask import Flask, render_template, request, redirect, session, url_for, jsonify
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
import yfinance as yf
import pandas as pd
import os
import json
from dotenv import load_dotenv
import plotly.graph_objs as go
from plotly.utils import PlotlyJSONEncoder
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# --- Flask App Setup ---
app = Flask(__name__)
load_dotenv()

app.secret_key = os.getenv("SECRET_KEY")
app.config["MONGO_URI"] = os.getenv("MONGO_URI")

mongo = PyMongo(app)
db = mongo.db
users = db.users

# Load CSV for screener
stocks_df = pd.read_csv("all_stocks.csv")

# Load model names
NIFTY_50 = [f.replace('.h5', '') for f in os.listdir("models") if f.endswith('.h5')]

# ---------- Routes ----------
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = users.find_one({'email': email})

        if user and check_password_hash(user['password'], password):
            session['user'] = user['name']
            return redirect(url_for('dashboard'))
        else:
            return render_template("login_register.html", show="login", error="Invalid credentials")

    return render_template("login_register.html", show="login")
 


@app.route('/register', methods=["POST"])
def register():
    name = request.form['name']
    email = request.form['email']
    password = request.form['password']

    if users.find_one({'email': email}):
        return render_template("login_register.html", show="register", error="Email already exists")

    hashed_pw = generate_password_hash(password)
    users.insert_one({'name': name, 'email': email, 'password': hashed_pw})
    session['user'] = name
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template("index.html", user=session['user'])

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ---------- Stock Analysis ----------
@app.route('/analyze_stock', methods=["POST"])
def analyze_stock():
    symbol = request.form.get("analysis_symbol", "").upper()
    exchange = request.form.get("analysis_exchange", "")

    if exchange == "NSE":
        symbol_full = symbol + ".NS"
    elif exchange == "BSE":
        symbol_full = symbol + ".BO"
    else:
        return render_template("index.html", analysis_error="Invalid exchange", scroll_to="analysis", user=session.get('user'))

    try:
        stock = yf.Ticker(symbol_full)
        hist = stock.history(period="1mo")
        if hist.empty:
            raise Exception("No historical data")

        fig = go.Figure(data=[go.Candlestick(
            x=hist.index.strftime('%Y-%m-%d'),
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close']
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

# ---------- Screener ----------
@app.route('/screener', methods=['POST'])
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

# ---------- Prediction ----------
@app.route("/predict_stock", methods=["POST"])
def predict_stock():
    symbol = request.form.get('stock_symbol', '').strip().upper()
    exchange = request.form.get('exchange', '').strip()

    if exchange == "NSE":
        symbol_ext = symbol + ".NS"
    elif exchange == "BSE":
        symbol_ext = symbol + ".BO"
    else:
        return render_template("index.html", error="Invalid exchange", scroll_to="predictor", user=session.get('user'))

    model_path = f"models/{symbol}.h5"
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=365 * 3)

    df = yf.download(symbol_ext, start=start, end=end)
    if df.empty or 'Close' not in df:
        return render_template("index.html", error="No data to predict", scroll_to="predictor", user=session.get('user'))

    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    if not os.path.exists(model_path):
        return render_template("index.html", error="Model not found. Train and retry.", scroll_to="predictor", user=session.get('user'))

    model = load_model(model_path)
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

# ---------- Portfolio ----------
@app.route("/portfolio")
def portfolio():
    if 'user' not in session:
        return redirect(url_for('login'))

    portfolio = db.portfolio.find({'user': session['user']})
    stocks = []
    for entry in portfolio:
        try:
            ticker = yf.Ticker(entry['symbol'] + ".NS")
            info = ticker.info
            stocks.append({
                'symbol': entry['symbol'],
                'price': info.get('currentPrice', 0),
                'high': info.get('fiftyTwoWeekHigh', 0),
                'low': info.get('fiftyTwoWeekLow', 0)
            })
        except Exception:
            continue

    return render_template("portfolio.html", stocks=stocks, user=session['user'])

@app.route("/add_to_portfolio", methods=["POST"])
def add_to_portfolio():
    if 'user' not in session:
        return redirect(url_for('login'))

    symbol = request.form.get('symbol', '').upper()
    if symbol and not db.portfolio.find_one({'user': session['user'], 'symbol': symbol}):
        db.portfolio.insert_one({'user': session['user'], 'symbol': symbol})
    return redirect(url_for('portfolio'))

@app.route("/remove_from_portfolio", methods=["POST"])
def remove_from_portfolio():
    if 'user' not in session:
        return redirect(url_for('login'))

    symbol = request.form.get('symbol', '').upper()
    db.portfolio.delete_one({'user': session['user'], 'symbol': symbol})
    return redirect(url_for('portfolio'))

# ---------- Run ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
