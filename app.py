from flask import Flask, render_template, request, redirect, session, url_for, jsonify, send_file
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
import yfinance as yf
import pandas as pd
import os
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import json
import plotly.graph_objs as go
from plotly.utils import PlotlyJSONEncoder
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()
app.secret_key =  os.getenv("SECRET_KEY")
# MongoDB config (if used)
app.config["MONGO_URI"] = os.environ.get("MONGO_URI")
mongo = PyMongo(app)
db = mongo.db
users = db.users
mongo = PyMongo(app)

# Load all stocks from CSV
stocks_df = pd.read_csv("all_stocks.csv")


# ---------------- ROUTES ---------------- #

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = users.find_one({'email': email})
        if user and user['password'] == password:
            session['user'] = user['name']
            return redirect(url_for('dashboard'))
        else:
            return render_template("login_register.html", show="login", error="Invalid credentials")
    return render_template("login_register.html", show="login")

@app.route('/register', methods=['POST'])
def register():
    name = request.form['name']
    email = request.form['email']
    password = request.form['password']

    if users.find_one({'email': email}):
        return render_template("login_register.html", show="register", error="Email already exists")
    
    users.insert_one({'name': name, 'email': email, 'password': password})
    session['user'] = name
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    if 'user' in session:
        return render_template('index.html', user=session['user'])
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


# ---------------- STOCK ANALYSIS ---------------- #

@app.route('/analyze_stock', methods=['POST'])
def analyze_stock():
    symbol = request.form['analysis_symbol'].upper()
    exchange = request.form['analysis_exchange']

    if exchange == 'NSE':
        ticker = symbol + ".NS"
    elif exchange == 'BSE':
        ticker = symbol + ".BO"
    else:
        return render_template('index.html', analysis_error="Invalid exchange selected.", scroll_to="analysis", user=session.get('user'))

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")

        if hist.empty:
            return render_template('index.html', analysis_error="No historical data found for this stock.", scroll_to="analysis", user=session.get('user'))

        stock_df = hist
        required_cols = {'Open', 'High', 'Low', 'Close'}
        if not required_cols.issubset(stock_df.columns):
            return render_template('index.html', analysis_error="Incomplete data for candlestick chart.", scroll_to="analysis", user=session.get('user'))

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

        fig = go.Figure(data=[go.Candlestick(
            x=stock_df.index.strftime('%Y-%m-%d'),
            open=stock_df['Open'],
            high=stock_df['High'],
            low=stock_df['Low'],
            close=stock_df['Close']
        )])

        fig.update_layout(
            title=f"{symbol.upper()} Candlestick Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark"
        )

        candlestick_json = json.dumps(fig, cls=PlotlyJSONEncoder)

        return render_template("index.html", analysis=stock_info, candlestick=candlestick_json, scroll_to="analysis", user=session.get('user'))

    except Exception as e:
        return render_template('index.html', analysis_error="Something went wrong: " + str(e), scroll_to="analysis", user=session.get('user'))


#------------ STOCK SCREENER --------------------#

@app.route('/screener', methods=['GET', 'POST'])
def screener():
    if 'user' not in session:
        return redirect(url_for('login'))

    filtered_results = []
    if request.method == 'POST':
        form = request.form

        # Get filter values
        min_marketcap = float(form.get("min_marketcap") or 0)
        min_volume = float(form.get("min_volume") or 0)
        max_pe = float(form.get("max_pe") or 1000)
        max_pb = float(form.get("max_pb") or 1000)
        min_dividend = float(form.get("min_dividend") or 0)
        min_bookvalue = float(form.get("min_bookvalue") or 0)

        # Filter all stocks
        for _, row in stocks_df.iterrows():
            symbol = row['Symbol'] + ".NS"
            try:
                stock = yf.Ticker(symbol)
                info = stock.info

                # Extract relevant metrics
                market_cap = info.get('marketCap', 0) / 1e7  # Convert to Cr
                volume = info.get('volume', 0)
                pe = info.get('trailingPE', 0)
                pb = info.get('priceToBook', 0)
                dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
                book_value = info.get('bookValue', 0)
                price = info.get('currentPrice', 0)
                high = info.get('fiftyTwoWeekHigh', 0)
                low = info.get('fiftyTwoWeekLow', 0)

                # Apply filters
                if (
                    market_cap >= min_marketcap and
                    volume >= min_volume and
                    pe <= max_pe and
                    pb <= max_pb and
                    dividend_yield >= min_dividend and
                    book_value >= min_bookvalue
                ):
                    filtered_results.append({
                        "symbol": row['Symbol'],
                        "price": price,
                        "marketCap": round(market_cap, 2),
                        "volume": volume,
                        "pe": pe,
                        "pb": pb,
                        "dividendYield": round(dividend_yield, 2),
                        "bookValue": book_value,
                        "high": high,
                        "low": low
                    })

            except Exception as e:
                continue

    return render_template("index.html", screener_data=filtered_results,scroll_to="screener", user=session.get('user'))


NIFTY_50 = [filename.replace('.h5', '') for filename in os.listdir('models') if filename.endswith('.h5')]
# ---------------- STOCK PREDICTION ---------------- #

@app.route('/predict_stock', methods=['POST'])
def predict_stock():
    symbol = request.form.get('stock_symbol', '').strip().upper()
    exchange = request.form.get('exchange', '').strip()
    user = session.get('user')

    if not symbol or not exchange:
        return render_template('index.html', user=user, error="Please enter both stock symbol and exchange.")

    symbol_key = symbol  # Used for naming model
    if exchange == 'NSE':
        symbol += ".NS"
    elif exchange == 'BSE':
        symbol += ".BO"

    model_path = f"models/{symbol_key}.h5"

    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=365 * 3)

    df = yf.download(symbol, start=start, end=end)
    if df.empty or 'Close' not in df:
        return render_template('index.html', user=user, error="Failed to fetch stock data.")

    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    if not os.path.exists(model_path):
        # ✅ Auto-train if model doesn't exist
        if len(scaled_data) < 100:
            return render_template('index.html', user=user, error="Not enough data to train a model.")
        
        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i])
            y.append(scaled_data[i])
        X, y = np.array(X), np.array(y)

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=5, batch_size=32, verbose=0,
                  callbacks=[EarlyStopping(monitor='loss', patience=2)])

        model.save(model_path)
    else:
        model = load_model(model_path)

    last_60 = scaled_data[-60:]
    input_seq = last_60
    predictions = []
    dates = []

    for i in range(7):
        pred_input = input_seq.reshape(1, 60, 1)
        pred = model.predict(pred_input, verbose=0)[0][0]
        predictions.append(pred)
        input_seq = np.append(input_seq, [[pred]], axis=0)[1:]
        forecast_date = (end + datetime.timedelta(days=i + 1)).strftime('%d %B %Y, %I:%M %p')
        dates.append(forecast_date)

    final_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    forecast = list(zip(dates, final_prices))

    return render_template('index.html', user=user, forecast=forecast, symbol=symbol_key, scroll_to='predictor')
# ---------------- AUTOCOMPLETE API ---------------- #

@app.route('/autocomplete')
def autocomplete():
    term = request.args.get('term', '').upper()
    suggestions = [s for s in NIFTY_50 if term in s]
    return jsonify(suggestions)


@app.route('/portfolio')
def portfolio():
    if 'user' not in session:
        return redirect(url_for('login'))

    portfolio = mongo.db.portfolio.find({'user': session['user']})
    stocks = []
    for entry in portfolio:
        symbol = entry['symbol'] + ".NS"
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            stocks.append({
                'symbol': entry['symbol'],
                'price': info.get('currentPrice', 0),
                'high': info.get('fiftyTwoWeekHigh', 0),
                'low': info.get('fiftyTwoWeekLow', 0)
            })
        except Exception:
            continue

    return render_template('portfolio.html', stocks=stocks, user=session['user'])

@app.route('/add_to_portfolio', methods=['POST'])
def add_to_portfolio():
    if 'user' not in session:
        return redirect(url_for('login'))

    symbol = request.form.get('symbol', '').upper()
    if symbol:
        existing = mongo.db.portfolio.find_one({'user': session['user'], 'symbol': symbol})
        if not existing:
            mongo.db.portfolio.insert_one({'user': session['user'], 'symbol': symbol})
    return redirect(url_for('portfolio'))

@app.route('/remove_from_portfolio', methods=['POST'])
def remove_from_portfolio():
    if 'user' not in session:
        return redirect(url_for('login'))

    symbol = request.form.get('symbol', '').upper()
    mongo.db.portfolio.delete_one({'user': session['user'], 'symbol': symbol})
    return redirect(url_for('portfolio'))

# ---------------- RUN ---------------- #

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Railway provides the port
    app.run(debug=False, host='0.0.0.0', port=port)

