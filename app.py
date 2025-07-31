from flask import Flask, render_template, request, redirect, session, url_for
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import os
from datetime import datetime, timedelta

# Fix: Add user-agent to avoid yfinance blocking
import requests
from yfinance import shared
shared._requests_session = requests.Session()
shared._requests_session.headers.update({'User-Agent': 'Mozilla/5.0'})

app = Flask(__name__)
app.secret_key = "finsight_secret_key"

app.config["MONGO_URI"] = "mongodb://localhost:27017/finsight"
mongo = PyMongo(app)

# Home Route
@app.route("/")
def index():
    if "user" in session:
        return render_template("index.html", user=session["user"])
    return redirect(url_for("login"))

# Register Route
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        users = mongo.db.users
        existing_user = users.find_one({"email": request.form["email"]})

        if existing_user is None:
            hashpass = generate_password_hash(request.form["password"])
            users.insert_one({
                "email": request.form["email"],
                "password": hashpass
            })
            session["user"] = request.form["email"]
            return redirect(url_for("index"))
        return render_template("register.html", error="Email already exists.")
    return render_template("register.html")

# Login Route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        users = mongo.db.users
        login_user = users.find_one({"email": request.form["email"]})

        if login_user and check_password_hash(login_user["password"], request.form["password"]):
            session["user"] = request.form["email"]
            return redirect(url_for("index"))
        return render_template("login.html", error="Invalid credentials.")
    return render_template("login.html")

# Logout Route
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# Market Analysis
@app.route("/analyze", methods=["POST"])
def analyze():
    if "user" not in session:
        return redirect(url_for("login"))
    try:
        symbol = request.form["analysis_symbol"].upper()
        exchange = request.form["analysis_exchange"]
        yf_symbol = f"{symbol}.NS" if exchange == "NSE" else f"{symbol}.BO"

        data = yf.download(yf_symbol, period="60d", interval="1d")
        info = yf.Ticker(yf_symbol).info

        if data.empty or not info:
            raise ValueError("No data found")

        candlestick = {
            "data": [{
                "x": data.index.strftime('%Y-%m-%d').tolist(),
                "open": data["Open"].tolist(),
                "high": data["High"].tolist(),
                "low": data["Low"].tolist(),
                "close": data["Close"].tolist(),
                "type": "candlestick",
                "name": yf_symbol
            }],
            "layout": {"title": f"{yf_symbol} Candlestick Chart"}
        }

        return render_template("index.html",
                               user=session["user"],
                               analysis={
                                   "symbol": symbol,
                                   "price": round(info.get("currentPrice", 0), 2),
                                   "open": round(info.get("open", 0), 2),
                                   "previousClose": round(info.get("previousClose", 0), 2),
                                   "dayHigh": round(info.get("dayHigh", 0), 2),
                                   "dayLow": round(info.get("dayLow", 0), 2),
                                   "volume": info.get("volume", 0),
                                   "fiftyTwoWeekHigh": round(info.get("fiftyTwoWeekHigh", 0), 2),
                                   "fiftyTwoWeekLow": round(info.get("fiftyTwoWeekLow", 0), 2)
                               },
                               candlestick=candlestick,
                               scroll_to="analysis")
    except Exception as e:
        return render_template("index.html", user=session["user"], analysis_error=f"Failed to fetch stock info: {str(e)}", scroll_to="analysis")

# Prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))
    try:
        symbol = request.form["predict_symbol"].upper()
        exchange = request.form["exchange"]
        yf_symbol = f"{symbol}.NS" if exchange == "NSE" else f"{symbol}.BO"

        model_path = f"models/{symbol}_model.h5"
        scaler_path = f"models/{symbol}_scaler.save"

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise ValueError(f"No data to predict for {symbol}: Model or Scaler not found for this symbol.")

        model = load_model(model_path)
        scaler = joblib.load(scaler_path)

        data = yf.download(yf_symbol, period="3mo")
        close_data = data["Close"].dropna().values[-60:]

        if len(close_data) < 60:
            raise ValueError(f"No data to predict for {symbol}: Not enough data")

        input_data = scaler.transform(close_data.reshape(-1, 1))
        input_sequence = input_data[-60:].reshape(1, 60, 1)

        predictions = []
        for _ in range(7):
            pred = model.predict(input_sequence)[0][0]
            predictions.append(pred)
            input_sequence = np.append(input_sequence[:, 1:, :], [[[pred]]], axis=1)

        forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        forecast_dates = [(datetime.today() + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(7)]

        return render_template("index.html", user=session["user"], forecast=zip(forecast_dates, forecast), symbol=symbol, scroll_to="predictor")
    except Exception as e:
        return render_template("index.html", user=session["user"], error=str(e), scroll_to="predictor")

# Screener
@app.route("/screener", methods=["POST"])
def screener():
    if "user" not in session:
        return redirect(url_for("login"))

    try:
        filters = {
            "min_marketcap": float(request.form.get("min_marketcap", 0)),
            "min_volume": float(request.form.get("min_volume", 0)),
            "max_pe": float(request.form.get("max_pe", float("inf"))),
            "max_pb": float(request.form.get("max_pb", float("inf"))),
            "min_dividend": float(request.form.get("min_dividend", 0)),
            "min_bookvalue": float(request.form.get("min_bookvalue", 0)),
        }

        symbols = ["RELIANCE", "INFY", "TCS", "HDFCBANK", "WIPRO", "ADANIPORTS"]
        results = []

        for sym in symbols:
            yf_symbol = f"{sym}.NS"
            try:
                info = yf.Ticker(yf_symbol).info
                if not info or "marketCap" not in info:
                    continue
                if (
                    info.get("marketCap", 0)/1e7 >= filters["min_marketcap"] and
                    info.get("volume", 0) >= filters["min_volume"] and
                    info.get("trailingPE", float("inf")) <= filters["max_pe"] and
                    info.get("priceToBook", float("inf")) <= filters["max_pb"] and
                    info.get("dividendYield", 0)*100 >= filters["min_dividend"] and
                    info.get("bookValue", 0) >= filters["min_bookvalue"]
                ):
                    results.append({
                        "symbol": sym,
                        "price": round(info.get("currentPrice", 0), 2),
                        "marketCap": round(info.get("marketCap", 0)/1e7, 2),
                        "volume": info.get("volume", 0),
                        "pe": round(info.get("trailingPE", 0), 2),
                        "pb": round(info.get("priceToBook", 0), 2),
                        "dividendYield": round(info.get("dividendYield", 0)*100, 2),
                        "bookValue": round(info.get("bookValue", 0), 2),
                        "high": round(info.get("fiftyTwoWeekHigh", 0), 2),
                        "low": round(info.get("fiftyTwoWeekLow", 0), 2)
                    })
            except Exception as e:
                print(f"Error processing {sym}: {str(e)}")
                continue

        if not results:
            return render_template("index.html", user=session["user"], screener_error="No data available for screener stocks.", scroll_to="screener")

        return render_template("index.html", user=session["user"], screener_data=results, scroll_to="screener")
    except Exception as e:
        return render_template("index.html", user=session["user"], screener_error=str(e), scroll_to="screener")

# Portfolio (Placeholder)
@app.route("/portfolio")
def portfolio():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("portfolio.html", user=session["user"])

# Add to Portfolio (Placeholder)
@app.route("/add_to_portfolio", methods=["POST"])
def add_to_portfolio():
    if "user" not in session:
        return redirect(url_for("login"))
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
