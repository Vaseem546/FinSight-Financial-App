from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from dotenv import load_dotenv
import os
import plotly.graph_objs as go
from datetime import datetime, timedelta

# Import our modules
from database import init_db, SessionLocal, User, Portfolio
from data_sync import get_stock_history, get_stock_metrics, NIFTY_50_SYMBOLS
from backtesting import predict_next_days, backtest_prediction, get_model_accuracy
from portfolio_analytics import get_user_portfolio, get_portfolio_analytics

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Initialize database on startup
init_db()

# Check if stock data exists, if not generate it
from database import StockMetrics
db = SessionLocal()
try:
    stock_count = db.query(StockMetrics).count()
    if stock_count == 0:
        print("[INIT] No stock data found, generating 60 stocks...")
        from data_sync import sync_all_stocks
        sync_all_stocks()
        print("[INIT] Stock data generation complete")
    else:
        print(f"[INFO] Found {stock_count} stocks in database")
finally:
    db.close()

# ========== AUTHENTICATION ==========
@app.route('/')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', user=session['user'])

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        db = SessionLocal()
        user = db.query(User).filter_by(email=email).first()
        db.close()
        
        if user and user.check_password(password):
            session['user'] = email
            return redirect(url_for('index'))
        
        return render_template('login_register.html', error="Invalid credentials", show="login")
    
    return render_template('login_register.html', show="login")

@app.route('/register', methods=['POST'])
def register():
    email = request.form['email']
    password = request.form['password']
    
    db = SessionLocal()
    existing = db.query(User).filter_by(email=email).first()
    
    if existing:
        db.close()
        return render_template('login_register.html', error="Email already registered", show="register")
    
    user = User(email=email)
    user.set_password(password)
    db.add(user)
    db.commit()
    db.close()
    
    session['user'] = email
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# ========== STOCK ANALYSIS ==========
@app.route('/analyze', methods=['POST'])
def analyze():
    symbol = request.form.get('analysis_symbol', '').upper()
    exchange = request.form.get('analysis_exchange')
    
    try:
        # Get data from local DB
        hist = get_stock_history(symbol, days=10)
        if hist is None or hist.empty:
            raise Exception(f"Stock data not available for {symbol}. Only 60 stocks are loaded.")
        
        # Get metrics
        metrics = get_stock_metrics(symbol)
        
        # Create candlestick chart
        candlestick_data = [go.Candlestick(
            x=hist['Date'].dt.strftime('%Y-%m-%d').tolist(),
            open=hist['Open'].tolist(),
            high=hist['High'].tolist(),
            low=hist['Low'].tolist(),
            close=hist['Close'].tolist()
        )]
        
        layout = go.Layout(
            title=f'{symbol} - FinSight Score: {metrics.score if metrics else "N/A"}',
            plot_bgcolor='#111111',
            paper_bgcolor='#111111',
            font=dict(color='white'),
            xaxis=dict(title='Date', color='white', showgrid=False),
            yaxis=dict(title='Price (₹)', color='white', showgrid=False)
        )
        
        fig = go.Figure(data=candlestick_data, layout=layout)
        
        analysis_data = {
            'symbol': symbol,
            'price': round(hist['Close'].iloc[-1], 2),
            'open': round(hist['Open'].iloc[-1], 2),
            'previousClose': round(hist['Close'].iloc[-2], 2) if len(hist) > 1 else 0,
            'dayHigh': round(hist['High'].iloc[-1], 2),
            'dayLow': round(hist['Low'].iloc[-1], 2),
            'volume': int(hist['Volume'].iloc[-1]),
            'fiftyTwoWeekHigh': round(metrics.week_52_high, 2) if metrics else 0,
            'fiftyTwoWeekLow': round(metrics.week_52_low, 2) if metrics else 0,
            'finsight_score': round(metrics.score, 1) if metrics else 0,
            'rsi': round(metrics.rsi, 1) if metrics else 0,
            'pe_ratio': round(metrics.pe_ratio, 1) if metrics else 0
        }
        
        return render_template('index.html',
            candlestick=fig.to_plotly_json(),
            analysis=analysis_data,
            analysis_symbol=symbol,
            exchange=exchange,
            user=session.get('user')
        )
        
    except Exception as e:
        return render_template('index.html', analysis_error=str(e), user=session.get('user'))

# ========== STOCK SCREENER ==========
@app.route('/screener', methods=['POST'])
def screener():
    try:
        filters = {
            'min_score': float(request.form.get('min_score') or 0),
            'max_pe': float(request.form.get('max_pe') or 999),
            'min_rsi': float(request.form.get('min_rsi') or 0),
            'max_rsi': float(request.form.get('max_rsi') or 100)
        }
        
        db = SessionLocal()
        from database import StockMetrics
        
        stocks = db.query(StockMetrics).filter(
            StockMetrics.score >= filters['min_score'],
            StockMetrics.pe_ratio <= filters['max_pe'],
            StockMetrics.rsi >= filters['min_rsi'],
            StockMetrics.rsi <= filters['max_rsi']
        ).order_by(StockMetrics.score.desc()).all()
        
        db.close()
        
        screener_data = [{
            'symbol': s.symbol,
            'price': round(s.current_price, 2),
            'score': round(s.score, 1),
            'rsi': round(s.rsi, 1),
            'pe': round(s.pe_ratio, 1),
            'high': round(s.week_52_high, 2),
            'low': round(s.week_52_low, 2)
        } for s in stocks]
        
        return render_template('index.html', screener_data=screener_data, scroll_to='screener', user=session.get('user'))
        
    except Exception as e:
        return render_template('index.html', screener_error=str(e), scroll_to='screener', user=session.get('user'))

# ========== STOCK PREDICTOR ==========
@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form['predict_symbol'].upper()
    
    try:
        # Get historical data from DB
        db = SessionLocal()
        from database import StockData
        
        data = db.query(StockData).filter_by(symbol=symbol).order_by(
            StockData.date.desc()
        ).limit(60).all()
        
        db.close()
        
        if len(data) < 60:
            raise Exception(f"Not enough historical data for {symbol}")
        
        # Simple trend-based prediction (fallback when LSTM fails)
        prices = [d.close for d in reversed(data)]
        recent_avg = sum(prices[-7:]) / 7
        trend = (prices[-1] - prices[-30]) / 30  # Daily trend
        
        forecast = []
        for i in range(1, 8):
            predicted_price = recent_avg + (trend * i)
            future_date = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
            forecast.append((future_date, round(predicted_price, 2)))
        
        # Try to get model accuracy if available
        accuracy = get_model_accuracy(symbol)
        
        return render_template('index.html',
            forecast=forecast,
            symbol=symbol,
            model_accuracy=accuracy,
            prediction_method="Trend Analysis",
            scroll_to='predictor',
            user=session.get('user')
        )
        
    except Exception as e:
        return render_template('index.html', error=str(e), scroll_to='predictor', user=session.get('user'))

# ========== PORTFOLIO ==========
@app.route('/portfolio')
def portfolio():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    stocks = get_user_portfolio(session['user'])
    analytics = get_portfolio_analytics(session['user'])
    
    return render_template('portfolio.html', stocks=stocks, analytics=analytics, user=session['user'])

@app.route('/add_to_portfolio', methods=['POST'])
def add_to_portfolio():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    symbol = request.form.get('symbol')
    exchange = request.form.get('exchange', 'NSE')
    
    db = SessionLocal()
    user = db.query(User).filter_by(email=session['user']).first()
    
    # Check if already exists
    existing = db.query(Portfolio).filter_by(user_id=user.id, symbol=symbol).first()
    if not existing:
        portfolio_item = Portfolio(user_id=user.id, symbol=symbol, exchange=exchange)
        db.add(portfolio_item)
        db.commit()
    
    db.close()
    return redirect(url_for('portfolio'))

@app.route('/remove_from_portfolio', methods=['POST'])
def remove_from_portfolio():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    symbol = request.form.get('symbol')
    
    db = SessionLocal()
    user = db.query(User).filter_by(email=session['user']).first()
    db.query(Portfolio).filter_by(user_id=user.id, symbol=symbol).delete()
    db.commit()
    db.close()
    
    return redirect(url_for('portfolio'))

# ========== ADMIN/SYNC ==========
@app.route('/regenerate_data')
def regenerate_data():
    """Regenerate local stock data (NO API)"""
    from data_sync import sync_all_stocks
    sync_all_stocks()
    return jsonify({'status': 'success', 'message': 'Data regenerated locally'})

@app.route('/backtest/<symbol>')
def backtest(symbol):
    """Backtest endpoint"""
    result = backtest_prediction(symbol.upper(), days_ago=7)
    if result:
        return jsonify(result)
    return jsonify({'error': 'Backtest failed'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
