from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from dotenv import load_dotenv
import os
import plotly.graph_objs as go

# Import our modules
from database import init_db, SessionLocal, User, Portfolio
from data_sync import fetch_stock, get_stock_history, get_stock_metrics, populate_screener_universe
from portfolio_analytics import get_user_portfolio, get_portfolio_analytics

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Flag to track if initialization is done
initialized = False

def initialize_app():
    """Initialize the database without fetching or generating market data."""
    global initialized
    if initialized:
        return

    try:
        init_db()
        initialized = True
    except Exception as e:
        print(f"[ERROR] Initialization failed: {e}", flush=True)

# ========== AUTHENTICATION ==========
@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'FinSight is running'}), 200

@app.route('/')
def index():
    initialize_app()
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
        hist, metrics = fetch_stock(symbol, exchange or 'NSE')
        if hist is None or hist.empty:
            raise Exception(f'Stock data not available for {symbol}')
        
        # Create candlestick chart
        candlestick_data = [go.Candlestick(
            x=hist['Date'].tolist(),
            open=hist['Open'].tolist(),
            high=hist['High'].tolist(),
            low=hist['Low'].tolist(),
            close=hist['Close'].tolist(),
            name=symbol,
            increasing=dict(line=dict(color='#16794b', width=1), fillcolor='#16794b'),
            decreasing=dict(line=dict(color='#b42318', width=1), fillcolor='#d92d20')
        )]
        
        layout = go.Layout(
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font=dict(color='#171717'),
            hovermode='x unified',
            margin=dict(l=58, r=18, t=20, b=48),
            xaxis=dict(title=None, color='#525252', type='date', showgrid=False,
                       tickformat='%b %Y', nticks=8,
                       rangeslider=dict(visible=False), linecolor='#d4d4d0'),
            yaxis=dict(title='Price (₹)', color='#525252', showgrid=True,
                       gridcolor='#ecece8', fixedrange=False, side='right')
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
            'pe_ratio': round(metrics.pe_ratio, 1) if metrics else 0,
            'updatedAt': metrics.updated_at.strftime('%d %b %Y, %H:%M UTC') if metrics and metrics.updated_at else 'Not cached'
        }
        
        return render_template('index.html',
            candlestick=fig.to_json(),
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
        def optional_float(name, default=None):
            value = request.form.get(name)
            return float(value) if value not in (None, '') else default

        filters = {
            'min_score': optional_float('min_score', 0),
            'max_pe': optional_float('max_pe'),
            'min_rsi': optional_float('min_rsi', 0),
            'max_rsi': optional_float('max_rsi', 100),
            'min_market_cap': optional_float('min_market_cap'),
            'max_market_cap': optional_float('max_market_cap'),
            'min_book_value': optional_float('min_book_value'),
            'min_dividend_yield': optional_float('min_dividend_yield'),
            'max_pb': optional_float('max_pb'),
            'near_high': request.form.get('near_high', 'any'),
            'near_low': request.form.get('near_low', 'any')
        }

        populate_screener_universe(limit=40)

        db = SessionLocal()
        from database import StockMetrics
        query = db.query(StockMetrics).filter(
            StockMetrics.score >= filters['min_score'],
            StockMetrics.rsi >= filters['min_rsi'],
            StockMetrics.rsi <= filters['max_rsi']
        )
        if filters['max_pe'] is not None:
            query = query.filter(StockMetrics.pe_ratio > 0, StockMetrics.pe_ratio <= filters['max_pe'])
        if filters['min_market_cap'] is not None:
            query = query.filter(StockMetrics.market_cap >= filters['min_market_cap'] * 10000000)
        if filters['max_market_cap'] is not None:
            query = query.filter(StockMetrics.market_cap <= filters['max_market_cap'] * 10000000)
        if filters['min_book_value'] is not None:
            query = query.filter(StockMetrics.book_value >= filters['min_book_value'])
        if filters['min_dividend_yield'] is not None:
            query = query.filter(StockMetrics.dividend_yield >= filters['min_dividend_yield'])
        if filters['max_pb'] is not None:
            query = query.filter(StockMetrics.pb_ratio > 0, StockMetrics.pb_ratio <= filters['max_pb'])

        stocks = query.order_by(StockMetrics.score.desc()).all()
        db.close()

        filtered = []
        for s in stocks:
            price = s.current_price or 0
            high = s.week_52_high or 0
            low = s.week_52_low or 0
            near_high = high > 0 and price >= high * 0.90
            near_low = low > 0 and price <= low * 1.10
            if filters['near_high'] == 'yes' and not near_high:
                continue
            if filters['near_high'] == 'no' and near_high:
                continue
            if filters['near_low'] == 'yes' and not near_low:
                continue
            if filters['near_low'] == 'no' and near_low:
                continue
            position = ((price - low) / (high - low) * 100) if high > low else 0
            filtered.append({
                'symbol': s.symbol, 'price': round(price, 2), 'score': round(s.score or 0, 1),
                'rsi': round(s.rsi or 0, 1), 'pe': round(s.pe_ratio or 0, 1),
                'pb': round(s.pb_ratio or 0, 2), 'book_value': round(s.book_value or 0, 2),
                'market_cap': round((s.market_cap or 0) / 10000000, 0),
                'dividend_yield': round(s.dividend_yield or 0, 2),
                'high': round(high, 2), 'low': round(low, 2),
                'near_high': 'Yes' if near_high else 'No', 'near_low': 'Yes' if near_low else 'No',
                'position': round(position, 1)
            })

        return render_template('index.html', screener_data=filtered, scroll_to='screener', user=session.get('user'))
    except Exception as e:
        return render_template('index.html', screener_error=str(e), scroll_to='screener', user=session.get('user'))

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

if __name__ == '__main__':
    app.run(debug=True, port=5000)
