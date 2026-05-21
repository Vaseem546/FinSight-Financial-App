import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from database import SessionLocal, StockData, StockMetrics

NIFTY_50_SYMBOLS = [
    # Nifty 50
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "ADANIENT",
    "KOTAKBANK", "SBIN", "ITC", "BHARTIARTL", "LT", "AXISBANK",
    "WIPRO", "HCLTECH", "SUNPHARMA", "BAJFINANCE", "MARUTI", "HINDUNILVR",
    "ASIANPAINT", "ULTRACEMCO", "TITAN", "NTPC", "POWERGRID", "ONGC",
    "COALINDIA", "BPCL", "GRASIM", "ADANIPORTS", "DRREDDY", "EICHERMOT",
    
    # Additional 30 Stocks (Total 60)
    "TATAMOTORS", "BAJAJFINSV", "TATASTEEL", "TECHM", "NESTLEIND", "HINDALCO",
    "INDUSINDBK", "JSWSTEEL", "BRITANNIA", "DIVISLAB", "APOLLOHOSP", "HEROMOTOCO",
    "CIPLA", "TATACONSUM", "SHREECEM", "UPL", "ADANIGREEN", "PIDILITIND",
    "SIEMENS", "GODREJCP", "DABUR", "HAVELLS", "BERGEPAINT", "MARICO",
    "BANDHANBNK", "TORNTPHARM", "AMBUJACEM", "BOSCHLTD", "COLPAL", "MCDOWELL-N"
]

# Base prices for realistic stock data (in INR)
BASE_PRICES = {
    "RELIANCE": 2450, "TCS": 3850, "INFY": 1650, "HDFCBANK": 1680, "ICICIBANK": 1150,
    "ADANIENT": 2800, "KOTAKBANK": 1850, "SBIN": 620, "ITC": 465, "BHARTIARTL": 1580,
    "LT": 3650, "AXISBANK": 1120, "WIPRO": 465, "HCLTECH": 1820, "SUNPHARMA": 1680,
    "BAJFINANCE": 7250, "MARUTI": 12500, "HINDUNILVR": 2380, "ASIANPAINT": 2850,
    "ULTRACEMCO": 10200, "TITAN": 3450, "NTPC": 355, "POWERGRID": 285, "ONGC": 245,
    "COALINDIA": 420, "BPCL": 595, "GRASIM": 2450, "ADANIPORTS": 1280, "DRREDDY": 6150,
    "EICHERMOT": 4850, "TATAMOTORS": 920, "BAJAJFINSV": 1680, "TATASTEEL": 165,
    "TECHM": 1720, "NESTLEIND": 2450, "HINDALCO": 645, "INDUSINDBK": 1450,
    "JSWSTEEL": 920, "BRITANNIA": 4850, "DIVISLAB": 5950, "APOLLOHOSP": 6850,
    "HEROMOTOCO": 4650, "CIPLA": 1450, "TATACONSUM": 1150, "SHREECEM": 26500,
    "UPL": 545, "ADANIGREEN": 1850, "PIDILITIND": 2950, "SIEMENS": 6450,
    "GODREJCP": 1180, "DABUR": 505, "HAVELLS": 1650, "BERGEPAINT": 485,
    "MARICO": 625, "BANDHANBNK": 195, "TORNTPHARM": 3350, "AMBUJACEM": 585,
    "BOSCHLTD": 34500, "COLPAL": 2850, "MCDOWELL-N": 1950
}

def generate_realistic_ohlcv(base_price, days=90):
    """Generate realistic OHLCV data without API"""
    np.random.seed(hash(base_price) % 2**32)
    dates = [datetime.now() - timedelta(days=days-i) for i in range(days)]
    
    prices = []
    current = base_price
    volatility = base_price * 0.02  # 2% daily volatility
    
    for _ in range(days):
        change = np.random.normal(0, volatility)
        current = max(current + change, base_price * 0.7)  # Don't drop below 70%
        prices.append(current)
    
    data = []
    for i, date in enumerate(dates):
        close = prices[i]
        daily_range = close * 0.03
        open_price = close + np.random.uniform(-daily_range, daily_range)
        high = max(open_price, close) + abs(np.random.uniform(0, daily_range))
        low = min(open_price, close) - abs(np.random.uniform(0, daily_range))
        volume = int(np.random.uniform(1000000, 10000000))
        
        data.append({
            'Date': date,
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': volume
        })
    
    return pd.DataFrame(data)

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else 50.0

def calculate_macd(data):
    """Calculate MACD"""
    exp1 = data.ewm(span=12, adjust=False).mean()
    exp2 = data.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    return macd.iloc[-1] if not macd.empty else 0.0

def calculate_finsight_score(metrics):
    """Custom FinSight Score Algorithm (0-100)"""
    score = 50  # Base score
    
    # RSI scoring (30-70 is healthy)
    if 30 <= metrics['rsi'] <= 70:
        score += 15
    elif metrics['rsi'] < 20 or metrics['rsi'] > 80:
        score -= 10
    
    # PE Ratio scoring (lower is better, but not too low)
    if 10 <= metrics['pe_ratio'] <= 25:
        score += 15
    elif metrics['pe_ratio'] > 40:
        score -= 10
    
    # Price vs 52W High (momentum indicator)
    if metrics['current_price'] and metrics['week_52_high']:
        price_ratio = (metrics['current_price'] / metrics['week_52_high']) * 100
        if price_ratio > 90:
            score += 10
        elif price_ratio < 50:
            score -= 10
    
    # MACD scoring (positive is bullish)
    if metrics['macd'] > 0:
        score += 10
    else:
        score -= 5
    
    return max(0, min(100, score))  # Clamp between 0-100

def sync_stock_data(symbol):
    """Sync single stock data to database (NO API - Local Generation)"""
    db = SessionLocal()
    try:
        base_price = BASE_PRICES.get(symbol, 1000)
        df = generate_realistic_ohlcv(base_price, days=90)
        
        if df is None or df.empty:
            print(f"[FAIL] Failed to generate data for {symbol}")
            return False
        
        # Store historical data
        for _, row in df.iterrows():
            stock_data = StockData(
                symbol=symbol,
                date=row['Date'],
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=int(row['Volume'])
            )
            db.merge(stock_data)
        
        # Calculate metrics
        rsi = calculate_rsi(df['Close'])
        macd = calculate_macd(df['Close'])
        current_price = df['Close'].iloc[-1]
        week_52_high = df['High'].max()
        week_52_low = df['Low'].min()
        
        # Generate realistic PE ratios
        pe_ratio = round(np.random.uniform(15, 35), 2)
        
        metrics_data = {
            'rsi': rsi,
            'macd': macd,
            'current_price': current_price,
            'market_cap': round(current_price * np.random.uniform(10000, 500000), 2),
            'pe_ratio': pe_ratio,
            'pb_ratio': round(np.random.uniform(2, 8), 2),
            'dividend_yield': round(np.random.uniform(0.5, 3.5), 2),
            'week_52_high': week_52_high,
            'week_52_low': week_52_low
        }
        
        metrics_data['score'] = calculate_finsight_score(metrics_data)
        
        metrics = db.query(StockMetrics).filter_by(symbol=symbol).first()
        if metrics:
            for key, value in metrics_data.items():
                setattr(metrics, key, value)
            metrics.updated_at = datetime.utcnow()
        else:
            metrics = StockMetrics(symbol=symbol, **metrics_data)
            db.add(metrics)
        
        db.commit()
        print(f"[OK] Synced {symbol} | Score: {metrics_data['score']:.1f}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error syncing {symbol}: {e}")
        db.rollback()
        return False
    finally:
        db.close()

def sync_all_stocks():
    """Sync all stocks (NO API - Local Generation)"""
    print("[SYNC] Starting local data generation...")
    success_count = 0
    for symbol in NIFTY_50_SYMBOLS:
        if sync_stock_data(symbol):
            success_count += 1
    print(f"[DONE] Sync complete: {success_count}/{len(NIFTY_50_SYMBOLS)} stocks generated")

def get_stock_history(symbol, days=10):
    """Get stock history from local DB"""
    db = SessionLocal()
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        data = db.query(StockData).filter(
            StockData.symbol == symbol,
            StockData.date >= cutoff_date
        ).order_by(StockData.date).all()
        
        if not data:
            return None
        
        df = pd.DataFrame([{
            'Date': d.date,
            'Open': d.open,
            'High': d.high,
            'Low': d.low,
            'Close': d.close,
            'Volume': d.volume
        } for d in data])
        
        return df
    finally:
        db.close()

def get_stock_metrics(symbol):
    """Get pre-computed metrics from DB"""
    db = SessionLocal()
    try:
        return db.query(StockMetrics).filter_by(symbol=symbol).first()
    finally:
        db.close()

if __name__ == '__main__':
    from database import init_db
    init_db()
    sync_all_stocks()
