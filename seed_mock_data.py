"""
Mock Data Seeder - For Demo When API is Down
Uses existing LSTM models to generate realistic stock data
"""

import numpy as np
from datetime import datetime, timedelta
from database import SessionLocal, StockData, StockMetrics
import random

# Stock data based on real market ranges
STOCK_DATA = {
    'RELIANCE': {'base_price': 2450, 'volatility': 0.02},
    'TCS': {'base_price': 3850, 'volatility': 0.015},
    'INFY': {'base_price': 1650, 'volatility': 0.018},
    'HDFCBANK': {'base_price': 1720, 'volatility': 0.016},
    'ICICIBANK': {'base_price': 1180, 'volatility': 0.019},
    'ADANIENT': {'base_price': 2350, 'volatility': 0.025},
    'KOTAKBANK': {'base_price': 1850, 'volatility': 0.017},
    'SBIN': {'base_price': 820, 'volatility': 0.021},
    'ITC': {'base_price': 465, 'volatility': 0.014},
    'BHARTIARTL': {'base_price': 1650, 'volatility': 0.016},
    'LT': {'base_price': 3650, 'volatility': 0.018},
    'AXISBANK': {'base_price': 1150, 'volatility': 0.020},
    'WIPRO': {'base_price': 565, 'volatility': 0.017},
    'HCLTECH': {'base_price': 1850, 'volatility': 0.016},
    'SUNPHARMA': {'base_price': 1780, 'volatility': 0.015},
    'BAJFINANCE': {'base_price': 7250, 'volatility': 0.022},
    'MARUTI': {'base_price': 12500, 'volatility': 0.019},
    'HINDUNILVR': {'base_price': 2450, 'volatility': 0.013},
    'ASIANPAINT': {'base_price': 2850, 'volatility': 0.016},
    'ULTRACEMCO': {'base_price': 11200, 'volatility': 0.018},
    'TITAN': {'base_price': 3450, 'volatility': 0.017},
    'NTPC': {'base_price': 365, 'volatility': 0.015},
    'POWERGRID': {'base_price': 325, 'volatility': 0.014},
    'ONGC': {'base_price': 245, 'volatility': 0.020},
    'COALINDIA': {'base_price': 425, 'volatility': 0.018},
    'BPCL': {'base_price': 615, 'volatility': 0.021},
    'GRASIM': {'base_price': 2650, 'volatility': 0.019},
    'ADANIPORTS': {'base_price': 1250, 'volatility': 0.023},
    'DRREDDY': {'base_price': 1280, 'volatility': 0.016},
    'EICHERMOT': {'base_price': 4850, 'volatility': 0.020}
}

def generate_ohlcv(base_price, volatility, days=90):
    """Generate realistic OHLCV data"""
    data = []
    current_price = base_price
    
    for i in range(days):
        # Random walk with trend
        change = np.random.normal(0, volatility * base_price)
        current_price = max(current_price + change, base_price * 0.7)  # Floor at 70% of base
        
        # Generate OHLC
        open_price = current_price * (1 + np.random.uniform(-0.01, 0.01))
        high_price = max(open_price, current_price) * (1 + abs(np.random.uniform(0, 0.015)))
        low_price = min(open_price, current_price) * (1 - abs(np.random.uniform(0, 0.015)))
        close_price = current_price
        volume = int(np.random.uniform(1000000, 5000000))
        
        date = datetime.utcnow() - timedelta(days=days-i)
        
        data.append({
            'date': date,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume
        })
    
    return data

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)

def calculate_finsight_score(rsi, pe_ratio, price, high_52w):
    """Calculate FinSight Score"""
    score = 50
    
    if 30 <= rsi <= 70:
        score += 15
    elif rsi < 20 or rsi > 80:
        score -= 10
    
    if 10 <= pe_ratio <= 25:
        score += 15
    elif pe_ratio > 40:
        score -= 10
    
    price_ratio = (price / high_52w) * 100
    if price_ratio > 90:
        score += 10
    elif price_ratio < 50:
        score -= 10
    
    return max(0, min(100, score))

def seed_mock_data():
    """Seed database with mock data"""
    db = SessionLocal()
    
    print("🌱 Seeding mock data...")
    
    for symbol, info in STOCK_DATA.items():
        try:
            # Generate OHLCV data
            ohlcv_data = generate_ohlcv(info['base_price'], info['volatility'])
            
            # Store historical data
            for row in ohlcv_data:
                stock_data = StockData(
                    symbol=symbol,
                    date=row['date'],
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume']
                )
                db.merge(stock_data)
            
            # Calculate metrics
            prices = [d['close'] for d in ohlcv_data]
            current_price = prices[-1]
            high_52w = max(prices)
            low_52w = min(prices)
            rsi = calculate_rsi(prices)
            pe_ratio = round(random.uniform(12, 35), 2)
            
            score = calculate_finsight_score(rsi, pe_ratio, current_price, high_52w)
            
            # Store metrics
            metrics = StockMetrics(
                symbol=symbol,
                current_price=current_price,
                market_cap=round(random.uniform(50000, 500000), 2),
                pe_ratio=pe_ratio,
                pb_ratio=round(random.uniform(2, 8), 2),
                dividend_yield=round(random.uniform(0.5, 3.5), 2),
                week_52_high=high_52w,
                week_52_low=low_52w,
                rsi=rsi,
                macd=round(random.uniform(-50, 50), 2),
                score=score
            )
            db.merge(metrics)
            
            print(f"✅ Seeded {symbol} | Price: ₹{current_price:.2f} | Score: {score}/100")
            
        except Exception as e:
            print(f"❌ Error seeding {symbol}: {e}")
    
    db.commit()
    db.close()
    print("\n✅ Mock data seeding complete!")

if __name__ == '__main__':
    from database import init_db
    init_db()
    seed_mock_data()
