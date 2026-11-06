import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

from database import SessionLocal, StockData, StockMetrics


def normalize_symbol(symbol, exchange=None):
    symbol = (symbol or '').strip().upper()
    symbol = symbol.replace('.NS', '').replace('.BO', '')
    if not symbol or any(char in symbol for char in [' ', '/', '\\']):
        raise ValueError('Enter a valid NSE or BSE stock ticker')
    return symbol, (exchange or 'NSE').upper()


def calculate_rsi(series, period=14):
    delta = series.diff()
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)
    average_gain = gains.rolling(period).mean()
    average_loss = losses.rolling(period).mean()
    if average_loss.iloc[-1] == 0:
        return 50.0
    rs = average_gain.iloc[-1] / average_loss.iloc[-1]
    return float(100 - (100 / (1 + rs))) if np.isfinite(rs) else 50.0


def calculate_macd(series):
    if len(series) < 26:
        return 0.0
    return float(series.ewm(span=12, adjust=False).mean().iloc[-1] - series.ewm(span=26, adjust=False).mean().iloc[-1])


def calculate_finsight_score(rsi, macd, pe_ratio, current_price, week_52_high):
    score = 50.0
    score += 15 if 30 <= rsi <= 70 else (-10 if rsi < 20 or rsi > 80 else 0)
    if pe_ratio:
        score += 15 if 10 <= pe_ratio <= 25 else (-10 if pe_ratio > 40 else 0)
    if week_52_high:
        price_ratio = current_price / week_52_high
        score += 10 if price_ratio > 0.9 else (-10 if price_ratio < 0.5 else 0)
    score += 10 if macd > 0 else -5
    return round(max(0, min(100, score)), 1)


def is_cache_fresh(metrics, hours=24):
    if metrics is None or metrics.updated_at is None:
        return False
    return metrics.updated_at >= datetime.utcnow() - timedelta(hours=hours)


def fetch_stock(symbol, exchange='NSE', cache_hours=24, force_refresh=False):
    symbol, exchange = normalize_symbol(symbol, exchange)

    db = SessionLocal()
    try:
        metrics = db.query(StockMetrics).filter_by(symbol=symbol).first()
        if not force_refresh and is_cache_fresh(metrics, cache_hours):
            cached_history = get_stock_history(symbol, days=365)
            if cached_history is not None and len(cached_history) >= 20:
                if metrics and (not metrics.market_cap or not metrics.pe_ratio or not metrics.book_value):
                    try:
                        suffix = '.NS' if exchange == 'NSE' else '.BO'
                        info = yf.Ticker(symbol + suffix).info or {}
                        pe_ratio = info.get('trailingPE') or info.get('forwardPE') or 0
                        metrics.market_cap = float(info.get('marketCap') or 0)
                        metrics.pe_ratio = float(pe_ratio or 0)
                        metrics.pb_ratio = float(info.get('priceToBook') or 0)
                        metrics.book_value = float(info.get('bookValue') or 0)
                        dividend_yield = float(info.get('dividendYield') or 0)
                        if dividend_yield and dividend_yield < 0.1:
                            dividend_yield *= 100
                        metrics.dividend_yield = dividend_yield
                        db.commit()
                        db.refresh(metrics)
                    except Exception:
                        pass
                return cached_history, metrics

        suffix = '.NS' if exchange == 'NSE' else '.BO'
        ticker_symbol = symbol + suffix
        ticker = yf.Ticker(ticker_symbol)
        history = ticker.history(period='1y', interval='1d', auto_adjust=False)

        if history.empty and exchange == 'NSE':
            ticker = yf.Ticker(symbol + '.BO')
            history = ticker.history(period='1y', interval='1d', auto_adjust=False)
        if history.empty:
            raise ValueError(f'No market data found for {symbol}. Check the ticker and exchange.')

        history = history.rename(columns=str.title).dropna(subset=['Close'])
        required = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required.issubset(history.columns):
            raise ValueError(f'Market data for {symbol} is incomplete')

        history.index = pd.to_datetime(history.index)
        if hasattr(history.index, 'tz') and history.index.tz is not None:
            history.index = history.index.tz_localize(None)

        for row_date, row in history.iterrows():
            stock = db.query(StockData).filter_by(symbol=symbol, date=row_date.to_pydatetime()).first()
            if stock is None:
                stock = StockData(symbol=symbol, date=row_date.to_pydatetime())
                db.add(stock)
            stock.open = float(row['Open'])
            stock.high = float(row['High'])
            stock.low = float(row['Low'])
            stock.close = float(row['Close'])
            stock.volume = int(row['Volume']) if pd.notna(row['Volume']) else 0

        close = history['Close'].astype(float)
        rsi = calculate_rsi(close)
        macd = calculate_macd(close)
        current_price = float(close.iloc[-1])
        week_52_high = float(history['High'].max())
        week_52_low = float(history['Low'].min())

        info = {}
        try:
            info = ticker.info or {}
        except Exception:
            pass

        pe_ratio = info.get('trailingPE') or info.get('forwardPE') or 0
        pb_ratio = info.get('priceToBook') or 0
        market_cap = info.get('marketCap') or 0
        dividend_yield = info.get('dividendYield') or 0
        book_value = info.get('bookValue') or 0
        if dividend_yield and dividend_yield < 0.1:
            dividend_yield *= 100

        pe_ratio = float(pe_ratio) if pe_ratio else 0
        pb_ratio = float(pb_ratio) if pb_ratio else 0
        book_value = float(book_value) if book_value else 0
        market_cap = float(market_cap) if market_cap else 0
        dividend_yield = float(dividend_yield) if dividend_yield else 0
        score = calculate_finsight_score(rsi, macd, pe_ratio, current_price, week_52_high)

        values = {
            'current_price': current_price,
            'market_cap': market_cap,
            'pe_ratio': pe_ratio,
            'pb_ratio': pb_ratio,
            'book_value': book_value,
            'dividend_yield': dividend_yield,
            'week_52_high': week_52_high,
            'week_52_low': week_52_low,
            'rsi': rsi,
            'macd': macd,
            'score': score,
            'updated_at': datetime.utcnow(),
        }

        if metrics is None:
            metrics = StockMetrics(symbol=symbol, **values)
            db.add(metrics)
        else:
            for key, value in values.items():
                setattr(metrics, key, value)

        db.commit()
        db.refresh(metrics)
        return history.reset_index().rename(columns={'index': 'Date', 'Datetime': 'Date'}), metrics
    finally:
        db.close()


def get_stock_history(symbol, days=365):
    db = SessionLocal()
    try:
        cutoff = datetime.utcnow() - timedelta(days=days)
        data = db.query(StockData).filter(
            StockData.symbol == symbol,
            StockData.date >= cutoff
        ).order_by(StockData.date).all()
        if not data:
            return None
        return pd.DataFrame([{
            'Date': item.date,
            'Open': item.open,
            'High': item.high,
            'Low': item.low,
            'Close': item.close,
            'Volume': item.volume,
        } for item in data])
    finally:
        db.close()


def get_stock_metrics(symbol):
    db = SessionLocal()
    try:
        return db.query(StockMetrics).filter_by(symbol=symbol).first()
    finally:
        db.close()


SCREENER_UNIVERSE = [
    'ADANIENT','ADANIPORTS','APOLLOHOSP','ASIANPAINT','AXISBANK','BAJAJ-AUTO','BAJFINANCE','BAJAJFINSV','BHARTIARTL','BPCL',
    'BRITANNIA','CIPLA','COALINDIA','DIVISLAB','DRREDDY','EICHERMOT','GRASIM','HCLTECH','HDFCBANK','HDFCLIFE',
    'HEROMOTOCO','HINDALCO','HINDUNILVR','ICICIBANK','INDUSINDBK','INFY','ITC','JSWSTEEL','KOTAKBANK','LT',
    'M&M','MARUTI','NESTLEIND','NTPC','ONGC','POWERGRID','RELIANCE','SBILIFE','SBIN','SHRIRAMFIN',
    'SUNPHARMA','TATACONSUM','TATAMOTORS','TATASTEEL','TCS','TECHM','TITAN','TRENT','ULTRACEMCO','WIPRO'
]


def populate_screener_universe(limit=40):
    """Populate the NSE screener with a curated benchmark universe on demand."""
    db = SessionLocal()
    populated = 0
    try:
        symbols = []
        for symbol in SCREENER_UNIVERSE[:limit]:
            metrics = db.query(StockMetrics).filter_by(symbol=symbol).first()
            if not is_cache_fresh(metrics, 24):
                symbols.append(symbol)
        if not symbols:
            return len(db.query(StockMetrics).all())

        tickers = [symbol + '.NS' for symbol in symbols]
        downloaded = yf.download(tickers, period='1y', interval='1d', group_by='ticker',
                                 auto_adjust=False, threads=True, progress=False)

        for symbol in symbols:
            try:
                if len(symbols) == 1:
                    history = downloaded
                else:
                    history = downloaded[symbol + '.NS'].copy()
                history = history.rename(columns=str.title).dropna(subset=['Close'])
                if len(history) < 20:
                    continue
                close = history['Close'].astype(float)
                rsi = calculate_rsi(close)
                macd = calculate_macd(close)
                current_price = float(close.iloc[-1])
                week_high = float(history['High'].max())
                week_low = float(history['Low'].min())
                for row_date, row in history.iterrows():
                    stock = db.query(StockData).filter_by(symbol=symbol, date=row_date.to_pydatetime()).first()
                    if stock is None:
                        stock = StockData(symbol=symbol, date=row_date.to_pydatetime())
                        db.add(stock)
                    stock.open = float(row['Open']); stock.high = float(row['High'])
                    stock.low = float(row['Low']); stock.close = float(row['Close'])
                    stock.volume = int(row['Volume']) if pd.notna(row['Volume']) else 0
                metrics = db.query(StockMetrics).filter_by(symbol=symbol).first()
                values = {
                    'current_price': current_price, 'market_cap': 0, 'pe_ratio': 0,
                    'pb_ratio': 0, 'book_value': 0, 'dividend_yield': 0,
                    'week_52_high': week_high, 'week_52_low': week_low,
                    'rsi': rsi, 'macd': macd,
                    'score': calculate_finsight_score(rsi, macd, 0, current_price, week_high),
                    'updated_at': datetime.utcnow()
                }
                if metrics is None:
                    db.add(StockMetrics(symbol=symbol, **values))
                else:
                    for key, value in values.items(): setattr(metrics, key, value)
                populated += 1
            except Exception:
                continue
        db.commit()
        return populated
    finally:
        db.close()
