import numpy as np
from database import SessionLocal, Portfolio, StockData, StockMetrics
from datetime import datetime, timedelta

def get_user_portfolio(user_email):
    """Get user's portfolio with current metrics"""
    db = SessionLocal()
    try:
        portfolio_items = db.query(Portfolio).join(Portfolio.user).filter(
            Portfolio.user.has(email=user_email)
        ).all()
        
        stocks = []
        for item in portfolio_items:
            metrics = db.query(StockMetrics).filter_by(symbol=item.symbol).first()
            if metrics:
                stocks.append({
                    'symbol': item.symbol,
                    'exchange': item.exchange,
                    'current_price': metrics.current_price,
                    'week_52_high': metrics.week_52_high,
                    'week_52_low': metrics.week_52_low,
                    'score': metrics.score,
                    'rsi': metrics.rsi,
                    'pe_ratio': metrics.pe_ratio
                })
        
        return stocks
    finally:
        db.close()

def calculate_portfolio_volatility(symbols):
    """Calculate portfolio volatility (risk metric)"""
    db = SessionLocal()
    try:
        volatilities = []
        for symbol in symbols:
            # Get last 30 days
            cutoff = datetime.utcnow() - timedelta(days=30)
            data = db.query(StockData).filter(
                StockData.symbol == symbol,
                StockData.date >= cutoff
            ).order_by(StockData.date).all()
            
            if len(data) < 10:
                continue
            
            prices = [d.close for d in data]
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            volatilities.append(volatility * 100)
        
        if not volatilities:
            return 0
        
        return round(np.mean(volatilities), 2)
    finally:
        db.close()

def calculate_diversification_score(symbols):
    """Calculate diversification score (0-100)"""
    if len(symbols) == 0:
        return 0
    
    # Simple diversification: more stocks = better diversification
    # Ideal portfolio: 8-15 stocks
    if len(symbols) >= 8:
        return 100
    elif len(symbols) >= 5:
        return 75
    elif len(symbols) >= 3:
        return 50
    else:
        return 25

def get_portfolio_analytics(user_email):
    """Get comprehensive portfolio analytics"""
    stocks = get_user_portfolio(user_email)
    
    if not stocks:
        return None
    
    symbols = [s['symbol'] for s in stocks]
    
    # Calculate metrics
    avg_score = sum(s['score'] for s in stocks) / len(stocks)
    volatility = calculate_portfolio_volatility(symbols)
    diversification = calculate_diversification_score(symbols)
    
    # Risk assessment
    if volatility < 20:
        risk_level = "Low"
    elif volatility < 40:
        risk_level = "Medium"
    else:
        risk_level = "High"
    
    # Portfolio health
    if avg_score >= 70:
        health = "Excellent"
    elif avg_score >= 60:
        health = "Good"
    elif avg_score >= 50:
        health = "Average"
    else:
        health = "Poor"
    
    return {
        'total_stocks': len(stocks),
        'avg_finsight_score': round(avg_score, 1),
        'volatility': volatility,
        'risk_level': risk_level,
        'diversification_score': diversification,
        'portfolio_health': health,
        'top_performer': max(stocks, key=lambda x: x['score'])['symbol'],
        'recommendations': generate_recommendations(stocks, diversification, volatility)
    }

def generate_recommendations(stocks, diversification, volatility):
    """Generate portfolio recommendations"""
    recommendations = []
    
    if len(stocks) < 5:
        recommendations.append("Consider adding more stocks to improve diversification")
    
    if volatility > 40:
        recommendations.append("High volatility detected. Consider adding stable blue-chip stocks")
    
    low_score_stocks = [s['symbol'] for s in stocks if s['score'] < 40]
    if low_score_stocks:
        recommendations.append(f"Review underperforming stocks: {', '.join(low_score_stocks)}")
    
    if not recommendations:
        recommendations.append("Portfolio looks healthy. Keep monitoring regularly.")
    
    return recommendations

if __name__ == '__main__':
    # Test analytics
    analytics = get_portfolio_analytics('test@example.com')
    if analytics:
        print(f"Portfolio Analytics: {analytics}")
