import numpy as np
import joblib
from datetime import datetime, timedelta
from database import SessionLocal, PredictionHistory, StockData
import os

def load_model_and_scaler(symbol):
    """Load LSTM model and scaler for a symbol"""
    model_path = f"models/{symbol}_model.h5"
    scaler_path = f"models/{symbol}_scaler.save"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    
    try:
        from tensorflow.keras.models import load_model
        # Load with custom objects to handle InputLayer issue
        from tensorflow.keras.layers import InputLayer
        model = load_model(model_path, compile=False, custom_objects={'InputLayer': InputLayer})
    except ImportError:
        print(f"TensorFlow not installed. Skipping LSTM load for {symbol}.")
        return None, None
    except Exception as e:
        print(f"Model load error for {symbol}: {e}")
        # Try alternative loading method
        try:
            import h5py
            from tensorflow.keras.models import load_model
            model = load_model(model_path, compile=False)
        except:
            return None, None
    
    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        print(f"Scaler load error for {symbol}: {e}")
        return None, None
        
    return model, scaler

def predict_next_days(symbol, days=7):
    """Predict next N days using LSTM model"""
    model, scaler = load_model_and_scaler(symbol)
    if model is None:
        return None
    
    db = SessionLocal()
    try:
        # Get last 60 days of data
        data = db.query(StockData).filter_by(symbol=symbol).order_by(
            StockData.date.desc()
        ).limit(60).all()
        
        if len(data) < 60:
            return None
        
        # Prepare data
        close_prices = np.array([d.close for d in reversed(data)]).reshape(-1, 1)
        scaled_data = scaler.transform(close_prices)
        x_input = scaled_data[-60:].reshape(1, 60, 1)
        
        # Generate predictions
        predictions = []
        for _ in range(days):
            pred = model.predict(x_input, verbose=0)[0][0]
            predictions.append(pred)
            x_input = np.append(x_input[:, 1:, :], [[[pred]]], axis=1)
        
        # Inverse transform
        forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        
        # Generate dates
        last_date = data[0].date
        future_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(days)]
        
        return list(zip(future_dates, forecast))
        
    finally:
        db.close()

def backtest_prediction(symbol, days_ago=7):
    """Backtest: Compare prediction made N days ago with actual price"""
    model, scaler = load_model_and_scaler(symbol)
    if model is None:
        return None
    
    db = SessionLocal()
    try:
        # Get data from days_ago + 60 days before
        cutoff_date = datetime.utcnow() - timedelta(days=days_ago + 60)
        prediction_date = datetime.utcnow() - timedelta(days=days_ago)
        
        historical_data = db.query(StockData).filter(
            StockData.symbol == symbol,
            StockData.date >= cutoff_date,
            StockData.date <= prediction_date
        ).order_by(StockData.date).all()
        
        if len(historical_data) < 60:
            return None
        
        # Make prediction using data from days_ago
        close_prices = np.array([d.close for d in historical_data[-60:]]).reshape(-1, 1)
        scaled_data = scaler.transform(close_prices)
        x_input = scaled_data[-60:].reshape(1, 60, 1)
        
        pred = model.predict(x_input, verbose=0)[0][0]
        predicted_price = scaler.inverse_transform([[pred]])[0][0]
        
        # Get actual price today
        actual_data = db.query(StockData).filter_by(symbol=symbol).order_by(
            StockData.date.desc()
        ).first()
        
        if not actual_data:
            return None
        
        actual_price = actual_data.close
        accuracy = 100 - abs((predicted_price - actual_price) / actual_price * 100)
        
        # Store in history
        history = PredictionHistory(
            symbol=symbol,
            prediction_date=prediction_date,
            predicted_price=predicted_price,
            actual_price=actual_price,
            accuracy=accuracy
        )
        db.add(history)
        db.commit()
        
        return {
            'symbol': symbol,
            'prediction_date': prediction_date.strftime('%Y-%m-%d'),
            'predicted_price': round(predicted_price, 2),
            'actual_price': round(actual_price, 2),
            'accuracy': round(accuracy, 2),
            'error': round(abs(predicted_price - actual_price), 2)
        }
        
    finally:
        db.close()

def get_model_accuracy(symbol):
    """Get average accuracy of past predictions"""
    db = SessionLocal()
    try:
        predictions = db.query(PredictionHistory).filter_by(symbol=symbol).all()
        if not predictions:
            return None
        
        accuracies = [p.accuracy for p in predictions if p.accuracy is not None]
        if not accuracies:
            return None
        
        return {
            'symbol': symbol,
            'avg_accuracy': round(sum(accuracies) / len(accuracies), 2),
            'total_predictions': len(predictions),
            'best_accuracy': round(max(accuracies), 2),
            'worst_accuracy': round(min(accuracies), 2)
        }
    finally:
        db.close()

if __name__ == '__main__':
    # Test backtesting
    result = backtest_prediction('RELIANCE', days_ago=7)
    if result:
        print(f"Backtest Result: {result}")
