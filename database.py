from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

Base = declarative_base()

# User Model
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    email = Column(String(120), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    portfolio = relationship('Portfolio', back_populates='user', cascade='all, delete-orphan')

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

# Portfolio Model
class Portfolio(Base):
    __tablename__ = 'portfolio'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    symbol = Column(String(20), nullable=False)
    exchange = Column(String(10), nullable=False)
    added_at = Column(DateTime, default=datetime.utcnow)
    user = relationship('User', back_populates='portfolio')

# Stock Data Model (Local Cache)
class StockData(Base):
    __tablename__ = 'stock_data'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    date = Column(DateTime, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    updated_at = Column(DateTime, default=datetime.utcnow)

# Stock Metrics (Pre-computed Technical Indicators)
class StockMetrics(Base):
    __tablename__ = 'stock_metrics'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), unique=True, nullable=False)
    current_price = Column(Float)
    market_cap = Column(Float)
    pe_ratio = Column(Float)
    pb_ratio = Column(Float)
    dividend_yield = Column(Float)
    week_52_high = Column(Float)
    week_52_low = Column(Float)
    rsi = Column(Float)  # Relative Strength Index
    macd = Column(Float)  # Moving Average Convergence Divergence
    score = Column(Float)  # Custom FinSight Score (0-100)
    updated_at = Column(DateTime, default=datetime.utcnow)

# Prediction History (Track Model Accuracy)
class PredictionHistory(Base):
    __tablename__ = 'prediction_history'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    prediction_date = Column(DateTime, nullable=False)
    predicted_price = Column(Float, nullable=False)
    actual_price = Column(Float)
    accuracy = Column(Float)  # Percentage accuracy
    created_at = Column(DateTime, default=datetime.utcnow)

# Database initialization
engine = create_engine('sqlite:///finsight.db', echo=False)
SessionLocal = sessionmaker(bind=engine)

def init_db():
    """Initialize database and create all tables"""
    Base.metadata.create_all(engine)
    print("[OK] Database initialized successfully")

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        return db
    finally:
        pass

if __name__ == '__main__':
    init_db()
