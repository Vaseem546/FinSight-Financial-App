from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import os

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
    __table_args__ = (UniqueConstraint('symbol', 'date', name='uq_stock_symbol_date'),)
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
    book_value = Column(Float)
    dividend_yield = Column(Float)
    week_52_high = Column(Float)
    week_52_low = Column(Float)
    rsi = Column(Float)  # Relative Strength Index
    macd = Column(Float)  # Moving Average Convergence Divergence
    score = Column(Float)  # Custom FinSight Score (0-100)
    updated_at = Column(DateTime, default=datetime.utcnow)

# Database initialization
database_path = os.environ.get('DATABASE_PATH')
if not database_path and os.environ.get('VERCEL'):
    database_path = '/tmp/finsight.db'
database_path = database_path or 'finsight.db'
engine = create_engine(f'sqlite:///{database_path}', echo=False)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db():
    """Initialize database and create or migrate all tables."""
    Base.metadata.create_all(engine)
    with engine.connect() as connection:
        columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(stock_metrics)").fetchall()}
        if 'book_value' not in columns:
            connection.exec_driver_sql("ALTER TABLE stock_metrics ADD COLUMN book_value FLOAT")
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
