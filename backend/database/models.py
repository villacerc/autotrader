from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, create_engine
from sqlalchemy.ext.declarative import declarative_base # for defining ORM models
from sqlalchemy.orm import sessionmaker # for creating database sessions 
from datetime import datetime, timezone

Base = declarative_base() # Base class for ORM models

class StockPrice(Base): # Represents stock price data, inherits from Base
    __tablename__ = "stock_prices" # Table name in the database
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    timestamp = Column(DateTime, default=datetime.now(timezone.utc))
    price = Column(Float, nullable=False)
    volume = Column(Integer, default=0)
    
class Trade(Base):
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    action = Column(String(4), nullable=False)  # BUY/SELL
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    agent_reason = Column(String(200))

class Portfolio(Base):
    __tablename__ = "portfolio"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, unique=True)
    quantity = Column(Integer, default=0)
    avg_cost = Column(Float, default=0.0)
    last_updated = Column(DateTime, default=datetime.utcnow)

# Database setup
DATABASE_URL = "sqlite:///./portfolio.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    Base.metadata.create_all(bind=engine)