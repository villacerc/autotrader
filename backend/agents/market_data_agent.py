import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
from agents.base_agent import BaseAgent
from database.models import StockPrice, SessionLocal

class MarketDataAgent(BaseAgent):
    def __init__(self, symbols: List[str]):
        super().__init__("MarketDataAgent")
        self.symbols = symbols
        self.current_prices = {}
        self.price_history = {}
        
    def fetch_prices(self, interval: str = None, period: str = "1d") -> Dict[str, float]:
        """Fetch current prices for all symbols"""
        try:
            # Create a space-separated string of symbols for yfinance
            symbols_str = " ".join(self.symbols)
            
            # Use yfinance library to download data in intervals, returns a pandas DataFrame
            tickers = yf.download(symbols_str, period=period, interval=interval, progress=False, auto_adjust=True)
            
            current_prices = {}
            
            # yfinance structure changes based on how many symbols you request:
            if len(self.symbols) == 1:
                # Single symbol case
                if not tickers.empty:
                    current_prices[self.symbols[0]] = float(tickers['Close'].iloc[-1])
            else:
                # Multiple symbols case
                for symbol in self.symbols:
                    if symbol in tickers['Close'].columns:
                        latest_price = tickers['Close'][symbol].dropna().iloc[-1]
                        current_prices[symbol] = float(latest_price)
            
            self.current_prices = current_prices
            self.log_action(f"Fetched prices: {current_prices}")
            return current_prices
            
        except Exception as e:
            self.log_action(f"Error fetching prices: {str(e)}")
            return {}
    
    def store_prices_to_db(self, prices: Dict[str, float]):
        """Store prices in database"""
        db = SessionLocal()
        try:
            for symbol, price in prices.items():
                db_price = StockPrice(
                    symbol=symbol,
                    price=price,
                    timestamp=datetime.utcnow()
                )
                db.add(db_price)
            db.commit()
            self.log_action(f"Stored {len(prices)} prices to database")
        except Exception as e:
            self.log_action(f"Database error: {str(e)}")
            db.rollback()
        finally:
            db.close()
    
    def get_price_history(self, symbol: str, days: int = 10) -> pd.DataFrame:
        """Get historical price data for technical analysis"""
        try:
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            hist = ticker.history(start=start_date, end=end_date)
            self.price_history[symbol] = hist
            
            self.log_action(f"Fetched {len(hist)} days of history for {symbol}")
            return hist
            
        except Exception as e:
            self.log_action(f"Error fetching history for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def is_market_open(self) -> bool:
        """Simple market hours check (US Eastern Time)"""
        now = datetime.now()
        # Simple check: weekday and between 9:30 AM and 4:00 PM
        if now.weekday() >= 5:  # Weekend
            return False
        
        hour = now.hour
        return 9 <= hour < 16  # Simplified market hours
    
    def process(self) -> Dict[str, Any]:
        """Main processing method"""
        if not self.is_market_open():
            self.log_action("Market is closed, fetching last available prices")
            if not self.current_prices:
                self.current_prices = self.fetch_prices(interval="1m", period="1d")
            return {"status": "market_closed", "prices": self.current_prices}
        
        # Fetch current prices
        prices = self.fetch_prices(interval="1m", period="1d")
        
        if prices:
            # Store to database
            self.store_prices_to_db(prices)
            
            # Publish price update event
            self.publish_event("PRICE_UPDATE", {
                "prices": prices,
                "timestamp": datetime.now(datetime.timezone.utc).isoformat()
            })
            
            self.last_update = datetime.now(datetime.timezone.utc)
            return {"status": "success", "prices": prices}
        else:
            return {"status": "error", "prices": {}}