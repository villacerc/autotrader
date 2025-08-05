import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Tuple
from agents.base_agent import BaseAgent
from database.models import StockPrice, SessionLocal

class MarketDataAgent(BaseAgent):
    def __init__(self, symbols: List[str]):
        super().__init__("MarketDataAgent")
        self.symbols = symbols
        self.current_prices = {}
        self.price_history = {}
        
    def fetch_current_prices(self) -> Dict[str, List[Tuple[float, datetime]]]:
        """Fetch the latest price and timestamp for all symbols"""
        try:
            symbols_str = " ".join(self.symbols)
            tickers = yf.download(
                symbols_str, period="1d", interval="1m", progress=False, auto_adjust=True
            )

            if tickers.empty:
                self.log_action("No data returned from yfinance (current)")
                return {}

            current_prices = {}

            if len(self.symbols) == 1:
                # Single symbol
                price = float(tickers['Close'].iloc[-1])
                timestamp = tickers.index[-1].to_pydatetime()
                current_prices[self.symbols[0]] = [(price, timestamp)]
            else:
                # Multiple symbols
                for symbol in self.symbols:
                    if symbol in tickers['Close'].columns:
                        close_series = tickers['Close'][symbol].dropna()
                        if not close_series.empty:
                            price = float(close_series.iloc[-1])
                            timestamp = close_series.index[-1].to_pydatetime()
                            current_prices[symbol] = [(price, timestamp)]

            self.current_prices = current_prices
            self.log_action(f"Fetched current prices: {current_prices}")
            return current_prices

        except Exception as e:
            self.log_action(f"Error fetching current prices: {str(e)}")
            return {}
    
    def store_prices_to_db(self, prices: Dict[str, List[Tuple[float, datetime]]]):
        """Store multiple prices per symbol in the database"""
        db = SessionLocal()
        try:
            for symbol, price_list in prices.items():
                for price, timestamp in price_list:
                    db_price = StockPrice(
                        symbol=symbol,
                        price=price,
                        timestamp=timestamp
                    )
                    db.add(db_price)
            db.commit()
            self.log_action(f"Stored prices to database")
        except Exception as e:
            self.log_action(f"Database error: {str(e)}")
            db.rollback()
        finally:
            db.close()

    def fetch_historical_prices(self, period: str = "60d", interval: str = "1d") -> Dict[str, List[Tuple[float, datetime]]]:
        """Fetch historical close prices for all symbols"""
        try:
            symbols_str = " ".join(self.symbols)
            tickers = yf.download(
                symbols_str, period=period, interval=interval, progress=False, auto_adjust=True
            )

            if tickers.empty:
                self.log_action("No data returned from yfinance (historical)")
                return {}

            historical_prices = {}

            if len(self.symbols) == 1:
                symbol = self.symbols[0]
                series = tickers['Close'].dropna()
                historical_prices[symbol] = [
                    (float(price), ts.to_pydatetime()) for ts, price in series.items()
                ]
            else:
                for symbol in self.symbols:
                    if symbol in tickers['Close'].columns:
                        series = tickers['Close'][symbol].dropna()
                        if not series.empty:
                            historical_prices[symbol] = [
                                (float(price), ts.to_pydatetime()) for ts, price in series.items()
                            ]

            self.log_action(f"Fetched historical prices for {len(historical_prices)} symbols")
            return historical_prices

        except Exception as e:
            self.log_action(f"Error fetching historical prices: {str(e)}")
            return {}

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
                self.current_prices = self.fetch_historical_prices()
                self.store_prices_to_db(self.current_prices)
            return {"status": "market_closed", "prices": self.current_prices}
        
        # Fetch current prices
        prices = self.fetch_current_prices()
        
        if prices:
            # Store to database
            self.store_prices_to_db(prices)
            
            # Publish price update event
            self.publish_event("PRICE_UPDATE", {
                "prices": prices,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            self.last_update = datetime.now(timezone.utc)
            return {"status": "success", "prices": prices}
        else:
            return {"status": "error", "prices": {}}