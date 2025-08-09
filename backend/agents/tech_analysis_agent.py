import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta, timezone
from agents.base_agent import BaseAgent
from database.models import SessionLocal, StockPrice

# Using pandas-ta as it's easier to install than TA-Lib
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    print("pandas-ta not found, using manual calculations")

class TechnicalAnalysisAgent(BaseAgent):
    def __init__(self, symbols: List[str]):
        super().__init__("TechnicalAnalysisAgent")
        self.symbols = symbols
        self.signals = {}
        self.current_indicators = {}
        
    def get_price_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get price data from database or fetch fresh data"""
        db = SessionLocal()
        try:
            # Get data from last N days
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            prices = db.query(StockPrice).filter(
                StockPrice.symbol == symbol,
                StockPrice.timestamp >= cutoff_date
            ).order_by(StockPrice.timestamp).all()
            
            if not prices:
                self.log_action(f"No price data found for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = pd.DataFrame([
                {
                    'timestamp': p.timestamp,
                    'close': p.price,
                    'volume': p.volume or 0
                } for p in prices
            ])
            
            data['timestamp'] = pd.to_datetime(data['timestamp']) # Converts the 'timestamp' column to proper datetime64[ns] format; important for time-based indexing and operations
            data.set_index('timestamp', inplace=True) # sets 'timestamp' as the index of the DataFrame.
            
            return data
            
        except Exception as e:
            self.log_action(f"Error fetching price data for {symbol}: {str(e)}")
            return pd.DataFrame()
        finally:
            db.close()

    # RSI (Relative Strength Index) to measures the speed and change of price movements
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        if HAS_PANDAS_TA:
            return ta.rsi(prices, length=period)
        
        # Manual RSI calculation
        delta = prices.diff() # calculates the difference between consecutive prices
        # Keeps positive differences (gain) or negative differences (loss) and sets all others to 0.
        # Then takes a rolling mean over the chosen period. This gives average gains
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss # Relative Strength (RS) is the ratio of average gain to average loss
        rsi = 100 - (100 / (1 + rs)) # RSI formula. Results range from 0 to 100.

        # common interpretation:
        # RSI below 30 is considered oversold (potential buy signal)
        # RSI above 70 is considered overbought (potential sell signal)
        return rsi
    
    # MACD (Moving Average Convergence Divergence) is a trend-following momentum indicator
    def calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if HAS_PANDAS_TA:
            macd_data = ta.macd(prices)
            return macd_data['MACD_12_26_9'], macd_data['MACDs_12_26_9'], macd_data['MACDh_12_26_9']
        
        # Manual MACD calculation
        exp1 = prices.ewm(span=12).mean() # 12 period Exponential Moving Average (EMA)
        exp2 = prices.ewm(span=26).mean() # 26 period Exponential Moving Average (EMA)
        macd = exp1 - exp2 # MACD line is the difference between the two EMAs
        signal = macd.ewm(span=9).mean() # Signal line is the 9 period EMA of the MACD line
        histogram = macd - signal # MACD histogram is the difference between MACD line and Signal line. Shows divergence/convergence.
        
        return macd, signal, histogram
    
    def calculate_moving_averages(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate various moving averages"""
        return {
            'sma_20': prices.rolling(window=20).mean(),
            'sma_50': prices.rolling(window=50).mean(),
            'ema_12': prices.ewm(span=12).mean(),
            'ema_26': prices.ewm(span=26).mean()
        }
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        return {
            'upper': sma + (std * 2),
            'middle': sma,
            'lower': sma - (std * 2)
        }
    
    def analyze_symbol(self, symbol: str) -> Dict:
        """Perform complete technical analysis for a symbol"""
        data = self.get_price_data(symbol, days=60)  # Get more data for better indicators
        
        if data.empty or len(data) < 26:  # Need at least 26 days for MACD
            return {
                'symbol': symbol,
                'signal': 'INSUFFICIENT_DATA',
                'confidence': 0.0,
                'indicators': {},
                'reasoning': 'Not enough price data for analysis'
            }
        
        prices = data['close']
        current_price = prices.iloc[-1]
        
        # Calculate all indicators
        rsi = self.calculate_rsi(prices)
        macd, macd_signal, macd_histogram = self.calculate_macd(prices)
        moving_averages = self.calculate_moving_averages(prices)
        bollinger = self.calculate_bollinger_bands(prices)
        
        # Get current values (most recent)
        current_rsi = rsi.iloc[-1] if not rsi.empty else 50
        current_macd = macd.iloc[-1] if not macd.empty else 0
        current_macd_signal = macd_signal.iloc[-1] if not macd_signal.empty else 0
        current_macd_histogram = macd_histogram.iloc[-1] if not macd_histogram.empty else 0
        
        # Store current indicators
        self.current_indicators[symbol] = {
            'rsi': current_rsi,
            'macd': current_macd,
            'macd_signal': current_macd_signal,
            'macd_histogram': current_macd_histogram,
            'price': current_price,
            'sma_20': moving_averages['sma_20'].iloc[-1] if len(moving_averages['sma_20']) > 0 else current_price,
            'sma_50': moving_averages['sma_50'].iloc[-1] if len(moving_averages['sma_50']) > 0 else current_price,
            'bb_upper': bollinger['upper'].iloc[-1] if len(bollinger['upper']) > 0 else current_price,
            'bb_lower': bollinger['lower'].iloc[-1] if len(bollinger['lower']) > 0 else current_price,
        }
        
        # Generate trading signal
        signal_analysis = self.generate_signal(symbol, self.current_indicators[symbol])
        
        return signal_analysis
    
    def generate_signal(self, symbol: str, indicators: Dict) -> Dict:
        """Generate buy/sell/hold signal based on technical indicators"""
        signals = []
        reasons = []
        
        # RSI Analysis
        rsi = indicators['rsi']
        if rsi < 30:
            signals.append('BUY')
            reasons.append(f'RSI oversold ({rsi:.1f})')
        elif rsi > 70:
            signals.append('SELL')
            reasons.append(f'RSI overbought ({rsi:.1f})')
        else:
            signals.append('NEUTRAL')
        
        # MACD Analysis
        macd = indicators['macd']
        macd_signal = indicators['macd_signal']
        macd_histogram = indicators['macd_histogram']
        
        if macd > macd_signal and macd_histogram > 0:
            signals.append('BUY')
            reasons.append('MACD bullish crossover')
        elif macd < macd_signal and macd_histogram < 0:
            signals.append('SELL')
            reasons.append('MACD bearish crossover')
        else:
            signals.append('NEUTRAL')
        
        # Moving Average Analysis
        price = indicators['price']
        sma_20 = indicators['sma_20']
        sma_50 = indicators['sma_50']
        
        if price > sma_20 > sma_50:
            signals.append('BUY')
            reasons.append('Price above moving averages (bullish trend)')
        elif price < sma_20 < sma_50:
            signals.append('SELL')
            reasons.append('Price below moving averages (bearish trend)')
        else:
            signals.append('NEUTRAL')
        
        # Bollinger Bands Analysis
        bb_upper = indicators['bb_upper']
        bb_lower = indicators['bb_lower']
        
        if price <= bb_lower:
            signals.append('BUY')
            reasons.append('Price at lower Bollinger Band (oversold)')
        elif price >= bb_upper:
            signals.append('SELL')
            reasons.append('Price at upper Bollinger Band (overbought)')
        else:
            signals.append('NEUTRAL')
        
        # Combine signals
        buy_count = signals.count('BUY')
        sell_count = signals.count('SELL')
        neutral_count = signals.count('NEUTRAL')
        
        total_signals = len(signals)
        
        # Determine overall signal
        if buy_count >= 3:
            overall_signal = 'STRONG_BUY'
            confidence = min(0.9, buy_count / total_signals)
        elif buy_count >= 2:
            overall_signal = 'BUY'
            confidence = buy_count / total_signals
        elif sell_count >= 3:
            overall_signal = 'STRONG_SELL'
            confidence = min(0.9, sell_count / total_signals)
        elif sell_count >= 2:
            overall_signal = 'SELL'
            confidence = sell_count / total_signals
        else:
            overall_signal = 'HOLD'
            confidence = neutral_count / total_signals
        
        return {
            'symbol': symbol,
            'signal': overall_signal,
            'confidence': round(confidence, 2),
            'indicators': indicators,
            'reasoning': '; '.join(reasons),
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'neutral_signals': neutral_count
        }
    
    def process(self) -> Dict:
        """Main processing method - analyze all symbols"""
        results = {}
        
        self.log_action("Starting technical analysis for all symbols")
        
        for symbol in self.symbols:
            try:
                analysis = self.analyze_symbol(symbol)
                results[symbol] = analysis
                self.signals[symbol] = analysis
                
                # Log the signal
                signal = analysis['signal']
                confidence = analysis['confidence']
                reasoning = analysis['reasoning']
                
                self.log_action(
                    f"{symbol}: {signal} (confidence: {confidence:.1%})",
                    {"reasoning": reasoning}
                )
                
                # Publish signal event if it's actionable
                if signal in ['STRONG_BUY', 'STRONG_SELL', 'BUY', 'SELL'] and confidence > 0.6:
                    self.publish_event("TECHNICAL_SIGNAL", {
                        'symbol': symbol,
                        'signal': signal,
                        'confidence': confidence,
                        'reasoning': reasoning,
                        'indicators': analysis['indicators']
                    })
                
            except Exception as e:
                self.log_action(f"Error analyzing {symbol}: {str(e)}")
                results[symbol] = {
                    'symbol': symbol,
                    'signal': 'ERROR',
                    'confidence': 0.0,
                    'error': str(e)
                }
        
        self.last_update = datetime.now(timezone.utc)
        return results
    
    def get_current_signals(self) -> Dict:
        """Get the most recent signals for all symbols"""
        return self.signals.copy()
    
    def get_signal_summary(self) -> Dict:
        """Get a summary of current signals"""
        if not self.signals:
            return {'total': 0, 'buy': 0, 'sell': 0, 'hold': 0}
        
        summary = {'total': len(self.signals), 'buy': 0, 'sell': 0, 'hold': 0}
        
        for symbol, analysis in self.signals.items():
            signal = analysis['signal']
            if signal in ['BUY', 'STRONG_BUY']:
                summary['buy'] += 1
            elif signal in ['SELL', 'STRONG_SELL']:
                summary['sell'] += 1
            else:
                summary['hold'] += 1
        
        return summary